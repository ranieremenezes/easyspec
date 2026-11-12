"""
easyspec gemini.gmos module.

Converts raw data of the Gemini GMOS-N and GMOS-S instruments into images that can be reduced with easyspec.cleaning() (imaging and
spectroscopy) and easyspec.extraction() (spectroscopy).

Why a converter? A raw GMOS frame is not a single image: each of the 3 CCDs is read through 1, 2 or 4 amplifiers, and each amplifier is
stored in a separate fits extension, with its own overscan region, bias level, gain and read noise. The three CCDs are separated by
gaps of ~3 arcsec. In addition, the calibrations of a GMOS program (bias, flats, arcs, standard star) come all together in the same
directory as the science frames, with different regions of interest (ROI), central wavelengths and dither positions.

For each raw frame, this module:

1. subtracts the overscan level of each amplifier (row by row) and removes the overscan regions;
2. converts each amplifier from ADU to electrons with its own gain (from the tables of the Gemini DRAGONS pipeline);
3. assembles the amplifiers and the three CCDs into a single image, with the gaps between the CCDs filled with NaN (so that the image
   has the true geometry of the focal plane), and with the pixels of the static bad pixel mask (if given) also set to NaN;
4. writes a single-extension fits file with a header that easyspec understands (GAIN, RDNOISE, EXPTIME, AIRMASS, DATE-OBS, FILTER,
   SATURATE, approximate wavelength range for spectroscopy, WCS for imaging...).

The converted frames are then sorted into one directory per instrument setup (e.g. one per filter in imaging, one per grating + central
wavelength + slit in spectroscopy), each with the subdirectories expected by cleaning.data_paths(): bias, flat, lamp, standard_star_NAME
and target_NAME (one target directory per spatial dither position in spectroscopy).

Typical use::

    from easyspec.gemini import gmos
    from easyspec.cleaning import cleaning

    setups = gmos.convert("raw_gemini_data", "gmos_converted")
    c = cleaning()
    raw = c.data_paths(**setups["longslit_B600_515nm_1.0arcsec_2x2"])

Supported: imaging and long-slit spectroscopy (tested with real data). MOS and IFU frames are also assembled and sorted (the format of
the detector is the same), but cutting the individual slitlets of a MOS mask or extracting the fibres of the IFU is not done here. The
conversion is not a replacement of DRAGONS: the CCDs are placed with a precision of one pixel (no resampling: the small rotations
between the CCDs, < 0.05 degrees, are ignored), which is what a reduction of point sources and long-slit spectra needs.

The detector tables come from DRAGONS (see easyspec/gemini/gemini_gmos_lookup.py). If you use this module, please cite DRAGONS (Labrie et al.
2023, RNAAS 7, 214).
"""

from pathlib import Path
import bz2
import gzip
import io
import re
import warnings

import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from . import gemini_gmos_lookup as lookup


MODULE_VERSION = 1
CUAR_LINE_LIST = str(Path(__file__).parent/"gemini_gmos_CuAr.dat")   # CuAr lines of the GMOS arc lamp, for identify_lamp_lines()
CCD_WIDTH = 2048                      # unbinned pixels, all GMOS detectors
ADC_LIMIT = 65535                     # ADU
RAW_EXTENSIONS = (".fits", ".fit", ".fits.bz2", ".fits.gz", ".fits.fz")

# Dispersion of the gratings: ~30.1/(lines per mm) nm per unbinned pixel of 15 microns (Hamamatsu CCDs)
DISPERSION_CONSTANT = 30.1
PIXEL_SIZE_RATIO = 13.5/15.0          # e2v and EEV CCDs (13.5 micron pixels)

# Header keywords copied from the first extension to the header of the converted frame are not needed: everything else comes from the
# primary header. These keywords of the primary header would be wrong for the converted image:
DROPPED_KEYWORDS = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO", "BSCALE", "NEXTEND", "CHECKSUM", "DATASUM")


# ======================================================================================================================
# Small helpers
# ======================================================================================================================

def _parse_section(section):

    """'[x1:x2,y1:y2]' (1-based, inclusive) -> (x1, x2, y1, y2) as 0-based, half-open limits."""

    numbers = [int(value) for value in re.findall(r"-?\d+", str(section))]
    if len(numbers) != 4:
        raise ValueError(f"Invalid image section: {section}")
    return numbers[0] - 1, numbers[1], numbers[2] - 1, numbers[3]


def _clean_name(name):

    """A name that is safe in file and directory names."""

    name = re.sub(r"[^A-Za-z0-9+\-.]+", "_", str(name).strip())
    return name.strip("_") or "unknown"


def _component(name):

    """'B600+_G5323' -> 'B600', 'i_G0302' -> 'i': the name of a GMOS filter or grating without its Gemini identifier."""

    return re.sub(r"\+$", "", str(name).split("_G")[0])


def _image_extensions(hdul):

    """The extensions of a raw GMOS file that hold the amplifiers (2D images with a DETSEC keyword)."""

    return [hdu for hdu in hdul[1:] if hdu.header.get("NAXIS", 0) == 2 and "DETSEC" in hdu.header]


def _open(file_name):

    """
    Open a raw file. The compressed files (.bz2, .gz) are decompressed in memory only once (reading the extensions of a compressed
    file one by one is very slow).
    """

    name = str(file_name).lower()
    if name.endswith(".bz2"):
        return fits.open(io.BytesIO(bz2.decompress(Path(file_name).read_bytes())))
    if name.endswith(".gz"):
        return fits.open(io.BytesIO(gzip.decompress(Path(file_name).read_bytes())))
    return fits.open(file_name)


def _read_headers(file_name):

    """Primary header and headers of the amplifier extensions, reading only the beginning of a compressed file when possible."""

    name = str(file_name).lower()
    if name.endswith(".bz2") or name.endswith(".gz"):
        opener = bz2.open if name.endswith(".bz2") else gzip.open
        with opener(file_name, "rb") as stream:     # sequential read: no data is loaded
            with fits.open(stream) as hdul:
                names = [hdu.name.upper() for hdu in hdul[1:]]
                return hdul[0].header.copy(), [hdu.header.copy() for hdu in _image_extensions(hdul)], names
    with fits.open(file_name) as hdul:
        names = [hdu.name.upper() for hdu in hdul[1:]]
        return hdul[0].header.copy(), [hdu.header.copy() for hdu in _image_extensions(hdul)], names


def _is_raw_file(path):

    name = path.name.lower()
    return path.is_file() and any(name.endswith(extension) for extension in RAW_EXTENSIONS)


def _stem(path):

    """File name without .fits, .fits.bz2, etc."""

    name = Path(path).name
    for extension in sorted(RAW_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(extension):
            return name[:-len(extension)]
    return Path(name).stem


# ======================================================================================================================
# Description of a raw frame
# ======================================================================================================================

def describe_frame(file_name):

    """
    Read the headers of a raw GMOS file and describe the frame: what it is (bias, flat, lamp, standard star, target...), the observing
    mode (imaging, long slit, MOS, IFU) and the instrument setup.

    Parameters
    ----------
    file_name: string or Path
        A raw GMOS file (.fits, .fits.bz2 or .fits.gz).

    Returns
    -------
    frame: dict
        "file", "category" ("bias", "dark", "flat", "twilight", "lamp", "standard_star", "target", "acquisition", "bad_pixel_mask",
        "mask_definition" or "unknown"), "mode" ("imaging", "longslit", "mos", "ifu"), "object", "instrument", "detector", "filter",
        "grating", "central_wavelength" (nm), "mask", "binning" (x, y), "roi" (x1, x2, y1, y2: the region of the detector that was read,
        in unbinned pixels, 0-based, half-open), "read_mode" (speed, gain setting), "exposure", "airmass", "date" (UT start, ISO),
        "p_offset" and "q_offset" (telescope offsets in arcsec, across and along the slit), "nod_and_shuffle" (True or False) and
        "observation_class". None if the file is not a GMOS frame.
    """

    file_name = Path(file_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primary, headers, extension_names = _read_headers(file_name)
    if not str(primary.get("INSTRUME", "")).upper().startswith("GMOS"):
        return None
    if len(headers) == 0:
        if str(primary.get("OBSTYPE", "")).upper() == "MASK" or "MDF" in extension_names:
            return {"file": str(file_name), "category": "mask_definition", "mode": None, "object": str(primary.get("OBJECT", ""))}
        return None

    sections = np.array([_parse_section(header["DETSEC"]) for header in headers])
    roi = (int(sections[:, 0].min()), int(sections[:, 1].max()), int(sections[:, 2].min()), int(sections[:, 3].max()))
    binning = tuple(int(value) for value in str(headers[0].get("CCDSUM", "1 1")).split()[:2])

    detector = str(primary.get("DETECTOR", ""))
    hamamatsu = "hamamatsu" in detector.lower() or str(primary.get("DETTYPE", "")).startswith("S10892")
    integration = primary.get("AMPINTEG")
    speed = None if integration is None else ("slow" if integration > (8000 if hamamatsu else 2000) else "fast")
    header_gain = headers[0].get("GAIN")
    gain_setting = None if header_gain is None else ("low" if float(header_gain) < 3.0 else "high")

    filters = [_component(primary.get(key, "")) for key in ("FILTER1", "FILTER2")]
    filters = [name for name in filters if name and not name.lower().startswith("open")]
    grating = _component(primary.get("GRATING", "MIRROR"))
    mask = str(primary.get("MASKNAME", "None")).strip()
    obstype = str(primary.get("OBSTYPE", "")).upper()
    obsclass = str(primary.get("OBSCLASS", "")).strip()
    target = str(primary.get("OBJECT", "")).strip()

    # ---- Observing mode
    if grating.upper() == "MIRROR":
        mode = "imaging"
    elif mask.upper().startswith("IFU"):
        mode = "ifu"
    elif "arcsec" in mask.lower():
        mode = "longslit"
    elif mask.lower() in ("none", ""):
        mode = "slitless"
    else:
        mode = "mos"
    nod_and_shuffle = mask.upper().startswith("NS") or mask.upper().startswith("IFU-NS") or "NODPIX" in primary

    # ---- What the frame is
    if obstype == "BPM":
        category = "bad_pixel_mask"
    elif obstype == "BIAS":
        category = "bias"
    elif obstype == "DARK":
        category = "dark"
    elif obstype == "ARC":
        category = "lamp"
    elif obstype == "FLAT":
        category = "flat"
    elif obstype == "OBJECT":
        if "twilight" in target.lower():
            category = "flat" if mode == "imaging" else "twilight"
        elif obsclass.lower() in ("acq", "acqcal"):
            category = "acquisition"
        elif obsclass.lower() in ("partnercal", "progcal"):
            category = "standard_star"
        elif obsclass.lower() == "daycal":
            category = "unknown"
        else:
            category = "target"
    else:
        category = "unknown"

    date = str(primary.get("DATE-OBS", "")).strip()
    clock = str(primary.get("UTSTART", primary.get("TIME-OBS", primary.get("UT", "")))).strip()
    if date and clock and "T" not in date:
        date = f"{date}T{clock}"

    def number(key, default=np.nan):
        try:
            return float(primary.get(key, default))
        except (TypeError, ValueError):
            return default

    return {"file": str(file_name), "category": category, "mode": mode, "object": target, "instrument": str(primary.get("INSTRUME", "")),
            "detector": str(primary.get("DETID", detector)).strip(), "detector_name": detector, "filter": "+".join(filters) or "open",
            "grating": grating, "central_wavelength": number("CENTWAVE", number("GRWLEN")), "mask": mask, "binning": binning,
            "roi": roi, "read_mode": (speed, gain_setting), "exposure": number("EXPTIME"), "airmass": number("AIRMASS"), "date": date,
            "p_offset": number("POFFSET", 0.0), "q_offset": number("QOFFSET", 0.0), "nod_and_shuffle": bool(nod_and_shuffle),
            "observation_class": obsclass}


def inventory(input_directory, verbose=True):

    """
    Table with all the GMOS frames of a directory (searched recursively) and what each one is. Use it to check your data before
    converting them.

    Parameters
    ----------
    input_directory: string or Path
        Directory with the raw GMOS files (.fits, .fits.bz2 or .fits.gz).
    verbose: boolean
        If True, the table is printed.

    Returns
    -------
    table: astropy.table.Table
        One row per frame, with the columns of describe_frame().
    """

    frames = _collect_frames(input_directory)
    table = _frames_table(frames)
    if verbose:
        table["file", "category", "mode", "object", "filter", "grating", "central_wavelength", "mask", "binning", "rows", "exposure",
              "q_offset"].pprint(max_lines=-1, max_width=-1)
    return table


def _collect_frames(input_directory):

    input_directory = Path(input_directory)
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_directory}")
    frames = []
    for path in sorted(input_directory.rglob("*")):
        if not _is_raw_file(path):
            continue
        try:
            frame = describe_frame(path)
        except Exception as error:   # a damaged or foreign file must not stop the conversion of the others
            warnings.warn(f"{path.name} could not be read ({error}): skipped.")
            continue
        if frame is not None:
            frames.append(frame)
    if len(frames) == 0:
        raise RuntimeError(f"No GMOS frames found in {input_directory}.")
    return frames


def _frames_table(frames):

    rows = []
    for frame in frames:
        if frame["category"] == "mask_definition":
            continue
        rows.append({"file": Path(frame["file"]).name, "category": frame["category"], "mode": frame["mode"], "object": frame["object"],
                     "filter": frame["filter"], "grating": frame["grating"], "central_wavelength": frame["central_wavelength"],
                     "mask": frame["mask"], "binning": f"{frame['binning'][0]}x{frame['binning'][1]}",
                     "rows": f"{frame['roi'][2] + 1}-{frame['roi'][3]}", "exposure": frame["exposure"], "airmass": frame["airmass"],
                     "p_offset": round(frame["p_offset"], 2), "q_offset": round(frame["q_offset"], 2), "date": frame["date"],
                     "read_mode": "-".join(str(value) for value in frame["read_mode"]),
                     "observation_class": frame["observation_class"]})
    return Table(rows=rows)


# ======================================================================================================================
# Detector properties
# ======================================================================================================================

def amplifier_properties(amplifier, date, read_mode, header_gain=None, header_read_noise=None):

    """
    Gain (electrons/ADU) and read noise (electrons) of a GMOS amplifier, from the tables of DRAGONS. The GAIN and RDNOISE keywords of the
    raw files are only nominal values, and must not be used unless the amplifier is not in the tables (then a warning is issued).

    Parameters
    ----------
    amplifier: string
        Name of the amplifier (AMPNAME keyword), e.g. "BI5-36-4k-2, 4".
    date: string
        UT date of the observation (ISO, e.g. "2017-10-22").
    read_mode: tuple
        (read-out speed, gain setting), e.g. ("slow", "low").
    header_gain, header_read_noise: float
        Nominal values of the header, used if the amplifier is not in the tables.

    Returns
    -------
    gain, read_noise: float
    from_table: boolean
        False if the nominal values of the header had to be used.
    """

    day = str(date)[:10]
    for start, table in lookup.AMPLIFIER_EPOCHS:
        if day >= start:
            values = table.get(str(amplifier).strip(), {}).get(tuple(read_mode))
            if values is not None and values[0] and values[0] > 0:
                read_noise = values[1] if values[1] and values[1] > 0 else header_read_noise
                return float(values[0]), (float(read_noise) if read_noise is not None else np.nan), True
            break
    gain = float(header_gain) if header_gain is not None else 1.0
    read_noise = float(header_read_noise) if header_read_noise is not None else np.nan
    return gain, read_noise, False


def _ccd_offsets(detector, binning):

    """
    Position (x, y), in binned pixels, of the origin of each of the three CCDs in the assembled image, rounded to the nearest pixel.
    """

    geometry = lookup.CCD_GEOMETRY.get(detector)
    if geometry is None:
        key = detector.replace(", ", ",")
        geometry = lookup.CCD_GEOMETRY.get(key)
    if geometry is None:
        gap = lookup.CCD_GAPS.get(detector, 37 if "EEV" in detector.upper() or "E2V" in detector.upper() else 61)
        geometry = [(-(CCD_WIDTH + gap), 0.0, 0.0), (CCD_WIDTH + gap, 0.0, 0.0)]
    (left_x, left_y, _), (right_x, right_y, _) = geometry
    x_bin, y_bin = binning
    middle = abs(left_x)
    x_origins = [0.0, middle, middle + right_x]
    y_origins = [left_y, 0.0, right_y]
    return [(int(round(x/x_bin)), int(round(y/y_bin))) for x, y in zip(x_origins, y_origins)]


def _overscan_level(overscan, method):

    """
    Bias level of each row of an amplifier, from its overscan region (rows x columns). "row": the median of each row, smoothed along the
    rows with a running median that also rejects the rows contaminated by saturated sources; "constant": a single value.
    """

    rows = np.nanmedian(overscan, axis=1)
    if method == "constant" or len(rows) < 16:
        return np.full(len(rows), np.nanmedian(rows))
    window = min(31, 2*(len(rows)//4) + 1)
    smooth = ndimage.median_filter(rows, size=window, mode="nearest")
    scatter = 1.4826*np.nanmedian(np.abs(rows - smooth))
    contaminated = np.abs(rows - smooth) > 5*max(scatter, 0.5)
    rows = np.where(contaminated, smooth, rows)
    # light smoothing: the noise of the median of a row is ~read noise/sqrt(columns), smaller than the noise of a pixel
    return ndimage.uniform_filter1d(ndimage.median_filter(rows, size=5, mode="nearest"), size=5, mode="nearest")


# ======================================================================================================================
# Assembling one frame
# ======================================================================================================================

def _layout(headers, shapes, detector, crop=None):

    """
    Where each amplifier goes in the assembled image. Returns, for each extension, the slices of its data region to use and the
    slices of the assembled image that receive it, and the shape of the assembled image. 'crop' = (y1, y2) restricts the rows of the
    detector (unbinned, 0-based, half-open).
    """

    x_bin, y_bin = (int(value) for value in str(headers[0].get("CCDSUM", "1 1")).split()[:2])
    offsets = _ccd_offsets(detector, (x_bin, y_bin))
    pieces = []
    for header, shape in zip(headers, shapes):
        dx1, dx2, dy1, dy2 = _parse_section(header["DETSEC"])
        if "DATASEC" in header:
            sx1, sx2, sy1, sy2 = _parse_section(header["DATASEC"])
        else:
            sx1, sx2, sy1, sy2 = 0, shape[1], 0, shape[0]
        if crop is not None:   # rows of the detector to keep
            low, high = max(dy1, crop[0]), min(dy2, crop[1])
            if high <= low:
                continue
            sy1, sy2 = sy1 + (low - dy1)//y_bin, sy1 + (high - dy1)//y_bin
            dy1, dy2 = low, high
        ccd = min(dx1//CCD_WIDTH, 2)
        x = offsets[ccd][0] + (dx1 - ccd*CCD_WIDTH)//x_bin
        y = offsets[ccd][1] + dy1//y_bin
        pieces.append({"source": (slice(sy1, sy2), slice(sx1, sx2)), "x": x, "y": y, "ccd": ccd,
                       "ny": sy2 - sy1, "nx": sx2 - sx1})
    if len(pieces) == 0:
        raise ValueError("The requested rows are outside the region of the detector that was read.")
    # Rows covered by all the CCDs (the CCDs may be shifted by a few rows with respect to each other):
    bottom = max(min(piece["y"] for piece in pieces if piece["ccd"] == ccd) for ccd in {piece["ccd"] for piece in pieces})
    top = min(max(piece["y"] + piece["ny"] for piece in pieces if piece["ccd"] == ccd) for ccd in {piece["ccd"] for piece in pieces})
    left = min(piece["x"] for piece in pieces)
    right = max(piece["x"] + piece["nx"] for piece in pieces)
    # Columns of the gaps between the CCDs
    present = sorted({piece["ccd"] for piece in pieces})
    edges = {ccd: (min(piece["x"] for piece in pieces if piece["ccd"] == ccd) - left,
                   max(piece["x"] + piece["nx"] for piece in pieces if piece["ccd"] == ccd) - left) for ccd in present}
    gaps = [(edges[first][1], edges[second][0]) for first, second in zip(present[:-1], present[1:]) if edges[second][0] > edges[first][1]]
    return pieces, (top - bottom, right - left), (left, bottom), gaps


def _place(canvas, piece, array, origin):

    """Copy the data region of an amplifier into the assembled image (rows outside the common rows of the CCDs are dropped)."""

    left, bottom = origin
    y1, y2 = piece["y"] - bottom, piece["y"] - bottom + piece["ny"]
    x1 = piece["x"] - left
    cut_low, cut_high = max(0, -y1), max(0, y2 - canvas.shape[0])
    canvas[y1 + cut_low:y2 - cut_high, x1:x1 + piece["nx"]] = array[cut_low:array.shape[0] - cut_high]


def _read_bad_pixel_mask(bad_pixel_mask, frame, crop):

    """Static bad pixel mask (a GMOS BPM file) assembled like the frame; None if it does not match the frame."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _open(bad_pixel_mask) as hdul:
            extensions = _image_extensions(hdul)
            if len(extensions) == 0:
                return None
            headers = [hdu.header for hdu in extensions]
            binning = tuple(int(value) for value in str(headers[0].get("CCDSUM", "1 1")).split()[:2])
            if binning != tuple(frame["binning"]):
                return None
            arrays = [np.asarray(hdu.data) for hdu in extensions]
    # only the amplifiers (and rows) that were read in the frame
    x1, x2, y1, y2 = frame["roi"]
    keep = [index for index, header in enumerate(headers) if x1 <= _parse_section(header["DETSEC"])[0] < x2]
    if len(keep) == 0:
        return None
    rows = (max(y1, crop[0]), min(y2, crop[1])) if crop is not None else (y1, y2)
    try:
        pieces, shape, origin, _ = _layout([headers[i] for i in keep], [arrays[i].shape for i in keep], frame["detector"], crop=rows)
    except ValueError:
        return None
    mask = np.zeros(shape, dtype=bool)
    for piece, index in zip(pieces, keep):
        _place(mask, piece, arrays[index][piece["source"]] != 0, origin)
    return mask


_MASK_CACHE = {}


def assemble(file_name, overscan="row", units="electron", bad_pixel_mask=None, crop_rows=None, blue_left=True, gap_value=np.nan):

    """
    Assemble a raw GMOS frame (one fits extension per amplifier) into a single image.

    Parameters
    ----------
    file_name: string or Path
        A raw GMOS file (.fits, .fits.bz2 or .fits.gz).
    overscan: string or None
        "row" (default): the bias level of each amplifier is measured in its overscan region for each row (smoothed along the rows) and
        subtracted. "constant": a single value per amplifier. None: no subtraction (the overscan regions are removed anyway).
    units: string
        "electron" (default): each amplifier is multiplied by its gain; the output image is in electrons and its GAIN keyword is 1.
        "adu": no gain correction (the output GAIN keyword is the mean gain; the different gains of the amplifiers must then be removed
        by the flat field).
    bad_pixel_mask: string, Path or None
        A GMOS static bad pixel mask (BPM file of the Gemini archive, with the same binning as the frame). Its bad pixels are set to NaN.
    crop_rows: tuple or None
        (y1, y2): rows of the detector to keep, in unbinned pixels (0-based, half-open), e.g. to cut a full-frame image to the rows of
        the "central spectrum" region of interest. Default: all the rows that were read.
    blue_left: boolean
        Only for spectroscopy. In raw GMOS spectra the wavelength decreases along the x axis; if True (default), the image is mirrored
        so that the wavelength increases along x (blue on the left), as in the other easyspec tutorials.
    gap_value: float
        Value of the pixels in the gaps between the CCDs. Default: NaN.

    Returns
    -------
    data: numpy.ndarray (float32)
        The assembled image.
    header: astropy.io.fits.Header
        Its header (see the module documentation).
    """

    if units not in ("electron", "adu"):
        raise ValueError("'units' must be 'electron' or 'adu'.")
    if overscan not in ("row", "constant", None):
        raise ValueError("'overscan' must be 'row', 'constant' or None.")
    frame = describe_frame(file_name)
    if frame is None or frame["category"] == "mask_definition":
        raise ValueError(f"{file_name} is not a raw GMOS image.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _open(file_name) as hdul:
            primary = hdul[0].header.copy()
            extensions = _image_extensions(hdul)
            headers = [hdu.header.copy() for hdu in extensions]
            arrays = [np.asarray(hdu.data, dtype=np.float32) for hdu in extensions]

    pieces, shape, origin, gaps = _layout(headers, [array.shape for array in arrays], frame["detector"], crop=crop_rows)
    used = [index for index, header in enumerate(headers)
            if crop_rows is None or max(_parse_section(header["DETSEC"])[2], crop_rows[0]) < min(_parse_section(header["DETSEC"])[3], crop_rows[1])]
    data = np.full(shape, gap_value, dtype=np.float32)

    gains, read_noises, levels, saturations, from_table = [], [], [], [], True
    x_bin, y_bin = frame["binning"]
    full_wells = lookup.FULL_WELL.get(frame["detector_name"], {})
    for piece, index in zip(pieces, used):
        header, array = headers[index], arrays[index]
        amplifier = str(header.get("AMPNAME", "")).strip()
        gain, read_noise, in_table = amplifier_properties(amplifier, frame["date"], frame["read_mode"], header.get("GAIN"),
                                                          header.get("RDNOISE"))
        from_table &= in_table
        region = array[piece["source"]]
        level = 0.0
        if "BIASSEC" in header:
            bx1, bx2, by1, by2 = _parse_section(header["BIASSEC"])
            # the columns next to the data (charge trailing) and at the edge of the overscan region are not used
            margin = 3 if bx2 - bx1 >= 12 else 0
            strip = array[piece["source"][0], bx1 + margin:bx2 - margin]
            level_rows = _overscan_level(strip, overscan or "row")
            level = float(np.nanmedian(level_rows))
            if overscan is not None:
                region = region - level_rows[:, None]
        if units == "electron":
            region = region*gain
        _place(data, piece, region, origin)
        gains.append(gain)
        read_noises.append(read_noise)
        levels.append(level)
        # Saturation: the limit of the analog-to-digital converter, or the full well of the pixels if it is lower
        limit = (ADC_LIMIT - (level if overscan is not None else 0.0))*gain
        well = full_wells.get(amplifier)
        if well is not None and x_bin*y_bin <= 2:
            limit = min(limit, well*x_bin*y_bin + (0.0 if overscan is not None else level*gain))
        saturations.append(limit if units == "electron" else limit/gain)

    if not from_table:
        warnings.warn(f"{Path(file_name).name}: amplifier(s) not found in the gain tables of GMOS for this date and read mode; the nominal "
                      "GAIN and RDNOISE of the header were used.")

    n_bad = 0
    if bad_pixel_mask is not None:
        key = (str(bad_pixel_mask), frame["detector"], frame["binning"], frame["roi"], crop_rows)
        if key not in _MASK_CACHE:
            _MASK_CACHE.clear()   # only the last mask is kept in memory
            _MASK_CACHE[key] = _read_bad_pixel_mask(bad_pixel_mask, frame, crop_rows)
        mask = _MASK_CACHE[key]
        if mask is None or mask.shape != data.shape:
            warnings.warn(f"{Path(file_name).name}: the bad pixel mask {Path(bad_pixel_mask).name} does not match this frame (binning or "
                          "detector region); it was not applied.")
        else:
            n_bad = int(np.count_nonzero(mask))
            data[mask] = np.nan

    spectroscopy = frame["mode"] != "imaging"
    mirrored = bool(blue_left and spectroscopy)
    if mirrored:
        data = data[:, ::-1]
        gaps = sorted((data.shape[1] - stop, data.shape[1] - start) for start, stop in gaps)

    # ------------------------------------------------------------------------------------------------ Header
    header = primary
    for key in DROPPED_KEYWORDS:
        header.remove(key, ignore_missing=True, remove_all=True)
    mean_gain = float(np.mean(gains))
    noise = np.array(read_noises, dtype=float)
    read_noise = float(np.sqrt(np.nanmean(noise**2))) if np.any(np.isfinite(noise)) else np.nan
    header["GAIN"] = (1.0 if units == "electron" else round(mean_gain, 4), "[e-/ADU] gain to apply to this image")
    header["GAINORIG"] = (round(mean_gain, 4), "[e-/ADU] mean gain of the amplifiers")
    if np.isfinite(read_noise):
        header["RDNOISE"] = (round(read_noise if units == "electron" else read_noise, 3), "[e-] read noise (rms of amplifiers)")
    header["BUNIT"] = ("electron" if units == "electron" else "adu", "units of the image")
    header["SATURATE"] = (round(float(np.min(saturations)), 1), "saturation level, in the units of the image")
    header["FILTER"] = (frame["filter"], "filter(s) in the beam")
    header["CCDSUM"] = (f"{x_bin} {y_bin}", "binning (x y)")
    scale = lookup.PIXEL_SCALE.get((frame["instrument"], str(primary.get("DETTYPE", "")).strip()))
    if scale is not None:
        header["PIXSCALX"] = (round(scale*x_bin, 5), "[arcsec/pixel] scale along x")
        header["PIXSCALY"] = (round(scale*y_bin, 5), "[arcsec/pixel] scale along y")
        if x_bin == y_bin:
            header["PIXSCALE"] = (round(scale*x_bin, 5), "[arcsec/pixel]")
    if frame["date"]:
        header["DATE-OBS"] = (frame["date"], "UT start of the exposure")
        try:
            header["MJD-OBS"] = (round(Time(frame["date"], scale="utc").mjd, 7), "MJD (UTC) at the start of the exposure")
        except ValueError:
            pass
    for number, (index, gain, noise_value, level) in enumerate(zip(used, gains, read_noises, levels), start=1):
        header[f"GAIN{number:02d}"] = (round(gain, 4), f"[e-/ADU] gain of amplifier {number}")
        if np.isfinite(noise_value):
            header[f"RON{number:02d}"] = (round(noise_value, 3), f"[e-] read noise of amplifier {number}")
        header[f"OVSC{number:02d}"] = (round(level, 2), f"[ADU] overscan level of amplifier {number}")

    header["ESGMOS"] = (MODULE_VERSION, "assembled by easyspec.gemini.gmos (version)")
    header["ESMODE"] = (frame["mode"], "GMOS observing mode")
    header["ESTYPE"] = (frame["category"], "frame type")
    header["ESOVSCAN"] = (str(overscan), "overscan subtraction")
    header["ESMIRROR"] = (mirrored, "image mirrored along x (blue on the left)")
    header["ESNBAD"] = (n_bad, "pixels set to NaN from the bad pixel mask")
    y0 = (crop_rows[0] if crop_rows is not None else frame["roi"][2])
    y1 = (crop_rows[1] if crop_rows is not None else frame["roi"][3])
    header["ESROWS"] = (f"{max(y0, frame['roi'][2]) + 1}:{min(y1, frame['roi'][3])}", "detector rows (unbinned) in this image")
    for number, (start, stop) in enumerate(gaps, start=1):
        header[f"ESGAP{number}"] = (f"{start}:{stop}", "columns of a CCD gap (0-based, half-open)")

    if spectroscopy:
        _add_wavelength_range(header, frame, data.shape[1], mirrored)
        for key in list(header.keys()):   # the celestial WCS of the raw extensions is meaningless for a dispersed image
            if re.match(r"^(CTYPE|CRPIX|CRVAL|CD\d_|CDELT|CUNIT)\d*", key):
                header.remove(key, ignore_missing=True, remove_all=True)
    else:
        _add_wcs(header, headers, pieces, used, origin)

    header["HISTORY"] = "Raw GMOS frame assembled by easyspec.gemini.gmos:"
    header["HISTORY"] = (f"overscan {'not subtracted' if overscan is None else 'subtracted (' + overscan + ')'}; "
                         f"{'converted to electrons (gain of each amplifier)' if units == 'electron' else 'left in ADU'}; "
                         "amplifiers and CCDs assembled (gaps = NaN)")
    return data, header


def _add_wavelength_range(header, frame, n_columns, mirrored):

    """
    Approximate wavelengths (Angstrom) at the first and at the last column of a spectrum, from the grating and the central wavelength:
    the input 'wavelength_range' of extraction.identify_lamp_lines().
    """

    ruling = re.search(r"(\d+)", frame["grating"])
    centre = frame["central_wavelength"]
    if ruling is None or not np.isfinite(centre) or centre <= 0:
        return
    dispersion = DISPERSION_CONSTANT/float(ruling.group(1))               # nm per unbinned Hamamatsu pixel
    if "hamamatsu" not in frame["detector_name"].lower():
        dispersion *= PIXEL_SIZE_RATIO
    dispersion *= 10*frame["binning"][0]                                   # Angstrom per pixel of the image
    half = 0.5*n_columns*dispersion
    blue, red = 10*centre - half, 10*centre + half
    first, last = (blue, red) if mirrored else (red, blue)
    header["ESWAVE1"] = (round(first, 1), "[A] approximate wavelength at the first column")
    header["ESWAVE2"] = (round(last, 1), "[A] approximate wavelength at the last column")
    header["ESDISP"] = (round(dispersion, 4), "[A/pixel] approximate dispersion")


def _add_wcs(header, headers, pieces, used, origin):

    """Celestial WCS of an assembled image, from the WCS of an amplifier of the central CCD."""

    candidates = [(piece, headers[index]) for piece, index in zip(pieces, used) if "CRPIX1" in headers[index] and "CD1_1" in headers[index]]
    if len(candidates) == 0:
        return
    central = [item for item in candidates if item[0]["ccd"] == 1] or candidates
    piece, source = central[0]
    for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "RADESYS", "EQUINOX"):
        if key in source:
            header[key] = source[key]
    header["CRPIX1"] = float(source["CRPIX1"]) - piece["source"][1].start + piece["x"] - origin[0]
    header["CRPIX2"] = float(source["CRPIX2"]) - piece["source"][0].start + piece["y"] - origin[1]


def wavelength_range(file_name):

    """
    Approximate wavelengths (in Angstrom) at the first and at the last column of a converted GMOS spectrum, computed from the grating and
    the central wavelength of the observation. This is the input 'wavelength_range' of extraction.identify_lamp_lines()::

        positions, wavelengths = extraction.identify_lamp_lines(lamp_spec_list, lamps=gmos.CUAR_LINE_LIST,
                                                                wavelength_range=gmos.wavelength_range(lamp_file))

    Parameters
    ----------
    file_name: string or Path
        A spectrum converted by this module (or any file produced from it by cleaning(), as long as the header was kept).

    Returns
    -------
    [first, last]: list
    """

    header = fits.getheader(file_name)
    if "ESWAVE1" not in header:
        raise KeyError(f"{file_name} has no ESWAVE1/ESWAVE2 keywords: it is not a GMOS spectrum converted by easyspec.gemini.gmos.")
    return [float(header["ESWAVE1"]), float(header["ESWAVE2"])]


def ccd_gaps(file_name):

    """
    Columns of the gaps between the CCDs of a converted GMOS image (keywords ESGAPn of its header), as a list of [first column, last
    column + 1]. This is the input 'ccd_gaps' of cleaning.norm_master_flat() (with keep_ccd_steps=True).

    Parameters
    ----------
    file_name: string or Path
        An image converted by this module (or a master file produced from it by cleaning(), as long as the header was kept).

    Returns
    -------
    gaps: list
    """

    header = fits.getheader(file_name)
    gaps = []
    number = 1
    while f"ESGAP{number}" in header:
        start, stop = (int(value) for value in str(header[f"ESGAP{number}"]).split(":"))
        gaps.append([start, stop])
        number += 1
    return gaps


# ======================================================================================================================
# Converting a whole directory
# ======================================================================================================================

def _setup_name(frame, with_rows=False):

    """Name of the instrument setup of a frame (the name of its output directory)."""

    binning = f"{frame['binning'][0]}x{frame['binning'][1]}"
    if frame["mode"] == "imaging":
        parts = ["imaging", frame["filter"], binning]
    else:
        parts = [frame["mode"], frame["grating"], f"{frame['central_wavelength']:g}nm", frame["mask"]]
        if frame["filter"] != "open":
            parts.append(frame["filter"])
        parts.append(binning)
    if with_rows:
        parts.append(f"rows{frame['roi'][2] + 1}-{frame['roi'][3]}")
    return _clean_name("_".join(parts))


def _family(frame):

    """Frames of the same family may share a detector region: same mode, grating, slit, filter and binning."""

    return (frame["instrument"], frame["detector"], frame["mode"], frame["grating"], frame["mask"], frame["filter"], frame["binning"])


def _dither_groups(frames, tolerance):

    """Group frames by telescope offset (P, Q), in order of observation. Returns a list with the group number of each frame."""

    centres, groups = [], []
    for frame in frames:
        position = np.array([frame["p_offset"], frame["q_offset"]])
        for number, centre in enumerate(centres):
            if np.hypot(*(position - centre)) <= tolerance:
                groups.append(number)
                break
        else:
            centres.append(position)
            groups.append(len(centres) - 1)
    return groups


def convert(input_directory, output_directory, bad_pixel_mask="auto", overscan="row", units="electron", common_rows=True,
            blue_left=True, dither_tolerance=0.5, calibration_time_limit=12.0, include_acquisition=False, overwrite=False, verbose=True):

    """
    Convert all the raw GMOS frames of a directory into single-extension images, sorted into the directories expected by
    easyspec.cleaning().

    The output directory has one subdirectory per instrument setup::

        imaging_FILTER_BINNING/                          e.g. imaging_i_2x2
        longslit_GRATING_WAVELENGTHnm_SLIT_BINNING/      e.g. longslit_B600_515nm_1.0arcsec_2x2
        mos_GRATING_WAVELENGTHnm_MASK_BINNING/, ifu_.../

    and each setup has the subdirectories (only those with data)::

        bias/  flat/  lamp/  dark/  standard_star_NAME/  target_NAME/  twilight/  acquisition/
        flat_standard_star/  lamp_standard_star/   (calibrations of a standard star observed in another night; see below)

    In spectroscopy, the exposures taken at different positions along the slit (spatial dithers) go to different directories
    (target_NAME_dither1, target_NAME_dither2...), and those taken with different central wavelengths (spectral dithers) to different
    setups, each one with its own flats and lamps. In imaging, all the dithered exposures of a field go to the same directory (they are
    aligned by cleaning.align()). The bias frames, which do not depend on the optical setup, are copied to every setup with the same
    binning, read mode and detector region.

    Parameters
    ----------
    input_directory: string or Path
        Directory with the raw GMOS files (.fits, .fits.bz2 or .fits.gz); it is searched recursively.
    output_directory: string or Path
        Where the converted images are written.
    bad_pixel_mask: string, Path, "auto" or None
        Static bad pixel mask (BPM file of the Gemini archive). "auto" (default): the BPM files found in the input directory are used for
        the frames of the same instrument and binning. Their bad pixels are set to NaN in the converted images. None: no mask.
    overscan, units, blue_left:
        See assemble().
    common_rows: boolean
        Only for long-slit spectroscopy. The standard stars are usually observed with a smaller region of interest ("central spectrum")
        than the targets ("full frame"), while cleaning() needs all the images of a reduction to have the same shape. If True (default),
        all the frames with the same grating and slit are cut to the rows of the detector that all the targets and standard stars
        have in common, so that the standard star, the targets and their calibrations can be reduced together. If False, the frames keep
        their original rows and each detector region gets its own setup directory.
    dither_tolerance: float
        Exposures whose telescope offsets differ by less than this value (arcsec) belong to the same dither position.
    calibration_time_limit: float
        Only for spectroscopy. GMOS flats and lamps are taken together with the exposures they calibrate (the spectra move on the detector
        with the position of the telescope). When the standard star was observed in another night, the flats and lamps taken with it (i.e.
        those whose closest exposure in time is the standard star, more than 'calibration_time_limit' hours away from any target) go to the
        directories flat_standard_star and lamp_standard_star, instead of being mixed with the calibrations of the targets. Default: 12
        hours.
    include_acquisition: boolean
        If True, the acquisition images are also converted (to the directory "acquisition" of their setup). Default: False.
    overwrite: boolean
        If False (default), an error is raised if a converted file already exists.
    verbose: boolean
        If True, a summary of the conversion is printed.

    Returns
    -------
    setups: dict
        For each setup, the dictionary of directories to give to cleaning.data_paths(): {"bias": ..., "flats": ..., "lamp": ..., "darks":
        ..., "standard_star": [...], "targets": [...]} (only the keys with data), e.g.::

            raw = cleaning.data_paths(**setups["longslit_B600_515nm_1.0arcsec_2x2"])

    Notes
    -----
    A table with every converted frame is saved in the output directory (gmos_conversion_log.csv) and returned by
    conversion_log(output_directory).
    """

    output_directory = Path(output_directory)
    frames = [frame for frame in _collect_frames(input_directory)]
    masks = [frame for frame in frames if frame["category"] == "bad_pixel_mask"]
    definitions = [frame for frame in frames if frame["category"] == "mask_definition"]
    frames = [frame for frame in frames if frame["category"] not in ("bad_pixel_mask", "mask_definition")]
    skipped = [frame for frame in frames if frame["category"] == "unknown" or (frame["category"] == "acquisition" and not include_acquisition)]
    frames = [frame for frame in frames if not any(frame is other for other in skipped)]
    if len(frames) == 0:
        raise RuntimeError("No bias, flat, lamp, standard star or target frames found.")

    def mask_for(frame):
        if bad_pixel_mask is None:
            return None
        if bad_pixel_mask != "auto":
            return bad_pixel_mask
        for mask in masks:
            if mask["instrument"] == frame["instrument"] and mask["binning"] == frame["binning"] and mask["detector"] == frame["detector"]:
                return mask["file"]
        return None

    # ------------------------------------------------------------------------------------------------ Rows of each family
    optical = [frame for frame in frames if frame["category"] not in ("bias", "dark")]
    crops = {}
    if common_rows:
        for family in {_family(frame) for frame in optical if frame["mode"] == "longslit"}:
            members = [frame for frame in optical if _family(frame) == family and frame["category"] in ("target", "standard_star")]
            if len(members) == 0:
                continue
            low, high = max(frame["roi"][2] for frame in members), min(frame["roi"][3] for frame in members)
            if high > low:
                crops[family] = (low, high)

    def rows_of(frame):
        return crops.get(_family(frame), (frame["roi"][2], frame["roi"][3]))

    def setup_of(frame):
        return _setup_name(frame, with_rows=(not common_rows and (frame["roi"][3] - frame["roi"][2]) < _full_height(frame)))

    # ------------------------------------------------------------------------------------------------ Plan
    plan = []          # (frame, setup, directory, rows)
    setups = {}
    for frame in optical:
        setup = setup_of(frame)
        rows = rows_of(frame)
        if frame["roi"][2] > rows[0] or frame["roi"][3] < rows[1]:
            skipped.append(frame)   # e.g. a flat of the central spectrum cannot calibrate a full-frame setup
            continue
        setups.setdefault(setup, {"frames": [], "rows": rows, "example": frame})
        setups[setup]["frames"].append(frame)

    # Dither positions: the same numbers in all the setups (central wavelengths) of an object
    dither = {}
    planned = sorted((frame for content in setups.values() for frame in content["frames"]), key=lambda frame: frame["date"])
    for key in {(_family(frame), frame["category"], frame["object"]) for frame in planned if frame["category"] in ("target", "standard_star")}:
        exposures = [frame for frame in planned if (_family(frame), frame["category"], frame["object"]) == key]
        groups = _dither_groups(exposures, dither_tolerance) if exposures[0]["mode"] != "imaging" else [0]*len(exposures)
        for frame, group in zip(exposures, groups):
            dither[frame["file"]] = (group, max(groups))

    for setup, content in setups.items():
        members = sorted(content["frames"], key=lambda frame: frame["date"])
        for category, prefix in (("target", "target"), ("standard_star", "standard_star")):
            for name in sorted({frame["object"] for frame in members if frame["category"] == category}):
                exposures = [frame for frame in members if frame["category"] == category and frame["object"] == name]
                for frame in exposures:
                    group, last = dither[frame["file"]]
                    suffix = f"_dither{group + 1}" if last > 0 else ""
                    plan.append((frame, setup, f"{prefix}_{_clean_name(name)}{suffix}", content["rows"]))
        # Flats and lamps: those taken with a standard star observed in another night go to their own directories
        science = [frame for frame in members if frame["category"] in ("target", "standard_star")]
        with_targets = any(frame["category"] == "target" for frame in science)
        for frame in members:
            if frame["category"] in ("flat", "lamp", "twilight", "acquisition"):
                directory = frame["category"]
                if frame["category"] in ("flat", "lamp") and with_targets and content["example"]["mode"] != "imaging":
                    nearest = _nearest_in_time(frame, science, calibration_time_limit)
                    if nearest is not None and nearest["category"] == "standard_star":
                        directory = f"{frame['category']}_standard_star"
                plan.append((frame, setup, directory, content["rows"]))
        # Bias and dark frames: same detector, binning and read mode (that of the targets), and all the rows of the setup
        example = science[0] if science else content["example"]
        for frame in frames:
            if frame["category"] not in ("bias", "dark"):
                continue
            same = (frame["instrument"] == example["instrument"] and frame["detector"] == example["detector"]
                    and frame["binning"] == example["binning"] and frame["read_mode"] == example["read_mode"]
                    and frame["roi"][0] <= example["roi"][0] and frame["roi"][1] >= example["roi"][1])
            rows = content["rows"]
            covers = frame["roi"][2] <= rows[0] and frame["roi"][3] >= rows[1]
            exact = (frame["roi"][2], frame["roi"][3]) == rows
            if same and (exact or (common_rows and covers)):
                plan.append((frame, setup, frame["category"], rows))

    # ------------------------------------------------------------------------------------------------ Conversion
    log = []
    for frame, setup, directory, rows in plan:
        destination = output_directory/setup/directory
        destination.mkdir(parents=True, exist_ok=True)
        output_file = destination/f"{_stem(frame['file'])}.fits"
        if output_file.exists() and not overwrite:
            raise FileExistsError(f"{output_file} already exists. Use overwrite=True to replace the converted files.")
        crop = rows if (rows[0] > frame["roi"][2] or rows[1] < frame["roi"][3]) else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data, header = assemble(frame["file"], overscan=overscan, units=units, bad_pixel_mask=mask_for(frame), crop_rows=crop,
                                    blue_left=blue_left)
        for warning in caught:
            warnings.warn(str(warning.message))
        header["ESSETUP"] = (setup[:68], "easyspec.gemini.gmos setup")
        fits.PrimaryHDU(data=data, header=header).writeto(output_file, overwrite=True, output_verify="silentfix")
        log.append({"file": Path(frame["file"]).name, "setup": setup, "directory": directory, "output": str(output_file),
                    "category": frame["category"], "object": frame["object"], "exposure": frame["exposure"], "airmass": frame["airmass"],
                    "central_wavelength": frame["central_wavelength"], "p_offset": round(frame["p_offset"], 2),
                    "q_offset": round(frame["q_offset"], 2), "date": frame["date"], "shape": f"{data.shape[0]}x{data.shape[1]}"})

    table = Table(rows=log)
    table.write(output_directory/"gmos_conversion_log.csv", format="ascii.csv", overwrite=True)

    result = _data_paths(table, output_directory)
    if verbose:
        _print_summary(table, result, setups, skipped, definitions, common_rows, crops)
    return result


def _hours(date):

    try:
        return Time(date, scale="utc").mjd*24.0
    except ValueError:
        return np.nan


def _nearest_in_time(frame, science, limit):

    """
    The target or standard star exposure closest in time to a calibration frame; None if a target was observed less than 'limit'
    hours from it (then it is a calibration of the targets).
    """

    time = _hours(frame["date"])
    gaps = np.array([abs(_hours(other["date"]) - time) for other in science])
    if len(gaps) == 0 or not np.any(np.isfinite(gaps)):
        return None
    targets = np.array([other["category"] == "target" for other in science])
    if np.any(targets) and np.nanmin(gaps[targets]) <= limit:
        return None
    return science[int(np.nanargmin(gaps))]


def _full_height(frame):

    """Number of rows (unbinned) of the full detector."""

    return 4224 if ("hamamatsu" in frame["detector_name"].lower() or frame["detector"].startswith("BI")) else 4608


def _data_paths(table, output_directory):

    """For each setup, the directories in the format of cleaning.data_paths()."""

    output_directory = Path(output_directory)
    result = {}
    for setup in sorted(str(name) for name in set(table["setup"])):
        directories = sorted(set(table["directory"][table["setup"] == setup]))
        paths = {}
        for directory in directories:
            path = str(output_directory/setup/directory)
            if directory == "bias":
                paths["bias"] = path
            elif directory == "flat":
                paths["flats"] = path
            elif directory == "lamp":
                paths["lamp"] = path

            elif directory == "dark":
                paths["darks"] = path
            elif directory.startswith("standard_star"):
                paths.setdefault("standard_star", []).append(path)
            elif directory.startswith("target"):
                paths.setdefault("targets", []).append(path)
        result[setup] = paths
    return result


def conversion_log(output_directory):

    """
    The table written by convert() (one row per converted frame: original file, setup, directory, type, object, exposure time, airmass,
    central wavelength, telescope offsets, date and image shape), and the directories of each setup in the format of
    cleaning.data_paths().

    Returns
    -------
    table: astropy.table.Table
    setups: dict
        See convert().
    """

    log_file = Path(output_directory)/"gmos_conversion_log.csv"
    if not log_file.exists():
        raise FileNotFoundError(f"{log_file} not found: run gmos.convert() first.")
    table = Table.read(log_file, format="ascii.csv")
    return table, _data_paths(table, output_directory)


def _print_summary(table, result, setups, skipped, definitions, common_rows, crops):

    print(f"{len(table)} frames converted into {len(result)} setup(s):")
    for setup, paths in result.items():
        selection = table[table["setup"] == setup]
        example = setups[setup]["example"]
        rows = setups[setup]["rows"]
        print(f"\n  {setup}   (image shape {selection['shape'][0]}, detector rows {rows[0] + 1}-{rows[1]})")
        for directory in sorted(set(selection["directory"])):
            members = selection[selection["directory"] == directory]
            detail = ""
            if directory.startswith(("target", "standard_star")):
                detail = (f"   exposure {np.mean(members['exposure']):g} s, airmass {np.mean(members['airmass']):.2f}"
                          + (f", offset along the slit {np.mean(members['q_offset']):+.1f} arcsec" if example["mode"] != "imaging" else ""))
            print(f"      {directory + '/':38s}{len(members):3d} frame(s){detail}")
        missing = [name for name, key in (("bias", "bias"), ("flat", "flats")) if key not in paths]
        if example["mode"] != "imaging" and "lamp" not in paths:
            missing.append("lamp")
        if any(str(directory).endswith("_standard_star") for directory in set(selection["directory"])):
            print("      note: flat_standard_star/ and lamp_standard_star/ were taken with the standard star, in another night: use them "
                  "(instead of flat/ and lamp/)\n            if you need the calibrations closest to the standard star")
        if "targets" in paths and "standard_star" not in paths and example["mode"] != "imaging":
            print("      note: no standard star in this setup (use the one of another central wavelength for the flux calibration)")
        if missing:
            print(f"      WARNING: no {', '.join(missing)} frames for this setup")
        if example["mode"] in ("mos", "ifu", "slitless"):
            print(f"      note: {example['mode'].upper()} frames are assembled and sorted only (no slitlet cutting or fibre extraction)")
        if any(frame["nod_and_shuffle"] for frame in setups[setup]["frames"]):
            print("      note: nod-and-shuffle frames: the shuffled sky is not subtracted by this module")
    if common_rows and crops:
        print("\nLong-slit frames were cut to the detector rows common to the targets and standard stars (common_rows=True).")
    if skipped:
        names = ", ".join(f"{Path(frame['file']).name} ({frame['category']})" for frame in skipped[:12])
        print(f"\n{len(skipped)} frame(s) not converted: {names}{'...' if len(skipped) > 12 else ''}")
    if definitions:
        print(f"{len(definitions)} mask definition file(s) (MDF) found: not used by this module.")
    print("\nUse: raw = cleaning.data_paths(**setups[\"<setup name>\"])")
