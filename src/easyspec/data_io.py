"""
easyspec data_io module.

This module defines the file formats shared by the three main steps of easyspec, so that cleaning (reduction),
extraction and analysis can live in separate classes and separate notebooks:

    cleaning  --[reduced image: data + VAR + MASK]-->  extraction  --[1D spectrum table]-->  analysis

Reduced images (e.g. master_target_*.fits)
------------------------------------------
- Primary HDU: the data (so fits.getdata(file) and fits.open(file)[0].data keep working), with the original header.
- Extension "VAR": variance of each pixel, in the data units squared (optional).
- Extension "MASK": data quality bit mask, 0 = good pixel (optional). The meaning of each bit is given in MASK_BITS.

1D spectra
----------
- Primary HDU: header only (target name, exposure time, airmass, etc.).
- Extension "SPECTRUM": binary table with the columns "wavelength", "flux" and, optionally, "flux_err", "mask" and any
  other column (e.g. systematic errors). Each column carries its physical unit.
read_spectrum() also reads the text files (.dat) written by previous easyspec versions.

All files written by this module carry the header keyword ESFORMAT (format version), so that future versions of
easyspec can read older files.
"""

from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.table import QTable


FORMAT_VERSION = 1
FORMAT_KEYWORD = "ESFORMAT"
VARIANCE_EXTNAME = "VAR"
MASK_EXTNAME = "MASK"
SPECTRUM_EXTNAME = "SPECTRUM"

# Bits of the data quality mask (a pixel may have several bits set; 0 = good pixel):
MASK_BITS = {1: "cosmic ray (pixel replaced by LACosmic)",
             2: "flat field not applied (low signal in the flat, e.g. outside the slit)",
             4: "non-finite data or variance",
             8: "no data (outside the original field of view after the alignment)"}


# ======================================================================================================================
# Reduced images
# ======================================================================================================================

def _clean_header(header):

    """Copy of a header without the keywords that would rescale the data or describe extensions."""

    header = fits.Header() if header is None else header.copy()
    for key in ("BZERO", "BSCALE", "BLANK", "EXTNAME", "EXTVER", "XTENSION", "PCOUNT", "GCOUNT"):
        header.remove(key, ignore_missing=True, remove_all=True)
    return header


def write_reduced_frame(file_name, data, header=None, variance=None, mask=None, units=None, overwrite=True, float_dtype=np.float32):
 
    """
    Save a reduced image (e.g. a master file or a fully reduced science frame) in the easyspec format.
 
    Parameters
    ----------
    file_name: string or Path
        Output file. Its directory is created if it does not exist.
    data: array
        Image data, saved in the primary HDU.
    header: astropy.io.fits.Header
        Header of the primary HDU (e.g. the header of the original raw frame). Keywords that would rescale the data
        (BZERO, BSCALE, BLANK) are removed.
    variance: array
        Optional. Variance of each pixel, in the data units squared. Saved in the "VAR" extension.
    mask: array
        Optional. Data quality bit mask (0 = good pixel, see MASK_BITS). Saved in the "MASK" extension.
    units: string
        Optional. Data units (e.g. "electron" or "adu"). Written in the BUNIT keyword of the primary HDU, and squared in
        the BUNIT keyword of the "VAR" extension.
    overwrite: bool
        If True (default), an existing file with the same name is replaced.
    float_dtype: numpy dtype
        Data type used to save floating-point data (default: np.float32, i.e. 4 bytes per pixel). Its ~7 significant digits are far more
        than enough for astronomical images, whose noise is much larger than the rounding error, and it halves the file size with respect
        to float64. Use np.float64 to keep double precision. Integer data are saved unchanged.
 
    Returns
    -------
    file_name: Path
        The path of the saved file.
    """
 
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating) and float_dtype is not None:
        data = data.astype(float_dtype)
    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)
 
    primary = fits.PrimaryHDU(data, header=_clean_header(header))
    primary.header[FORMAT_KEYWORD] = (FORMAT_VERSION, "easyspec file format version")
    if units is not None:
        primary.header["BUNIT"] = (str(units), "Data units")
    hdul = fits.HDUList([primary])
 
    if variance is not None:
        variance = np.asarray(variance)
        if variance.shape != data.shape:
            raise ValueError(f"The variance has shape {variance.shape}, but the data have shape {data.shape}. The variance must undergo "
                             "the same geometric operations as the data (trim, rotate, align, ...).")
        var_hdu = fits.ImageHDU(variance.astype(np.float32), name=VARIANCE_EXTNAME)
        var_hdu.header["COMMENT"] = "Variance of each pixel of the primary HDU, in its units squared."
        if units is not None:
            var_hdu.header["BUNIT"] = (f"{units}2", "Variance units")
        hdul.append(var_hdu)
 
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != data.shape:
            raise ValueError(f"The mask has shape {mask.shape}, but the data have shape {data.shape}. The mask must undergo the same "
                             "geometric operations as the data (trim, rotate, align, ...).")
        mask_hdu = fits.ImageHDU(mask.astype(np.uint8), name=MASK_EXTNAME)
        mask_hdu.header["COMMENT"] = "Data quality bit mask (0 = good pixel):"
        for bit, meaning in MASK_BITS.items():
            mask_hdu.header["COMMENT"] = f"bit {bit}: {meaning}"
        hdul.append(mask_hdu)
 
    hdul.writeto(file_name, overwrite=overwrite)
    return file_name


def _find_keyword(hdul, keyword):

    """Return the value of 'keyword' from the first HDU that contains it, or None."""

    for hdu in hdul:
        if keyword in hdu.header and hdu.header[keyword] is not None:
            return hdu.header[keyword]
    return None


def read_reduced_frame(file_name, estimate_variance=True, gain=None, readnoise=None, gain_header_entry="GAIN",
                       readnoise_header_entry="RDNOISE"):

    """
    Read a reduced image saved in the easyspec format (or any fits image with the data in the primary HDU).

    If the file has no "VAR" extension (e.g. files created by previous easyspec versions) and estimate_variance=True,
    the variance is estimated with a simple noise model (Poisson noise + read noise), using the gain and read noise
    given as arguments or found in the header:

        data in electrons (BUNIT = "electron"):  Var = max(data, 0) + RN^2
        data in ADU (any other BUNIT):           Var = max(data, 0)/gain + (RN/gain)^2

    This estimate neglects the noise of the calibration masters (bias, dark, flat), so it is a lower limit; a warning
    is issued. If the file has no "MASK" extension, an all-zero (all good) mask is returned.

    Parameters
    ----------
    file_name: string or Path
        Path to the fits file.
    estimate_variance: bool
        If True (default), estimate the variance when the file has no "VAR" extension. If False, None is returned instead.
    gain: float
        Gain in e-/ADU, used only to estimate the variance of ADU images. If None, it is read from the header.
    readnoise: float
        Read noise in electrons, used only to estimate the variance. If None, it is read from the header.
    gain_header_entry, readnoise_header_entry: string
        Header keywords for the gain and read noise.

    Returns
    -------
    data: array
        The image (primary HDU), as floating point.
    variance: array or None
        The variance (in the data units squared), or None if it is neither stored nor estimable.
    mask: array
        Data quality bit mask (uint8, 0 = good pixel; see MASK_BITS).
    header: astropy.io.fits.Header
        Header of the primary HDU.
    """

    with fits.open(file_name) as hdul:
        data = np.array(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
        variance = np.array(hdul[VARIANCE_EXTNAME].data, dtype=float) if VARIANCE_EXTNAME in hdul else None
        mask = np.array(hdul[MASK_EXTNAME].data, dtype=np.uint8) if MASK_EXTNAME in hdul else np.zeros(data.shape, dtype=np.uint8)
        if gain is None:
            gain = _find_keyword(hdul, gain_header_entry)
        if readnoise is None:
            readnoise = _find_keyword(hdul, readnoise_header_entry)

    if variance is None and estimate_variance:
        in_electrons = str(header.get("BUNIT", "")).strip().lower() in ("electron", "electrons", "e-")
        if readnoise is None or (not in_electrons and gain is None):
            warnings.warn(f"{Path(file_name).name} has no variance extension, and the gain and/or read noise were not found, so the "
                          "variance could not be estimated. Please provide them with the parameters 'gain' and 'readnoise'.")
        else:
            readnoise = float(readnoise)
            if in_electrons:
                variance = np.clip(data, 0, None) + readnoise**2
            else:
                gain = float(gain)
                variance = np.clip(data, 0, None) / gain + (readnoise / gain)**2
            warnings.warn(f"{Path(file_name).name} has no variance extension. The variance was estimated from a simple noise model "
                          "(Poisson + read noise), which neglects the noise of the calibration masters and is therefore a lower limit.")

    return data, variance, mask, header


def read_master(file_name):

    """
    Read a master (or normalized master flat) file, returning its data and, if available, its variance.

    Unlike read_reduced_frame(), the variance is never estimated: for masters it can only be measured from the scatter
    between the combined frames, which is done by cleaning.master().

    Parameters
    ----------
    file_name: string or Path
        Path to the fits file, e.g. "master_bias.fits" or "norm_master_flat.fits".

    Returns
    -------
    data: array
        The master data (primary HDU).
    variance: array or None
        The variance stored in the "VAR" extension, or None if the file has no such extension.
    """

    data, variance, _, _ = read_reduced_frame(file_name, estimate_variance=False)
    return data, variance


# ======================================================================================================================
# 1D spectra
# ======================================================================================================================

DEFAULT_WAVELENGTH_UNIT = u.AA
DEFAULT_FLUX_UNIT = u.erg / u.cm**2 / u.s / u.AA

# Column descriptions used in the text (.dat) files written by previous easyspec versions:
_LEGACY_COLUMNS = {"wavelength": "wavelength",
                   "flux": "flux",
                   "systematic wavelength error": "wavelength_sys_err",
                   "systematic flux error": "flux_sys_err"}


def _as_quantity(values, default_unit):

    """Return 'values' as an astropy Quantity, attaching default_unit if it has no unit."""

    if isinstance(values, u.Quantity):
        return values
    return np.asarray(values, dtype=float) * (u.dimensionless_unscaled if default_unit is None else default_unit)


def write_spectrum(file_name, wavelength, flux, flux_err=None, mask=None, header=None, extra_columns=None, overwrite=True):

    """
    Save a 1D spectrum in the easyspec format: a fits file with the metadata in the primary header and a binary table
    (extension "SPECTRUM") with one row per wavelength.

    Parameters
    ----------
    file_name: string or Path
        Output file (".fits"). Its directory is created if it does not exist.
    wavelength: array or astropy Quantity
        Wavelengths. If no unit is attached, Angstrom is assumed.
    flux: array or astropy Quantity
        Flux (or counts). If no unit is attached, erg/cm2/s/Angstrom is assumed.
    flux_err: array or astropy Quantity
        Optional. 1-sigma statistical uncertainty of the flux. If no unit is attached, the unit of 'flux' is assumed.
    mask: array
        Optional. Bit mask of each wavelength bin (0 = good).
    header: astropy.io.fits.Header or dict
        Optional. Metadata for the primary header (e.g. OBJECT, EXPTIME, AIRMASS).
    extra_columns: dict
        Optional. Other columns to save, as {name: array or Quantity}, e.g. {"flux_sys_err": ..., "wavelength_sys_err": ...}.
        Scalars are broadcast to all rows.
    overwrite: bool
        If True (default), an existing file with the same name is replaced.

    Returns
    -------
    file_name: Path
        The path of the saved file.
    """

    wavelength = _as_quantity(wavelength, DEFAULT_WAVELENGTH_UNIT)
    flux = _as_quantity(flux, DEFAULT_FLUX_UNIT)
    n_rows = len(wavelength)
    if len(flux) != n_rows:
        raise ValueError(f"'wavelength' has {n_rows} elements, but 'flux' has {len(flux)}.")

    table = QTable()
    table["wavelength"] = wavelength
    table["flux"] = flux
    if flux_err is not None:
        table["flux_err"] = _as_quantity(flux_err, flux.unit)
    if mask is not None:
        table["mask"] = np.asarray(mask, dtype=np.uint8)
    for name, values in (extra_columns or {}).items():
        values = values if isinstance(values, u.Quantity) else np.asarray(values)
        if np.ndim(values) == 0:
            values = values * np.ones(n_rows)
        table[name] = values
    for name in table.colnames:
        if len(table[name]) != n_rows:
            raise ValueError(f"Column '{name}' has {len(table[name])} elements, but the spectrum has {n_rows}.")

    primary = fits.PrimaryHDU(header=_clean_header(fits.Header(header) if isinstance(header, dict) else header))
    primary.header[FORMAT_KEYWORD] = (FORMAT_VERSION, "easyspec file format version")
    table_hdu = fits.table_to_hdu(table)
    table_hdu.name = SPECTRUM_EXTNAME

    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, table_hdu]).writeto(file_name, overwrite=overwrite)
    return file_name


def _read_legacy_spectrum(file_name):

    """Read a text spectrum (.dat) written by previous easyspec versions, identifying the columns by the header line."""

    descriptions = []
    with open(file_name) as text_file:
        first_line = text_file.readline()
    if first_line.startswith("#"):
        descriptions = [item.strip() for item in first_line.lstrip("#").split(",")]

    data = np.atleast_2d(np.loadtxt(file_name))
    n_columns = data.shape[1]
    if len(descriptions) != n_columns:
        # No (or inconsistent) header: e.g. stacked spectra, saved with two columns and no header.
        if n_columns < 2:
            raise ValueError(f"{Path(file_name).name} must have at least two columns (wavelength and flux).")
        descriptions = ["wavelength (Angstrom)", "Flux (erg/cm2/s/Angstrom)"] + [f"col{i} ()" for i in range(2, n_columns)]

    table = QTable()
    for column, description in zip(data.T, descriptions):
        if "(" in description:
            label, unit_string = description.split("(", 1)
            unit_string = unit_string.rsplit(")", 1)[0]
        else:
            label, unit_string = description, ""
        label = label.strip()
        name = _LEGACY_COLUMNS.get(label.lower(), label.lower().replace(" ", "_"))
        try:
            unit = u.Unit(unit_string) if unit_string else u.dimensionless_unscaled
        except ValueError:
            unit = u.dimensionless_unscaled
        table[name] = column * unit
    return table


def read_spectrum(file_name):

    """
    Read a 1D spectrum written by write_spectrum() (".fits") or by previous easyspec versions (text ".dat" files).

    Parameters
    ----------
    file_name: string or Path
        Path to the spectrum file.

    Returns
    -------
    spectrum: astropy.table.QTable
        Table with (at least) the columns "wavelength" and "flux", as astropy Quantities, plus any other saved column
        (e.g. "flux_err", "mask", "flux_sys_err", "wavelength_sys_err"). The primary header of fits files is available in
        spectrum.meta["header"].
    """

    file_name = Path(file_name)
    if file_name.suffix.lower() in (".fits", ".fit", ".fts"):
        with fits.open(file_name) as hdul:
            header = hdul[0].header.copy()
        spectrum = QTable.read(file_name, hdu=SPECTRUM_EXTNAME)
        spectrum.meta["header"] = header
    else:
        spectrum = _read_legacy_spectrum(file_name)
        spectrum.meta["header"] = fits.Header()

    for required in ("wavelength", "flux"):
        if required not in spectrum.colnames:
            raise ValueError(f"{file_name.name} has no '{required}' column. Columns found: {spectrum.colnames}.")
    return spectrum
