"""
Automatic identification of the lines of arc-lamp spectra, for the wavelength calibration of easyspec (used by
extraction.identify_lamp_lines()).

The user only needs to know which lamp(s) were used (e.g. "HgArNe") and, approximately, the wavelength range covered by the spectrum
(e.g. from the instrument setup: grating and central wavelength). The lines are then identified in three steps:

1. Detection: the peaks of the lamp spectrum are found above a robust estimate of its continuum and noise, and their positions are refined
   to sub-pixel precision.
2. First guess (pattern matching): for a grid of dispersions around the expected one, every pair (detected peak, catalogue line) votes for
   the offset of a linear wavelength solution (a 1D Hough transform). The (dispersion, offset) with the largest number of coincidences is
   the first guess. This works without any initial identification, and is insensitive to missing lines, spurious peaks and blends.
3. Refinement: the lines are matched to the catalogue with decreasing tolerances, while the order of the polynomial wavelength solution is
   increased up to the requested one, with iterative rejection of outliers.

The line lists (in the folder "arc_lines") were taken from PypeIt (https://github.com/pypeit/PypeIt), Copyright (c) 2018-2023, PypeIt
Developers, BSD-3-Clause license (see arc_lines/LICENSE_PypeIt.txt). Their wavelengths are in vacuum, and are converted to air by default.
"""

import re
import warnings
from pathlib import Path
import numpy as np
from scipy import ndimage, signal
from easyspec.extraction import optimal_extraction

ARC_LINES_PATH = Path(__file__).parent.resolve() / "arc_lines"

# Lamp elements that can be given in a string such as "HgArNe" (longest names first, so that "ThAr" is not read as "Th" + "Ar"):
_ELEMENTS = ["ThAr", "Hg", "Ne", "Ar", "He", "Cu", "Cd", "Zn", "Kr", "Xe", "Fe"]


def available_lamps():

    """Names of the line lists distributed with easyspec (e.g. 'HgI', 'NeI', 'ArI', ...)."""

    return sorted(p.stem for p in ARC_LINES_PATH.glob("*.dat"))


def vacuum_to_air(wavelength):

    """Convert vacuum wavelengths (Angstrom, > 2000 A) to air, with the IAU standard formula (Morton 2000, ApJS 130, 403)."""

    wavelength = np.asarray(wavelength, dtype=float)
    s2 = (1e4/wavelength)**2
    n = 1 + 0.0000834254 + 0.02406147/(130 - s2) + 0.00015998/(38.9 - s2)
    return wavelength/n


def air_to_vacuum(wavelength):

    """Convert air wavelengths (Angstrom) to vacuum, inverting vacuum_to_air() iteratively (accuracy < 1e-6 A)."""

    air = np.asarray(wavelength, dtype=float)
    vacuum = air.copy()
    for _ in range(5):
        vacuum = vacuum + (air - vacuum_to_air(vacuum))
    return vacuum


def _lamp_files(lamps):

    """Translate a lamp specification ('HgArNe', 'Hg+Ar', ['HgI', 'NeI'], ...) into the list of line-list names."""

    if isinstance(lamps, str):
        tokens = [t for t in re.split(r"[\s,+/_-]+", lamps) if t]
        names = []
        for token in tokens:
            if token in available_lamps():                  # e.g. "HgI"
                names.append(token)
                continue
            position = 0
            while position < len(token):                     # e.g. "HgArNe" -> Hg, Ar, Ne
                for element in _ELEMENTS:
                    if token[position:position + len(element)].lower() == element.lower():
                        names.append(element if element == "ThAr" else element + "I")
                        position += len(element)
                        break
                else:
                    raise ValueError(f"Unknown lamp element in '{lamps}' (at '{token[position:]}'). Available line lists: "
                                     + ", ".join(available_lamps()) + ".")
        return names
    return list(lamps)


def load_line_list(lamps, medium="air", wavelength_range=None, min_amplitude=0.0):

    """
    Load the catalogue of lamp lines.

    Parameters
    ----------
    lamps: string, list, path or array
        - Lamp names, e.g. "HgArNe", "Hg+Ar+Ne", "CuHeAr" or ["HgI", "NeI"] (see available_lamps()); or
        - the path to a text file with the wavelengths (first column, in Angstrom, in the requested medium) and, optionally, relative
          intensities (second column); or
        - an array with the wavelengths (in the requested medium).
    medium: string
        "air" (default) or "vacuum": the medium of the returned wavelengths (the built-in lists are converted from vacuum if needed).
    wavelength_range: list
        Optional. Only lines within [min, max] are returned.
    min_amplitude: float
        Optional. Only lines with relative amplitude above this value are returned (built-in lists only).

    Returns
    -------
    wavelengths, amplitudes, names: arrays sorted by wavelength.
    """

    if medium not in ("air", "vacuum"):
        raise ValueError("'medium' must be 'air' or 'vacuum'.")
    if isinstance(lamps, (str, Path)) and Path(str(lamps)).is_file():
        table = np.loadtxt(lamps, ndmin=2, usecols=None)
        wavelengths = table[:, 0]
        amplitudes = table[:, 1] if table.shape[1] > 1 else np.ones(len(wavelengths))
        names = np.full(len(wavelengths), Path(str(lamps)).stem)
    elif not isinstance(lamps, str) and np.ndim(lamps) == 1 and len(lamps) > 0 and np.isscalar(lamps[0]) and not isinstance(lamps[0], str):
        wavelengths = np.asarray(lamps, dtype=float)
        amplitudes = np.ones(len(wavelengths))
        names = np.full(len(wavelengths), "user")
    else:
        wavelengths, amplitudes, names = [], [], []
        for name in _lamp_files(lamps):
            path = ARC_LINES_PATH / f"{name}.dat"
            if not path.exists():
                raise ValueError(f"Unknown line list '{name}'. Available: " + ", ".join(available_lamps()) + ".")
            rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
            vacuum = np.array([float(r[0]) for r in rows])
            amplitude = np.array([float(r[1]) for r in rows])
            wavelengths.append(vacuum_to_air(vacuum) if medium == "air" else vacuum)
            amplitudes.append(np.where(amplitude > 0, amplitude, np.nan))
            names.append(np.array([r[2] for r in rows]))
        wavelengths, amplitudes, names = np.concatenate(wavelengths), np.concatenate(amplitudes), np.concatenate(names)
        # Lines without amplitude get the median amplitude of their list:
        amplitudes = np.where(np.isfinite(amplitudes), amplitudes, np.nanmedian(amplitudes) if np.any(np.isfinite(amplitudes)) else 1.0)
        keep = amplitudes >= min_amplitude
        wavelengths, amplitudes, names = wavelengths[keep], amplitudes[keep], names[keep]
    if wavelength_range is not None:
        keep = (wavelengths >= min(wavelength_range)) & (wavelengths <= max(wavelength_range))
        wavelengths, amplitudes, names = wavelengths[keep], amplitudes[keep], names[keep]
    order = np.argsort(wavelengths)
    return wavelengths[order], amplitudes[order], names[order]


def detect_lamp_lines(spectrum, detection_sigma=5.0, min_distance=3):

    """
    Detect the emission lines of a lamp spectrum.

    The continuum is a running median (wider than the lines), and the noise is the robust scatter of the spectrum around it, so that only
    peaks more significant than 'detection_sigma' are kept. The positions are refined to sub-pixel precision with Gaussian fits.

    Returns (positions, heights above the continuum, significance, FWHM of the lines in pixels).
    """

    spectrum = np.nan_to_num(np.asarray(spectrum, dtype=float))
    continuum = ndimage.median_filter(spectrum, size=max(31, len(spectrum)//40), mode="nearest")
    residual = spectrum - continuum
    noise = 1.4826*np.median(np.abs(np.diff(residual)))/np.sqrt(2)   # Differences are insensitive to the lines' wings
    if not noise > 0:
        noise = np.std(residual) if np.std(residual) > 0 else 1.0
    peaks, properties = signal.find_peaks(residual, height=detection_sigma*noise, prominence=detection_sigma*noise,
                                          distance=max(1, int(min_distance)))
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.nan
    positions = optimal_extraction.refine_peak_positions(spectrum, peaks)
    widths = signal.peak_widths(residual, peaks, rel_height=0.5)[0]
    fwhm = float(np.median(widths)) if len(widths) else np.nan
    heights = properties["peak_heights"]
    return positions, heights, heights/noise, fwhm


def _hough_linear_guesses(x_peaks, peak_weights, line_wavelengths, line_weights, nx, wavelength_range, range_tolerance, tolerance_pixels,
                          n_candidates=8):

    """
    First guesses of a linear wavelength solution lambda(x) = offset + dispersion*x, by a Hough transform: for each trial dispersion, each pair
    (peak, catalogue line) votes for an offset, and the (dispersion, offset) bins with the largest (weighted) number of coincidences are the
    candidates. Several distinct candidates are returned (best first), because with rich line lists a wrong solution may occasionally collect
    as many chance coincidences as the right one; the candidates are then refined and compared (see identify_lines()).

    Returns a list of (dispersion, offset, score).
    """

    first, last = wavelength_range
    span = abs(last - first)
    expected = (last - first)/(nx - 1)
    step = tolerance_pixels*abs(expected)/(2.0*nx)                 # The prediction at the detector edge changes by < tolerance/2 per step
    dispersions = np.arange(expected*(1 - range_tolerance), expected*(1 + range_tolerance), np.sign(expected)*step)
    offset_min, offset_max = first - range_tolerance*span, first + range_tolerance*span
    pair_weight = np.sqrt(peak_weights)[:, None]*np.sqrt(line_weights)[None, :]
    trials = []
    for dispersion in dispersions:
        bin_width = tolerance_pixels*abs(dispersion)
        offsets = line_wavelengths[None, :] - dispersion*x_peaks[:, None]
        inside = (offsets >= offset_min) & (offsets <= offset_max)
        if not np.any(inside):
            continue
        for shift in (0.0, 0.5):                                    # Two staggered grids, so that coincidences are not split by bin edges
            edges = np.arange(offset_min - shift*bin_width, offset_max + bin_width, bin_width)
            index = np.digitize(offsets[inside], edges)
            score = np.bincount(index, weights=pair_weight[inside], minlength=len(edges) + 1)
            for k in np.argsort(score)[::-1][:3]:
                if score[k] > 0:
                    center = np.average(offsets[inside][index == k], weights=pair_weight[inside][index == k])
                    trials.append((score[k], dispersion, center))
    trials.sort(key=lambda t: -t[0])
    candidates = []
    for score, dispersion, offset in trials:
        # Distinct solutions: their predictions differ by more than 2 tolerances somewhere on the detector
        distinct = all(max(abs(offset - o), abs(offset + dispersion*(nx - 1) - o - d*(nx - 1))) > 2*tolerance_pixels*abs(dispersion)
                       for d, o, _ in candidates)
        if distinct:
            candidates.append((dispersion, offset, score))
        if len(candidates) >= n_candidates:
            break
    return candidates


def _match(x_peaks, solution, line_wavelengths, tolerance_angstrom):

    """One-to-one matching of peaks and catalogue lines: each peak gets the nearest line within the tolerance (and vice versa)."""

    predicted = solution(x_peaks)
    matched = np.full(len(x_peaks), np.nan)
    if len(line_wavelengths) == 0:
        return matched
    nearest = np.clip(np.searchsorted(line_wavelengths, predicted), 1, len(line_wavelengths) - 1)
    left, right = line_wavelengths[nearest - 1], line_wavelengths[nearest]
    candidate = np.where(np.abs(predicted - left) < np.abs(predicted - right), left, right)
    distance = np.abs(predicted - candidate)
    for i in np.argsort(distance):                                  # Closest pairs first; each catalogue line is used only once
        if distance[i] <= tolerance_angstrom and not np.any(matched == candidate[i]):
            matched[i] = candidate[i]
    return matched


def _clip_outliers(x, wavelengths, valid, order, nx, dispersion, clip_sigma=3.5, max_iterations=10):

    """Polynomial fit with simultaneous rejection of the lines beyond clip_sigma times the robust scatter (never below 0.1 pixel)."""

    used = valid.copy()
    for _ in range(max_iterations):
        polynomial = np.polynomial.Polynomial.fit(x[used], wavelengths[used], order, domain=[0, nx - 1])
        residuals = wavelengths - polynomial(x)
        scale = 1.4826*np.median(np.abs(residuals[used]))
        new_used = valid & (np.abs(residuals) <= max(clip_sigma*scale, 0.1*dispersion))
        if np.count_nonzero(new_used) < order + 2 or np.array_equal(new_used, used):
            break
        used = new_used
    return used


def identify_lines(spectrum, line_wavelengths, line_amplitudes, wavelength_range=None, poly_order=3, detection_sigma=5.0, range_tolerance=0.1,
                   tolerance=None, initial_solution=None, max_peaks=60, line_names=None, max_rms_fwhm=0.1, max_rms_pixels=0.15, min_fraction=0.25):

    """
    Identify the lines of a lamp spectrum and return their positions and wavelengths.

    Parameters
    ----------
    spectrum: array
        The lamp spectrum (counts per pixel along the dispersion axis).
    line_wavelengths, line_amplitudes: arrays
        Catalogue of lamp lines (see load_line_list()).
    wavelength_range: list
        Approximate wavelengths [lambda(first pixel), lambda(last pixel)] (in the order of the pixels; use a decreasing pair if the
        wavelength decreases along x). Not needed if 'initial_solution' is given.
    poly_order: int
        Order of the final polynomial.
    detection_sigma: float
        Detection threshold of the lamp lines, in units of the noise.
    range_tolerance: float
        Relative uncertainty of 'wavelength_range' (default 0.1 = 10% of the range, for both the dispersion and the starting wavelength).
    tolerance: float
        Final matching tolerance in pixels. Default: the FWHM of the lamp lines (at least 1.5 pixels).
    initial_solution: callable
        Optional. A previous wavelength solution lambda(x) (e.g. from another trace), used instead of the pattern matching.
    max_peaks: int
        Maximum number of (brightest) peaks used in the pattern matching.
    line_names: array
        Optional. Name of the element of each catalogue line (see load_line_list()); used to identify blended lines.
    max_rms_fwhm: float
        A solution is accepted only if the RMS of its residuals is smaller than this fraction of the FWHM of the lamp lines (default 0.1),
        and smaller than max_rms_pixels (default 0.15 pixels; increase it for low-resolution spectra with many blended lines).
    min_fraction: float
        A solution is accepted only if it identifies at least this fraction of the detected peaks (default 0.25), and if its lines cover at
        least half of the detector.

    Returns
    -------
    result: dict
        "positions" (all detected peaks), "wavelengths" (catalogue wavelength of each peak, NaN if not identified), "polynomial" (the
        wavelength solution), "rms" (Angstrom), "fwhm" (pixels), "n_identified", "heights".
    """

    spectrum = np.asarray(spectrum, dtype=float)
    nx = len(spectrum)
    positions, heights, significance, fwhm = detect_lamp_lines(spectrum, detection_sigma=detection_sigma)
    if len(positions) < poly_order + 3:
        raise RuntimeError(f"Only {len(positions)} lamp lines were detected above {detection_sigma} sigma, which is not enough to identify them "
                           f"(at least {poly_order + 3} are needed). Try a lower 'detection_sigma'.")
    if not np.isfinite(fwhm):
        fwhm = 3.0
    final_tolerance = tolerance if tolerance is not None else max(1.5, fwhm)

    def refine(solution, start_tolerance, start_order, start_fraction=1.0):
        """
        Iterative refinement: match, fit (with rejection of outliers), increase the order and decrease the tolerance. The region of the
        detector where lines are matched grows from the central 'start_fraction' to the whole detector, so that a linear first guess
        (accurate near the center) is extended progressively to the edges, where the true solution deviates from a straight line.
        """
        order = start_order
        used, matched = None, None
        tolerances = np.concatenate([np.geomspace(start_tolerance, final_tolerance, 6), [final_tolerance]*3])
        fractions = np.concatenate([np.linspace(start_fraction, 1.0, 4), np.ones(len(tolerances) - 4)])
        for tolerance_pixels, fraction in zip(tolerances, fractions):
            dispersion = np.median(np.abs(np.gradient(solution(np.arange(nx)))))
            matched = _match(positions, solution, line_wavelengths, tolerance_pixels*dispersion)
            matched[np.abs(positions - (nx - 1)/2) > fraction*nx/2] = np.nan
            valid = np.isfinite(matched)
            current_order = min(order, poly_order, max(1, np.count_nonzero(valid) - 3))
            if np.count_nonzero(valid) < current_order + 2:
                return None
            used = _clip_outliers(positions, matched, valid, current_order, nx, dispersion)
            solution = np.polynomial.Polynomial.fit(positions[used], matched[used], current_order, domain=[0, nx - 1])
            if order < poly_order and np.count_nonzero(used) >= order + 4:
                order += 1
        if np.count_nonzero(used) < poly_order + 2 or order < poly_order:
            return None
        dispersion = np.median(np.abs(np.gradient(solution(np.arange(nx)))))
        residuals = (matched - solution(positions))[used]
        dof = max(1, np.count_nonzero(used) - (poly_order + 1))
        rms = float(np.sqrt(np.sum(residuals**2)/dof))
        return {"polynomial": solution, "used": used, "matched": matched, "rms": rms, "rms_pixels": rms/dispersion, "dispersion": dispersion}

    results = []
    if initial_solution is None:
        if wavelength_range is None or len(wavelength_range) != 2:
            raise ValueError("Please give the approximate 'wavelength_range' = [lambda(first pixel), lambda(last pixel)] covered by the spectrum.")
        first, last = float(wavelength_range[0]), float(wavelength_range[1])
        span = abs(last - first)
        margin = (range_tolerance + 0.05)*span
        catalogue = (line_wavelengths >= min(first, last) - margin) & (line_wavelengths <= max(first, last) + margin)
        if np.count_nonzero(catalogue) < poly_order + 3:
            raise RuntimeError(f"The line list has only {np.count_nonzero(catalogue)} lines between {min(first, last) - margin:.0f} and "
                               f"{max(first, last) + margin:.0f} A. Check 'lamps' and 'wavelength_range'.")
        # Catalogue lines blended at the resolution of the spectrum (a neighbour within 1.2 FWHM with at least 10% of their amplitude) are
        # not used: the measured position of a blend is not the wavelength of either line.
        # (the amplitudes are only compared within the list of the same element, since they are not comparable between lists)
        blend_width = 1.2*fwhm*abs(last - first)/(nx - 1)
        isolated = np.ones(len(line_wavelengths), dtype=bool)
        groups = line_names if line_names is not None else np.zeros(len(line_wavelengths))
        for i in range(len(line_wavelengths)):
            neighbours = (np.abs(line_wavelengths - line_wavelengths[i]) < blend_width) & (groups == groups[i])
            neighbours[i] = False
            if np.any(neighbours & (line_amplitudes >= 0.1*line_amplitudes[i])):
                isolated[i] = False
        if np.count_nonzero(isolated & catalogue) >= poly_order + 3:
            line_wavelengths, line_amplitudes = line_wavelengths[isolated], line_amplitudes[isolated]
            catalogue = catalogue[isolated]
        # The pattern matching uses the brightest peaks in the central 60% of the detector, where a linear solution is a good approximation
        central = np.abs(positions - (nx - 1)/2) <= 0.3*nx
        pool = np.nonzero(central)[0] if np.count_nonzero(central) >= 10 else np.arange(len(positions))
        brightest = pool[np.argsort(heights[pool])[::-1][:max_peaks]]
        # The pattern matching is done with two tolerances, since a linear solution deviates from the true (curved) one near the edges:
        for hough_tolerance in (2.0*final_tolerance, 4.0*final_tolerance):
            for dispersion, offset, score in _hough_linear_guesses(positions[brightest], significance[brightest], line_wavelengths[catalogue],
                                                                   line_amplitudes[catalogue], nx, (first, last), range_tolerance, hough_tolerance):
                result = refine(np.polynomial.Polynomial([offset, dispersion]), hough_tolerance, 1, start_fraction=0.6)
                if result is not None:
                    results.append(result)
    else:
        # The solution of another trace differs from this one mainly by a small shift (and stretch) along x, due to the tilt of the lines on
        # the detector. So the lines are identified with the other solution transformed as lambda(x) = initial_solution(a + b*x), which has
        # only two free parameters and is therefore not distorted by a few blended or missing lines; a, b are fitted to the coincidences.
        catalogue_sorted = np.sort(line_wavelengths)
        grid = np.arange(nx, dtype=float)
        reference_wavelengths = initial_solution(grid)
        increasing = reference_wavelengths[-1] > reference_wavelengths[0]
        def reference_x(wavelength):   # Inverse of the initial solution
            if increasing:
                return np.interp(wavelength, reference_wavelengths, grid, left=np.nan, right=np.nan)
            return np.interp(wavelength, reference_wavelengths[::-1], grid[::-1], left=np.nan, right=np.nan)
        dispersion = np.median(np.abs(np.gradient(reference_wavelengths)))
        best_shift, best_count = 0.0, -1
        for shift in np.arange(-10, 10.01, 0.1):
            count = np.count_nonzero(np.isfinite(_match(positions + shift, initial_solution, catalogue_sorted, 0.5*final_tolerance*dispersion)))
            if count > best_count:
                best_shift, best_count = shift, count
        a_coefficient, b_coefficient = best_shift, 1.0
        for tolerance_pixels in (final_tolerance, 0.75*final_tolerance, 0.5*final_tolerance):
            transformed = lambda x: initial_solution(a_coefficient + b_coefficient*x)
            matched = _match(positions, transformed, catalogue_sorted, tolerance_pixels*dispersion)
            valid = np.isfinite(matched)
            if np.count_nonzero(valid) < 4:
                break
            x_reference = reference_x(matched[valid])
            keep = np.isfinite(x_reference)
            for _ in range(5):   # Robust linear fit x_reference = a + b*x
                b_coefficient, a_coefficient = np.polyfit(positions[valid][keep], x_reference[keep], 1)
                residuals = x_reference - (a_coefficient + b_coefficient*positions[valid])
                scale = 1.4826*np.median(np.abs(residuals[keep]))
                new_keep = np.isfinite(x_reference) & (np.abs(residuals) <= max(3.5*scale, 0.05))
                if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 4:
                    break
                keep = new_keep
        transformed = lambda x: initial_solution(a_coefficient + b_coefficient*x)
        matched = _match(positions, transformed, catalogue_sorted, 0.5*final_tolerance*dispersion)
        valid = np.isfinite(matched)
        if np.count_nonzero(valid) >= poly_order + 3:
            # A line is kept if it agrees with the transformed reference solution (robust near the edges, where a polynomial fit can be
            # pulled by a blend) or with a polynomial fit of the lines of this spectrum (lines not used in the reference solution).
            x_reference = reference_x(matched)
            residuals = x_reference - (a_coefficient + b_coefficient*positions)
            scale = 1.4826*np.nanmedian(np.abs(residuals[valid]))
            agrees_with_reference = valid & np.isfinite(residuals) & (np.abs(np.nan_to_num(residuals, nan=np.inf)) <= max(3.5*scale, 0.1))
            used = agrees_with_reference | _clip_outliers(positions, matched, valid, poly_order, nx, dispersion)
            solution = np.polynomial.Polynomial.fit(positions[used], matched[used], poly_order, domain=[0, nx - 1])
            residuals = (matched - solution(positions))[used]
            dof = max(1, np.count_nonzero(used) - (poly_order + 1))
            rms = float(np.sqrt(np.sum(residuals**2)/dof))
            results.append({"polynomial": solution, "used": used, "matched": matched, "rms": rms, "rms_pixels": rms/dispersion,
                            "dispersion": dispersion})

    # Wrong solutions with few coincidences are also excluded: a correct solution identifies a good fraction of the detected peaks, and
    # its lines cover most of the detector.
    for r in results:
        x_used = positions[r["used"]]
        r["coverage"] = (x_used.max() - x_used.min())/(nx - 1) if len(x_used) else 0.0
        r["fraction"] = np.count_nonzero(r["used"])/len(positions)
    results = [r for r in results if r["fraction"] >= min_fraction and r["coverage"] >= 0.5]
    best_rms_pixels = min((r["rms_pixels"] for r in results), default=np.inf)
    rms_limit = min(max_rms_fwhm*fwhm, max_rms_pixels)
    good = [r for r in results if r["rms_pixels"] < min(rms_limit, max(0.3*rms_limit, 1.5*best_rms_pixels))]
    if not good:
        best_rms = min((r["rms_pixels"] for r in results), default=np.nan)   # among the solutions with enough lines
        raise RuntimeError("The lamp lines could not be identified reliably (best RMS = " + (f"{best_rms:.2f} px" if np.isfinite(best_rms) else "n/a")
                           + f", for lines with FWHM = {fwhm:.1f} px"
                           + "). Please check the lamp(s) (only the elements present in the lamp should be given), the 'wavelength_range' (and its "
                           "orientation: [lambda(first pixel), lambda(last pixel)]), or increase 'range_tolerance'.")
    best = max(good, key=lambda r: (np.count_nonzero(r["used"]), -r["rms_pixels"]))
    solution, used, matched = best["polynomial"], best["used"], best["matched"]

    # Recovery of the lines beyond the first/last identified line: near the edges of the detector the solution is extrapolated during the
    # refinement, so lines there may be lost (e.g. Hg 3650 A at the blue end). The peaks beyond the identified range are added one at a time,
    # starting from the strongest, if they match a catalogue line within max(3.5 x the scatter of the residuals, 0.1 FWHM); the solution is refitted
    # after each addition. Only the edges are searched, which keeps the chance of a false identification small; the final rejection of
    # outliers is done by wavelength_calibration().
    used, matched = used.copy(), matched.copy()
    dispersion = np.median(np.abs(np.gradient(solution(np.arange(nx, dtype=float)))))
    catalogue_all = np.sort(line_wavelengths)
    for side in (-1, 1):
        while True:
            x_edge = positions[used].min() if side < 0 else positions[used].max()
            candidates = np.nonzero(((positions < x_edge) if side < 0 else (positions > x_edge)) & ~used)[0]
            residuals_used = (matched[used] - solution(positions[used]))/dispersion
            limit = max(3.5*1.4826*np.median(np.abs(residuals_used)), 0.1*fwhm)
            added = False
            for peak in candidates[np.argsort(heights[candidates])[::-1]]:   # The strongest peaks first (blends are usually weaker)
                line = _match(positions[[peak]], solution, catalogue_all, limit*dispersion)[0]
                if np.isfinite(line) and not np.any(matched[used] == line):
                    matched[peak], used[peak] = line, True
                    solution = np.polynomial.Polynomial.fit(positions[used], matched[used], poly_order, domain=[0, nx - 1])
                    added = True
                    break
            if not added:
                break
    wavelengths = np.where(used, matched, np.nan)
    residuals = (wavelengths - solution(positions))[used]
    dof = max(1, np.count_nonzero(used) - (poly_order + 1))
    return {"positions": positions, "wavelengths": wavelengths, "polynomial": solution, "rms": float(np.sqrt(np.sum(residuals**2)/dof)),
            "fwhm": fwhm, "n_identified": int(np.count_nonzero(used)), "heights": heights, "n_candidates": len(results),
            "rms_pixels": float(np.sqrt(np.sum(residuals**2)/dof))/np.median(np.abs(np.gradient(solution(np.arange(nx, dtype=float)))))}