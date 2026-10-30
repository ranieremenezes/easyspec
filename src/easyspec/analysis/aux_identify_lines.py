"""
Auxiliary functions of analysis.identify_lines(): automatic identification of the lines found by find_lines() and estimate of the redshift.

The redshift is found by pattern matching: every pair (detected line, strong reference line) gives a candidate redshift, and each candidate
is scored by how many detected lines it explains (weighted by the typical strength of the reference lines and by the significance of the
detections), with penalties for strong detections left unexplained and for strong emission lines that should be in the spectrum but were
not detected. The best candidate is refined with the lines it explains. All the detected lines are then named with the reference lines, or
with the easyspec line database (lines/astro_lines.dat), and the remaining absorption lines are checked against telluric bands and against
the doublets of intervening absorbers (Mg II, C IV, Si IV, Ca II).
"""

import numpy as np

SPEED_OF_LIGHT = 299792.458   # km/s

# Strong lines used to find the redshift: (name, rest wavelength, weight). Rest wavelengths follow the easyspec convention (vacuum below
# 2000 A, air above). Blended doublets (unresolved in most spectra) are given at their mean wavelength. The weight is the typical
# prominence of the line (10 = usually the strongest line in its spectral range).
EMISSION_REFERENCES = [
    ("H_Ly_beta", 1025.72, 3), ("O_VI", 1033.82, 4), ("H_Ly_alpha", 1215.67, 10), ("N_V", 1240.14, 4), ("O_I", 1304.35, 2),
    ("C_II", 1335.31, 2), ("Si_IV+O_IV]", 1399.8, 5), ("C_IV", 1549.06, 9), ("He_II", 1640.42, 3), ("O_III]", 1663.48, 2),
    ("Al_III", 1858.75, 2), ("C_III]", 1908.73, 7), ("C_II]", 2325.4, 2), ("Mg_II", 2797.9, 8), ("[Ne_V]", 3425.9, 3),
    ("[O_II]", 3727.4, 7), ("[Ne_III]", 3868.76, 4), ("H_delta", 4101.74, 3), ("H_gamma", 4340.47, 5), ("[O_III]", 4363.21, 2),
    ("He_II", 4685.7, 2), ("H_beta", 4861.33, 8), ("[O_III]", 4958.91, 5), ("[O_III]", 5006.84, 9), ("He_I", 5875.62, 3),
    ("[O_I]", 6300.30, 3), ("[N_II]", 6548.05, 3), ("H_alpha", 6562.80, 10), ("[N_II]", 6583.45, 6), ("[S_II]", 6716.44, 4),
    ("[S_II]", 6730.82, 4), ("[S_III]", 9068.6, 3), ("[S_III]", 9531.1, 3)]

ABSORPTION_REFERENCES = [
    ("H_Ly_alpha", 1215.67, 4), ("Si_IV", 1398.26, 2), ("C_IV", 1549.06, 3), ("Fe_II", 2382.04, 2), ("Fe_II", 2599.40, 2), ("Mg_I", 2852.13, 2),
    ("Mg_II", 2797.9, 4), ("Ca_II_K", 3933.66, 7), ("Ca_II_H", 3968.47, 6), ("H_delta", 4101.74, 4), ("G_band", 4304.4, 3),
    ("H_gamma", 4340.47, 5), ("H_beta", 4861.33, 6), ("Mg_I_b", 5175.4, 4), ("Na_I_D", 5892.9, 5), ("H_alpha", 6562.80, 7),
    ("Ca_II", 8498.02, 3), ("Ca_II", 8542.09, 4), ("Ca_II", 8662.14, 3)]

# Atmospheric absorption bands (observed frame, Angstrom): absorption lines inside them are flagged as telluric and not used.
TELLURIC_BANDS = [(6860, 6925, "O2 B band"), (7160, 7340, "H2O"), (7585, 7700, "O2 A band"), (8120, 8350, "H2O"), (8950, 9800, "H2O")]

# Doublets of intervening absorbers: (name, blue line, red line) in the easyspec convention.
ABSORBER_DOUBLETS = [("Mg_II", 2795.528, 2802.705), ("C_IV", 1548.187, 1550.772), ("Si_IV", 1393.755, 1402.770),
                     ("Ca_II", 3933.66, 3968.47)]


def to_vacuum(wavelength, air_to_vacuum):
    """Rest wavelengths of the easyspec convention converted to vacuum (only those above 2000 A are in air)."""
    wavelength = np.asarray(wavelength, dtype=float)
    return np.where(wavelength < 2000, wavelength, air_to_vacuum(wavelength))


def read_line_database(path):
    """The easyspec line database (rest wavelength, name), e.g. lines/astro_lines.dat."""
    table = np.loadtxt(path, dtype=str, delimiter=",")
    return table[:, 0].astype(float), [name.strip() for name in table[:, 1]]


def telluric_band(observed_wavelength):
    """Name of the telluric band containing an observed wavelength (None if it is outside all bands)."""
    for low, high, name in TELLURIC_BANDS:
        if low <= observed_wavelength <= high:
            return name
    return None


def _match(observed_vacuum, strength, kinds, z, references, tolerance):
    """
    Greedy matching (strongest detections first) of the detected lines to the reference lines at redshift z. Returns, for each detected
    line, the index of its reference line (or -1). Each reference line explains at most one detection.
    """

    match = np.full(len(observed_vacuum), -1)
    used = set()
    for i in np.argsort(-strength):
        table = references[kinds[i]]
        if table is None:
            continue
        velocity = SPEED_OF_LIGHT*np.abs(observed_vacuum[i]/((1 + z)*table["vacuum"]) - 1)
        for j in np.argsort(velocity):
            if velocity[j] > tolerance[i]:
                break
            if (kinds[i], j) not in used:
                match[i] = j
                used.add((kinds[i], j))
                break
    return match


def _score(observed_vacuum, strength, kinds, z, references, tolerance, observed_range):
    """Score of a candidate redshift (see the module docstring), and the matches."""

    match = _match(observed_vacuum, strength, kinds, z, references, tolerance)
    score = 0.0
    for i, j in enumerate(match):
        if j >= 0:
            score += references[kinds[i]]["weight"][j]*strength[i]
        else:
            score -= 2.0*strength[i]
    # Strong emission lines that should be in the spectrum but were not detected (only if the spectrum shows emission lines):
    if np.count_nonzero(np.asarray(kinds) == "emission") >= 2:
        table = references["emission"]
        predicted = table["vacuum"]*(1 + z)
        matched = set(match[(np.asarray(kinds) == "emission") & (match >= 0)])
        inside = (predicted > observed_range[0]*1.01) & (predicted < observed_range[1]*0.99)
        for j in np.flatnonzero(inside & (table["weight"] >= 7)):
            if j not in matched:
                score -= 0.3*table["weight"][j]
    return score, match


def find_redshift(observed_vacuum, significance, kinds, references, tolerance, observed_range, redshift_range):
    """
    Best redshift and the matches of each detected line, plus the list of the best distinct candidates [(z, score, n_matched), ...].
    """

    strength = np.log10(1 + np.abs(significance))
    candidates = []
    for i, kind in enumerate(kinds):
        table = references[kind]
        if table is None:
            continue
        z = observed_vacuum[i]/table["vacuum"] - 1
        candidates.extend(z[(z >= redshift_range[0]) & (z <= redshift_range[1])])
    if not candidates:
        return None, np.full(len(observed_vacuum), -1), []

    results = []
    for z in np.unique(np.round(candidates, 5)):
        score, match = _score(observed_vacuum, strength, kinds, z, references, tolerance, observed_range)
        n_matched = np.count_nonzero(match >= 0)
        if n_matched == 0:
            continue
        # Refined redshift: weighted median of the redshifts of the matched lines
        z_lines = np.array([observed_vacuum[i]/references[kinds[i]]["vacuum"][j] - 1 for i, j in enumerate(match) if j >= 0])
        weights = np.array([references[kinds[i]]["weight"][j]*strength[i] for i, j in enumerate(match) if j >= 0])
        order = np.argsort(z_lines)
        cumulative = np.cumsum(weights[order])
        z_refined = float(z_lines[order][np.searchsorted(cumulative, 0.5*cumulative[-1])])
        score, match = _score(observed_vacuum, strength, kinds, z_refined, references, tolerance, observed_range)
        results.append((z_refined, score, int(np.count_nonzero(match >= 0)), match))
    if not results:
        return None, np.full(len(observed_vacuum), -1), []

    results.sort(key=lambda item: -item[1])
    # Distinct solutions (redshifts separated by more than the tolerance):
    distinct = []
    typical_tolerance = float(np.median(tolerance))
    for z, score, n_matched, match in results:
        if all(SPEED_OF_LIGHT*abs(z - other[0])/(1 + other[0]) > 2*typical_tolerance for other in distinct):
            distinct.append((z, score, n_matched, match))
    best = distinct[0]
    return best[0], best[3], [(z, score, n) for z, score, n, _ in distinct[:5]]


def intervening_doublets(observed, kinds, available, pixel_tolerance, air_to_vacuum, redshift_max):
    """
    Pairs of absorption lines (among 'available') whose wavelength ratio matches the doublet of an intervening absorber. Returns a list of
    (index_blue, index_red, name, redshift).
    """

    found = []
    indices = [i for i in available if kinds[i] == "absorption"]
    observed_vacuum = air_to_vacuum(np.asarray(observed, dtype=float))
    for a in indices:
        for b in indices:
            if observed[b] <= observed[a]:
                continue
            for name, blue, red in ABSORBER_DOUBLETS:
                blue_vacuum, red_vacuum = to_vacuum([blue, red], air_to_vacuum)
                z = observed_vacuum[a]/blue_vacuum - 1
                if not 0 <= z <= redshift_max:
                    continue
                expected_red = red_vacuum*(1 + z)
                if abs(observed_vacuum[b] - expected_red) <= pixel_tolerance:
                    found.append((a, b, name, float(z)))
    return found
