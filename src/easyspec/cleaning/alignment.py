"""
easyspec alignment module.

This module contains the machinery used by cleaning.align() to align photometric images and long-slit spectra,
allowing for translations, rotations and (optionally) changes of plate scale. Users normally do not need to call
these functions directly: please see the documentation of cleaning.align().

Coordinate convention: points are (x, y) = (column, row) pixel coordinates. A transformation maps an image onto its
reference as p_ref = matrix @ p_img + offset.
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from astropy.stats import sigma_clipped_stats


VALID_METHODS = ["auto", "stars", "correlation"]


def align_images(data, groups, method="auto", reference=None, allow_rotation=True, allow_scale=False, subpixel=True,
                 interpolation_order=3, crop=True, fill_value=0.0, fwhm=4.0, detection_sigma=5.0, max_stars=60,
                 match_tolerance=2.0, min_matches=6, median_prefilter=3, plot=False, verbose=True):

    """
    Align groups of images, each one onto the reference image of its own group. This is the engine behind
    cleaning.align(); see that function for a full description of the parameters.

    Parameters
    ----------
    data: dict
        A dictionary containing the paths to the data files (keys) and the image data (values). All entries are
        returned, but only those listed in 'groups' are aligned. The remaining ones are only cropped (if crop=True).
    groups: dict
        Dictionary {group name or directory path: list of keys of 'data'} with the images to be aligned together.
    Other parameters:
        See cleaning.align().

    Returns
    -------
    aligned_data: dict
        Dictionary with the same keys as 'data', containing the aligned (and possibly cropped) images.
    transforms: dict
        Transformation found for each aligned image (see transform_record()).
    crop_box: tuple or None
        (x1, x2, y1, y2) limits of the crop applied to all images, or None if crop=False.
    """

    # ---------------------- Input validation ----------------------
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Options are: " + ", ".join(f'"{m}"' for m in VALID_METHODS) + ".")
    if not isinstance(interpolation_order, (int, np.integer)) or not 0 <= interpolation_order <= 5:
        raise ValueError("'interpolation_order' must be an integer between 0 and 5.")
    if not subpixel and (allow_rotation or allow_scale):
        warnings.warn("subpixel=False: only integer-pixel translations will be applied. "
                      "'allow_rotation' and 'allow_scale' are ignored.")
        allow_rotation, allow_scale = False, False

    align_list = [name for files in groups.values() for name in files]
    reference_path = None
    if isinstance(reference, str):
        reference_path = str(Path(reference).resolve())
        if reference_path not in align_list:
            raise ValueError("The reference image given in 'reference' is not among the images to be aligned.")
    elif reference is not None and not isinstance(reference, (int, np.integer)):
        raise TypeError("'reference' must be None, an integer, or a string with the path to an image.")

    # ---------------------- Alignment ----------------------
    aligned_data = {name: np.asarray(image) for name, image in data.items()}
    footprints = {}
    transforms = {}
    reference_of = {}
    common_shape = None

    for group_name, files in groups.items():
        if reference_path is not None and reference_path in files:
            reference_name = reference_path
        elif isinstance(reference, (int, np.integer)):
            if not -len(files) <= reference < len(files):
                raise IndexError(f"'reference'={reference} is out of range for the group {Path(group_name).name}, "
                                 f"which has {len(files)} images.")
            reference_name = files[reference]
        else:
            reference_name = files[0]

        reference_image = np.nan_to_num(np.asarray(data[reference_name], dtype=float))
        reference_estimate = prefilter(reference_image, median_prefilter)
        shape = reference_image.shape
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape and crop:
            raise RuntimeError("Images in different groups have different shapes, so they cannot be cropped "
                               "to a common box. Please use crop=False or align each Type separately.")

        if verbose:
            print(f"Group '{Path(group_name).name}' - reference image: {Path(reference_name).name}")

        reference_stars = None
        center = np.array([(shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0])

        for name in files:
            reference_of[name] = reference_name
            if name == reference_name:
                transforms[name] = transform_record(np.eye(2), np.zeros(2), center, "reference", None, None)
                aligned_data[name] = reference_image.copy()
                footprints[name] = np.ones(shape, dtype=bool)
                continue

            image = np.nan_to_num(np.asarray(data[name], dtype=float))
            if image.shape != shape:
                raise RuntimeError(f"Image {Path(name).name} has shape {image.shape}, while its reference has "
                                   f"shape {shape}.")
            image_estimate = prefilter(image, median_prefilter)

            solution = None
            if method in ("auto", "stars"):
                if reference_stars is None:
                    reference_stars = detect_stars(reference_estimate, fwhm, detection_sigma, max_stars)
                solution = transform_from_stars(reference_stars, image_estimate, fwhm, detection_sigma, max_stars,
                                                match_tolerance, min_matches, allow_rotation, allow_scale)
                if solution is None:
                    if method == "stars":
                        raise RuntimeError(f"Could not find a reliable match (at least {min_matches} stars, not all "
                                           f"collinear) between {Path(name).name} and its reference. Try "
                                           "method='correlation', or adjust 'fwhm', 'detection_sigma', "
                                           "'match_tolerance' or 'min_matches'.")
                    if verbose:
                        print(f"  {Path(name).name}: no reliable star match, switching to method='correlation'.")

            if solution is None:
                solution = transform_from_correlation(reference_estimate, image_estimate, allow_rotation)

            matrix, offset, info = solution
            if not subpixel:
                matrix = np.eye(2)
                offset = np.round(offset)
            order = interpolation_order if subpixel else 0

            aligned_data[name] = warp(image, matrix, offset, shape, order=order, cval=0.0)
            footprint = warp(np.ones(shape), matrix, offset, shape, order=min(order, 1), cval=0.0) > 0.999
            if order > 1:
                # Spline interpolation may ring close to the edges of the field of view: we discard a thin margin.
                footprint = ndimage.binary_erosion(footprint, iterations=order // 2 + 1)
            footprints[name] = footprint

            record = transform_record(matrix, offset, center, info["method"], info.get("n_matches"), info.get("rms"))
            transforms[name] = record
            if verbose:
                message = (f"  {Path(name).name}: method = {record['method']}, rotation = {record['rotation_deg']:+.3f} deg, "
                           f"center shift (dx, dy) = ({record['center_shift'][0]:+.2f}, {record['center_shift'][1]:+.2f}) px")
                if allow_scale:
                    message += f", scale = {record['scale']:.5f}"
                if record["n_matches"] is not None:
                    message += f", matched stars = {record['n_matches']}, rms = {record['rms']:.3f} px"
                print(message)

    # ---------------------- Crop or fill ----------------------
    crop_box = None
    if crop:
        common_valid = np.ones(common_shape, dtype=bool)
        for footprint in footprints.values():
            common_valid &= footprint
        y1, y2, x1, x2 = largest_valid_box(common_valid)
        crop_box = (x1, x2, y1, y2)
        for name in aligned_data:
            if aligned_data[name].shape == common_shape:
                aligned_data[name] = aligned_data[name][y1:y2, x1:x2]
            else:
                warnings.warn(f"{Path(name).name} was not cropped because its shape differs from the aligned images.")
        if verbose:
            print(f"Final crop box (x1, x2, y1, y2): ({x1}, {x2}, {y1}, {y2}). "
                  f"New shape: {y2 - y1} x {x2 - x1} (rows x columns).")
    else:
        for name, footprint in footprints.items():
            aligned_data[name] = aligned_data[name].copy()
            aligned_data[name][~footprint] = fill_value

    if plot:
        plot_alignment(aligned_data, transforms, reference_of)

    return aligned_data, transforms, crop_box


def plot_alignment(aligned_data, transforms, reference_of):

    """
    For each aligned image, plot a color composite of the reference image (magenta) and the aligned image (green).
    Well-aligned sources appear white/gray; misalignments show colored doublets.
    """

    for name, reference_name in reference_of.items():
        if name == reference_name:
            continue
        reference_view = display_scale(aligned_data[reference_name])
        aligned_view = display_scale(aligned_data[name])
        composite = np.dstack([reference_view, aligned_view, reference_view])
        plt.figure(figsize=(8, 8 * composite.shape[0] / composite.shape[1]))
        record = transforms[name]
        plt.title(f"{Path(name).name} (green) vs. reference (magenta)\n"
                  f"rotation = {record['rotation_deg']:+.3f} deg, center shift = "
                  f"({record['center_shift'][0]:+.2f}, {record['center_shift'][1]:+.2f}) px")
        plt.imshow(composite, origin="lower")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
    plt.show()


def prefilter(image, size):

    """Median-filtered copy of the image, used only to estimate the transformation (see 'median_prefilter')."""

    if size is None or size <= 1:
        return image
    return ndimage.median_filter(image, size=int(size), mode="nearest")


def transform_record(matrix, offset, center, method, n_matches, rms):

    """Summarize a transformation p_ref = matrix @ p_img + offset in a dictionary."""

    scale = float(np.sqrt(abs(np.linalg.det(matrix))))
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    center_shift = matrix @ center + offset - center
    return {"matrix": matrix, "offset": offset, "rotation_deg": rotation, "scale": scale,
            "center_shift": center_shift, "method": method, "n_matches": n_matches, "rms": rms}


def detect_stars(image, fwhm, detection_sigma, max_stars):

    """
    Detect point sources and return their centroids as an (N, 2) array of (x, y) pixel coordinates, sorted from
    the brightest to the faintest source.
    """

    image = np.nan_to_num(np.asarray(image, dtype=float))
    step = max(1, int(np.sqrt(image.size / 250000)))  # Subsampling only to speed up the background statistics
    _, background, _ = sigma_clipped_stats(image[::step, ::step], sigma=3.0, maxiters=5)

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    smoothed = ndimage.gaussian_filter(image - background, sigma)  # Matched filter for a Gaussian-like PSF
    _, smoothed_median, smoothed_std = sigma_clipped_stats(smoothed[::step, ::step], sigma=3.0, maxiters=5)

    size = 2 * int(np.ceil(fwhm)) + 1
    peaks = (smoothed == ndimage.maximum_filter(smoothed, size=size, mode="nearest"))
    peaks &= smoothed > smoothed_median + detection_sigma * smoothed_std
    border = int(np.ceil(2 * fwhm))
    peaks[:border, :] = False
    peaks[-border:, :] = False
    peaks[:, :border] = False
    peaks[:, -border:] = False

    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return np.empty((0, 2))
    brightest = np.argsort(smoothed[ys, xs])[::-1][:max_stars]
    ys, xs = ys[brightest], xs[brightest]

    # Intensity-weighted centroids inside a small box around each peak:
    half = max(2, int(np.ceil(fwhm)))
    offsets = np.arange(-half, half + 1)
    positions = []
    for y, x in zip(ys, xs):
        stamp = image[y - half:y + half + 1, x - half:x + half + 1] - background
        stamp = np.clip(stamp, 0.0, None)
        total = stamp.sum()
        if total <= 0:
            positions.append([float(x), float(y)])
            continue
        positions.append([x + np.sum(stamp.sum(axis=0) * offsets) / total,
                          y + np.sum(stamp.sum(axis=1) * offsets) / total])
    return np.array(positions)


def triangles(points, n_neighbors=4):

    """
    Build triangles from each point and its nearest neighbors. Returns the vertex indices of each triangle,
    ordered canonically (vertex opposite to the longest side first), and two invariants (ratios of the sorted
    side lengths) that do not change under translation, rotation or scaling.
    """

    n_points = len(points)
    if n_points < 3:
        return np.empty((0, 3), dtype=int), np.empty((0, 2))

    k = min(n_neighbors + 1, n_points)
    _, neighbors = cKDTree(points).query(points, k=k)
    triangles = set()
    for row in np.atleast_2d(neighbors):
        for combination in combinations(row, 3):
            triangles.add(tuple(sorted(combination)))
    triangles = np.array(sorted(triangles), dtype=int)

    p0, p1, p2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    opposite_sides = np.stack([np.linalg.norm(p1 - p2, axis=1),   # side opposite to vertex 0
                               np.linalg.norm(p0 - p2, axis=1),   # side opposite to vertex 1
                               np.linalg.norm(p0 - p1, axis=1)],  # side opposite to vertex 2
                              axis=1)
    order = np.argsort(opposite_sides, axis=1)[:, ::-1]  # longest side first
    sides = np.take_along_axis(opposite_sides, order, axis=1)
    triangles = np.take_along_axis(triangles, order, axis=1)

    valid = sides[:, 2] > 1e-3 * sides[:, 0]  # discard degenerate (collinear) triangles
    invariants = np.stack([sides[:, 1] / sides[:, 0], sides[:, 2] / sides[:, 1]], axis=1)
    return triangles[valid], invariants[valid]


def fit_transform(source, destination, allow_rotation, allow_scale):

    """
    Least-squares fit of destination = A @ source + t (Umeyama 1991), where A is a rotation (optionally times
    an isotropic scale). Reflections are not allowed.
    """

    source_mean, destination_mean = source.mean(axis=0), destination.mean(axis=0)
    if not allow_rotation:
        return np.eye(2), destination_mean - source_mean

    source_c, destination_c = source - source_mean, destination - destination_mean
    covariance = destination_c.T @ source_c / len(source)
    U, singular_values, Vt = np.linalg.svd(covariance)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    rotation = U @ D @ Vt
    scale = 1.0
    if allow_scale:
        variance = np.mean(np.sum(source_c ** 2, axis=1))
        scale = np.trace(np.diag(singular_values) @ D) / variance
    matrix = scale * rotation
    return matrix, destination_mean - matrix @ source_mean


def transform_from_stars(reference_stars, image, fwhm, detection_sigma, max_stars, match_tolerance,
                    min_matches, allow_rotation, allow_scale, max_candidates=3000):

    """
    Estimate the image -> reference transformation by matching triangle asterisms (RANSAC-like search over the
    candidate triangle pairs, followed by an iterative least-squares refinement on all matched stars).
    Returns (matrix, offset, info) or None if the match fails.
    """

    image_stars = detect_stars(image, fwhm, detection_sigma, max_stars)
    if len(reference_stars) < 3 or len(image_stars) < 3:
        return None

    reference_triangles, reference_invariants = triangles(reference_stars)
    image_triangles, image_invariants = triangles(image_stars)
    if len(reference_triangles) == 0 or len(image_triangles) == 0:
        return None

    # Candidate pairs of similar triangles:
    distances, indices = cKDTree(image_invariants).query(reference_invariants, k=min(5, len(image_invariants)),
                                                         distance_upper_bound=0.03)
    distances, indices = np.atleast_2d(distances.T).T, np.atleast_2d(indices.T).T
    reference_index, column = np.nonzero(np.isfinite(distances))
    if len(reference_index) == 0:
        return None
    image_index = indices[reference_index, column]
    best_first = np.argsort(distances[reference_index, column])[:max_candidates]

    reference_tree = cKDTree(reference_stars)
    best_count, best_solution = 0, None
    for candidate in best_first:
        source = image_stars[image_triangles[image_index[candidate]]]
        destination = reference_stars[reference_triangles[reference_index[candidate]]]
        matrix, offset = fit_transform(source, destination, allow_rotation, allow_scale)
        if np.max(np.linalg.norm(source @ matrix.T + offset - destination, axis=1)) > 3 * match_tolerance:
            continue
        distance, _ = reference_tree.query(image_stars @ matrix.T + offset)
        count = int(np.sum(distance < match_tolerance))
        if count > best_count:
            best_count, best_solution = count, (matrix, offset)
            if count >= 0.8 * min(len(image_stars), len(reference_stars)):
                break  # Excellent solution, no need to keep searching

    if best_solution is None or best_count < min_matches:
        return None

    # Iterative least-squares refinement with all matched stars (one-to-one matches only):
    matrix, offset = best_solution
    for _ in range(3):
        distance, match = reference_tree.query(image_stars @ matrix.T + offset)
        good = distance < match_tolerance
        if np.sum(good) < min_matches:
            return None
        image_index_good, reference_index_good, distance_good = np.nonzero(good)[0], match[good], distance[good]
        keep = {}
        for i, j, d in zip(image_index_good, reference_index_good, distance_good):
            if j not in keep or d < keep[j][1]:
                keep[j] = (i, d)
        pairs = np.array([[i, j] for j, (i, _) in keep.items()])
        matrix, offset = fit_transform(image_stars[pairs[:, 0]], reference_stars[pairs[:, 1]],
                                                   allow_rotation, allow_scale)

    residuals = np.linalg.norm(image_stars[pairs[:, 0]] @ matrix.T + offset - reference_stars[pairs[:, 1]], axis=1)
    if len(pairs) < min_matches:
        return None

    # Geometry check: if the matched sources are nearly collinear (e.g. emission knots along a single spectral
    # trace), rotation and scale are not constrained and the solution cannot be trusted.
    matched = reference_stars[pairs[:, 1]]
    spread = np.linalg.svd(matched - matched.mean(axis=0), compute_uv=False)
    if spread[1] < 0.1 * spread[0]:
        return None
    return matrix, offset, {"method": "stars", "n_matches": int(len(pairs)),
                            "rms": float(np.sqrt(np.mean(residuals ** 2)))}


def prepare(image):

    """Remove the background and apply a Hann window to reduce edge effects in the Fourier transforms."""

    image = np.nan_to_num(np.asarray(image, dtype=float))
    image = image - np.median(image)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    return image * window


def phase_correlation(reference, image):

    """
    Phase correlation between two images of the same shape. Returns the shift (dx, dy) such that
    reference(p) ~ image(p - shift), with sub-pixel precision, and the height of the correlation peak.
    """

    cross_power = np.fft.fft2(reference) * np.conj(np.fft.fft2(image))
    cross_power /= np.abs(cross_power) + 1e-12
    correlation = np.real(np.fft.ifft2(cross_power))
    correlation = ndimage.gaussian_filter(correlation, 1.0, mode="wrap")  # suppresses noise-dominated frequencies

    ny, nx = correlation.shape
    py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = correlation[py, px]

    def parabolic(minus, center, plus):
        denominator = minus - 2 * center + plus
        return 0.0 if denominator == 0 else 0.5 * (minus - plus) / denominator

    dy = py + parabolic(correlation[(py - 1) % ny, px], peak, correlation[(py + 1) % ny, px])
    dx = px + parabolic(correlation[py, (px - 1) % nx], peak, correlation[py, (px + 1) % nx])
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return np.array([dx, dy]), peak


def rotation_from_fourier(reference, image, n_angles=720, snr_threshold=3.0):

    """
    Estimate the rotation angle (degrees, modulo 180) between two prepared images from the polar-resampled
    amplitudes of their Fourier transforms, which are insensitive to translations.

    Only spatial frequencies where the amplitude is clearly above the white-noise floor (estimated at the highest
    frequencies) are used. Frequencies beyond the PSF cutoff carry only noise and would wash out the signal, and
    the cutoff depends on the seeing, so it is determined from the data.
    """

    ny, nx = reference.shape
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)), np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
    cosine = np.cos(np.pi * fx) * np.cos(np.pi * fy)
    high_pass = (1.0 - cosine) * (2.0 - cosine)  # suppresses the dominant (uninformative) lowest frequencies

    amplitudes = [np.abs(np.fft.fftshift(np.fft.fft2(prepared))) for prepared in (reference, image)]

    # Polar grid in frequency units (cycles/pixel). Using frequency units makes it valid for non-square images.
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    radii = np.linspace(0.02, 0.48, max(100, min(ny, nx) // 2))
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    coordinates = [ny // 2 + rr * np.sin(aa) * ny, nx // 2 + rr * np.cos(aa) * nx]

    # Keep only the radii where both images have signal above the noise floor:
    keep = np.ones(len(radii), dtype=bool)
    for amplitude in amplitudes:
        profile = np.median(ndimage.map_coordinates(amplitude, coordinates, order=1, mode="nearest"), axis=1)
        noise_floor = np.median(profile[radii > 0.42])
        keep &= profile > snr_threshold * noise_floor
    if np.count_nonzero(keep) < 8:
        keep = radii <= 0.2  # Very low signal-to-noise: fall back to a conservative frequency range

    polar = []
    for amplitude in amplitudes:
        sampled = ndimage.map_coordinates(amplitude * high_pass, coordinates, order=1, mode="nearest")[keep]
        sampled -= sampled.mean(axis=1, keepdims=True)
        polar.append(sampled)

    # Circular cross-correlation along the angle axis, summed over all radii:
    spectrum = np.sum(np.fft.fft(polar[0], axis=1) * np.conj(np.fft.fft(polar[1], axis=1)), axis=0)
    correlation = np.real(np.fft.ifft(spectrum))
    k = int(np.argmax(correlation))
    minus, center, plus = correlation[(k - 1) % n_angles], correlation[k], correlation[(k + 1) % n_angles]
    denominator = minus - 2 * center + plus
    k = k + (0.0 if denominator == 0 else 0.5 * (minus - plus) / denominator)
    angle = k * 180.0 / n_angles
    if angle >= 90.0:
        angle -= 180.0
    return angle


def transform_from_correlation(reference, image, allow_rotation):

    """
    Fourier-based estimate of the image -> reference transformation (rotation + translation).
    Returns (matrix, offset, info).
    """

    shape = reference.shape
    center = np.array([(shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0])
    prepared_reference = prepare(reference)

    if not allow_rotation:
        shift, _ = phase_correlation(prepared_reference, prepare(image))
        return np.eye(2), shift, {"method": "correlation"}

    angle = rotation_from_fourier(prepared_reference, prepare(image))

    # The Fourier amplitudes cannot distinguish between theta and theta + 180 deg. We test both.
    best = None
    for candidate in (angle, angle + 180.0):
        theta = np.radians(candidate)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotation_offset = center - rotation @ center  # rotation about the image center
        rotated = warp(image, rotation, rotation_offset, shape, order=1, cval=0.0)
        shift, peak = phase_correlation(prepared_reference, prepare(rotated))
        if best is None or peak > best[0]:
            best = (peak, rotation, rotation_offset + shift)

    _, matrix, offset = best
    return matrix, offset, {"method": "correlation"}


def warp(image, matrix, offset, output_shape, order=3, cval=0.0):

    """
    Resample 'image' onto the reference grid, given p_ref = matrix @ p_img + offset with p = (x, y).
    """

    inverse = np.linalg.inv(matrix)
    # scipy.ndimage works with (row, column) = (y, x), so we swap both axes of the matrix and the offset:
    matrix_yx = inverse[::-1, ::-1]
    offset_yx = -(inverse @ offset)[::-1]
    return ndimage.affine_transform(image, matrix_yx, offset=offset_yx, output_shape=output_shape, order=order,
                                    mode="constant", cval=cval, prefilter=order > 1)


def largest_valid_box(valid):

    """
    Find a large axis-aligned box containing only valid pixels. The valid region (intersection of affine
    footprints) is convex, so the box is valid once its four edges are; we shrink greedily the worst edge.
    Returns (y1, y2, x1, x2).
    """

    y1, y2, x1, x2 = 0, valid.shape[0], 0, valid.shape[1]
    while y2 - y1 > 1 and x2 - x1 > 1:
        invalid = [np.count_nonzero(~valid[y1, x1:x2]), np.count_nonzero(~valid[y2 - 1, x1:x2]),
                   np.count_nonzero(~valid[y1:y2, x1]), np.count_nonzero(~valid[y1:y2, x2 - 1])]
        if max(invalid) == 0:
            return y1, y2, x1, x2
        fraction = [invalid[0] / (x2 - x1), invalid[1] / (x2 - x1), invalid[2] / (y2 - y1), invalid[3] / (y2 - y1)]
        worst = int(np.argmax(fraction))
        if worst == 0:
            y1 += 1
        elif worst == 1:
            y2 -= 1
        elif worst == 2:
            x1 += 1
        else:
            x2 -= 1
    raise RuntimeError("The aligned images have no common valid region. Please check the alignment solutions.")


def display_scale(image):

    """Asinh stretch between the 1st and 99.7th percentiles, mapped to [0, 1], for display purposes."""

    low, high = np.nanpercentile(image, [1.0, 99.7])
    scaled = np.clip((image - low) / max(high - low, 1e-12), 0.0, 1.0)
    return np.arcsinh(10 * scaled) / np.arcsinh(10)


# ======================================================================================================================
# Propagation of variance frames and data quality masks through the alignment (used by cleaning.align())
# ======================================================================================================================
 
def interpolation_footprint(matrix, offset, shape, order):
 
    """
    Boolean map of the output pixels that are fully covered by the input image after the transformation. This is the
    same criterion used by align_images() to define the valid region of each aligned image.
    """
 
    footprint = warp(np.ones(shape), matrix, offset, shape, order=min(order, 1), cval=0.0) > 0.999
    if order > 1:
        footprint = ndimage.binary_erosion(footprint, iterations=order // 2 + 1)
    return footprint
 
 
def source_coordinates(matrix, offset, output_shape):
 
    """
    For each pixel of the output (reference) grid, return the (x, y) coordinates of the point of the input image that
    is mapped onto it, given p_ref = matrix @ p_img + offset.
    """
 
    inverse = np.linalg.inv(matrix)
    y_out, x_out = np.indices(output_shape, dtype=float)
    x_shifted, y_shifted = x_out - offset[0], y_out - offset[1]
    x_source = inverse[0, 0] * x_shifted + inverse[0, 1] * y_shifted
    y_source = inverse[1, 0] * x_shifted + inverse[1, 1] * y_shifted
    return x_source, y_source
 
 
_SPLINE_KERNELS = {}
 
def interpolation_kernel(order):
 
    """
    Return (half_width, eta), where eta(t) is the weight given by the spline interpolation of the given order to a
    sample located at a distance t (in pixels) from the interpolation point, and half_width is the number of samples
    on each side that carry non-negligible weight. For order >= 2, scipy's spline interpolation is equivalent to a
    convolution with the "cardinal spline" eta, which is computed here numerically as the response to a unit impulse.
    """
 
    if order not in _SPLINE_KERNELS:
        if order == 0:
            _SPLINE_KERNELS[order] = (1, lambda t: (np.abs(t) < 0.5).astype(float))
        elif order == 1:
            _SPLINE_KERNELS[order] = (1, lambda t: np.clip(1.0 - np.abs(t), 0.0, None))
        else:
            half_width = 6 if order <= 3 else 9
            length = 8 * half_width + 1
            impulse = np.zeros(length)
            impulse[length // 2] = 1.0
            t_grid = np.linspace(-half_width, half_width, 4000 * half_width + 1)
            values = ndimage.map_coordinates(impulse, [length // 2 + t_grid], order=order, mode="mirror")
            _SPLINE_KERNELS[order] = (half_width, lambda t, t_grid=t_grid, values=values: np.interp(t, t_grid, values, left=0.0, right=0.0))
    return _SPLINE_KERNELS[order]
 
 
def warp_variance(variance, matrix, offset, output_shape, order, mode="aperture"):
 
    """
    Transform a variance frame with the same geometric transformation applied to its image by warp().
 
    Interpolation makes each output pixel a weighted sum of input pixels, I_out = sum_k w_k I_k, which has two
    consequences: the variance of a single output pixel is sum_k w_k^2 V_k, but neighbouring output pixels share input
    pixels, so their noise is correlated. Two conventions are therefore available:
 
    - mode="aperture" (default): V_out = sum_k w_k V_k (bilinear weights). Each input pixel distributes its variance
      among the output pixels exactly as its flux, so the total variance is conserved and the variance of a sum over
      any region larger than ~2 pixels (an extraction aperture, a photometric aperture, a spectral bin) is correct
      when the output pixels are treated as independent. Single-pixel variances are overestimated (by up to 4x for
      half-pixel shifts with bilinear interpolation).
    - mode="pixel": V_out = sum_k w_k^2 V_k, with the exact weights of the interpolation of the given order. Correct
      for each individual pixel, but summing these values over a region underestimates the variance of the sum.
 
    With order=0 (integer shifts, i.e. align(..., subpixel=False)) both conventions coincide and are exact.
    """
 
    variance = np.asarray(variance, dtype=float)
    if mode not in ("aperture", "pixel"):
        raise ValueError(f"Invalid variance mode '{mode}'. Options are: 'aperture' and 'pixel'.")
 
    if order == 0:
        return warp(variance, matrix, offset, output_shape, order=0, cval=0.0)
    if mode == "aperture":
        return warp(variance, matrix, offset, output_shape, order=1, cval=0.0)
 
    # mode == "pixel": explicit sum of squared weights over the samples around each interpolation point.
    half_width, eta = interpolation_kernel(order)
    x_source, y_source = source_coordinates(matrix, offset, output_shape)
    x_base, y_base = np.floor(x_source).astype(int), np.floor(y_source).astype(int)
    x_fraction, y_fraction = x_source - x_base, y_source - y_base
    ny, nx = variance.shape
    result = np.zeros(output_shape)
    taps = range(-half_width + 1, half_width + 1)
    x_weights = {m: eta(x_fraction - m)**2 for m in taps}
    for n in taps:
        y_index = y_base + n
        y_weight = eta(y_fraction - n)**2
        y_inside = (y_index >= 0) & (y_index < ny)
        y_clipped = np.clip(y_index, 0, ny - 1)
        for m in taps:
            x_index = x_base + m
            inside = y_inside & (x_index >= 0) & (x_index < nx)
            result += np.where(inside, y_weight * x_weights[m] * variance[y_clipped, np.clip(x_index, 0, nx - 1)], 0.0)
    return result
 
 
def warp_mask(mask, matrix, offset, output_shape, order):
 
    """
    Transform a data quality bit mask with the same geometric transformation applied to its image by warp(). Each bit
    is treated separately, and an output pixel inherits a bit if any of the input pixels that dominate its
    interpolation (the nearest one for order=0, the 2x2 nearest ones otherwise) has that bit set.
    """
 
    mask = np.asarray(mask, dtype=np.uint8)
    result = np.zeros(output_shape, dtype=np.uint8)
    for bit in (1 << i for i in range(8)):
        plane = (mask & bit) > 0
        if not plane.any():
            continue
        if order == 0:
            warped = warp(plane.astype(float), matrix, offset, output_shape, order=0, cval=0.0) > 0.5
        else:
            warped = warp(plane.astype(float), matrix, offset, output_shape, order=1, cval=0.0) > 1e-6
        result[warped] |= bit
    return result
 
 
def align_variance_and_mask(variance, mask, transforms, crop_box, original_shape, order, crop=True, variance_mode="aperture"):
 
    """
    Apply to variance frames and data quality masks the same transformations and crop that align_images() applied to
    the corresponding images.
 
    Parameters
    ----------
    variance, mask: dict or None
        Dictionaries {file path: array} with the variance frames and/or bit masks, with the geometry of the images
        BEFORE the alignment. Entries of images that were not aligned (e.g. lamps) are only cropped.
    transforms: dict
        Transformations returned by align_images().
    crop_box: tuple or None
        (x1, x2, y1, y2) crop returned by align_images(), or None if crop=False.
    original_shape: tuple
        Shape of the aligned images before the crop.
    order: int
        Interpolation order used for the images (0 for integer shifts).
    crop: bool
        The same value passed to align_images(). If False, output pixels outside the original field of view get infinite
        variance (i.e. zero weight) and bit 8 of the mask.
    variance_mode: string
        "aperture" or "pixel" (see warp_variance()).
 
    Returns
    -------
    aligned_variance, aligned_mask: dict or None
        The transformed dictionaries (None if the corresponding input is None).
    """
 
    def transform(dictionary, function):
        if dictionary is None:
            return None
        output = {}
        for name, array in dictionary.items():
            array = np.asarray(array)
            record = transforms.get(name)
            if record is not None and record["method"] != "reference":
                if array.shape != tuple(original_shape):
                    raise ValueError(f"The variance/mask of {Path(name).name} has shape {array.shape}, but the image had shape "
                                     f"{tuple(original_shape)} before the alignment.")
                array = function(array, record)
                if not crop:
                    outside = ~interpolation_footprint(record["matrix"], record["offset"], original_shape, order)
                    if np.issubdtype(array.dtype, np.integer):
                        array[outside] |= 8
                    else:
                        array[outside] = np.inf
            if crop and crop_box is not None and array.shape == tuple(original_shape):
                x1, x2, y1, y2 = crop_box
                array = array[y1:y2, x1:x2]
            output[name] = array
        return output
 
    aligned_variance = transform(variance, lambda array, record: warp_variance(array, record["matrix"], record["offset"], original_shape,
                                                                               order, mode=variance_mode))
    aligned_mask = transform(mask, lambda array, record: warp_mask(array, record["matrix"], record["offset"], original_shape, order))
    return aligned_variance, aligned_mask