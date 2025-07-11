import math
import warnings
import numpy as np


def _get_all_maxima(data_array):
    """finds the indices of all local maxima, including plateaus."""
    mids = []
    lefts = []
    rights = []
    i = 1
    i_max = data_array.shape[0] - 1
    while i < i_max:
        if data_array[i - 1] < data_array[i]:
            i_ahead = i + 1
            while i_ahead < i_max and data_array[i_ahead] == data_array[i]:
                i_ahead += 1
            if data_array[i_ahead] < data_array[i]:
                left_edge = i
                right_edge = i_ahead - 1
                midpoint = (left_edge + right_edge) // 2
                mids.append(midpoint)
                lefts.append(left_edge)
                rights.append(right_edge)
                i = i_ahead
        i += 1
    return (
        np.array(mids, dtype=np.intp),
        np.array(lefts, dtype=np.intp),
        np.array(rights, dtype=np.intp),
    )


def _filter_by_distance(indices, priorities, min_separation):
    """
    removes peaks that are too close to higher-priority peaks.

    it walks through peaks from highest to lowest priority, invalidating
    any neighbors that fall within the `min_separation` window.
    """
    num_indices = indices.shape[0]
    separation = math.ceil(min_separation)
    is_valid = np.ones(num_indices, dtype=bool)

    priority_order = np.argsort(priorities)

    for i in range(num_indices - 1, -1, -1):
        j = priority_order[i]
        if not is_valid[j]:
            continue

        k = j - 1
        while k >= 0 and indices[j] - indices[k] < separation:
            is_valid[k] = False
            k -= 1

        k = j + 1
        while k < num_indices and indices[k] - indices[j] < separation:
            is_valid[k] = False
            k += 1
    return is_valid


def _calculate_prominence(data_array, peak_indices, window_size):
    """
    calculates the prominence for each peak.

    prominence is a measure of how much a peak stands out. it's the vertical
    distance between the peak and its lowest contour line.
    """
    needs_warning = False
    prominences = np.empty(peak_indices.shape[0], dtype=np.float64)
    left_bases = np.empty(peak_indices.shape[0], dtype=np.intp)
    right_bases = np.empty(peak_indices.shape[0], dtype=np.intp)

    for i, current_peak in enumerate(peak_indices):
        i_min, i_max = 0, data_array.shape[0] - 1
        if not i_min <= current_peak <= i_max:
            raise ValueError(
                f"Index {current_peak} is out of bounds for the signal array."
            )

        if window_size >= 2:
            i_min = max(current_peak - window_size // 2, i_min)
            i_max = min(current_peak + window_size // 2, i_max)

        cursor = current_peak
        left_bases[i] = current_peak
        left_min_val = data_array[current_peak]
        while i_min <= cursor and data_array[cursor] <= data_array[current_peak]:
            if data_array[cursor] < left_min_val:
                left_min_val = data_array[cursor]
                left_bases[i] = cursor
            cursor -= 1

        cursor = current_peak
        right_bases[i] = current_peak
        right_min_val = data_array[current_peak]
        while cursor <= i_max and data_array[cursor] <= data_array[current_peak]:
            if data_array[cursor] < right_min_val:
                right_min_val = data_array[cursor]
                right_bases[i] = cursor
            cursor += 1

        prominences[i] = data_array[current_peak] - max(left_min_val, right_min_val)
        if prominences[i] == 0:
            needs_warning = True

    if needs_warning:
        warnings.warn(
            "Zero prominence calculated for one or more peaks.",
            UserWarning,
            stacklevel=2,
        )
    return prominences, left_bases, right_bases


def _calculate_width(
    data_array, peak_indices, relative_height, prominences, left_bases, right_bases
):
    """
    calculates the width of each peak.

    the width is measured at a vertical level defined as a percentage of
    the peak's prominence.
    """
    if relative_height < 0:
        raise ValueError("`relative_height` must be non-negative.")
    if not (
        peak_indices.shape[0]
        == prominences.shape[0]
        == left_bases.shape[0]
        == right_bases.shape[0]
    ):
        raise ValueError("Input data arrays must have matching shapes.")

    needs_warning = False
    widths = np.empty(peak_indices.shape[0], dtype=np.float64)
    width_levels = np.empty(peak_indices.shape[0], dtype=np.float64)
    left_intercepts = np.empty(peak_indices.shape[0], dtype=np.float64)
    right_intercepts = np.empty(peak_indices.shape[0], dtype=np.float64)

    for i, current_peak in enumerate(peak_indices):
        i_min, i_max = left_bases[i], right_bases[i]
        if not 0 <= i_min <= current_peak <= i_max < data_array.shape[0]:
            raise ValueError(
                f"Invalid prominence data for peak at index {current_peak}"
            )

        evaluation_height = data_array[current_peak] - prominences[i] * relative_height
        width_levels[i] = evaluation_height

        cursor = current_peak
        while i_min < cursor and evaluation_height < data_array[cursor]:
            cursor -= 1
        left_ip = float(cursor)
        if data_array[cursor] < evaluation_height:
            left_ip += (evaluation_height - data_array[cursor]) / (
                data_array[cursor + 1] - data_array[cursor]
            )

        cursor = current_peak
        while cursor < i_max and evaluation_height < data_array[cursor]:
            cursor += 1
        right_ip = float(cursor)
        if data_array[cursor] < evaluation_height:
            right_ip -= (evaluation_height - data_array[cursor]) / (
                data_array[cursor - 1] - data_array[cursor]
            )

        widths[i] = right_ip - left_ip
        if widths[i] == 0:
            needs_warning = True
        left_intercepts[i] = left_ip
        right_intercepts[i] = right_ip

    if needs_warning:
        warnings.warn(
            "Zero width calculated for one or more peaks.", UserWarning, stacklevel=2
        )
    return widths, width_levels, left_intercepts, right_intercepts


def _as_float_array(arr):
    """validates and prepares the main signal array."""
    prepared_arr = np.asarray(arr, order="C", dtype=np.float64)
    if prepared_arr.ndim != 1:
        raise ValueError("Input signal must be a 1-D array.")
    return prepared_arr


def _validate_window_size(val):
    if val is None:
        return -1
    if val > 1:
        return np.intp(math.ceil(val))
    raise ValueError(f"Window length must be larger than 1, but got {val}.")


def _unpack_boundary(bounds, signal_array, peak_indices):
    """unpacks boundary arguments, which can be a number or a (min, max) tuple."""
    try:
        lower, upper = bounds
    except (TypeError, ValueError):
        lower, upper = (bounds, None)

    if isinstance(lower, np.ndarray):
        if lower.size != signal_array.size:
            raise ValueError("Lower boundary array must match signal array size.")
        lower = lower[peak_indices]
    if isinstance(upper, np.ndarray):
        if upper.size != signal_array.size:
            raise ValueError("Upper boundary array must match signal array size.")
        upper = upper[peak_indices]
    return lower, upper


def _filter_by_range(properties, min_val, max_val):
    is_valid = np.ones(properties.size, dtype=bool)
    if min_val is not None:
        is_valid &= min_val <= properties
    if max_val is not None:
        is_valid &= properties <= max_val
    return is_valid


def _filter_by_descent(signal_array, peak_indices, min_thresh, max_thresh):
    """creates a boolean mask based on a vertical distance threshold (w.r.t. neighbors)."""
    v_dist = np.vstack(
        [
            signal_array[peak_indices] - signal_array[peak_indices - 1],
            signal_array[peak_indices] - signal_array[peak_indices + 1],
        ]
    )
    is_valid = np.ones(peak_indices.size, dtype=bool)
    if min_thresh is not None:
        is_valid &= min_thresh <= np.min(v_dist, axis=0)
    if max_thresh is not None:
        is_valid &= np.max(v_dist, axis=0) <= max_thresh
    return is_valid, v_dist[0], v_dist[1]


def find_peaks(
    x,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
):
    """
    finds peaks in a 1D signal with advanced filtering options.

    this function identifies all local maxima and allows filtering by properties
    like height, prominence, distance, and width.
    """
    signal_array = _as_float_array(x)
    if distance is not None and distance < 1:
        raise ValueError("`distance` must be 1 or greater.")

    peak_indices, left_edges, right_edges = _get_all_maxima(signal_array)
    properties = {}

    def _apply_selection(mask):
        """apply boolean mask to all collected data."""
        nonlocal peak_indices, properties
        peak_indices = peak_indices[mask]
        properties = {key: val[mask] for key, val in properties.items()}

    if plateau_size is not None:
        p_sizes = right_edges - left_edges + 1
        min_val, max_val = _unpack_boundary(plateau_size, signal_array, peak_indices)
        selection_mask = _filter_by_range(p_sizes, min_val, max_val)

        properties["plateau_sizes"] = p_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        _apply_selection(selection_mask)

    if height is not None:
        amplitudes = signal_array[peak_indices]
        min_val, max_val = _unpack_boundary(height, signal_array, peak_indices)
        selection_mask = _filter_by_range(amplitudes, min_val, max_val)

        properties["peak_heights"] = amplitudes
        _apply_selection(selection_mask)

    if threshold is not None:
        min_val, max_val = _unpack_boundary(threshold, signal_array, peak_indices)
        selection_mask, left_thresh, right_thresh = _filter_by_descent(
            signal_array, peak_indices, min_val, max_val
        )

        properties["left_thresholds"] = left_thresh
        properties["right_thresholds"] = right_thresh
        _apply_selection(selection_mask)

    if distance is not None:
        priorities = signal_array[peak_indices]
        selection_mask = _filter_by_distance(peak_indices, priorities, distance)
        _apply_selection(selection_mask)

    if prominence is not None or width is not None:
        win_len = _validate_window_size(wlen)
        prominences, left_bases, right_bases = _calculate_prominence(
            signal_array, peak_indices, win_len
        )
        properties.update(
            {
                "prominences": prominences,
                "left_bases": left_bases,
                "right_bases": right_bases,
            }
        )

    if prominence is not None:
        min_val, max_val = _unpack_boundary(prominence, signal_array, peak_indices)
        selection_mask = _filter_by_range(properties["prominences"], min_val, max_val)
        _apply_selection(selection_mask)

    if width is not None:
        w_vals, w_levels, l_ips, r_ips = _calculate_width(
            signal_array,
            peak_indices,
            rel_height,
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        )
        properties.update(
            {
                "widths": w_vals,
                "width_heights": w_levels,
                "left_ips": l_ips,
                "right_ips": r_ips,
            }
        )
        min_val, max_val = _unpack_boundary(width, signal_array, peak_indices)
        selection_mask = _filter_by_range(properties["widths"], min_val, max_val)
        _apply_selection(selection_mask)

    return peak_indices, properties
