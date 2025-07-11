# chordlib/features/peaks.py

import numpy as np
from .._internal.peak_finding import find_peaks
from typing import Tuple

def spectral_peaks(
    mag_spectrum: np.ndarray,
    sample_rate: int,
    min_freq: float = 40.0,
    max_freq: float = 5000.0,
    mag_threshold: float = 0.0,
    max_peaks: int = 100,
    min_peak_distance_hz: float = 20.0,
    interpolate: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects local spectral peaks with optional parabolic interpolation.

    Steps:
      1. Convert magnitude spectrum to decibels for uniform peak picking.
      2. Compute frequency bin width and minimum bin distance.
      3. Find peaks above threshold separated by min_peak_distance_hz.
      4. Filter peaks to the desired frequency range.
      5. Keep only the top max_peaks by height.
      6. (Optional) Refine each peak’s frequency via parabolic interpolation.
      7. Convert dB heights back to linear amplitude.
      8. Sort and return by ascending frequency.
    """
    # 1) Convert to dB, avoiding log(0)
    eps = 1e-12
    safe_mag = np.maximum(mag_spectrum, eps)
    db_spectrum = 20.0 * np.log10(safe_mag)

    # 2) Frequency bin width (Hz per bin)
    nyquist = sample_rate / 2.0
    num_bins = len(db_spectrum) - 1
    freq_bin_width = nyquist / num_bins

    # 3) Minimum separation in bins
    min_distance_bins = int(round(min_peak_distance_hz / freq_bin_width))
    
    # --- FIX ---
    # Ensure distance is at least 1 to satisfy find_peaks requirement.
    # This prevents an error when frequency resolution is low.
    if min_distance_bins < 1:
        min_distance_bins = 1
    # --- END FIX ---

    # 4) Initial peak finding in dB domain
    peak_indices, properties = find_peaks(
        db_spectrum,
        height=mag_threshold,
        distance=min_distance_bins
    )
    peak_db_heights = properties['peak_heights']

    # 5) Convert indices to frequencies
    peak_freqs = peak_indices * freq_bin_width

    # 6) Filter by frequency range
    freq_mask = (peak_freqs >= min_freq) & (peak_freqs <= max_freq)
    filtered_indices = peak_indices[freq_mask]
    filtered_heights = peak_db_heights[freq_mask]
    filtered_freqs = peak_freqs[freq_mask]

    # 7) Select top peaks by height
    if len(filtered_indices) > 0:
        sort_order = np.argsort(filtered_heights)[::-1]
        top_order = sort_order[:max_peaks]
        selected_indices = filtered_indices[top_order]
        selected_heights_db = filtered_heights[top_order]
    else: # Handle case with no peaks found
        selected_indices = np.array([], dtype=int)
        selected_heights_db = np.array([], dtype=float)


    # 8) Parabolic interpolation (optional)
    if interpolate and len(selected_indices) > 0:
        refined_freqs = []
        for idx in selected_indices:
            if idx <= 0 or idx >= num_bins:
                refined_freqs.append(idx * freq_bin_width)
                continue
            y1, y2, y3 = db_spectrum[idx - 1], db_spectrum[idx], db_spectrum[idx + 1]
            denom = y1 - 2 * y2 + y3
            delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
            refined_bin = idx + delta
            refined_freqs.append(refined_bin * freq_bin_width)
        final_freqs = np.array(refined_freqs)
    else:
        # Create frequencies from selected indices directly if not interpolating or no peaks
        final_freqs = selected_indices * freq_bin_width

    # 9) Convert back to linear amplitude
    linear_amps = 10.0 ** (selected_heights_db / 20.0)

    # 10) Sort by frequency and return
    if len(final_freqs) > 0:
        freq_sort = np.argsort(final_freqs)
        sorted_freqs = final_freqs[freq_sort]
        sorted_amps = linear_amps[freq_sort]
    else:
        sorted_freqs = np.array([], dtype=float)
        sorted_amps = np.array([], dtype=float)


    return sorted_freqs, sorted_amps