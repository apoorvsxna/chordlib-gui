# inspired by essentia's C++ PCP algorithm

import numpy as np
from .._internal.dsp import stft_analysis
from .peaks import spectral_peaks


def compute_peak_weight(distance: float, window_size: float, kind: str) -> float:
    """
    Compute contribution weight for a given peak at `distance` from a bin center.
    """
    if abs(distance) > window_size:
        return 0.0
    if kind == "none":
        return 1.0
    # cosine taper
    angle = (np.pi / 2.0) * (distance / window_size)
    base = np.cos(angle)
    return base**2 if kind == "squaredCosine" else base


def compute_hpcp_vector(
    freqs: np.ndarray,
    mags: np.ndarray,
    num_bins: int,
    ref_freq: float,
    harmonics: int,
    band_preset: bool,
    band_split_freq: float,
    min_freq: float,
    max_freq: float,
    weight_kind: str,
    apply_nonlinear: bool,
    window_size: float,
    normalization: str,
) -> np.ndarray:
    """
    Build one HPCP vector by spreading each spectral peak (and its harmonics)
    into chroma bins with a weighting window.
    """
    main_bins = np.zeros(num_bins, dtype=float)
    low_bins = np.zeros(num_bins, dtype=float) if band_preset else None
    high_bins = np.zeros(num_bins, dtype=float) if band_preset else None
    for f0, m0 in zip(freqs, mags):
        for h in range(harmonics + 1):
            freq_h = f0 * (h + 1)
            if not (min_freq <= freq_h <= max_freq):
                continue

            semitone_offset = num_bins * np.log2(freq_h / ref_freq)
            center = semitone_offset % num_bins

            start_bin = int(np.floor(center - window_size))
            end_bin = int(np.ceil(center + window_size))
            for b in range(start_bin, end_bin + 1):
                dist = b - center
                w = compute_peak_weight(dist, window_size, weight_kind)
                energy = m0 * w
                target = b % num_bins
                if band_preset:
                    if freq_h < band_split_freq:
                        low_bins[target] += energy
                    else:
                        high_bins[target] += energy
                else:
                    main_bins[target] += energy

    hpcp = (low_bins + high_bins) if band_preset else main_bins

    if apply_nonlinear:
        mask = hpcp < 0.6
        hpcp[mask] = np.sin((np.pi / 2.0) * hpcp[mask]) ** 2

    if normalization == "unitMax":
        peak_val = np.max(hpcp)
        if peak_val > 0:
            hpcp = hpcp / peak_val
    elif normalization == "unitSum":
        total = np.sum(hpcp)
        if total > 0:
            hpcp = hpcp / total

    return hpcp


def extract_pcp(
    audio_signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    sp_min_freq: float = 40.0,
    sp_max_freq: float = 5000.0,
    sp_mag_threshold: float = 0.0,
    sp_max_peaks: int = 100,
    sp_min_peak_distance_hz: float = 20.0,
    sp_interpolate: bool = True,
    hp_size: int = 12,
    hp_ref_freq: float = 440.0,
    hp_harmonics: int = 0,
    hp_band_preset: bool = True,
    hp_band_split: float = 500.0,
    hp_min_freq: float = 40.0,
    hp_max_freq: float = 5000.0,
    hp_weight: str = "squaredCosine",
    hp_non_linear: bool = False,
    hp_window_size: float = 1.0,
    hp_norm: str = "unitMax",
    hp_max_shifted: bool = False,
) -> np.ndarray:
    """
    perform STFT to spectral peaks to HPCP vector for each frame.
    Returns an (n_frames x hp_size) matrix.
    """
    # compute magnitude spectrogram with overlapping windows
    stft_matrix = stft_analysis(y=audio_signal, n_fft=n_fft, hop_length=hop_length)
    mag_spec = np.abs(stft_matrix)
    num_bins, num_frames = mag_spec.shape

    # Prepare output matrix
    pcp_matrix = np.zeros((num_frames, hp_size), dtype=float)

    for t in range(num_frames):
        frame_mags = mag_spec[:, t]

        # spectral peak picking
        freqs, amps = spectral_peaks(
            frame_mags,
            sample_rate,
            sp_min_freq,
            sp_max_freq,
            sp_mag_threshold,
            sp_max_peaks,
            sp_min_peak_distance_hz,
            sp_interpolate,
        )

        # build HPCP vector
        hpcp_vec = compute_hpcp_vector(
            freqs,
            amps,
            hp_size,
            hp_ref_freq,
            hp_harmonics,
            hp_band_preset,
            hp_band_split,
            hp_min_freq,
            hp_max_freq,
            hp_weight,
            hp_non_linear,
            hp_window_size,
            hp_norm,
        )

        # 3) Optional rotation so strongest bin is first
        if hp_max_shifted:
            shift_amt = int(np.argmax(hpcp_vec))
            hpcp_vec = np.roll(hpcp_vec, -shift_amt)

        pcp_matrix[t, :] = hpcp_vec

    return pcp_matrix
