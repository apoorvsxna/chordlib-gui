import numpy as np
import scipy
import scipy.signal
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from typing_extensions import Literal
from numpy.typing import DTypeLike

from .utils import (
    normalize,
    frame,
    pad_center,
    expand_to,
    validate_audio,
    is_positive_int,
    sync,
    dtype_r2c,
    magnitude_square,
    time_to_frames,
    fft_frequencies,
    mel_frequencies,
    tempo_frequencies,
    MAX_MEM_BLOCK,
)


def compute_autocorrelation(
    y: np.ndarray, *, max_size: Optional[int] = None, axis: int = -1
) -> np.ndarray:
    """
    computes the autocorrelation of a signal.

    this is done by taking the inverse fourier transform of power spectrum.
    (aka wiener-khinchin theorem)

    Args:
        y: the input signal array. can be multi-dimensional too.
        max_size: maximum number of lags to compute. if none, we compute up
            to the full length of the signal.
        axis: the axis along which autocorrelation should be calculated.
    """
    if max_size is None:
        max_size = y.shape[axis]
    max_size = int(min(max_size, y.shape[axis]))
    fft = scipy.fft
    real = not np.iscomplexobj(y)
    n_pad = scipy.fft.next_fast_len(2 * y.shape[axis] - 1, real=real)
    if real:
        powspec = magnitude_square(fft.rfft(y, n=n_pad, axis=axis))
        autocorr = fft.irfft(powspec, n=n_pad, axis=axis)
    else:
        powspec = magnitude_square(fft.fft(y, n=n_pad, axis=axis))
        autocorr = fft.ifft(powspec, n=n_pad, axis=axis)
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)
    autocorr_slice: np.ndarray = autocorr[tuple(subslice)]
    return autocorr_slice


def make_window(window: Any, Nx: int, *, fftbins: Optional[bool] = True) -> np.ndarray:
    """
    generates a window of a given type and length.

    basically a flexible wrapper that accepts a string (e.g., 'hann'),
    a callable, or a pre-computed numpy array.
    """
    if callable(window):
        return window(Nx)
    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)
        return win
    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)
        raise ValueError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ValueError(f"Invalid window specification: {window!r}")


def make_mel_bank(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[Union[Literal["slaney"], float]] = "slaney",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """
    creates a filter bank to convert fft bins to mel-scale bins.

    Args:
        sr: audio sample rate.
        n_fft: number of fft components.
        n_mels: number of mel bands to generate.
        fmin: lowest frequency (in hz).
        fmax: highest frequency (in hz).
        htk: if true, use the htk formula for mel scale conversion.
        norm: how to normalize filter weights. 'slaney' normalizes to constant area.
    """
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    if isinstance(norm, str):
        if norm == "slaney":
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
        else:
            raise ValueError(f"Unsupported norm={norm}")
    else:
        weights = normalize(weights, norm=norm, axis=-1)
    return weights


def stft_analysis(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Any = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: Any = "constant",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    computes the short-time fourier transform (stft) of a signal.

    this function represents a signal in the time-frequency domain by
    computing ffts over short, overlapping windows.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not is_positive_int(hop_length):
        raise ValueError(f"hop_length={hop_length} must be a positive integer")

    validate_audio(y)
    fft_window = make_window(window, win_length, fftbins=True)
    fft_window = pad_center(fft_window, size=n_fft)
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        start_k = int(np.ceil(n_fft // 2 / hop_length))
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1
        if tail_k <= start_k:
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            y_frames_pre = y_frames_pre[..., :start_k]
            extra = y_frames_pre.shape[-1]
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = frame(y_post, frame_length=n_fft, hop_length=hop_length)
                extra += y_frames_post.shape[-1]
            else:
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        start = 0
        extra = 0

    fft = scipy.fft
    if dtype is None:
        dtype = dtype_r2c(y.dtype)
    y_frames = frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    else:
        stft_matrix = out[..., : shape[-1]]

    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)
        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0
    n_columns = int(MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)
    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])
        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix


def _spectrogram_from_audio(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    power: float = 1,
    win_length: Optional[int] = None,
    window: Any = "hann",
    center: bool = True,
    pad_mode: Any = "constant",
) -> Tuple[np.ndarray, int]:
    if S is not None:
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        if n_fft is None:
            raise ValueError(f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ValueError("Input signal must be provided to compute a spectrogram")
        S = (
            np.abs(
                stft_analysis(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )
    return S, n_fft


def power_to_decibels(
    S: Any,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> Any:
    """
    converts a power or amplitude spectrogram to decibel (db) units.

    Args:
        ref: reference value for the db calculation.
        amin: a small constant to avoid `log10(0)`.
        top_db: if not none, thresholds the output to be no more than `top_db`
            below the maximum value.
    """
    S = np.asarray(S)
    if amin <= 0:
        raise ValueError("amin must be strictly positive")
    if np.issubdtype(S.dtype, np.complexfloating):
        magnitude = np.abs(S)
    else:
        magnitude = S
    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def compute_melspectrogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: Any = "hann",
    center: bool = True,
    pad_mode: Any = "constant",
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:
    """computes a mel-frequency spectrogram."""
    S, n_fft = _spectrogram_from_audio(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    mel_basis = make_mel_bank(sr=sr, n_fft=n_fft, **kwargs)
    melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
    return melspec


def compute_band_onset_strength(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[np.ndarray] = None,
    detrend: bool = False,
    center: bool = True,
    feature: Optional[Callable] = None,
    aggregate: Optional[Union[Callable, bool]] = None,
    channels: Optional[Union[Sequence[int], Sequence[slice]]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    computes an onset strength envelope for one or more frequency bands.

    this is done by calculating the spectral flux: a measure of the temporal
    change in the spectrum, which often corresponds to note onsets.
    """
    if feature is None:
        feature = compute_melspectrogram
        kwargs.setdefault("fmax", 0.5 * sr)
    if aggregate is None:
        aggregate = np.mean
    if not is_positive_int(lag):
        raise ValueError(f"lag={lag} must be a positive integer")
    if not is_positive_int(max_size):
        raise ValueError(f"max_size={max_size} must be a positive integer")
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
        S = power_to_decibels(S)
    assert S is not None
    S = np.atleast_2d(S)
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ValueError(
            f"Reference spectrum shape {ref.shape} must match input spectrum {S.shape}"
        )
    onset_env = S[..., lag:] - ref[..., :-lag]
    onset_env = np.maximum(0.0, onset_env)
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False
    if callable(aggregate):
        onset_env = sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=-2)
    pad_width = lag
    if center:
        pad_width += n_fft // (2 * hop_length)
    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode="constant")
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)
    if center:
        onset_env = onset_env[..., : S.shape[-1]]
    return onset_env


def compute_onset_strength(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[np.ndarray] = None,
    detrend: bool = False,
    center: bool = True,
    feature: Optional[Callable] = None,
    aggregate: Optional[Union[Callable, bool]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """computes a single onset strength envelope by aggregating all frequency bands."""
    if aggregate is False:
        raise ValueError(
            "aggregate parameter cannot be False when computing full-spectrum onset strength."
        )
    odf_all = compute_band_onset_strength(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )
    return odf_all[..., 0, :]


def compute_tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: Any = "hann",
    norm: Optional[float] = np.inf,
) -> np.ndarray:
    """
    computes a tempogram from an onset strength envelope.

    a tempogram measures the periodicity of the onset envelope's energy,
    which reveals the dominant tempi in the signal. it's essentially an
    autocorrelation of the onset envelope over time.
    """
    if win_length < 1:
        raise ValueError("win_length must be a positive integer")
    ac_window = make_window(window, win_length, fftbins=True)
    if onset_envelope is None:
        if y is None:
            raise ValueError("Either y or onset_envelope must be provided")
        onset_envelope = compute_onset_strength(y=y, sr=sr, hop_length=hop_length)
    n = onset_envelope.shape[-1]
    if center:
        padding = [(0, 0) for _ in onset_envelope.shape]
        padding[-1] = (int(win_length // 2),) * 2
        onset_envelope = np.pad(
            onset_envelope, padding, mode="linear_ramp", end_values=[0, 0]
        )
    odf_frame = frame(onset_envelope, frame_length=win_length, hop_length=1)
    if center:
        odf_frame = odf_frame[..., :n]
    ac_window = expand_to(ac_window, ndim=odf_frame.ndim, axes=-2)
    return normalize(
        compute_autocorrelation(odf_frame * ac_window, axis=-2), norm=norm, axis=-2
    )


def estimate_tempo(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    tg: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: Optional[float] = 320.0,
    aggregate: Optional[Callable[..., Any]] = np.mean,
    prior: Optional[scipy.stats.rv_continuous] = None,
) -> np.ndarray:
    """
    estimates the global tempo of a signal.

    this is done by finding the most prominent periodicity in the tempogram,
    biased by a "prior" centered at `start_bpm`.
    """
    if start_bpm <= 0:
        raise ValueError("start_bpm must be strictly positive")
    if tg is None:
        win_length = time_to_frames(ac_size, sr=sr, hop_length=hop_length).item()
        tg = compute_tempogram(
            y=y,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            win_length=win_length,
        )
    else:
        win_length = tg.shape[-2]
    if aggregate is not None:
        tg = aggregate(tg, axis=-1, keepdims=True)
    assert tg is not None
    bpms = tempo_frequencies(win_length, hop_length=hop_length, sr=sr)
    if prior is None:
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    else:
        logprior = prior.logpdf(bpms)
    if max_tempo is not None:
        max_idx = int(np.argmax(bpms < max_tempo))
        logprior[:max_idx] = -np.inf
    logprior = expand_to(logprior, ndim=tg.ndim, axes=-2)
    best_period = np.argmax(np.log1p(1e6 * tg) + logprior, axis=-2)
    tempo_est: np.ndarray = np.take(bpms, best_period)
    return tempo_est
