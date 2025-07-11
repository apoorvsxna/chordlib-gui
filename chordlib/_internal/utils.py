import numpy as np
from numpy.lib.stride_tricks import as_strided
import numba
from numba import stencil, guvectorize
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from numpy.typing import DTypeLike

MAX_MEM_BLOCK = 2**8 * 2**10  # maximum memory block size for batch processing.


def tiny(x: Union[float, np.ndarray]) -> float:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)
    return np.finfo(dtype).tiny


def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    """
    normalizes a matrix along a specified axis.

    Args:
        norm: the norm to use. `np.inf` for max-norm, `1` for L1-norm, etc.
        axis: the axis to normalize over.
        threshold: values below this are handled specially to avoid division by zero.
        fill: how to handle vectors with norms below the threshold.
    """
    if threshold is None:
        threshold = tiny(S)
    elif threshold <= 0:
        raise ValueError(f"threshold={threshold} must be strictly positive")
    if fill not in [None, False, True]:
        raise ValueError(f"fill={fill} must be None or boolean")
    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    mag = np.abs(S).astype(float)
    fill_norm = 1
    if norm is None:
        return S
    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)
    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")
        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)
        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)
    else:
        raise ValueError(f"Unsupported norm: {repr(norm)}")

    small_idx = length < threshold
    Snorm = np.empty_like(S)
    if fill is None:
        length[small_idx] = 1.0
        Snorm[:] = S / length
    elif fill:
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        length[small_idx] = np.inf
        Snorm[:] = S / length
    return Snorm


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    """
    slices a signal into overlapping frames.
    returns a view of the original array using stride tricks, avoiding data duplication.
    """
    x = np.array(x, copy=False, subok=subok)
    if x.shape[axis] < frame_length:
        raise ValueError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )
    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length:d}")
    out_strides = x.strides + tuple([x.strides[axis]])
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def pad_center(
    data: np.ndarray, *, size: int, axis: int = -1, **kwargs: Any
) -> np.ndarray:
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")
    return np.pad(data, lengths, **kwargs)


def expand_to(
    x: np.ndarray, *, ndim: int, axes: Union[int, slice, Sequence[int], Sequence[slice]]
) -> np.ndarray:
    try:
        axes_tup = tuple(axes)
    except TypeError:
        axes_tup = tuple([axes])
    if len(axes_tup) != x.ndim:
        raise ValueError(
            f"Shape mismatch between axes={axes_tup} and input x.shape={x.shape}"
        )
    if ndim < x.ndim:
        raise ValueError(
            f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}"
        )
    shape: List[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]
    return x.reshape(shape)


def validate_audio(y: np.ndarray) -> bool:
    if not isinstance(y, np.ndarray):
        raise TypeError("Audio data must be of type numpy.ndarray")
    if not np.issubdtype(y.dtype, np.floating):
        raise TypeError("Audio data must be floating-point")
    if y.ndim == 0:
        raise ValueError(
            f"Audio data must be at least one-dimensional, given y.shape={y.shape}"
        )
    if not np.isfinite(y).all():
        raise ValueError("Audio buffer is not finite everywhere")
    return True


def is_positive_int(x: float) -> bool:
    return isinstance(x, (int, np.integer)) and (x > 0)


def sanitize_frame_bounds(
    frames: Sequence[int],
    *,
    x_min: Optional[int] = 0,
    x_max: Optional[int] = None,
    pad: bool = True,
) -> np.ndarray:
    """
    cleans up a list of frame indices.

    ensures they are sorted, unique, non-negative, and optionally
    clipped/padded to a given range.
    """
    frames = np.asarray(frames)
    if np.any(frames < 0):
        raise ValueError("Negative frame index detected")
    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)
    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((np.asarray(pad_data), frames))
    if x_min is not None:
        frames = frames[frames >= x_min]
    if x_max is not None:
        frames = frames[frames <= x_max]
    unique: np.ndarray = np.unique(frames).astype(int)
    return unique


@stencil
def _stencil_localmax(x):
    return (x[0] > x[-1]) & (x[0] >= x[1])


@guvectorize(
    [
        "void(int16[:], bool_[:])",
        "void(int32[:], bool_[:])",
        "void(int64[:], bool_[:])",
        "void(float32[:], bool_[:])",
        "void(float64[:], bool_[:])",
    ],
    "(n)->(n)",
    cache=True,
    nopython=True,
)
def _gufunc_localmax(x, y):
    y[:] = _stencil_localmax(x)


def find_local_maxima(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """
    finds local maxima in an array, accelerated with numba.

    a point is a local max if its value is > its left neighbor and >= its
    right neighbor.
    """
    xi = x.swapaxes(-1, axis)
    lmax = np.empty_like(x, dtype=bool)
    lmaxi = lmax.swapaxes(-1, axis)
    _gufunc_localmax(xi, lmaxi)
    lmaxi[..., -1] = xi[..., -1] > xi[..., -2]
    return lmax


def bounds_to_slices(
    idx: Sequence[int],
    *,
    idx_min: Optional[int] = None,
    idx_max: Optional[int] = None,
    step: Optional[int] = None,
    pad: bool = True,
) -> List[slice]:
    idx_fixed = sanitize_frame_bounds(idx, x_min=idx_min, x_max=idx_max, pad=pad)
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]


def sync(
    data: np.ndarray,
    idx: Union[Sequence[int], Sequence[slice]],
    *,
    aggregate: Optional[Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """
    synchronizes a feature matrix to a new time basis.

    useful for aggregating frame-wise features (e.g. chromagram)
    into beat-synchronous segments by taking the mean/median of the frames
    that fall between each beat.
    """
    if aggregate is None:
        aggregate = np.mean
    shape = list(data.shape)
    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.integer) for _ in idx]):
        slices = bounds_to_slices(
            np.asarray(idx), idx_min=0, idx_max=shape[axis], pad=pad
        )
    else:
        raise ValueError(f"Invalid index set: {idx}")
    agg_shape = list(shape)
    agg_shape[axis] = len(slices)
    data_agg = np.empty(
        agg_shape, order="F" if np.isfortran(data) else "C", dtype=data.dtype
    )
    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim
    for i, segment in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)
    return data_agg


def dtype_r2c(d: DTypeLike, *, default: Optional[type] = np.complex64) -> DTypeLike:
    """finds the complex-valued equivalent of a real-valued data type."""
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt
    return np.dtype(mapping.get(dt, default))


@numba.vectorize(
    ["float32(complex64)", "float64(complex128)"], nopython=True, cache=True, identity=0
)
def _complex_mag_square(x):
    return x.real**2 + x.imag**2


def magnitude_square(x: Any, dtype: Optional[DTypeLike] = None) -> Any:
    """
    computes the squared magnitude of the input array.
    """
    if np.iscomplexobj(x):
        y = _complex_mag_square(x)
        if dtype is None:
            return y
        else:
            return y.astype(dtype)
    else:
        return np.square(x, dtype=dtype)


def frames_to_samples(
    frames: Any, *, hop_length: int = 512, n_fft: Optional[int] = None
) -> Union[np.integer, np.ndarray]:
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def time_to_samples(times: Any, *, sr: float = 22050) -> Union[np.integer, np.ndarray]:
    return (np.asanyarray(times) * sr).astype(int)


def samples_to_frames(
    samples: Any, *, hop_length: int = 512, n_fft: Optional[int] = None
) -> Union[np.integer, np.ndarray]:
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    samples = np.asanyarray(samples)
    return np.asarray(np.floor((samples - offset) // hop_length), dtype=int)


def time_to_frames(
    times: Any, *, sr: float = 22050, hop_length: int = 512, n_fft: Optional[int] = None
) -> Union[np.integer, np.ndarray]:
    samples = time_to_samples(times, sr=sr)
    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


def samples_to_time(
    samples: Any, *, sr: float = 22050
) -> Union[np.floating, np.ndarray]:
    return np.asanyarray(samples) / float(sr)


def frames_to_time(
    frames: Any,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.floating, np.ndarray]:
    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples_to_time(samples, sr=sr)


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def hz_to_mel(frequencies: Any, *, htk: bool = False) -> Union[np.floating, np.ndarray]:
    frequencies = np.asanyarray(frequencies)
    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: Any, *, htk: bool = False) -> Union[np.floating, np.ndarray]:
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def mel_frequencies(
    n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> np.ndarray:
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def tempo_frequencies(
    n_bins: int, *, hop_length: int = 512, sr: float = 22050
) -> np.ndarray:
    """computes the tempo (in BPM) for each bin of a tempogram."""
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)
    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))
    return bin_frequencies
