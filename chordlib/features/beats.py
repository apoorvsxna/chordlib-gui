import numpy as np
import scipy.stats
import numba
from typing import Optional, Tuple, Union

from .._internal.utils import (
    tiny, find_local_maxima, frames_to_samples, frames_to_time, expand_to
)
from .._internal.dsp import compute_onset_strength, estimate_tempo

def _normalize_strength(onsets):
    """normalizes an onset strength envelope by its standard deviation."""
    norm = onsets.std(ddof=1, axis=-1, keepdims=True)
    return onsets / (norm + tiny(onsets))

@numba.guvectorize(["void(float32[:], float32[:], float32[:])", "void(float64[:], float64[:], float64[:])",], "(t),(n)->(t)", nopython=True, cache=False)
def _score_beats(onset_envelope, frames_per_beat, localscore):
    """
    computes a local score for each frame based on its proximity to a beat.

    done by applying a gaussian window centered at each frame, where
    the window width is determined by the local tempo.
    """
    N = len(onset_envelope)
    if len(frames_per_beat) == 1:
        window = np.exp(-0.5 * (np.arange(-frames_per_beat[0], frames_per_beat[0] + 1) * 32.0 / frames_per_beat[0]) ** 2)
        K = len(window)
        for i in range(len(onset_envelope)):
            localscore[i] = 0.
            for k in range(max(0, i + K // 2 - N + 1), min(i + K // 2, K)):
                localscore[i] += window[k] * onset_envelope[i + K//2 -k]
    elif len(frames_per_beat) == len(onset_envelope):
        for i in range(len(onset_envelope)):
            window = np.exp(-0.5 * (np.arange(-frames_per_beat[i], frames_per_beat[i] + 1) * 32.0 / frames_per_beat[i]) ** 2)
            K = 2 * int(frames_per_beat[i]) + 1
            localscore[i] = 0.
            for k in range(max(0, i + K // 2 - N + 1), min(i + K // 2, K)):
                localscore[i] += window[k] * onset_envelope[i + K // 2 - k]

@numba.guvectorize(["void(float32[:], float32[:], float32, int32[:], float32[:])", "void(float64[:], float64[:], float32, int32[:], float64[:])",], "(t),(n),()->(t),(t)", nopython=True, cache=True)
def _dp_beat_path(localscore, frames_per_beat, tightness, backlink, cumscore):
    """
    uses DP to find the optimal beat sequence.

    this function builds a cumulative "score array" and a "backlink array" to
    trace the most likely path of beats. a "tightness" parameter puts a penalty on
    deviations from the local tempo.
    """
    score_thresh = 0.01 * localscore.max()
    first_beat = True
    backlink[0] = -1
    cumscore[0] = localscore[0]
    tv = int(len(frames_per_beat) > 1)
    for i, score_i in enumerate(localscore):
        best_score = - np.inf
        beat_location = -1
        for loc in range(i - int(np.round(frames_per_beat[tv * i] / 2)), i - int(2 * frames_per_beat[tv * i]) - 1, - 1):
            if loc < 0:
                break
            score = cumscore[loc] - tightness * (np.log(i - loc) - np.log(frames_per_beat[tv * i]))**2
            if score > best_score:
                best_score = score
                beat_location = loc
        if beat_location >= 0:
            cumscore[i] = score_i + best_score
            backlink[i] = beat_location
        else:
            cumscore[i] = score_i
            backlink[i] = -1
        if first_beat and score_i < score_thresh:
            backlink[i] = -1
        else:
            first_beat = False

@numba.guvectorize(["void(float32[:], bool_[:], bool_, bool_[:])", "void(float64[:], bool_[:], bool_, bool_[:])"], "(t),(t),()->(t)", nopython=True, cache=True)
def _trim_leading_trailing_beats(localscore, beats, trim, beats_trimmed):
    beats_trimmed[:] = beats
    if not np.any(beats):
        return
    w = np.hanning(5)
    smooth_boe = np.convolve(localscore[beats], w)[len(w)//2:len(localscore)+len(w)//2]
    if trim:
        threshold = 0.5 * ((smooth_boe**2).mean()**0.5)
    else:
        threshold = 0.0
    n = 0
    while n < len(localscore) and localscore[n] <= threshold:
        beats_trimmed[n] = False
        n += 1
    n = len(localscore) - 1
    while n >= 0 and localscore[n] <= threshold:
        beats_trimmed[n] = False
        n -= 1
    pass

@numba.guvectorize(["void(float32[:], bool_[:], float32, int64[:])", "void(float64[:], bool_[:], float64, int64[:])",], "(t),(t),()->()", nopython=True, cache=True)
def _find_best_last_beat(cumscore, mask, threshold, out):
    n = len(cumscore) - 1
    out[0] = n
    while n >= 0:
        if not mask[n] and cumscore[n] >= threshold:
            out[0] = n
            break
        else:
            n -= 1

def _get_last_beat_index(cumscore):
    mask = ~find_local_maxima(cumscore, axis=-1)
    masked_scores = np.ma.masked_array(data=cumscore, mask=mask)
    medians = np.ma.median(masked_scores, axis=-1)
    thresholds = 0.5 * np.ma.getdata(medians)
    tail = np.empty(shape=cumscore.shape[:-1], dtype=int)
    _find_best_last_beat(cumscore, mask, thresholds, tail)
    return tail

@numba.guvectorize(["void(int32[:], int32, bool_[:])", "void(int64[:], int64, bool_[:])"], "(t),()->(t)", nopython=True, cache=True)
def _backtrack_from_end(backlinks, tail, beats):
    n = tail
    while n >= 0:
        beats[n] = True
        n = backlinks[n]

def _run_beat_tracker(onset_envelope: np.ndarray, bpm: np.ndarray, frame_rate: float, tightness: float, trim: bool) -> np.ndarray:
    if np.any(bpm <= 0):
        raise ValueError(f"bpm={bpm} must be strictly positive")
    if tightness <= 0:
        raise ValueError("tightness must be strictly positive")
    if bpm.shape[-1] not in (1, onset_envelope.shape[-1]):
        raise ValueError(f"Invalid bpm shape={bpm.shape} does not match onset envelope shape={onset_envelope.shape}")

    frames_per_beat = np.round(frame_rate * 60.0 / bpm)
    localscore = _score_beats(_normalize_strength(onset_envelope), frames_per_beat)
    backlink, cumscore = _dp_beat_path(localscore, frames_per_beat, tightness)
    tail = _get_last_beat_index(cumscore)
    beats = np.zeros_like(onset_envelope, dtype=bool)
    _backtrack_from_end(backlink, tail, beats)
    beats: np.ndarray = _trim_leading_trailing_beats(localscore, beats, trim)
    return beats

def track_beats(*, y: Optional[np.ndarray] = None, sr: float = 22050, onset_envelope: Optional[np.ndarray] = None, hop_length: int = 512, start_bpm: float = 120.0, tightness: float = 100, trim: bool = True, bpm: Optional[Union[float, np.ndarray]] = None, prior: Optional[scipy.stats.rv_continuous] = None, units: str = "frames", sparse: bool = True) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """
    finds the beat locations in an audio signal.

    this function uses dynamic programming to find a sequence of
    beat times that best aligns with the signal's onset strength envelope.

    Args:
        y: input audio signal.
        sr: audio sample rate.
        onset_envelope: a pre-computed onset strength envelope.
        hop_length: number of samples per frame.
        start_bpm: a prior for the initial tempo guess.
        tightness: how strictly the tracker adheres to the estimated tempo.
        trim: if true, remove beats from silent sections at the start/end.
        bpm: a pre-computed tempo. if none, it is estimated automatically.
        prior: a `scipy.stats` distribution for the tempo prior.
        units: the units for the output beat locations ('frames', 'samples', 'time').
        sparse: if true, returns an array of indices. if false, returns a boolean mask.
    """
    if onset_envelope is None:
        if y is None:
            raise ValueError("y or onset_envelope must be provided")
        onset_envelope = compute_onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)

    if sparse and onset_envelope.ndim != 1:
        raise ValueError(f"sparse=True (default) does not support "
                             f"{onset_envelope.ndim}-dimensional inputs. "
                             f"Either set sparse=False or convert the signal to mono.")

    if not onset_envelope.any():
        if sparse:
            return (np.array(0.0), np.array([], dtype=int))
        else:
            return (np.zeros(shape=onset_envelope.shape[:-1], dtype=float), np.zeros_like(onset_envelope, dtype=bool))

    if bpm is None:
        bpm = estimate_tempo(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length, start_bpm=start_bpm, prior=prior)

    _bpm = np.atleast_1d(bpm)
    bpm_expanded = expand_to(_bpm, ndim=onset_envelope.ndim, axes=range(_bpm.ndim))

    beats = _run_beat_tracker(onset_envelope, bpm_expanded, float(sr) / hop_length, tightness, trim)

    if sparse:
        beats = np.flatnonzero(beats)
        if units == "frames":
            pass
        elif units == "samples":
            return (bpm, frames_to_samples(beats, hop_length=hop_length))
        elif units == "time":
            return (bpm, frames_to_time(beats, hop_length=hop_length, sr=sr))
        else:
            raise ValueError(f"Invalid unit type: {units}")
    return (bpm, beats)