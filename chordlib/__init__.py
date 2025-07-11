import numpy as np
from typing import List, Dict, Any, Literal

from .features.pcp import extract_pcp
from .detection.frame_wise import ChordsDetection
from .detection.beat_synchronous import ChordsDetectionBeats
from .analysis.grouping import group_chord_events


def recognize(
    audio_signal: np.ndarray,
    sample_rate: int,
    strategy: Literal["beat_synchronous", "frame_wise"] = "beat_synchronous",
    n_fft: int = 4096,
    hop_length: int = 1024,
    min_duration_sec: float = 0.1,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    recognize chords in an audio signal from start to finish.

    this is the main entry point for the library. it orchestrates the
    full pipeline:
      1. extracts pitch class profile (pcp) features.
      2. detects a raw sequence of chords using the specified strategy.
      3. groups consecutive identical chords into timed events.

    Args:
        audio_signal: the input audio waveform.
        sample_rate: the sample rate of the audio signal.
        strategy: the chord detection strategy to use.
            ['beat_synchronous', 'frame_wise']
        n_fft: the fft window size for spectral analysis.
        hop_length: the number of samples between consecutive analysis frames.
        min_duration_sec: minimum duration for a chord event to be kept.
        **kwargs: advanced configuration options passed to the underlying
            algorithms. (e.g., `sp_*` and `hp_*` for `extract_pcp`).
    """
    pcp_kwargs = {k: v for k, v in kwargs.items() if k.startswith(("sp_", "hp_"))}

    pcp_sequence = extract_pcp(
        audio_signal, sample_rate, n_fft=n_fft, hop_length=hop_length, **pcp_kwargs
    )

    if strategy == "beat_synchronous":
        detector_kwargs = {
            k: v for k, v in kwargs.items() if k in ["chroma_pick", "profile_type"]
        }
        detector = ChordsDetectionBeats(
            sample_rate=sample_rate, hop_size=hop_length, **detector_kwargs
        )
        times, chords, strengths, _ = detector.detect(audio_signal, pcp_sequence)

    elif strategy == "frame_wise":

        num_frames = pcp_sequence.shape[0]
        frame_indices = np.arange(num_frames)
        times = frame_indices * hop_length / sample_rate

        detector_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["window_size_sec", "profile_type", "pcp_threshold"]
        }
        detector = ChordsDetection(
            sample_rate=sample_rate, hop_size=hop_length, **detector_kwargs
        )
        chords, strengths, _ = detector.detect(pcp_sequence)
    else:
        raise ValueError(
            f"Unknown detection strategy: '{strategy}'. Must be 'beat_synchronous' or 'frame_wise'."
        )

    events = group_chord_events(
        np.array(times), chords, strengths, min_duration_sec=min_duration_sec
    )

    return events
