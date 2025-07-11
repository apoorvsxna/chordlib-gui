import numpy as np
from ..features.beats import track_beats
from .._internal.utils import frames_to_time
from ..profiles.detector import ChordTemplateMatcher
from typing import List, Tuple


class ChordsDetectionBeats:

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 2048,
        chroma_pick: str = "interbeat_median",
        profile_type: str = "tonictriad",
    ):
        valid = ("interbeat_median", "starting_beat")
        if chroma_pick not in valid:
            raise ValueError(f"chroma_pick must be one of {valid}")
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.chroma_pick = chroma_pick
        self.chord_matcher = ChordTemplateMatcher(
            profile_type=profile_type, use_polyphony=False, use_three_chords=True
        )

    def detect(
        self, audio_data: np.ndarray, pcp_sequence: np.ndarray
    ) -> Tuple[np.ndarray, List[str], List[float], List[float]]:

        # beat tracking
        _, beat_frames = track_beats(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_size
        )

        # segment boundaries
        total = pcp_sequence.shape[0]
        frame_bounds = [0] + beat_frames.tolist() + [total]

        # convert to times
        times = frames_to_time(
            np.array(frame_bounds), sr=self.sample_rate, hop_length=self.hop_size
        )

        chords: List[str] = []
        strengths: List[float] = []
        rels: List[float] = []

        # analyze each segment
        for i in range(len(frame_bounds) - 1):
            start_f = frame_bounds[i]
            end_f = frame_bounds[i + 1]
            if end_f <= start_f:
                continue

            if self.chroma_pick == "starting_beat":
                vec = pcp_sequence[start_f]
            else:
                vec = np.median(pcp_sequence[start_f:end_f], axis=0)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            root, scale, strength, rel = self.chord_matcher.detect(vec)
            chord_label = root + ("m" if scale == "minor" else "")

            chords.append(chord_label)
            strengths.append(strength)
            rels.append(rel)

        return times[:-1], chords, strengths, rels
