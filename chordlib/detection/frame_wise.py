# This is not used in the GUI/anywhere else, but exists as a reference implementation to show 
# that we can detect chords using a straightforward sliding window without tracking beats. 
# But the beats one is more suitable (musically) for analysis on fixed files.
# This could be used in a future addition, such as for real-time detection using the microphone.

import numpy as np
from ..profiles.detector import ChordTemplateMatcher
from typing import List, Tuple


class ChordsDetection:
    """Frame-wise chord detection using median smoothing."""

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 2048,
        window_size_sec: float = 1.0,
        profile_type: str = "tonictriad",
        pcp_threshold: float = 1e-3,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.window_size_sec = window_size_sec
        # Compute half-window in frames
        frames_per_second = sample_rate / hop_size
        window_frames = int(window_size_sec * frames_per_second)
        self.half_window = max(window_frames // 2, 0)
        self.pcp_threshold = pcp_threshold
        self.chord_matcher = ChordTemplateMatcher(
            profile_type=profile_type, use_polyphony=False, use_three_chords=True
        )

    def detect(
        self, pcp_sequence: np.ndarray
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        detect chords frame by frame (sliding window) from a PCP sequence (pitch class profiles).
        """
        chords: List[str] = []
        strengths: List[float] = []
        rels: List[float] = []

        total_frames = len(pcp_sequence)
        for idx in range(total_frames):
            start = max(0, idx - self.half_window)
            end = min(total_frames, idx + self.half_window + 1)

            # median smoothing
            window = pcp_sequence[start:end]
            median_vec = np.median(window, axis=0)

            # threshold check
            max_val = np.max(median_vec)
            if max_val < self.pcp_threshold:
                chords.append("N")
                strengths.append(0.0)
                rels.append(0.0)
                continue

            # normalize to unit max
            normalized = median_vec / max_val

            # detect chord root/scale
            root, scale, strength, rel = self.chord_matcher.detect(normalized)
            chord_label = root + ("m" if scale == "minor" else "")

            chords.append(chord_label)
            strengths.append(strength)
            rels.append(rel)

        return chords, strengths, rels
