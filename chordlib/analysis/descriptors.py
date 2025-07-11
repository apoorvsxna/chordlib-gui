import numpy as np
from collections import Counter
from typing import List, Dict, Any


class ChordDescriptors:
    """
    computes the following descriptors for a chord progression:
      - chords_changes_rate
      - chords_number_rate
      - chords_key and chords_scale
      - chords_histogram_labels & values (rotated to key)
    """

    def __init__(self):
        # circle of fifths labels and alternates
        self.circle = [
            "C",
            "Em",
            "G",
            "Bm",
            "D",
            "F#m",
            "A",
            "C#m",
            "E",
            "Abm",
            "B",
            "Ebm",
            "F#",
            "Bbm",
            "C#",
            "Fm",
            "Ab",
            "Cm",
            "Eb",
            "Gm",
            "Bb",
            "Dm",
            "F",
            "Am",
        ]
        self.alt = [
            "C",
            "Em",
            "G",
            "Bm",
            "D",
            "Gbm",
            "A",
            "Dbm",
            "E",
            "G#m",
            "B",
            "D#m",
            "Gb",
            "A#m",
            "Db",
            "Fm",
            "G#",
            "Cm",
            "D#",
            "Gm",
            "A#",
            "Dm",
            "F",
            "Am",
        ]

        self.chord_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.circle)
        }
        for alt_name, idx in zip(self.alt, range(len(self.alt))):
            self.chord_to_index[alt_name] = idx

    def describe(self, chords: List[str]) -> Dict[str, Any]:
        """
        analyze chord list to compute the following:
          - change rate
          - unique chord rate
          - inferred key & scale
          - rotated histogram
        """
        prog = [c for c in chords if c != "N"]
        if not prog:
            raise ValueError("Cannot describe an empty chord progression.")

        total = len(prog)

        changes = sum(1 for i in range(1, total) if prog[i] != prog[i - 1])
        changes_rate = changes / total

        # histogram on circle
        raw_hist = np.zeros(len(self.circle), dtype=float)
        for c in prog:
            idx = self.chord_to_index.get(c)
            if idx is not None:
                raw_hist[idx] += 1
        hist_percent = (raw_hist / total) * 100.0

        # most frequent chord = the key of the music (not always true but works most of the time)
        counts = Counter(prog)
        most_freq, _ = max(
            counts.items(),
            key=lambda item: (item[1], -self.chord_to_index.get(item[0], 0)),
        )
        if most_freq.endswith("m"):
            key_root = most_freq[:-1]
            scale = "minor"
        else:
            key_root = most_freq
            scale = "major"

        # rotate histogram so inferred key is at index 0
        key_idx = self.chord_to_index.get(most_freq, 0)
        rotated = np.roll(hist_percent, -key_idx)

        # unique- chord rate: bins >1% / total chords
        unique_count = np.sum(rotated > 1.0)
        number_rate = unique_count / total

        return {
            "chords_changes_rate": changes_rate,
            "chords_number_rate": number_rate,
            "chords_key": key_root,
            "chords_scale": scale,
            "chords_histogram_labels": self.circle,
            "chords_histogram_values": rotated.tolist(),
        }
