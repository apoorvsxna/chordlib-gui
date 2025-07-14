import numpy as np
import math
from typing import Tuple, List, Dict

class ChordTemplateMatcher:
    """
    Detects the best-matching chord (root and scale) for a given PCP/HPCP
    vector by correlating it against a set of ideal chord templates.
    """
    def __init__(
        self,
        profile_type: str = 'tonictriad',
        use_polyphony: bool = False,
        use_three_chords: bool = True,
        num_harmonics: int = 4,
        slope: float = 0.6,
        use_majmin: bool = False,
        pcp_size: int = 12
    ):
        # Config
        self.profile_type = profile_type
        self.use_polyphony = use_polyphony
        self.use_three_chords = use_three_chords
        self.num_harmonics = num_harmonics
        self.slope = slope
        self.use_majmin = use_majmin
        self.pcp_size = pcp_size

        # Base 12-bin major/minor profiles
        self.base_profiles: Dict[str, Tuple[List[float], List[float]]] = {
            'tonictriad': (
                [1,0,0,0,1,0,0,1,0,0,0,0],
                [1,0,0,1,0,0,0,1,0,0,0,0]
            )
        }
        self.other_profiles: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

        self._note_names = ['A','Bb','B','C','C#','D','Eb','E','F','F#','G','Ab']

        self._build_profiles()

    def _build_profiles(self):
        """Expand, optionally apply polyphony, and compute stats."""
        # 1) Select base 12-bin profiles
        if self.profile_type in self.base_profiles:
            major_12, minor_12 = self.base_profiles[self.profile_type]
            other_12 = [0.0] * 12
        else:
            major_12, minor_12, other_12 = self.other_profiles[self.profile_type]

        # 2) Polyphony expansion
        if self.use_polyphony:
            poly_M = [0.0]*12
            poly_m = [0.0]*12
            self._add_major_contributions(major_12, poly_M)
            self._add_minor_contributions(minor_12, poly_m)
            major_12, minor_12 = poly_M, poly_m

        # 3) Resize to pcp_size
        self._profile_M = self._resize_profile(major_12)
        self._profile_m = self._resize_profile(minor_12)
        self._profile_O = (
            self._resize_profile(other_12)
            if self.use_majmin else
            [0.0]*len(self._profile_M)
        )

        # 4) Compute mean/std for each
        self._mean_M, self._std_M = float(np.mean(self._profile_M)), float(np.std(self._profile_M))
        self._mean_m, self._std_m = float(np.mean(self._profile_m)), float(np.std(self._profile_m))
        self._mean_O, self._std_O = float(np.mean(self._profile_O)), float(np.std(self._profile_O))

    def _resize_profile(self, base12: List[float]) -> List[float]:
        """
        Linearly interpolate a 12-element profile into pcp_size bins.
        """
        bins_per_semi = self.pcp_size // 12
        extended = [0.0] * (bins_per_semi * 12)
        for s in range(12):
            v0 = base12[s]
            v1 = base12[(s + 1) % 12]
            for j in range(bins_per_semi):
                frac = j / bins_per_semi
                extended[s * bins_per_semi + j] = v0 + (v1 - v0) * frac
        return extended

    def _add_harmonics(self, root: int, weight: float, out: List[float]):
        """Distribute a weight across harmonic bins with cosine taper."""
        for h in range(1, self.num_harmonics + 1):
            idx_f = root + 12 * math.log2(h)
            f0, f1 = math.floor(idx_f), math.ceil(idx_f)
            i0, i1 = int(f0) % 12, int(f1) % 12
            if i0 < i1:
                d0 = idx_f - f0
                d1 = f1 - idx_f
                out[i0] += (math.cos(0.5*math.pi*d0)**2) * weight
                out[i1] += (math.cos(0.5*math.pi*d1)**2) * weight
            else:
                out[i0] += weight
            weight *= self.slope

    def _add_major_contributions(self, src: List[float], dst: List[float]):
        """Add harmonics for root, major third, and perfect fifth."""
        for root, w in enumerate(src):
            if w == 0.0:
                continue
            for interval in (0, 4, 7):
                pitch = (root + interval) % 12
                self._add_harmonics(pitch, w, dst)

    def _add_minor_contributions(self, src: List[float], dst: List[float]):
        """Add harmonics for root, minor third, and perfect fifth."""
        for root, w in enumerate(src):
            if w == 0.0:
                continue
            for interval in (0, 3, 7):
                pitch = (root + interval) % 12
                self._add_harmonics(pitch, w, dst)

    def detect(self, pcp: np.ndarray) -> Tuple[str, str, float, float]:
        """
        Slide each profile across the PCP vector to find the best match.
        Returns (root, scale, strength, relative_strength).
        """
        mean_p = float(np.mean(pcp))
        std_p = float(np.std(pcp))
        if std_p == 0.0:
            return 'N', 'major', 0.0, 0.0

        bestM, secondM, idxM = -1e9, -1e9, 0
        bestm, secondm, idxm = -1e9, -1e9, 0
        bestO, secondO, idxO = -1e9, -1e9, 0

        L = len(pcp)
        for shift in range(L):
            # Major correlation
            profM = np.roll(self._profile_M, shift)
            numM = np.dot(pcp-mean_p, profM-self._mean_M)
            denM = std_p * self._std_M
            corrM = numM/denM if denM != 0 else 0.0
            if corrM > bestM:
                secondM, bestM, idxM = bestM, corrM, shift

            # Minor correlation
            profm = np.roll(self._profile_m, shift)
            numm = np.dot(pcp-mean_p, profm-self._mean_m)
            denm = std_p * self._std_m
            corrm = numm/denm if denm != 0 else 0.0
            if corrm > bestm:
                secondm, bestm, idxm = bestm, corrm, shift

            # Optional “other” profiles
            if self.use_majmin:
                profO = np.roll(self._profile_O, shift)
                numO = np.dot(pcp-mean_p, profO-self._mean_O)
                denO = std_p * self._std_O
                corrO = numO/denO if denO != 0 else 0.0
                if corrO > bestO:
                    secondO, bestO, idxO = bestO, corrO, shift

        # Choose the winning scale
        if bestM >= bestm and bestM >= bestO:
            scale, strength, second, idx_win = 'major', bestM, secondM, idxM
        elif bestm >= bestM and bestm >= bestO:
            scale, strength, second, idx_win = 'minor', bestm, secondm, idxm
        else:
            scale, strength, second, idx_win = 'majmin', bestO, secondO, idxO

        # Map shift to key index 0–11 and get root name
        root_idx = int(round(idx_win * 12.0 / L)) % 12
        root = self._note_names[root_idx]

        # Relative strength measure
        rel = (strength - second) / strength if strength != 0 else 0.0
        return root, scale, float(strength), float(rel)