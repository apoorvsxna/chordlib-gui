import numpy as np
import librosa
from PyQt5.QtCore import QThread, pyqtSignal
from config import SAMPLE_RATE

try:
    from chordlib.features.pcp import extract_pcp
    from chordlib.features.beats import track_beats
    from chordlib.detection.beat_synchronous import ChordsDetectionBeats
    from chordlib.analysis.grouping import group_chord_events
    from chordlib._internal import dsp
except ImportError as e:
    print("Warning: 'chordlib' not found. Using dummy analysis functions.")
    print(e)
    class ChordTemplateMatcher:
        def __init__(self, *args, **kwargs):
            self._profile_M = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
            self._profile_m = np.array([1,0,0,1,0,0,0,1,0,0,0,0])
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def detect(self, *args, **kwargs): return np.array([0,1,2]), ['C', 'G', 'Am'], [11.2, 10.8, 10.1], None
    ChordsDetectionBeats = Dummy
    def extract_pcp(y, sr, n_fft, hop_length, **kwargs): return np.random.rand(int(len(y)/hop_length), 12)
    def track_beats(y, sr, hop_length, sparse, **kwargs): return 0, np.linspace(0, int(len(y)/hop_length)-1, 20).astype(int)
    def group_chord_events(times, chords, strengths, min_duration_sec=0.1):
        events = []
        if not any(chords): return events
        sorted_indices = np.argsort(times)
        times, chords, strengths = np.array(times)[sorted_indices], np.array(chords)[sorted_indices], np.array(strengths)[sorted_indices]
        for i in range(len(times)):
            start = times[i]
            end = times[i+1] if i + 1 < len(times) else times[i] + 1
            if end - start >= min_duration_sec:
                 events.append({'start': start, 'end': end, 'chord': chords[i], 'strength': strengths[i]})
        return events
    class dsp:
        @staticmethod
        def compute_onset_strength(y, sr, hop_length, **kwargs): return np.random.rand(int(len(y)/hop_length))
    class utils:
        @staticmethod
        def frames_to_time(frames, sr, hop_length, **kwargs): return frames * hop_length / sr

class AnalysisWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, file_path, n_fft, hop_length, min_duration_sec, chord_matcher):
        super().__init__()
        self.file_path = file_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_duration_sec = min_duration_sec
        self.chord_matcher = chord_matcher

    def run(self):
        try:
            results = {}
            self.progress.emit("Loading audio file...", 10)
            y, sr = librosa.load(self.file_path, sr=SAMPLE_RATE, mono=True)
            results['waveform'] = y; results['sr'] = sr; results['duration'] = librosa.get_duration(y=y, sr=sr)
            self.progress.emit("1/3: Calculating Onset Strength & Beats...", 30)
            onset_env = dsp.compute_onset_strength(y=y, sr=sr, hop_length=self.hop_length, aggregate=np.median)
            _, beats = track_beats(y=y, sr=sr, hop_length=self.hop_length, sparse=True)
            results['onset_envelope'] = onset_env; results['beats'] = beats
            self.progress.emit("2/3: Extracting PCP Chromagram...", 60)
            pcp_sequence = extract_pcp(y, sr, n_fft=self.n_fft, hop_length=self.hop_length)
            results['pcp'] = pcp_sequence
            self.progress.emit("3/3: Detecting and grouping chords...", 85)
            detector = ChordsDetectionBeats(sample_rate=sr, hop_size=self.hop_length)
            detector.chord_matcher = self.chord_matcher
            times, chords, strengths, _ = detector.detect(y, pcp_sequence)
            grouped_chords = group_chord_events(np.array(times), chords, strengths, min_duration_sec=self.min_duration_sec)
            
            results['chord_start_times'] = [event['start'] for event in grouped_chords]
            results['grouped_chords'] = grouped_chords
            results['hop_length'] = self.hop_length
            
            self.progress.emit("Analysis complete!", 100)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"An error occurred during analysis: {e}")