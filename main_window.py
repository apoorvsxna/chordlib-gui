import os
import json
import csv
import bisect
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QStyle
)
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from config import APP_TITLE, UPDATE_INTERVAL_MS
from ui_builder import UIBuilder
from visuals_manager import VisualsManager
from ui_components import SettingsDialog
from analysis import AnalysisWorker, ChordTemplateMatcher

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1600, 900)
        
        # --- App State ---
        self.analysis_results = {}
        self.current_file_path = ""
        self.auto_scroll_enabled = True
        self.current_pcp_display = np.zeros(12)
        self.n_fft = 4096
        self.hop_length = 2048
        self.min_duration_sec = 0.1
        self.chord_matcher = ChordTemplateMatcher(profile_type='tonictriad')
        self.chord_templates = self._get_chord_templates()

        # --- Media Player and Timers ---
        self.media_player = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.update_timer = QTimer()

        # --- UI and Visuals ---
        self.ui = UIBuilder(self)
        self.ui.setup_ui()
        self.visuals = VisualsManager(self)
        
        self._connect_signals()
        self._apply_stylesheet()
        
        self.status_bar.showMessage("Ready. Please open an audio file.")
        self.show()

    def _get_chord_templates(self):
        templates = {}
        note_names_sharp = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        note_names_flat = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']
        for i in range(12):
            maj_template = np.roll(self.chord_matcher._profile_M, i)
            templates[note_names_sharp[i]] = maj_template
            if note_names_sharp[i] != note_names_flat[i]: templates[note_names_flat[i]] = maj_template
            min_template = np.roll(self.chord_matcher._profile_m, i)
            templates[note_names_sharp[i] + 'm'] = min_template
            if note_names_sharp[i] != note_names_flat[i]: templates[note_names_flat[i] + 'm'] = min_template
        return templates

    def _connect_signals(self):
        # top controls
        self.open_button.clicked.connect(self.open_file)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.export_button.clicked.connect(self.export_chords)
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.auto_scroll_button.clicked.connect(self.toggle_auto_scroll)
        
        # player controls
        self.play_button.clicked.connect(self.toggle_play)
        self.position_slider.valueChanged.connect(self.set_player_position)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_change)
        self.media_player.stateChanged.connect(self.handle_player_state_change)
        
        # table and plots
        self.chord_table.cellClicked.connect(self.jump_to_chord)
        self.p_timeline.scene().sigMouseClicked.connect(self.seek_from_plot_click)
        self.p_chroma.scene().sigMouseClicked.connect(self.seek_from_plot_click)
        self.p_waveform.scene().sigMouseClicked.connect(self.seek_from_plot_click)

        # timer
        self.update_timer.timeout.connect(self.smooth_update)
        self.update_timer.setInterval(UPDATE_INTERVAL_MS)

    def _apply_stylesheet(self):
        pg.setConfigOption('background', '#1F1F1F')
        pg.setConfigOption('foreground', 'w')
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2D2D2D; color: #E0E0E0; font-family: Arial, sans-serif; }
            QPushButton { background-color: #555; border: 1px solid #777; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
            QPushButton:pressed { background-color: #444; }
            QPushButton:disabled { background-color: #404040; color: #888; }
            QPushButton:checked { background-color: #007ACC; border-color: #007ACC; }
            QLabel { padding: 5px; }
            QSlider::groove:horizontal { border: 1px solid #444; height: 5px; background: #444; margin: 2px 0; border-radius: 2px; }
            QSlider::handle:horizontal { background: #007ACC; border: 1px solid #007ACC; width: 16px; margin: -6px 0; border-radius: 8px; }
            QStatusBar { background-color: #252525; }
            QTableWidget { background-color: #3C3C3C; gridline-color: #555; selection-background-color: #007ACC; }
            QHeaderView::section { background-color: #555; padding: 4px; border: 1px solid #666; }
            QSplitter::handle { background: #555; }
            QDialog { background-color: #3C3C3C; }
            QSpinBox, QDoubleSpinBox { background-color: #2D2D2D; padding: 2px; border: 1px solid #555; border-radius: 3px; }
        """)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.aac)")
        if file_path:
            self.current_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.analyze_button.setEnabled(True)
            self.status_bar.showMessage(f"Loaded '{os.path.basename(file_path)}'. Ready to analyze.")
            self._reset_application_state()

    def start_analysis(self):
        if not self.current_file_path: return
        self.analyze_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.status_bar.showMessage("Starting analysis...")
        self.worker = AnalysisWorker(
            self.current_file_path, self.n_fft, self.hop_length, self.min_duration_sec, self.chord_matcher)
        self.worker.progress.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.handle_analysis_results)
        self.worker.error.connect(self.handle_analysis_error)
        self.worker.start()

    def handle_analysis_results(self, results):
        self.analysis_results = results
        self.status_bar.showMessage("Analysis complete. Populating visuals...")
        
        self.visuals.draw_initial_plots(results)
        
        self.status_bar.showMessage("Analysis complete. Loading audio for playback...")
        self.play_button.setEnabled(False)
        self.position_slider.setEnabled(False)
        self.export_button.setEnabled(True)
        
        url = QUrl.fromLocalFile(self.current_file_path)
        self.media_player.setMedia(QMediaContent(url))
        self.open_button.setEnabled(True)

    def handle_analysis_error(self, message): 
        self.status_bar.showMessage(message)
        self.analyze_button.setEnabled(True)
        self.open_button.setEnabled(True)

    def open_settings_dialog(self):
        dialog = SettingsDialog(self, self.n_fft, self.hop_length, self.min_duration_sec)
        if dialog.exec_() == SettingsDialog.Accepted:
            settings = dialog.get_values()
            self.n_fft, self.hop_length, self.min_duration_sec = settings.values()
            self.status_bar.showMessage(f"Settings updated: N_FFT={self.n_fft}, Hop={self.hop_length}, MinDur={self.min_duration_sec}s")

    def export_chords(self):
        if not self.analysis_results.get('grouped_chords'):
            self.status_bar.showMessage("No analysis results to export.")
            return
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        default_path = os.path.join(os.path.dirname(self.current_file_path), f"{base_name}_chords")
        filePath, selected_filter = QFileDialog.getSaveFileName(self, "Export Chords", default_path, "CSV Files (*.csv);;JSON Files (*.json)")
        if not filePath: return
        try:
            if 'csv' in selected_filter:
                with open(filePath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["start_time", "end_time", "chord", "strength"])
                    for event in self.analysis_results['grouped_chords']:
                        writer.writerow([f"{event['start']:.3f}", f"{event['end']:.3f}", event['chord'], event['strength']])
            elif 'json' in selected_filter:
                with open(filePath, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results['grouped_chords'], f, indent=4)
            self.status_bar.showMessage(f"Chords successfully exported to {os.path.basename(filePath)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error exporting file: {e}")

    def smooth_update(self):
        if not self.analysis_results or self.media_player.state() != QMediaPlayer.PlayingState:
            return
        
        current_time_s = self.media_player.position() / 1000.0
        
        self.visuals.update_playhead(current_time_s)
        self.visuals.update_live_pcp(current_time_s, self.analysis_results)

        if self.auto_scroll_enabled:
            self.update_auto_scroll(current_time_s)
            
        chord_index = self._find_chord_index_at_time(current_time_s)
        self.visuals.update_table_highlight(chord_index)

        if chord_index != -1:
            chord_name = self.analysis_results['grouped_chords'][chord_index]['chord']
            self.visuals.update_inspector_template(chord_name)
        else:
            self.visuals.update_inspector_template(None)

    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState: self.media_player.pause()
        else: self.media_player.play()

    def handle_player_state_change(self, state):
        if state == QMediaPlayer.PlayingState: 
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.update_timer.start()
        else: 
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.update_timer.stop()
            
    def handle_media_status_change(self, status):
        if status == QMediaPlayer.LoadedMedia: 
            self.play_button.setEnabled(True); self.position_slider.setEnabled(True)
            self.status_bar.showMessage("Ready to play.")
        elif status == QMediaPlayer.InvalidMedia: self.status_bar.showMessage("Error: Could not load the audio file.")
        elif status == QMediaPlayer.StalledMedia: self.status_bar.showMessage("Playback stalled, buffering...")

    def update_position(self, position):
        self.position_slider.blockSignals(True); self.position_slider.setValue(position); self.position_slider.blockSignals(False)
        duration = self.media_player.duration()
        pos_secs, dur_secs = position // 1000, duration // 1000
        self.time_label.setText(f"{pos_secs//60:02d}:{pos_secs%60:02d} / {dur_secs//60:02d}:{dur_secs%60:02d}")
    
    def update_duration(self, duration): self.position_slider.setRange(0, duration)
    def set_player_position(self, position): self.media_player.setPosition(position)
    def toggle_auto_scroll(self):
        self.auto_scroll_enabled = self.auto_scroll_button.isChecked()
        self.auto_scroll_button.setText("Auto-Scroll: ON" if self.auto_scroll_enabled else "Auto-Scroll: OFF")

    def jump_to_chord(self, row, column):
        start_time_str = self.chord_table.item(row, 0).text()
        self.media_player.setPosition(int(float(start_time_str) * 1000))
        if self.media_player.state() != QMediaPlayer.PlayingState: self.media_player.play()
            
    def seek_from_plot_click(self, event):
        if event.button() == Qt.LeftButton and self.analysis_results:
            mouse_point = self.p_timeline.vb.mapSceneToView(event.scenePos())
            time_s = mouse_point.x()
            duration_s = self.analysis_results.get('duration', 0)
            if 0 <= time_s <= duration_s: self.media_player.setPosition(int(time_s * 1000))
                
    def closeEvent(self, event):
        self.update_timer.stop(); self.media_player.stop()
        if hasattr(self, 'worker') and self.worker.isRunning(): self.worker.quit(); self.worker.wait()
        event.accept()

    # private helpers

    def _reset_application_state(self):
        self.visuals.reset_all()
        self.play_button.setEnabled(False)
        self.position_slider.setEnabled(False)
        self.export_button.setEnabled(False)
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        self.update_timer.stop()
        if self.media_player.state() == QMediaPlayer.PlayingState: self.media_player.stop()

    def _find_chord_index_at_time(self, time_s):
        if 'chord_start_times' not in self.analysis_results: return -1
        start_times = self.analysis_results['chord_start_times']
        chords = self.analysis_results['grouped_chords']
        index = bisect.bisect_right(start_times, time_s) - 1
        if 0 <= index < len(chords) and time_s < chords[index]['end']: return index
        return -1

    def update_auto_scroll(self, current_time_s):
        view_box = self.p_timeline.vb
        view_range = view_box.viewRange()[0]
        view_width = view_range[1] - view_range[0]
        if view_width <= 0: return

        dead_zone_left = view_range[0] + view_width * 0.3
        dead_zone_right = view_range[0] + view_width * 0.7
        target_center = None
        if current_time_s > dead_zone_right: target_center = current_time_s - view_width * 0.2
        elif current_time_s < dead_zone_left: target_center = current_time_s + view_width * 0.2

        if target_center is not None:
            current_center = view_range[0] + view_width / 2.0
            new_center = current_center + (target_center - current_center) * 0.05
            duration = self.analysis_results['duration']
            half_view = view_width / 2.0
            new_center = max(half_view, min(new_center, duration - half_view))
            new_start = new_center - half_view
            view_box.setXRange(new_start, new_start + view_width, padding=0)