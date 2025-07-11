import numpy as np
import librosa
import pyqtgraph as pg
from PyQt5.QtWidgets import QTableWidgetItem, QGraphicsRectItem, QAbstractItemView
from config import NOTE_NAMES, INTERPOLATION_FACTOR

class VisualsManager:
    def __init__(self, main_window):
        self.win = main_window
        self.chord_viz_items = []

    def draw_initial_plots(self, results):
        self._draw_waveform(results)
        self._draw_chromagram(results)
        self.populate_chord_table(results)
        self.draw_chord_timeline(results)
        
        duration = results.get('duration', 0)
        self.win.p_timeline.setXRange(0, duration, padding=0)
        self.win.p_timeline.setYRange(-1, 1)

    def _draw_waveform(self, results):
        y = results.get('waveform', np.array([]))
        duration = results.get('duration', 0)
        time_axis = np.linspace(0, duration, len(y))
        self.win.p_waveform.plot(time_axis, y, pen=pg.mkPen('w', width=1), clear=True)

    def _draw_chromagram(self, results):
        y = results.get('waveform', np.array([]))
        sr = results.get('sr', 22050)
        hop_length = results.get('hop_length', 2048)
        duration = results.get('duration', 0)

        chromagram_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
        chromagram_reordered = np.roll(chromagram_cens, shift=-9, axis=0)
        
        self.win.img_chroma.setImage(chromagram_reordered.T)
        self.win.img_chroma.setRect(pg.QtCore.QRectF(0, 0, duration, 12))
        colormap = pg.colormap.get('plasma')
        self.win.img_chroma.setLookupTable(colormap.getLookupTable())

    def populate_chord_table(self, results):
        self.win.chord_table.setRowCount(0)
        for i, event in enumerate(results.get('grouped_chords', [])):
            self.win.chord_table.insertRow(i)
            self.win.chord_table.setItem(i, 0, QTableWidgetItem(f"{event['start']:.2f}"))
            self.win.chord_table.setItem(i, 1, QTableWidgetItem(f"{event['end']:.2f}"))
            self.win.chord_table.setItem(i, 2, QTableWidgetItem(event['chord']))
            
            raw_strength = event['strength']
            confidence_pct = max(0, raw_strength / 12.0) * 100
            self.win.chord_table.setItem(i, 3, QTableWidgetItem(f"{confidence_pct:.1f}%"))

    def draw_chord_timeline(self, results):
        for item in self.chord_viz_items: self.win.p_timeline.removeItem(item)
        self.chord_viz_items.clear()
        
        alpha = 150
        base_colors = [ (255, 0, 0), (0, 180, 0), (0, 0, 255), (0, 200, 200), (200, 0, 200), (220, 220, 0), (255, 165, 0), (255, 192, 203), (0, 100, 0), (100, 0, 100), (0, 100, 100), (100, 100, 0) ]
        brushes = [pg.mkBrush(c[0], c[1], c[2], alpha) for c in base_colors]
        note_to_brush = {note: brushes[i] for i, note in enumerate(NOTE_NAMES)}

        for event in results.get('grouped_chords', []):
            start, end, chord_name = event['start'], event['end'], event['chord']
            duration = end - start
            root_note = chord_name.replace('m', '').replace('7', '').replace('dim', '').replace('aug', '')
            brush = note_to_brush.get(root_note, pg.mkBrush(128, 128, 128, alpha))
            
            bar = QGraphicsRectItem(start, -0.5, duration, 1)
            bar.setBrush(brush); bar.setPen(pg.mkPen(None))
            self.win.p_timeline.addItem(bar)
            self.chord_viz_items.append(bar)
            
            text = pg.TextItem(text=chord_name, color=(255, 255, 255), anchor=(0.5, 0.5))
            text.setPos(start + duration / 2, 0)
            self.win.p_timeline.addItem(text)
            self.chord_viz_items.append(text)

    def update_playhead(self, time_s):
        self.win.playhead_timeline.setPos(time_s)
        self.win.playhead_chroma.setPos(time_s)
        self.win.playhead_waveform.setPos(time_s)

    def update_live_pcp(self, time_s, results):
        sr = results.get('sr', 1)
        hop = results.get('hop_length', 1)
        pcp_sequence = results.get('pcp', np.array([]))
        
        frame_index = int(time_s * sr / hop)
        if 0 <= frame_index < len(pcp_sequence):
            target_pcp = pcp_sequence[frame_index]
            self.win.current_pcp_display = (1 - INTERPOLATION_FACTOR) * self.win.current_pcp_display + INTERPOLATION_FACTOR * target_pcp
            self.win.pcp_bargraph.setOpts(height=self.win.current_pcp_display)

    def update_inspector_template(self, chord_name):
        if chord_name and chord_name in self.win.chord_templates:
            template = self.win.chord_templates[chord_name]
            self.win.template_bargraph.setOpts(height=template)
        else:
            self.win.template_bargraph.setOpts(height=np.zeros(12))

    def update_table_highlight(self, chord_index):
        if chord_index >= 0 and self.win.chord_table.currentRow() != chord_index:
            self.win.chord_table.selectRow(chord_index)
            self.win.chord_table.scrollToItem(self.win.chord_table.item(chord_index, 0), QAbstractItemView.PositionAtCenter)
        elif chord_index == -1 and self.win.chord_table.currentRow() != -1:
            self.win.chord_table.clearSelection()

    def reset_all(self):
        for item in self.chord_viz_items: self.win.p_timeline.removeItem(item)
        self.chord_viz_items.clear()
        self.win.p_waveform.clear()
        self.win.img_chroma.clear()
        self.win.chord_table.setRowCount(0)
        self.win.current_pcp_display = np.zeros(12)
        self.win.pcp_bargraph.setOpts(height=self.win.current_pcp_display)
        self.win.template_bargraph.setOpts(height=np.zeros(12))