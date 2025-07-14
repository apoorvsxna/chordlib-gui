import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStyle,
    QStatusBar,
    QTableWidget,
    QAbstractItemView,
    QHeaderView,
    QSplitter,
)
from PyQt5.QtCore import Qt
from config import NOTE_NAMES
from ui_components import ClickableSlider


class UIBuilder:
    def __init__(self, main_window):
        self.win = main_window

    def setup_ui(self):
        self.win.central_widget = QWidget()
        self.win.setCentralWidget(self.win.central_widget)
        main_layout = QVBoxLayout(self.win.central_widget)

        self._setup_top_controls()
        top_controls_layout = self.win.top_controls_layout

        self._setup_main_views()
        main_splitter = self.win.main_splitter

        self._setup_player_controls()
        player_layout = self.win.player_layout

        self.win.status_bar = QStatusBar()
        self.win.setStatusBar(self.win.status_bar)

        main_layout.addLayout(top_controls_layout)
        main_layout.addWidget(main_splitter, 1)
        main_layout.addLayout(player_layout)

        self._setup_playheads()

    def _setup_top_controls(self):
        self.win.top_controls_layout = QHBoxLayout()
        self.win.open_button = QPushButton("Open Audio File")
        self.win.analyze_button = QPushButton("Analyze")
        self.win.export_button = QPushButton("Export Chords...")
        self.win.settings_button = QPushButton("Settings...")
        self.win.auto_scroll_button = QPushButton("Auto-Scroll: ON")
        self.win.file_label = QLabel("No file selected.")

        self.win.analyze_button.setEnabled(False)
        self.win.export_button.setEnabled(False)
        self.win.auto_scroll_button.setCheckable(True)
        self.win.auto_scroll_button.setChecked(True)
        self.win.file_label.setStyleSheet("color: #AAA;")

        self.win.top_controls_layout.addWidget(self.win.open_button)
        self.win.top_controls_layout.addWidget(self.win.analyze_button)
        self.win.top_controls_layout.addWidget(self.win.export_button)
        self.win.top_controls_layout.addWidget(self.win.settings_button)
        self.win.top_controls_layout.addWidget(self.win.auto_scroll_button)
        self.win.top_controls_layout.addWidget(self.win.file_label, 1)

    def _setup_main_views(self):
        self.win.main_splitter = QSplitter(Qt.Horizontal)

        # left side: visualizations
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        self.win.plot_widget = pg.GraphicsLayoutWidget()
        pg.setConfigOptions(antialias=True)

        self.win.p_timeline = self.win.plot_widget.addPlot(row=0, col=0)
        self.win.p_chroma = self.win.plot_widget.addPlot(row=1, col=0)
        self.win.p_waveform = self.win.plot_widget.addPlot(row=2, col=0)
        vis_layout.addWidget(self.win.plot_widget)

        # right side: inspector + table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.win.p_inspector = pg.PlotWidget()
        self.win.chord_table = QTableWidget()
        right_layout.addWidget(QLabel("Chord Progression Details"), 0)
        right_layout.addWidget(self.win.chord_table, 2)
        right_layout.addWidget(self.win.p_inspector, 1)

        self.win.main_splitter.addWidget(vis_widget)
        self.win.main_splitter.addWidget(right_widget)
        self.win.main_splitter.setSizes([1200, 400])

        self._configure_plots()

    def _configure_plots(self):
        # timeline plot
        self.win.p_timeline.setTitle("Chord Timeline")
        self.win.p_timeline.getAxis("left").hide()

        # chromagram plot
        self.win.p_chroma.setTitle("Harmonic Content (Chromagram)")
        self.win.img_chroma = pg.ImageItem()
        self.win.p_chroma.addItem(self.win.img_chroma)
        y_axis_chroma = self.win.p_chroma.getAxis("left")
        y_axis_chroma.setTicks([[(i, name) for i, name in enumerate(NOTE_NAMES)]])

        # waveform plot
        self.win.p_waveform.setTitle("Audio Waveform")
        self.win.p_waveform.setLabel("left", "Amplitude")
        self.win.p_waveform.setLabel("bottom", "Time (s)")

        # link x-axes
        self.win.p_chroma.setXLink(self.win.p_timeline)
        self.win.p_waveform.setXLink(self.win.p_timeline)

        # stretch factors
        self.win.plot_widget.ci.layout.setRowStretchFactor(0, 1)
        self.win.plot_widget.ci.layout.setRowStretchFactor(1, 3)
        self.win.plot_widget.ci.layout.setRowStretchFactor(2, 1)

        # inspector plot
        self.win.p_inspector.setTitle("Chord Matching Inspector")
        self.win.p_inspector.setYRange(0, 1.1)
        self.win.p_inspector.showGrid(x=True, y=True, alpha=0.3)
        inspector_xaxis = self.win.p_inspector.getAxis("bottom")
        inspector_xaxis.setTicks([[(i, name) for i, name in enumerate(NOTE_NAMES)]])
        self.win.p_inspector.addLegend(offset=(10, 10))
        self.win.template_bargraph = pg.BarGraphItem(
            x=np.arange(12) - 0.2,
            height=np.zeros(12),
            width=0.4,
            brush=pg.mkBrush(255, 255, 255, 100),
            name="Ideal Template",
        )
        self.win.pcp_bargraph = pg.BarGraphItem(
            x=np.arange(12) + 0.2,
            height=np.zeros(12),
            width=0.4,
            brush=pg.mkBrush(0, 150, 255, 200),
            name="Live Audio",
        )
        self.win.p_inspector.addItem(self.win.template_bargraph)
        self.win.p_inspector.addItem(self.win.pcp_bargraph)

        # chord table
        self.win.chord_table.setColumnCount(4)
        self.win.chord_table.setHorizontalHeaderLabels(
            ["Start (s)", "End (s)", "Chord", "Confidence"]
        )
        self.win.chord_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.win.chord_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.win.chord_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

    def _setup_player_controls(self):
        self.win.player_layout = QHBoxLayout()
        self.win.play_button = QPushButton()
        self.win.position_slider = ClickableSlider(Qt.Horizontal)
        self.win.time_label = QLabel("00:00 / 00:00")

        self.win.play_button.setIcon(self.win.style().standardIcon(QStyle.SP_MediaPlay))
        self.win.play_button.setEnabled(False)
        self.win.position_slider.setEnabled(False)

        self.win.player_layout.addWidget(self.win.play_button)
        self.win.player_layout.addWidget(self.win.position_slider)
        self.win.player_layout.addWidget(self.win.time_label)

    def _setup_playheads(self):
        playhead_pen = pg.mkPen("y", width=2)
        self.win.playhead_timeline = pg.InfiniteLine(
            angle=90, movable=False, pen=playhead_pen
        )
        self.win.playhead_chroma = pg.InfiniteLine(
            angle=90, movable=False, pen=playhead_pen
        )
        self.win.playhead_waveform = pg.InfiniteLine(
            angle=90, movable=False, pen=playhead_pen
        )
        self.win.p_timeline.addItem(self.win.playhead_timeline, ignoreBounds=True)
        self.win.p_chroma.addItem(self.win.playhead_chroma, ignoreBounds=True)
        self.win.p_waveform.addItem(self.win.playhead_waveform, ignoreBounds=True)
