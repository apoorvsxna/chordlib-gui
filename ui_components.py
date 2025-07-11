from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QSlider, QStyle, QStyleOptionSlider
)
from PyQt5.QtCore import Qt

class ClickableSlider(QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            if handle_rect.contains(event.pos()):
                super().mousePressEvent(event)
                return
            if self.orientation() == Qt.Horizontal:
                val = event.pos().x() / self.width()
            else:
                val = (self.height() - event.pos().y()) / self.height()
            new_value = self.minimum() + val * (self.maximum() - self.minimum())
            self.setValue(int(new_value))
        else:
            super().mousePressEvent(event)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, n_fft=4096, hop_length=2048, min_duration_sec=0.1):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.n_fft_spinbox = QSpinBox()
        self.n_fft_spinbox.setRange(256, 16384); self.n_fft_spinbox.setSingleStep(256); self.n_fft_spinbox.setValue(n_fft)
        self.n_fft_spinbox.setToolTip("FFT window size. Higher values give better frequency resolution but are slower.")
        
        self.hop_length_spinbox = QSpinBox()
        self.hop_length_spinbox.setRange(128, 8192); self.hop_length_spinbox.setSingleStep(128); self.hop_length_spinbox.setValue(hop_length)
        self.hop_length_spinbox.setToolTip("Hop length between frames. Smaller values are slower but provide better time resolution.")

        self.min_duration_spinbox = QDoubleSpinBox()
        self.min_duration_spinbox.setRange(0.01, 1.0); self.min_duration_spinbox.setSingleStep(0.01); self.min_duration_spinbox.setValue(min_duration_sec); self.min_duration_spinbox.setDecimals(2)
        self.min_duration_spinbox.setToolTip("Filter out detected chords shorter than this duration (in seconds).")

        form_layout.addRow("N_FFT:", self.n_fft_spinbox)
        form_layout.addRow("Hop Length:", self.hop_length_spinbox)
        form_layout.addRow("Min. Chord Duration (s):", self.min_duration_spinbox)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        
        layout.addLayout(form_layout)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            'n_fft': self.n_fft_spinbox.value(), 
            'hop_length': self.hop_length_spinbox.value(),
            'min_duration_sec': self.min_duration_spinbox.value()
        }