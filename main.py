import sys
import tempfile
import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QInputDialog, QMessageBox, QComboBox, QLineEdit, QSlider, QFormLayout, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QRect, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from scipy.io.wavfile import write as wav_write, read as wav_read
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
import simpleaudio as sa
import threading
import vlc
import scipy.signal

class WaveformLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection = None  # (start_x, end_x) in pixels
        self.selecting = False
        self.audio_duration = 1.0  # seconds, default to avoid div by zero
        self.selection_changed_callback = None
        self.waveform_pixmap = None

    def set_audio_duration(self, duration):
        self.audio_duration = duration

    def set_waveform(self, pixmap):
        self.waveform_pixmap = pixmap
        self.setPixmap(pixmap)
        self.selection = None
        self.update()

    def mousePressEvent(self, event):
        if self.waveform_pixmap is None:
            return
        if event.button() == Qt.LeftButton:
            self.selecting = True
            self.selection = (event.x(), event.x())
            self.update()

    def mouseMoveEvent(self, event):
        if self.selecting and self.waveform_pixmap is not None:
            self.selection = (self.selection[0], event.x())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.selecting and self.waveform_pixmap is not None:
            self.selection = (self.selection[0], event.x())
            self.selecting = False
            self.update()
            if self.selection_changed_callback:
                start_px, end_px = self.selection
                start_px, end_px = min(start_px, end_px), max(start_px, end_px)
                width = self.width()
                start_sec = max(0, min(start_px / width * self.audio_duration, self.audio_duration))
                end_sec = max(0, min(end_px / width * self.audio_duration, self.audio_duration))
                if abs(end_sec - start_sec) < 0.01:
                    self.selection = None
                    self.update()
                    self.selection_changed_callback(None)
                else:
                    self.selection_changed_callback((start_sec, end_sec))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.waveform_pixmap and self.selection:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            start_px, end_px = self.selection
            x1, x2 = min(start_px, end_px), max(start_px, end_px)
            rect = QRect(int(x1), 0, int(x2-x1), self.height())
            painter.setBrush(QColor(0, 120, 255, 80))
            painter.setPen(Qt.NoPen)
            painter.drawRect(rect)
            painter.end()

class AudioPlaybackThread(QThread):
    playback_finished = Signal()
    def __init__(self, segment):
        super().__init__()
        self.segment = segment
        self._stop = False
        self.play_obj = None

    def run(self):
        import numpy as np
        import simpleaudio as sa
        samples = np.array(self.segment.get_array_of_samples())
        if self.segment.channels > 1:
            samples = samples.reshape((-1, self.segment.channels))
        audio_data = samples.astype(np.int16).tobytes()
        self.play_obj = sa.play_buffer(audio_data, num_channels=self.segment.channels, bytes_per_sample=self.segment.sample_width, sample_rate=self.segment.frame_rate)
        while self.play_obj.is_playing():
            if self._stop:
                self.play_obj.stop()
                break
            self.msleep(100)
        self.playback_finished.emit()

    def stop(self):
        self._stop = True
        if self.play_obj:
            self.play_obj.stop()

class ListenerThread(QThread):
    cue_detected = Signal()
    listening_status = Signal(str)
    corr_value = Signal(float)
    def __init__(self, cue_path, fs, threshold=0.7):
        super().__init__()
        self.cue_path = cue_path
        self.fs = fs
        self.threshold = threshold
        self._stop = False

    def run(self):
        import sounddevice as sd
        from pydub import AudioSegment
        import numpy as np
        cue_audio = AudioSegment.from_file(self.cue_path).set_channels(1).set_frame_rate(self.fs)
        cue_samples = np.array(cue_audio.get_array_of_samples()).astype(np.float32)
        cue_samples = cue_samples / (np.max(np.abs(cue_samples)) + 1e-8)
        cue_len = len(cue_samples)
        buffer = np.zeros(cue_len * 3, dtype=np.float32)
        self.listening_status.emit(f"Listening... (Threshold: {self.threshold:.2f})")
        try:
            with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32') as stream:
                while not self._stop:
                    audio_chunk, _ = stream.read(cue_len // 2)
                    audio_chunk = audio_chunk.flatten()
                    buffer = np.roll(buffer, -len(audio_chunk))
                    buffer[-len(audio_chunk):] = audio_chunk
                    # Normalize buffer
                    norm_buffer = buffer / (np.max(np.abs(buffer)) + 1e-8)
                    # Cross-correlation
                    corr = scipy.signal.correlate(norm_buffer, cue_samples, mode='valid')
                    corr /= (np.linalg.norm(norm_buffer) * np.linalg.norm(cue_samples) + 1e-8)
                    max_corr = float(np.max(corr))
                    self.corr_value.emit(max_corr)
                    if max_corr > self.threshold:
                        self.listening_status.emit("Cue detected! Triggering action...")
                        self.cue_detected.emit()
                        break
        except Exception as e:
            self.listening_status.emit(f"Listening error: {e}")
        self.listening_status.emit("Stopped listening.")

    def stop(self):
        self._stop = True

class QueueSection(QGroupBox):
    def __init__(self, idx, parent=None, vlc_instance=None):
        super().__init__(f"Audio Queue {idx+1}", parent)
        self.setCheckable(True)
        self.setChecked(idx == 0)
        self.idx = idx
        self.audio_path = None
        self.copied_segment = None
        self.selection = None
        self.detection_threshold = 0.5
        self.trigger_action_type = None
        self.trigger_action_value = None
        self.listener_thread = None
        self.fs = 44100
        self.max_record_sec = 20
        self.recording = False
        self.record_thread = None
        self.record_stop_event = threading.Event()
        self.last_corr = 0.0
        self.vlc_instance = vlc_instance
        self.vlc_player = None
        layout = QVBoxLayout()
        self.label = QLabel(f"Audio Queue {idx+1} controls")
        layout.addWidget(self.label)
        button_layout = QHBoxLayout()
        self.record_btn = QPushButton("Record Cue")
        self.stop_record_btn = QPushButton("Stop Recording")
        self.stop_record_btn.setEnabled(False)
        self.load_btn = QPushButton("Load Audio File")
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.stop_record_btn)
        button_layout.addWidget(self.load_btn)
        layout.addLayout(button_layout)
        self.waveform_label = WaveformLabel()
        self.waveform_label.setText("[Waveform Editor Placeholder]")
        self.waveform_label.setAlignment(Qt.AlignCenter)
        self.waveform_label.setStyleSheet("border: 1px solid #888; padding: 40px;")
        self.waveform_label.selection_changed_callback = self.on_waveform_selection
        layout.addWidget(self.waveform_label)
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.pause_btn)
        playback_layout.addWidget(self.stop_btn)
        layout.addLayout(playback_layout)
        edit_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select")
        self.cut_btn = QPushButton("Cut")
        self.copy_btn = QPushButton("Copy")
        self.paste_btn = QPushButton("Paste")
        self.delete_btn = QPushButton("Delete")
        self.trim_btn = QPushButton("Trim (≤3s)")
        for btn in [self.select_btn, self.cut_btn, self.copy_btn, self.paste_btn, self.delete_btn, self.trim_btn]:
            edit_layout.addWidget(btn)
        layout.addLayout(edit_layout)
        threshold_layout = QFormLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(int(self.detection_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_label = QLabel(f"Detection Threshold: {self.detection_threshold:.2f}")
        self.corr_label = QLabel(f"Last Correlation: {self.last_corr:.2f}")
        threshold_layout.addRow(self.threshold_label, self.threshold_slider)
        threshold_layout.addRow(self.corr_label)
        layout.addLayout(threshold_layout)
        trigger_layout = QHBoxLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Select Action", "Run Program", "Play Sound File"])
        self.action_value_edit = QLineEdit()
        self.action_value_edit.setPlaceholderText("No action assigned")
        self.action_value_edit.setReadOnly(True)
        self.action_browse_btn = QPushButton("Browse...")
        self.action_test_btn = QPushButton("Test Trigger Action")
        trigger_layout.addWidget(self.action_combo)
        trigger_layout.addWidget(self.action_value_edit)
        trigger_layout.addWidget(self.action_browse_btn)
        trigger_layout.addWidget(self.action_test_btn)
        layout.addLayout(trigger_layout)
        self.setLayout(layout)
        # Connect controls
        self.record_btn.clicked.connect(self.on_record)
        self.stop_record_btn.clicked.connect(self.on_stop_recording)
        self.load_btn.clicked.connect(self.on_load)
        self.select_btn.clicked.connect(self.on_select)
        self.cut_btn.clicked.connect(self.on_cut)
        self.copy_btn.clicked.connect(self.on_copy)
        self.paste_btn.clicked.connect(self.on_paste)
        self.delete_btn.clicked.connect(self.on_delete)
        self.trim_btn.clicked.connect(self.on_trim)
        self.play_btn.clicked.connect(self.on_play)
        self.pause_btn.clicked.connect(self.on_pause)
        self.stop_btn.clicked.connect(self.on_stop)
        self.action_combo.currentIndexChanged.connect(self.on_action_type_changed)
        self.action_browse_btn.clicked.connect(self.on_action_browse)
        self.action_test_btn.clicked.connect(self.on_action_test)
    def on_record(self):
        # Implementation of on_record method
        pass
    def on_stop_recording(self):
        # Implementation of on_stop_recording method
        pass
    def on_load(self):
        # Implementation of on_load method
        pass
    def on_select(self):
        # Implementation of on_select method
        pass
    def on_cut(self):
        # Implementation of on_cut method
        pass
    def on_copy(self):
        # Implementation of on_copy method
        pass
    def on_paste(self):
        # Implementation of on_paste method
        pass
    def on_delete(self):
        # Implementation of on_delete method
        pass
    def on_trim(self):
        # Implementation of on_trim method
        pass
    def on_play(self):
        # Implementation of on_play method
        pass
    def on_pause(self):
        # Implementation of on_pause method
        pass
    def on_stop(self):
        # Implementation of on_stop method
        pass
    def on_action_type_changed(self):
        # Implementation of on_action_type_changed method
        pass
    def on_action_browse(self):
        # Implementation of on_action_browse method
        pass
    def on_action_test(self):
        # Implementation of on_action_test method
        pass
    def on_waveform_selection(self, selection):
        # Implementation of on_waveform_selection method
        pass
    def on_threshold_changed(self, value):
        # Implementation of on_threshold_changed method
        pass
    def start_listening(self, on_detected_callback):
        if not self.audio_path or not self.trigger_action_type or not self.trigger_action_value:
            return False
        if self.listener_thread and self.listener_thread.isRunning():
            return False
        self.listener_thread = ListenerThread(self.audio_path, self.fs, threshold=self.detection_threshold)
        self.listener_thread.cue_detected.connect(lambda: on_detected_callback(self))
        self.listener_thread.listening_status.connect(self.label.setText)
        self.listener_thread.corr_value.connect(lambda v: self.corr_label.setText(f"Last Correlation: {v:.2f}"))
        self.listener_thread.start()
        return True
    def stop_listening(self):
        if self.listener_thread and self.listener_thread.isRunning():
            self.listener_thread.stop()
            self.listener_thread.wait()
            self.listener_thread = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Queue Trigger")
        self.setGeometry(100, 100, 900, 900)
        self.vlc_instance = vlc.Instance()
        self.vlc_player = None
        self.auto_listen_enabled = False
        self.queue_sections = [QueueSection(i, vlc_instance=self.vlc_instance) for i in range(3)]
        main_layout = QVBoxLayout()
        self.label = QLabel("Welcome to Audio Queue Trigger!", self)
        self.label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)
        for section in self.queue_sections:
            main_layout.addWidget(section)
        listen_layout = QHBoxLayout()
        self.listen_btn = QPushButton("Start Listening")
        self.stop_listen_btn = QPushButton("Stop Listening")
        self.stop_listen_btn.setEnabled(False)
        listen_layout.addWidget(self.listen_btn)
        listen_layout.addWidget(self.stop_listen_btn)
        main_layout.addLayout(listen_layout)
        scroll = QScrollArea()
        container = QWidget()
        container.setLayout(main_layout)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        for i, section in enumerate(self.queue_sections):
            section.toggled.connect(lambda checked, idx=i: self.on_section_toggled(idx, checked))
        self.listen_btn.clicked.connect(self.on_start_listening)
        self.stop_listen_btn.clicked.connect(self.on_stop_listening)
        self.listening = False
    def on_section_toggled(self, idx, checked):
        if checked:
            for i, section in enumerate(self.queue_sections):
                if i != idx:
                    section.setChecked(False)
    def on_start_listening(self):
        self.listening = True
        self.listen_btn.setEnabled(False)
        self.stop_listen_btn.setEnabled(True)
        self.label.setText("Listening for all enabled cues...")
        for section in self.queue_sections:
            if section.isChecked():
                section.start_listening(self.on_any_cue_detected)
    def on_stop_listening(self):
        self.listening = False
        self.listen_btn.setEnabled(True)
        self.stop_listen_btn.setEnabled(False)
        self.label.setText("Stopped listening.")
        for section in self.queue_sections:
            section.stop_listening()
    def on_any_cue_detected(self, section):
        self.label.setText(f"Cue detected in Queue {section.idx+1}! Triggering action...")
        section.on_action_test()
        # Resume listening for all enabled queues after a short delay
        if self.listening:
            for s in self.queue_sections:
                s.stop_listening()
            QTimer.singleShot(500, self.on_start_listening)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 