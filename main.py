import os
import io
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QSlider, QGroupBox, QComboBox, QLineEdit, QFormLayout,
    QScrollArea
)
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage
import vlc
from pydub import AudioSegment

class Config:
    """Configuration manager for the application"""
    def __init__(self, app_dir):
        import json  # Ensure json is imported within the class
        self.json = json  # Store reference to json module
        self.config_file = os.path.join(app_dir, "config.json")
        self.data = self._load_config()
        print(f"Initialized config from: {self.config_file}")
        print(f"Config data: {self.data}")
    
    def _load_config(self):
        """Load configuration from file"""
        import json  # Import json here as well for safety
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    print(f"Successfully loaded config with {len(config_data.get('queues', {}))} queues")
                    return config_data
            except Exception as e:
                print(f"Error loading config: {str(e)}")
        return {"queues": {}}
    
    def _save_config(self):
        """Save configuration to file"""
        import json  # Import json here as well for safety
        try:
            # Create a backup of the existing config file
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.bak"
                try:
                    import shutil
                    shutil.copy2(self.config_file, backup_file)
                    print(f"Created config backup: {backup_file}")
                except Exception as e:
                    print(f"Failed to create config backup: {str(e)}")
            
            # Save the current configuration
            with open(self.config_file, 'w') as f:
                json.dump(self.data, f, indent=2)
                print(f"Saved config to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {str(e)}")
    
    def get_queue_data(self, idx):
        """Get configuration data for a specific queue"""
        idx_str = str(idx)
        if "queues" not in self.data:
            self.data["queues"] = {}
        if idx_str not in self.data["queues"]:
            self.data["queues"][idx_str] = {}
        return self.data["queues"][idx_str]
    
    def update_queue(self, idx, updates):
        """Update configuration for a specific queue"""
        queue_data = self.get_queue_data(idx)
        queue_data.update(updates)
        print(f"Updated queue {idx} config: {updates}")
        self._save_config()

class WaveformLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection = None
        self.selection_changed_callback = None
        self.audio_duration = 0
        self.is_selecting = False
        self.cache_filename = None
        self.saved_filename = None
        self.is_short_clip = False
        self.waveform_pixmap = None
    
    def set_waveform(self, pixmap):
        """Set the waveform pixmap to display"""
        self.waveform_pixmap = pixmap
        self.setPixmap(pixmap)
        self.update()
    
    def set_cache_filename(self, filename):
        self.cache_filename = filename
        self.update()
    
    def set_saved_filename(self, filename):
        self.saved_filename = filename
        # Check if it's a short clip (≤3s)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(filename)
            self.is_short_clip = len(audio) <= 3000  # 3 seconds in milliseconds
        except:
            self.is_short_clip = False
        self.update()
    
    def set_audio_duration(self, duration):
        self.audio_duration = duration
        self.update()
    
    def mousePressEvent(self, event):
        if self.audio_duration > 0:
            x = event.position().x()
            time_pos = (x / self.width()) * self.audio_duration
            self.selection = [time_pos, time_pos]
            self.is_selecting = True
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.is_selecting and self.audio_duration > 0:
            x = event.position().x()
            time_pos = max(0, min((x / self.width()) * self.audio_duration, self.audio_duration))
            self.selection[1] = time_pos
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self.is_selecting:
            self.is_selecting = False
            if self.selection and self.selection_changed_callback:
                # Sort selection points
                sorted_selection = sorted(self.selection)
                self.selection_changed_callback(sorted_selection)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Draw cache filename in yellow
        if self.cache_filename:
            painter.setPen(QColor("yellow"))
            painter.drawText(10, 20, f"Cache: {os.path.basename(self.cache_filename)}")
        
        # Draw saved filename in green or red based on duration
        if self.saved_filename:
            if self.is_short_clip:
                painter.setPen(QColor("green"))
            else:
                painter.setPen(QColor("red"))
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
            painter.drawText(10, 40, f"File: {os.path.basename(self.saved_filename)}")
        
        # Draw selection if exists
        if self.selection:
            start_pos = int((min(self.selection) / self.audio_duration) * self.width())
            end_pos = int((max(self.selection) / self.audio_duration) * self.width())
            
            highlight_rect = QRect(start_pos, 0, end_pos - start_pos, self.height())
            painter.fillRect(highlight_rect, QColor(100, 100, 255, 50))
            
            # Draw selection boundaries
            painter.setPen(QColor(100, 100, 255))
            painter.drawLine(start_pos, 0, start_pos, self.height())
            painter.drawLine(end_pos, 0, end_pos, self.height())
            
            # Draw selection time
            selection_time = max(self.selection) - min(self.selection)
            painter.drawText(
                (start_pos + end_pos) // 2 - 40, 
                self.height() - 10, 
                f"{selection_time:.2f}s"
            )

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
        self.play_obj = self.vlc_instance.media_player_new()
        media = self.vlc_instance.media_new_memory_from_array(audio_data)
        self.play_obj.set_media(media)
        self.play_obj.play()
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
        print(f"ListenerThread initialized with cue: {cue_path}, threshold: {threshold}")

    def run(self):
        import sounddevice as sd
        from pydub import AudioSegment
        import numpy as np
        import scipy.signal
        
        try:
            # Load the cue audio
            self.listening_status.emit(f"Loading cue audio: {os.path.basename(self.cue_path)}")
            cue_audio = AudioSegment.from_file(self.cue_path).set_channels(1).set_frame_rate(self.fs)
            cue_samples = np.array(cue_audio.get_array_of_samples()).astype(np.float32)
            
            # Normalize cue samples
            cue_samples = cue_samples / (np.max(np.abs(cue_samples)) + 1e-8)
            cue_len = len(cue_samples)
            
            # Create buffer for incoming audio
            buffer = np.zeros(cue_len * 3, dtype=np.float32)
            
            self.listening_status.emit(f"Listening... (Threshold: {self.threshold:.2f})")
            print(f"Starting to listen with threshold: {self.threshold}")
            
            # Start audio input stream
            with sd.InputStream(samplerate=self.fs, channels=1, dtype='float32') as stream:
                while not self._stop:
                    # Read audio chunk
                    audio_chunk, _ = stream.read(cue_len // 2)
                    audio_chunk = audio_chunk.flatten()
                    
                    # Update buffer with new audio
                    buffer = np.roll(buffer, -len(audio_chunk))
                    buffer[-len(audio_chunk):] = audio_chunk
                    
                    # Normalize buffer
                    norm_buffer = buffer / (np.max(np.abs(buffer)) + 1e-8)
                    
                    # Cross-correlation
                    corr = scipy.signal.correlate(norm_buffer, cue_samples, mode='valid')
                    corr /= (np.linalg.norm(norm_buffer) * np.linalg.norm(cue_samples) + 1e-8)
                    max_corr = float(np.max(corr))
                    
                    # Emit correlation value for UI update
                    self.corr_value.emit(max_corr)
                    
                    # Check if correlation exceeds threshold
                    if max_corr > self.threshold:
                        print(f"Cue detected! Correlation: {max_corr}")
                        self.listening_status.emit(f"Cue detected! Correlation: {max_corr:.2f}")
                        self.cue_detected.emit()
                        break
                    
                    # Small sleep to reduce CPU usage
                    sd.sleep(10)
                    
        except Exception as e:
            print(f"Listening error: {str(e)}")
            self.listening_status.emit(f"Listening error: {str(e)}")
        
        self.listening_status.emit("Stopped listening.")
        print("Listener thread stopped")

    def stop(self):
        self._stop = True

class QueueSection(QGroupBox):
    def __init__(self, idx, parent=None, vlc_instance=None, config=None):
        super().__init__(f"Audio Queue {idx+1}", parent)
        self.setCheckable(True)
        self.setChecked(idx == 0)
        self.idx = idx
        self.audio_path = None
        self.saved_path = None
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
        self.edit_history = []  # For undo functionality
        self.config = config
        
        # Create cache directory if it doesn't exist
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory: {self.cache_dir}")
        else:
            print(f"Using existing cache directory: {self.cache_dir}")
        
        # Load configuration if available
        if config:
            queue_data = config.get_queue_data(idx)
            if "saved_path" in queue_data:
                self.saved_path = queue_data["saved_path"]
                print(f"Loaded saved path from config: {self.saved_path}")
            if "trigger_action_type" in queue_data:
                self.trigger_action_type = queue_data["trigger_action_type"]
            if "trigger_action_value" in queue_data:
                self.trigger_action_value = queue_data["trigger_action_value"]
            if "threshold" in queue_data:
                self.detection_threshold = queue_data["threshold"]
        
        # Create the UI layout
        self._create_ui()
        
        # Load saved file if available
        if self.saved_path and os.path.exists(self.saved_path):
            self._load_file(self.saved_path)
    
    def _create_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 10)  # Reduce top margin for better alignment
        
        # Title label
        self.label = QLabel(f"Audio Queue {self.idx+1} controls")
        self.label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.label)
        
        # Record buttons (centered above waveform with microphone icons)
        record_layout = QHBoxLayout()
        record_layout.addStretch()
        
        # Add microphone icon on left
        mic_icon_left = QLabel("🎤    ")
        mic_icon_left.setStyleSheet("font-size: 18px;")
        record_layout.addWidget(mic_icon_left)
        
        # Record and stop buttons
        self.record_btn = QPushButton("🔴 Record")
        self.record_btn.setStyleSheet("background-color: #ffdddd; font-weight: bold;color: #f00e1a;")
        self.stop_record_btn = QPushButton("⏹️ Stop")
        self.stop_record_btn.setStyleSheet("background-color: #ffdddd; font-weight: bold;color: black; ")
        self.stop_record_btn.setEnabled(False)
        record_layout.addWidget(self.record_btn)
        record_layout.addWidget(self.stop_record_btn)
        
        # Add microphone icon on right
        mic_icon_right = QLabel("    🎤")
        mic_icon_right.setStyleSheet("font-size: 18px;")
        record_layout.addWidget(mic_icon_right)
        
        record_layout.addStretch()
        layout.addLayout(record_layout)
        
        # Load/Save buttons (above waveform, left aligned)
        file_buttons_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Load")
        self.save_btn = QPushButton("💾 Save")
        file_buttons_layout.addWidget(self.load_btn)
        file_buttons_layout.addWidget(self.save_btn)
        file_buttons_layout.addStretch()
        
        # Edit buttons (above waveform, right aligned)
        self.cut_btn = QPushButton("✂️")
        self.copy_btn = QPushButton("📋")
        self.paste_btn = QPushButton("📌")
        self.delete_btn = QPushButton("🗑️")
        for btn in [self.cut_btn, self.copy_btn, self.paste_btn, self.delete_btn]:
            btn.setMaximumWidth(40)
            file_buttons_layout.addWidget(btn)
        
        layout.addLayout(file_buttons_layout)
        
        # Waveform display
        self.waveform_label = WaveformLabel()
        self.waveform_label.setText("[Waveform Editor Placeholder]")
        self.waveform_label.setAlignment(Qt.AlignCenter)
        self.waveform_label.setStyleSheet("border: 1px solid #888; padding: 40px;")
        self.waveform_label.selection_changed_callback = self.on_waveform_selection
        layout.addWidget(self.waveform_label)
        
        # Playback controls (beneath waveform, left aligned with high contrast)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # More visible playback controls
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.setStyleSheet("font-weight: bold;color: #a0d9a5;")
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.setStyleSheet("font-weight: bold;color: #a0d9a5;")
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setStyleSheet("font-weight: bold;color: #a0d9a5;")
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        # Crop, trim and undo buttons (centered beneath waveform)
        self.crop_btn = QPushButton("✂️🔍 Crop")
        self.trim_btn = QPushButton("✂️3s Trim")
        self.undo_btn = QPushButton("↩️ Undo")
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(self.crop_btn)
        center_layout.addWidget(self.trim_btn)
        center_layout.addWidget(self.undo_btn)
        center_layout.addStretch()
        controls_layout.addLayout(center_layout)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Threshold controls
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
        
        # Trigger action controls
        trigger_layout = QHBoxLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Select Action", "Run Program", "Play Sound File"])
        if self.trigger_action_type:
            index = self.action_combo.findText(self.trigger_action_type)
            if index >= 0:
                self.action_combo.setCurrentIndex(index)
        
        self.action_value_edit = QLineEdit()
        self.action_value_edit.setPlaceholderText("No action assigned")
        if self.trigger_action_value:
            self.action_value_edit.setText(self.trigger_action_value)
        else:
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
        self.save_btn.clicked.connect(self.on_save)
        self.cut_btn.clicked.connect(self.on_cut)
        self.copy_btn.clicked.connect(self.on_copy)
        self.paste_btn.clicked.connect(self.on_paste)
        self.delete_btn.clicked.connect(self.on_delete)
        self.crop_btn.clicked.connect(self.on_crop)
        self.trim_btn.clicked.connect(self.on_trim)
        self.undo_btn.clicked.connect(self.on_undo)
        self.play_btn.clicked.connect(self.on_play)
        self.pause_btn.clicked.connect(self.on_pause)
        self.stop_btn.clicked.connect(self.on_stop)
        self.action_combo.currentIndexChanged.connect(self.on_action_type_changed)
        self.action_browse_btn.clicked.connect(self.on_action_browse)
        self.action_test_btn.clicked.connect(self.on_action_test)
        
        # Initially disable edit buttons until we have audio loaded
        for btn in [self.cut_btn, self.copy_btn, self.delete_btn, self.crop_btn, self.trim_btn]:
            btn.setEnabled(False)
    
    def on_save(self):
        if not hasattr(self, 'audio_segment'):
            self.label.setText("No audio to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", 
                                                 "Audio Files (*.wav)")
        if file_path:
            try:
                # Save audio to the selected path
                self.audio_segment.export(file_path, format="wav")
                print(f"Saved audio to: {file_path}")
                
                # Update saved path
                self.saved_path = file_path
                self.waveform_label.set_saved_filename(file_path)
                
                # Update configuration
                if self.config:
                    self.config.update_queue(self.idx, {
                        "saved_path": file_path
                    })
                
                self.label.setText(f"Saved to: {file_path}")
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                self.label.setText(f"Error saving: {str(e)}")

    def on_play(self):
        if not hasattr(self, 'audio_segment') or not self.audio_path:
            self.label.setText("No audio to play")
            return
            
        try:
            if self.vlc_instance:
                if self.vlc_player:
                    self.vlc_player.stop()
                
                self.vlc_player = self.vlc_instance.media_player_new()
                media = self.vlc_instance.media_new(self.audio_path)
                self.vlc_player.set_media(media)
                self.vlc_player.play()
                self.label.setText(f"Playing: {os.path.basename(self.audio_path)}")
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
            self.label.setText(f"Error playing audio: {str(e)}")
    def on_pause(self):
        if self.vlc_player:
            self.vlc_player.pause()
            
    def on_stop(self):
        if self.vlc_player:
            self.vlc_player.stop()
    def on_select(self):
        # Implementation for select button
        if self.audio_path:
            self.label.setText("Select a portion of the waveform")
            
    def on_cut(self):
        if self.audio_path and self.selection:
            print(f"Cutting selection: {self.selection}")
            self._save_state_for_undo()
            start_sec, end_sec = self.selection
            # Store cut segment
            self.copied_segment = self.audio_segment[start_sec*1000:end_sec*1000]
            
            # Save copied segment to file
            copy_file = os.path.join(self.cache_dir, f"clipboard_{self.idx}.wav")
            self.copied_segment.export(copy_file, format="wav")
            print(f"Saved cut segment to: {copy_file}")
            
            # Remove selected portion
            self.audio_segment = self.audio_segment[:start_sec*1000] + self.audio_segment[end_sec*1000:]
            
            # Save modified audio
            self.audio_segment.export(self.audio_path, format="wav")
            print(f"Saved modified audio to: {self.audio_path}")
            
            self._update_waveform_display()
            self.label.setText(f"Cut {end_sec-start_sec:.2f}s segment")
            self.selection = None
            
    def on_copy(self):
        if self.audio_path and self.selection:
            print(f"Copying selection: {self.selection}")
            start_sec, end_sec = self.selection
            self.copied_segment = self.audio_segment[start_sec*1000:end_sec*1000]
            
            # Save copied segment to file
            copy_file = os.path.join(self.cache_dir, f"clipboard_{self.idx}.wav")
            self.copied_segment.export(copy_file, format="wav")
            print(f"Saved copied segment to: {copy_file}")
            
            self.label.setText(f"Copied {end_sec-start_sec:.2f}s segment")
            
    def on_paste(self):
        if self.audio_path and self.copied_segment:
            print("Pasting copied segment")
            self._save_state_for_undo()
            
            # If there's a selection, paste at selection start
            if self.selection:
                start_sec = min(self.selection)
                print(f"Pasting at selection: {start_sec}s")
                self.audio_segment = self.audio_segment[:start_sec*1000] + self.copied_segment + self.audio_segment[start_sec*1000:]
            else:
                # Otherwise paste at the end
                print("Pasting at end")
                self.audio_segment = self.audio_segment + self.copied_segment
            
            # Save modified audio
            self.audio_segment.export(self.audio_path, format="wav")
            print(f"Saved modified audio to: {self.audio_path}")
            
            self._update_waveform_display()
            self.label.setText(f"Pasted {self.copied_segment.duration_seconds:.2f}s segment")
            self.selection = None
            
    def on_delete(self):
        if self.audio_path and self.selection:
            print(f"Deleting selection: {self.selection}")
            self._save_state_for_undo()
            start_sec, end_sec = self.selection
            self.audio_segment = self.audio_segment[:start_sec*1000] + self.audio_segment[end_sec*1000:]
            
            # Save modified audio
            self.audio_segment.export(self.audio_path, format="wav")
            print(f"Saved modified audio to: {self.audio_path}")
            
            self._update_waveform_display()
            self.label.setText(f"Deleted {end_sec-start_sec:.2f}s segment")
            self.selection = None
            
    def on_trim(self):
        if self.audio_path and self.selection:
            start_sec, end_sec = self.selection
            if end_sec - start_sec <= 3:  # Ensure segment is ≤3s
                self.audio_segment = self.audio_segment[start_sec*1000:end_sec*1000]
                self._update_waveform_display()
                self.label.setText(f"Trimmed to {end_sec-start_sec:.2f}s segment")
                self.selection = None
            else:
                self.label.setText("Selection too long. Trim limited to ≤3s segments")
            
    def on_action_type_changed(self):
        action_type = self.action_combo.currentText()
        if action_type != "Select Action":
            self.trigger_action_type = action_type
            self.action_value_edit.setReadOnly(False)
            
            # Update configuration
            if self.config:
                self.config.update_queue(self.idx, {
                    "trigger_action_type": action_type
                })
        else:
            self.trigger_action_type = None
            self.action_value_edit.setReadOnly(True)

    def on_action_browse(self):
        action_type = self.action_combo.currentText()
        if action_type == "Run Program":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Program", "", "Executable Files (*.exe *.bat *.cmd)")
            if file_path:
                self.action_value_edit.setText(file_path)
                self.trigger_action_value = file_path
                
                # Update configuration
                if self.config:
                    self.config.update_queue(self.idx, {
                        "trigger_action_value": file_path
                    })
        elif action_type == "Play Sound File":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Sound File", "", "Audio Files (*.mp3 *.wav *.m4a *.ogg)")
            if file_path:
                self.action_value_edit.setText(file_path)
                self.trigger_action_value = file_path
                
                # Update configuration
                if self.config:
                    self.config.update_queue(self.idx, {
                        "trigger_action_value": file_path
                    })
                
    def on_action_test(self):
        if not self.trigger_action_type or not self.trigger_action_value:
            self.label.setText("No action configured to test")
            return
        
        self.label.setText(f"Testing action: {self.trigger_action_type}")
        print(f"Testing action: {self.trigger_action_type}, value: {self.trigger_action_value}")
        
        try:
            if self.trigger_action_type == "Run Program":
                # Run the program
                import subprocess
                subprocess.Popen(self.trigger_action_value, shell=True)
                self.label.setText(f"Launched program: {os.path.basename(self.trigger_action_value)}")
            
            elif self.trigger_action_type == "Play Sound File":
                # Play the sound file
                if self.vlc_instance and os.path.exists(self.trigger_action_value):
                    # Create a separate player for the action sound
                    action_player = self.vlc_instance.media_player_new()
                    media = self.vlc_instance.media_new(self.trigger_action_value)
                    action_player.set_media(media)
                    action_player.play()
                    self.label.setText(f"Playing sound: {os.path.basename(self.trigger_action_value)}")
                else:
                    self.label.setText(f"Error: Sound file not found or VLC not initialized")
        except Exception as e:
            self.label.setText(f"Error executing action: {str(e)}")
            print(f"Error executing action: {str(e)}")
    def start_listening(self, on_detected_callback):
        # Use the original saved file path for listening if available, otherwise use the cached audio path
        cue_path = self.saved_path if self.saved_path and os.path.exists(self.saved_path) else self.audio_path
        
        if not cue_path or not self.trigger_action_type or not self.trigger_action_value:
            self.label.setText("Cannot listen: Missing audio or action configuration")
            return False
        
        # Stop any existing listener thread
        self.stop_listening()
        
        # Create and start a new listener thread
        self.label.setText("Starting to listen for audio cue...")
        print(f"Using cue path for listening: {cue_path}")
        self.listener_thread = ListenerThread(cue_path, self.fs, threshold=self.detection_threshold)
        self.listener_thread.cue_detected.connect(lambda: on_detected_callback(self))
        self.listener_thread.listening_status.connect(self.label.setText)
        self.listener_thread.corr_value.connect(self.update_correlation_value)
        self.listener_thread.start()
        print(f"Started listening thread for queue {self.idx} with threshold {self.detection_threshold}")
        return True
    
    def stop_listening(self):
        if self.listener_thread and self.listener_thread.isRunning():
            self.listener_thread.stop()
            self.listener_thread.wait()
            self.listener_thread = None
            return True
        return False
    
    def update_correlation_value(self, value):
        self.last_corr = value
        self.corr_label.setText(f"Last Correlation: {value:.2f}")
        # If value is close to threshold, change color to indicate
        if value > (self.detection_threshold * 0.8):
            self.corr_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.corr_label.setStyleSheet("")
    def _load_audio_waveform(self):
        if not self.audio_path:
            print("No audio path to load waveform from")
            return
        
        try:
            print(f"Loading waveform from: {self.audio_path}")
            # Load audio file using pydub
            self.audio_segment = AudioSegment.from_file(self.audio_path)
            print(f"Loaded audio: {self.audio_segment.duration_seconds}s, {self.audio_segment.channels} channels")
            self.waveform_label.set_audio_duration(self.audio_segment.duration_seconds)
            
            # Generate and display waveform
            self._update_waveform_display()
        except Exception as e:
            print(f"Error loading waveform: {str(e)}")
            self.label.setText(f"Error loading waveform: {str(e)}")

    def _update_waveform_display(self):
        if not hasattr(self, 'audio_segment'):
            return
        
        try:
            # Convert to numpy array for plotting
            samples = np.array(self.audio_segment.get_array_of_samples())
            if self.audio_segment.channels > 1:
                samples = samples.reshape((-1, self.audio_segment.channels)).mean(axis=1)
            
            # Create a figure for the waveform
            plt.figure(figsize=(8, 2))
            plt.plot(samples, color='blue')
            plt.axis('off')
            
            # Save to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Convert buffer to QPixmap
            buf.seek(0)
            img = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(img)
            
            # Set the pixmap to the label
            self.waveform_label.set_waveform(pixmap)
            
            # Update cache filename
            self.waveform_label.set_cache_filename(self.audio_path)
            
            # Enable buttons now that we have audio
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.paste_btn.setEnabled(hasattr(self, 'copied_segment'))
            
        except Exception as e:
            print(f"Error updating waveform display: {str(e)}")
            self.label.setText(f"Error displaying waveform: {str(e)}")

    def _save_state_for_undo(self):
        if hasattr(self, 'audio_segment'):
            try:
                # Create a unique filename for this undo state
                undo_file = os.path.join(self.cache_dir, f"undo_{len(self.edit_history)}_{self.idx}.wav")
                print(f"Saving undo state to: {undo_file}")
                
                # Export audio to the undo file
                self.audio_segment.export(undo_file, format="wav")
                print(f"Exported undo state, size: {os.path.getsize(undo_file)} bytes")
                
                # Add the filename to the edit history
                self.edit_history.append(undo_file)
                print(f"Added to edit history, now {len(self.edit_history)} items")
                
                # Limit history size to prevent disk space issues
                if len(self.edit_history) > 10:
                    oldest_file = self.edit_history.pop(0)
                    if os.path.exists(oldest_file):
                        os.remove(oldest_file)
                        print(f"Removed oldest undo file: {oldest_file}")
            except Exception as e:
                print(f"Error saving undo state: {str(e)}")

    def on_undo(self):
        if self.edit_history:
            # Get the last state filename
            last_state_file = self.edit_history.pop()
            print(f"Undoing to previous state: {last_state_file}")
            
            # Load the previous state
            if os.path.exists(last_state_file):
                self.audio_segment = AudioSegment.from_file(last_state_file)
                
                # Save to current audio path
                self.audio_segment.export(self.audio_path, format="wav")
                print(f"Restored audio to: {self.audio_path}")
                
                self._update_waveform_display()
                self.label.setText("Undo successful")
            else:
                print(f"Undo file not found: {last_state_file}")
                self.label.setText("Undo file not found")

    def on_crop(self):
        if self.audio_path and self.selection:
            print(f"Cropping to selection: {self.selection}")
            self._save_state_for_undo()
            start_sec, end_sec = self.selection
            # Keep only the selected portion
            self.audio_segment = self.audio_segment[start_sec*1000:end_sec*1000]
            
            # Save modified audio
            self.audio_segment.export(self.audio_path, format="wav")
            print(f"Saved cropped audio to: {self.audio_path}")
            
            self._update_waveform_display()
            self.label.setText(f"Cropped to {end_sec-start_sec:.2f}s segment")
            self.selection = None

    def on_record(self):
        if self.recording:
            return
        
        self.recording = True
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.label.setText("Recording... (max 20s)")
        
        # Reset stop event
        self.record_stop_event.clear()
        
        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def on_stop_recording(self):
        if not self.recording:
            return
        
        self.label.setText("Stopping recording...")
        self.record_stop_event.set()
        
        # Wait for recording thread to finish
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
            self.record_thread = None
    
    def _record_audio(self):
        try:
            # Create a filename for the recording in the cache directory
            record_file = os.path.join(self.cache_dir, f"recording_{self.idx}.wav")
            
            # Record audio
            fs = self.fs  # Sample rate
            duration = self.max_record_sec  # Max recording duration
            
            # Create recording stream
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            
            # Wait for recording to complete or stop event
            for i in range(int(duration * 10)):  # Check every 100ms
                if self.record_stop_event.is_set():
                    break
                sd.sleep(100)
            
            # Stop recording
            sd.stop()
            actual_duration = min(i/10, duration)
            
            # Save only the recorded portion
            actual_samples = recording[:int(actual_duration * fs)]
            wav_write(record_file, fs, actual_samples)
            
            # Load the recorded audio
            self.audio_path = record_file
            self.waveform_label.set_cache_filename(record_file)
            self.label.setText(f"Recorded {actual_duration:.1f}s audio")
            
            # Clear cache/buffer
            self.copied_segment = None
            self.selection = None
            self.edit_history = []
            
            # Load the waveform
            self._load_audio_waveform()
            
        except Exception as e:
            self.label.setText(f"Recording error: {str(e)}")
        finally:
            self.recording = False
            self.record_btn.setEnabled(True)
            self.stop_record_btn.setEnabled(False)

    def on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Audio File", "", 
                                                  "Audio Files (*.mp3 *.wav *.m4a *.ogg)")
        if file_path:
            self._load_file(file_path)
    
    def _load_file(self, file_path):
        try:
            # Create a cache copy of the loaded file
            cache_file = os.path.join(self.cache_dir, f"loaded_{self.idx}.wav")
            print(f"Loading file: {file_path}")
            print(f"Creating cache file: {cache_file}")
            
            # Load and convert to WAV in cache
            audio = AudioSegment.from_file(file_path)
            audio.export(cache_file, format="wav")
            print(f"Exported to cache: {cache_file}, size: {os.path.getsize(cache_file)} bytes")
            
            # Use the cached file as our audio path
            self.audio_path = cache_file
            self.waveform_label.set_cache_filename(cache_file)
            
            # If this is a file we're loading from outside, set it as saved path
            if file_path != self.audio_path:
                self.saved_path = file_path
                self.waveform_label.set_saved_filename(file_path)
                
                # Update configuration
                if self.config:
                    # Only update the saved_path field, preserving other config values
                    self.config.update_queue(self.idx, {
                        "saved_path": file_path
                    })
                    print(f"Updated config with saved_path: {file_path}")
            
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
            
            # Clear cache/buffer
            self.copied_segment = None
            self.selection = None
            self.edit_history = []
            
            # Load the waveform
            self._load_audio_waveform()
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            self.label.setText(f"Error: {str(e)}")

    def on_waveform_selection(self, selection):
        """Called when the user makes a selection on the waveform"""
        self.selection = selection
        start_sec, end_sec = selection
        duration = end_sec - start_sec
        
        # Update UI to show selection information
        self.label.setText(f"Selected {duration:.2f}s segment ({start_sec:.2f}s - {end_sec:.2f}s)")
        
        # Enable/disable buttons based on selection
        has_selection = duration > 0
        self.cut_btn.setEnabled(has_selection)
        self.copy_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.crop_btn.setEnabled(has_selection)
        
        # Special case for trim button - only enable if selection is ≤3s
        self.trim_btn.setEnabled(has_selection and duration <= 3)

    def on_threshold_changed(self, value):
        """Called when the threshold slider value changes"""
        self.detection_threshold = value / 100.0
        self.threshold_label.setText(f"Detection Threshold: {self.detection_threshold:.2f}")
        
        # Update configuration
        if self.config:
            self.config.update_queue(self.idx, {
                "threshold": self.detection_threshold
            })

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Queue Trigger")
        self.setGeometry(100, 100, 980, 600)
        
        # Initialize configuration
        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = Config(app_dir)
        print(f"MainWindow initialized with config from {app_dir}")
        
        # Initialize VLC with options to suppress warnings
        vlc_args = ['--quiet', '--no-xlib']
        self.vlc_instance = vlc.Instance(vlc_args)
        self.vlc_player = None
        self.auto_listen_enabled = False
        
        # Create queue sections
        self.queue_sections = [QueueSection(0, vlc_instance=self.vlc_instance, config=self.config)]
        print(f"Created {len(self.queue_sections)} queue sections")
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 10)  # Reduce top margin
        main_layout.setSpacing(5)  # Reduce spacing between elements
        
        # Add queue sections first for top alignment
        for section in self.queue_sections:
            main_layout.addWidget(section)
        
        # Add welcome label below the queue sections
        self.label = QLabel("Welcome to Audio Queue Trigger!", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        main_layout.addWidget(self.label)
        
        # Add listening controls at the bottom
        listen_layout = QHBoxLayout()
        listen_layout.addStretch()
        self.listen_btn = QPushButton("🎧 🛑 Start Listening")
        self.listen_btn.setStyleSheet("background-color: #ffdddd; font-weight: bold;color: #f00e1a; padding: 8px 16px;")
        self.stop_listen_btn = QPushButton("⏹️ Stop Listening")
        self.stop_listen_btn.setStyleSheet("background-color: #ffdddd; font-weight: bold;color: black; padding: 8px 16px;")
        self.stop_listen_btn.setEnabled(False)
        listen_layout.addWidget(self.listen_btn)
        listen_layout.addWidget(self.stop_listen_btn)
        listen_layout.addStretch()
        main_layout.addLayout(listen_layout)
        
        # Add some spacing at the bottom
        main_layout.addStretch()
        
        scroll = QScrollArea()
        container = QWidget()
        container.setLayout(main_layout)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        
        # Connect signals
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
        
        # Start listening for all checked queue sections
        listening_started = False
        for section in self.queue_sections:
            if section.isChecked():
                if section.start_listening(self.on_any_cue_detected):
                    listening_started = True
                    print(f"Started listening for Queue {section.idx+1}")
        
        if not listening_started:
            self.label.setText("No valid cues to listen for. Please configure at least one queue.")
            self.listening = False
            self.listen_btn.setEnabled(True)
            self.stop_listen_btn.setEnabled(False)
    
    def on_stop_listening(self):
        self.listening = False
        self.listen_btn.setEnabled(True)
        self.stop_listen_btn.setEnabled(False)
        self.label.setText("Stopped listening.")
        for section in self.queue_sections:
            section.stop_listening()
    
    def on_any_cue_detected(self, section):
        # This method is called when a cue is detected in any queue section
        print(f"Cue detected in Queue {section.idx+1}! Triggering action...")
        self.label.setText(f"Cue detected in Queue {section.idx+1}! Triggering action...")
        
        # Execute the action for the section that detected the cue
        section.on_action_test()
        
        # Stop listening in all sections
        for s in self.queue_sections:
            s.stop_listening()
        
        # Resume listening after a short delay if still in listening mode
        if self.listening:
            QTimer.singleShot(2000, self.on_start_listening)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()    
    app.setStyle("fusion")
    window.show()
    sys.exit(app.exec()) 
