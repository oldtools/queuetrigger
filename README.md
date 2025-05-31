# Audio Queue Trigger App

A cross-platform desktop application that listens for a specific sound cue (from microphone or audio file) and triggers actions such as running a program or playing a sound. Built with Python and Qt6 (PySide6).

## Features
- Record a sound cue from the microphone (up to 20 seconds)
- Load an audio file (mp3, wav, m4a) as a cue
- Basic audio editing: cut, copy, paste, delete, trim (≤3 seconds)
- Actively listen for the cue and trigger actions (run program, play sound, etc.)
- Cross-platform: Windows, macOS, Linux

## Setup
1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python main.py
   ```

## Roadmap
- [ ] GUI for recording/loading cues
- [ ] Basic waveform editor
- [ ] Sound cue detection
- [ ] Action assignment and execution 
