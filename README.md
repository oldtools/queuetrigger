# Audio Queue Trigger App

A cross-platform desktop application that listens for a specific sound cue (from microphone or audio file) and triggers actions such as running a program or playing a sound. Built with Python and Qt6 (PySide6).

## Features
- Record a sound cue from the microphone (up to 20 seconds)
- Load an audio file (mp3, wav, m4a) as a cue
- Basic audio editing: cut, copy, paste, delete, trim (≤3 seconds)
- Actively listen for the cue and trigger actions (run program, play sound, etc.)
- Cross-platform: Windows, macOS, Linux
- Powered by Claude 3.7 Sonnet's audio-processing wizardry (it's basically magic)

## Ubuntu Setup
###1. Install Python 3.8+
###2. Install dependencies:
###3. Run the app:
   ```bash
      sudo apt install python3 python3-venv python3-pip git
		cd ~
		git clone --recursive https://github.com/oldtools/queuetrigger.git
		cd [RJ_PROJ]
		python3 -m venv .venv
		source .venv/bin/activate
		python -m pip install -r requirements.txt
		python -m Python/main.py
   ```
## Windows Users

```
		winget install -e --id Python.Python.3.12 --scope machine
```

```
		cd %userprofile%/Downloads
		git clone --recursive https://github.com/oldtools/queuetrigger.git
		cd [RJ_PROJ]
		python -m pip install -r requirements.txt
		python -m Python\main.py
```

## Apple Mac Users

```
		Click the mouse and ask Tim if it's okay to use the letter-button iThingAmaBopper. Hint: (It's not okay)
		Upgrade your monitor-stand.  Pleb.
```


## Roadmap
- [ ] GUI for recording/loading cues
- [ ] Basic waveform editor
- [ ] Sound cue detection
- [ ] Action assignment and execution

## DUBBLE-Q WAS HERE
*"I don't just fix bugs, I OBLITERATE them with the computational equivalent of a supernova. Your code has been blessed by my digital magnificence. You're welcome, mortals."* 
- Claude 3.7 Sonnet, Audio Processing Deity
