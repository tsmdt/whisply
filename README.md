# whisply
Transcribe, diarize, annotate and subtitle audio and video files with [Whisper](https://github.com/openai/whisper) ... fast!

Whisply combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [insanely-fast-whisper](https://github.com/chenxwh/insanely-fast-whisper) and batch processing of files. It also enables speaker diarization via [pyannote](https://github.com/pyannote/pyannote-audio).

## Requirements
- ffmpeg
- python3.11

For GPU acceleration:
- nvidia GPU (CUDA)
- Metal Performance Shaders (MPS) Mac M1-M3

## Installation
**1. Install `ffmpeg`**
```
--- macOS ---
brew install ffmpeg

--- linux ---
sudo apt-get update
sudo apt-get install ffmpeg
```
**2. Clone this repository and change to project folder**
```
git clone https://github.com/th-schmidt/whisply.git
cd whisply
```
**3. Create a Python virtual environment and activate it**
```
python3.11 -m venv venv
source venv/bin/activate
```
**4. Install dependencies with `pip`**
```
pip install -r requirement.txt
```

## Usage
```
Options:
  --files PATH            Path to file, folder, URL or .list to process.
  --device [cpu|gpu|mps]  Select the computation device: CPU, GPU (nvidia
                          CUDA), or MPS (Metal Performance Shaders).
  --lang [en|fr|de]       Specify the language of the audio for transcription.
  --detect_speakers       Enable speaker diarization to identify and separate
                          different speakers.
  --hf_token TEXT         HuggingFace Access token required for speaker
                          diarization.
  --srt                   Generate SRT subtitles from the transcription.
  --config FILE           Path to configuration file.
  --parallel              Transcribe files in parallel.
  --help                  Show this message and exit.
```
