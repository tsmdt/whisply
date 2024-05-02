# whisply
Transcribe, diarize, annotate and subtitle audio and video files with [Whisper](https://github.com/openai/whisper) ... fast!

Whisply combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [insanely-fast-whisper](https://github.com/chenxwh/insanely-fast-whisper) and batch processing of files. It also enables speaker detection and annotation via [pyannote](https://github.com/pyannote/pyannote-audio).

## Requirements
- [FFmpeg](https://ffmpeg.org/)
- python3.11

For GPU acceleration:
- nvidia GPU (CUDA)
- Metal Performance Shaders (MPS) Mac M1-M3

For speaker detection / diarization:
- HuggingFace access token

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
                          Default: auto-detection.
  --detect_speakers       Enable speaker diarization to identify and separate
                          different speakers.
  --hf_token TEXT         HuggingFace Access token required for speaker
                          diarization.
  --srt                   Generate SRT subtitles from the transcription.
  --config FILE           Path to configuration file.
  --parallel              Transcribe files in parallel.
  --help                  Show this message and exit.
```
**Speaker Detection / Diarization**<br>
To use `--detect_speakers` you need to provide a valid [HuggingFace](https://huggingface.co) access token by using the `--hf_token` flag. In addition to this you have to accept *both* `pyannote` user conditions for version 3.0 and 3.1 of the segmentation model. Follow the instructions in the section *Requirements* of the [pyannote model page on HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1).

**Using config files**<br>
You can provide a .json config file by using the `--config` which makes processing more user-friendly. An example config looks like this:
```
{
    "files": "path/to/files",
    "device": "cpu",
    "lang": null,
    "detect_speakers": true,
    "hf_token": "Hugging Face Access Token",
    "srt": true,
    "parallel": false
}
```
**Using .list files for batch processing**<br>
Instead of providing a file, folder or URL by using the `--files` option, you can pass a `.list` with a mix of files, folders and URLs for processing. Example:
```
cat my_files.list
video_01.mp4
video_02.mp4
./my_files/
https://youtu.be/KtOayYXEsN4?si=-0MS6KXbEWXA7dqo
```
