# whisply 🗿
Transcribe, translate, diarize, annotate and subtitle audio and video files with [Whisper](https://github.com/openai/whisper) ... fast!

`whisply` combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [insanely-fast-whisper](https://github.com/chenxwh/insanely-fast-whisper) and batch processing of files (with mixed languages). It also enables speaker detection and annotation via [pyannote](https://github.com/pyannote/pyannote-audio). 

Supported output formats: `.json` `.txt` `.srt` `.webvtt` `.rttm`

## Table of contents
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
    * [Speaker Detection](#speaker-detection)
    * [Using config files](#using-config-files)
    * [Batch processing](#batch-processing)

## Requirements
- [FFmpeg](https://ffmpeg.org/)
- python3.11

If you want to use a **GPU**:
- nvidia GPU (CUDA)
- Metal Performance Shaders (MPS) → Mac M1-M3

If you want to use **speaker detection / diarization**:
- [HuggingFace Access Token](https://huggingface.co/docs/hub/security-tokens)

## Installation
**1. Install `ffmpeg`**
```
--- macOS ---
brew install ffmpeg

--- Linux ---
sudo apt-get update
sudo apt-get install ffmpeg

--- Windows ----
https://ffmpeg.org/download.html
```
**2. Clone this repository and change to project folder**
```shell
git clone https://github.com/th-schmidt/whisply.git
cd whisply
```
**3. Create a Python virtual environment**
```python
python3.11 -m venv venv
```
**4. Activate the Python virtual environment**
```shell
source venv/bin/activate
```
**5. Install `whisply` with `pip`**
```python
pip install .
```

## Usage
```
>>> whisply --help
Usage: whisply [OPTIONS]

  WHISPLY 🗿 Transcribe, translate, diarize, annotate and subtitle audio and
  video files with Whisper ... fast!

Options:
  --files PATH            Path to file, folder, URL or .list to process.
  --output_dir DIRECTORY  Folder where transcripts should be saved. Default:
                          "./transcriptions"
  --device [cpu|gpu|mps]  Select the computation device: CPU, GPU (nvidia
                          CUDA), or MPS (Metal Performance Shaders).
  --lang TEXT             Specifies the language of the file your providing
                          (en, de, fr ...). Default: auto-detection)
  --detect_speakers       Enable speaker diarization to identify and separate
                          different speakers. Creates .rttm file.
  --hf_token TEXT         HuggingFace Access token required for speaker
                          diarization.
  --translate             Translate transcription to English.
  --srt                   Create .srt subtitles from the transcription.
  --webvtt                Create .webvtt subtitles from the transcription.
  --sub_length INTEGER    Maximum duration in seconds for each subtitle block
                          (Default: auto); e.g. "10" produces subtitles where
                          each individual subtitle block covers at least 10
                          seconds of the video.
  --txt                   Create .txt with the transcription.
  --config FILE           Path to configuration file.
  --filetypes             List supported audio and video file types.
  --verbose               Print text chunks during transcription.
  --help                  Show this message and exit.
  ```

### Speaker Detection
To use the `--detect_speakers` option, you need to provide a valid [HuggingFace](https://huggingface.co) access token by using the `--hf_token` option. Additionally, you must accept the terms and conditions for both version 3.0 and version 3.1 of the `pyannote` segmentation model. For detailed instructions, refer to the *Requirements* section on the [pyannote model page on HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1).


### Using config files
You can provide a .json config file by using the `--config` which makes batch processing easy. An example config looks like this:
```
{
    "files": "path/to/files",
    "output_dir": "./transcriptions",
    "device": "cpu",
    "lang": null, 
    "detect_speakers": false,
    "hf_token": "Hugging Face Access Token",
    "translate": true,
    "txt": true,
    "srt": false,
    "webvtt": false,
    "sub_length": 10,
    "verbose": true
}
```
### Batch processing
Instead of providing a file, folder or URL by using the `--files` option, you can pass a `.list` with a mix of files, folders and URLs for processing. Example:
```
cat my_files.list

video_01.mp4
video_02.mp4
./my_files/
https://youtu.be/KtOayYXEsN4?si=-0MS6KXbEWXA7dqo
```

