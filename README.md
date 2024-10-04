# whisply

<img src="https://github.com/user-attachments/assets/3a15509b-b60e-4581-af87-ecaa8e5089a3" width="25%">

*Transcribe, translate, annotate and subtitle audio and video files with OpenAI's [Whisper](https://github.com/openai/whisper) ... fast!*

`whisply` combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [insanely-fast-whisper](https://github.com/chenxwh/insanely-fast-whisper) to offer an easy-to-use solution for batch processing files. It also enables word-level speaker annotation by integrating [whisperX](https://github.com/m-bain/whisperX) and [pyannote](https://github.com/pyannote/pyannote-audio).

## Table of contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
    * [Speaker annotation and diarization](#speaker-annotation-and-diarization)
    * [Batch processing](#batch-processing)

## Features

* 🚴‍♂️ **Performance**: Depending on your hardware `whisply` will use the fastest `Whisper` implementation:
  * CPU: `fast-whisper` or `whisperX`
  * GPU (Nvidia CUDA) and MPS (Metal Performance Shaders, Apple M1-M3): `insanely-fast-whisper` or `whisperX`

* ✅ **Auto device selection**: When performing transcription or translation tasks without speaker annotation or subtitling, `faster-whisper` (CPU) or `insanely-fast-whisper` (MPS, Nvidia GPUs) will be selected automatically based on your hardware if you do not provide a device by using the `--device` option.

* 🗣️ **Word-level annotations**: If you choose to `--subtitle` or `--annotate`, `whisperX` will be used, which supports word-level segmentation and speaker annotations. Depending on your hardware, `whisperX` can run either on CPU or Nvidia GPU (but not on Apple MPS). Out of the box `whisperX` will not provide timestamps for words containing only numbers (e.g. "1.5" or "2024"): `whisply` fixes those instances through timestamp approximation.

* 💬 **Subtitles**: Generating subtitles is customizable. You can specify the number of words per subtitle block (e.g., choosing "5" will generate `.srt` and `.webvtt` files where each subtitle block exactly 5 words per segment with the corresponding timestamps).

* 🧺 **Batch processing**: `whisply` can process single files, whole folders, URLs or a combination of all by combining paths in a `.list` document. See the [Batch processing](#batch-processing) section for more information.

* ⚙️ **Supported output formats**: `.json` `.txt` `.txt (annotated)` `.srt` `.webvtt` `.rttm`

## Requirements

* [FFmpeg](https://ffmpeg.org/)
* python3.11
* **GPU processing** requires Nvidia GPU (CUDA) or Apple Metal Performance Shaders (MPS) (Mac M1-M3)
* **Speaker annotation** requires a [HuggingFace Access Token](https://huggingface.co/docs/hub/security-tokens)

## Installation

**1. Install `ffmpeg`**

```markdown
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
git clone https://github.com/tsmdt/whisply.git
```

```shell
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

```markdown
Usage: whisply [OPTIONS]

  WHISPLY 💬 Transcribe, translate, annotate and subtitle audio and video files
  with OpenAI's Whisper ... fast!

Options:
  --files PATH                 Path to file, folder, URL or .list to process.
  --output_dir DIRECTORY       Folder where transcripts should be saved.
                               Default: "./transcriptions".
  --device [auto|cpu|gpu|mps]  Select the computation device: auto (default),
                               CPU, GPU (NVIDIA CUDA), or MPS (Mac M1-M3).
  --model TEXT                 Select the whisper model to use (Default:
                               large-v2). Refers to whisper model size:
                               https://huggingface.co/collections/openai
  --lang TEXT                  Specify the language of the file(s) you provide
                               (en, de, fr ... Default: auto-detection).
  --annotate                   Enable speaker detection to identify and
                               annotate different speakers. Creates .rttm
                               file.
  --hf_token TEXT              HuggingFace Access token required for speaker
                               detection.
  --translate                  Translate transcription to English.
  --subtitle                   Create .srt and .webvtt subtitles from the
                               transcription.
  --sub_length INTEGER         Subtitle length in words for each subtitle
                               block (Default: 5); e.g. "10" produces
                               subtitles where each individual subtitle block
                               covers exactly 10 words.
  --config FILE                Path to configuration file.
  --filetypes                  List supported audio and video file types.
  --verbose                    Print text chunks during transcription.
  --help                       Show this message and exit.
  ```

### Speaker annotation and diarization

#### Requirements

In order to annotate speakers using `--annotate` you need to provide a valid [HuggingFace](https://huggingface.co) access token using the `--hf_token` option. Additionally, you must accept the terms and conditions for both version 3.0 and version 3.1 of the `pyannote` segmentation model. For detailed instructions, refer to the *Requirements* section on the [pyannote model page on HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1).

Whithout passing the `--hf_token` option, `whisply` will try to automatically read an existing HuggingFace access token from your shell environment that you have previously exported like this:

```shell
export HF_TOKEN=hf_abcieo...
```

#### How speaker annotation works

`whisply` uses [whisperX](https://github.com/m-bain/whisperX) for speaker diarization and annotation. Instead of returning chunk-level timestamps like the standard `Whisper` implementation `whisperX` is able to return word-level timestamps as well as annotating speakers word by word, thus returning much more precise annotations.

Out of the box `whisperX` will not provide timestamps for words containing only numbers (e.g. "1.5" or "2024"): `whisply` fixes those instances through timestamp approximation. Other known limitations of `whisperX` include:

* inaccurate speaker diarization if multiple speakers talk at the same time
* to provide word-level timestamps and annotations `whisperX` uses language specific alignment models; out of the box `whisperX` supports these languages: `en, fr, de, es, it, ja, zh, nl, uk, pt`.

Refer to the [whisperX GitHub page](https://github.com/m-bain/whisperX) for more information.


### Batch processing

Instead of providing a file, folder or URL by using the `--files` option you can pass a `.list` with a mix of files, folders and URLs for processing. 

Example:

```shell
$ cat my_files.list

video_01.mp4
video_02.mp4
./my_files/
https://youtu.be/KtOayYXEsN4?si=-0MS6KXbEWXA7dqo
```

#### Using config files for batch processing

You can provide a `.json` config file by using the `--config` option which makes batch processing easy. An example config looks like this:

```markdown
{
    "files": "./files/my_files.list",          # Path to your files
    "output_dir": "./transcriptions",          # Output folder where transcriptions are saved
    "device": "auto",                          # AUTO, GPU, MPS or CPU
    "model": "large-v2",                       # Whisper model to use
    "lang": null,                              # Null for auto-detection or language codes ("en", "de", ...)
    "annotate": false,                         # Annotate speakers 
    "hf_token": "HuggingFace Access Token",    # Your HuggingFace Access Token (needed for annotations)
    "translate": false,                        # Translate to English
    "subtitle": false,                         # Subtitle file(s)
    "sub_length": 10,                          # Length of each subtitle block in number of words
    "verbose": false                           # Print transcription segments while processing 
}
```
