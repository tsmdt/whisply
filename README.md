# whisply

<img src="https://github.com/user-attachments/assets/3a15509b-b60e-4581-af87-ecaa8e5089a3" width="25%">

*Transcribe, translate, annotate and subtitle audio and video files with OpenAI's [Whisper](https://github.com/openai/whisper) ... fast!*

`whisply` combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [insanely-fast-whisper](https://github.com/chenxwh/insanely-fast-whisper) to offer an easy-to-use solution for batch processing files. It also enables word-level speaker annotation by integrating [whisperX](https://github.com/m-bain/whisperX) and [pyannote](https://github.com/pyannote/pyannote-audio).

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Speaker annotation and diarization](#speaker-annotation-and-diarization)
    - [Requirements](#requirements-1)
    - [How speaker annotation works](#how-speaker-annotation-works)
  - [Batch processing](#batch-processing)
    - [Using config files for batch processing](#using-config-files-for-batch-processing)

## Features

* 🚴‍♂️ **Performance**: Depending on your hardware `whisply` will use the fastest `Whisper` implementation:
  * CPU: `fast-whisper` or `whisperX`
  * GPU (Nvidia CUDA) and MPS (Metal Performance Shaders, Apple M1-M3): `insanely-fast-whisper` or `whisperX`

* ✅ **Auto device selection**: When performing transcription or translation tasks without speaker annotation or subtitling, `faster-whisper` (CPU) or `insanely-fast-whisper` (MPS, Nvidia GPUs) will be selected automatically based on your hardware if you do not provide a device by using the `--device` option.

* 🗣️ **Word-level annotations**: If you choose to `--subtitle` or `--annotate`, `whisperX` will be used, which supports word-level segmentation and speaker annotations. Depending on your hardware, `whisperX` can run either on CPU or Nvidia GPU (but not on Apple MPS). Out of the box `whisperX` will not provide timestamps for words containing only numbers (e.g. "1.5" or "2024"): `whisply` fixes those instances through timestamp approximation.

* 💬 **Subtitles**: Generating subtitles is customizable. You can specify the number of words per subtitle block (e.g., choosing "5" will generate `.srt` and `.webvtt` files where each subtitle block exactly 5 words per segment with the corresponding timestamps).

* 🧺 **Batch processing**: `whisply` can process single files, whole folders, URLs or a combination of all by combining paths in a `.list` document. See the [Batch processing](#batch-processing) section for more information.

* ⚙️ **Supported output formats**: `.json` `.txt` `.txt (annotated)` `.srt` `.webvtt` `.vtt` `.rttm`

## Requirements

* [FFmpeg](https://ffmpeg.org/)
* \>= Python3.10
* GPU  processing requires:
  * Nvidia GPU (CUDA: cuBLAS and cuDNN 8 for CUDA 12) 
  * Apple Metal Performance Shaders (MPS) (Mac M1-M3)
* Speaker annotation requires a [HuggingFace Access Token](https://huggingface.co/docs/hub/security-tokens)

<details>
<summary><b>GPU Fix</b> for <i>Could not load library libcudnn_ops_infer.so.8.</i> (<b>click to expand</b>)</summary>
<br>If you use <b>whisply</b> on a Linux system with a Nivida GPU and get this error:<br><br>

```shell
"Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory"
```

Run the following line in your CLI:

```shell
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

Add this line to your Python environment to make it permanent:

```shell
echo "export LD_LIBRARY_PATH=\`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))'\`" >> path/to/your/python/env
```

For more information please refer to the <a href="https://github.com/SYSTRAN/faster-whisper" target="_blank">faster-whisper</a> GitHub page.

</details>

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

```shell
 Usage: whisply [OPTIONS]

 WHISPLY 💬 Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --files               -f       TEXT                Path to file, folder, URL or .list to process. [default: None]                                          │
│ --output_dir          -o       DIRECTORY           Folder where transcripts should be saved. [default: transcriptions]                                     │
│ --device              -d       [auto|cpu|gpu|mps]  Select the computation device: CPU, GPU (NVIDIA), or MPS (Mac M1-M3). [default: auto]                   │
│ --model               -m       TEXT                Whisper model to use (List models via --list_models). [default: large-v2]                               │
│ --lang                -l       TEXT                Language of provided file(s) ("en", "de") (Default: auto-detection). [default: None]                    │
│ --annotate            -a                           Enable speaker annotation (Saves .rttm).                                                                │
│ --hf_token            -hf      TEXT                HuggingFace Access token required for speaker annotation. [default: None]                               │
│ --translate           -t                           Translate transcription to English.                                                                     │
│ --subtitle            -s                           Create subtitles (Saves .srt, .vtt and .webvtt).                                                        │
│ --sub_length                   INTEGER             Subtitle segment length in words [default: 5]                                                           │
│ --verbose             -v                           Print text chunks during transcription.                                                                 │
│ --config                       PATH                Path to configuration file. [default: None]                                                             │
│ --list_filetypes                                   List supported audio and video file types.                                                              │
│ --list_models                                      List available models.                                                                                  │
│ --install-completion                               Install completion for the current shell.                                                               │
│ --show-completion                                  Show completion for the current shell, to copy it or customize the installation.                        │
│ --help                                             Show this message and exit.                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
  ```

### Speaker annotation and diarization

#### Requirements

In order to annotate speakers using `--annotate` you need to provide a valid [HuggingFace](https://huggingface.co) access token using the `--hf_token` option. Additionally, you must accept the terms and conditions for both version 3.0 and version 3.1 of the `pyannote` segmentation model. For detailed instructions, refer to the *Requirements* section on the [pyannote model page on HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1).

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
    "model": "large-v3-turbo",                 # Whisper model to use
    "lang": null,                              # Null for auto-detection or language codes ("en", "de", ...)
    "annotate": false,                         # Annotate speakers 
    "hf_token": "HuggingFace Access Token",    # Your HuggingFace Access Token (needed for annotations)
    "translate": false,                        # Translate to English
    "subtitle": false,                         # Subtitle file(s)
    "sub_length": 10,                          # Length of each subtitle block in number of words
    "verbose": false                           # Print transcription segments while processing 
}
```
