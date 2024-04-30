# whisply
Transcribe, annotate, subtitle audio and video with Whisper ... Fast!

## Requirements
- ffmpeg
- python3.11

For GPU acceleration:
- CUDA


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
