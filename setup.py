from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="whisply",
    author="Thomas Schmidt, Renat Shigapov",
    version='0.6.1',
    packages=find_packages(),
    license="Apache-2.0",
    description="Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsmdt/whisply",
    install_requires=[
        'typer==0.12.5',
        'numpy==1.26.4',
        'faster-whisper==1.1.0',
        'ffmpeg-python==0.2.0',
        'optimum==1.19.1',
        'pyannote.audio==3.1.1',
        'pyannote.core==5.0.0',
        'pyannote.database==5.1.0',
        'pyannote.metrics==3.2.1',
        'pyannote.pipeline==3.0.1',
        'rich==13.7.1',
        'torch==2.3.0',
        'torch-audiomentations==0.11.1',
        'torch-pitch-shift==1.2.4',
        'torchaudio==2.3.0',
        'torchmetrics==1.3.2',
        'transformers==4.39.3',
        'validators==0.28.1',
        'yt-dlp>=2024.07.07',
        'whisperx',
        'gradio==5.6.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'whisply=whisply.cli:run',
        ],
    },
)
