from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="whisply",
    author="Thomas Schmidt, Renat Shigapov",
    version='0.10.4',
    packages=find_packages(),
    license="MIT",
    description="Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsmdt/whisply",
    install_requires=[
        'typer==0.15.2',
        'numpy==2.0.2',
        'faster-whisper==1.1.1',
        'ffmpeg-python==0.2.0',
        'optimum==1.24.0',
        'pyannote.audio==3.3.2',
        'whisperx==3.3.4',
        'rich==13.7.1',
        'torch==2.7.0',
        'torchaudio==2.7.0',
        'transformers==4.50.0',
        'validators==0.28.1',
        'yt-dlp==2025.4.30',
        'gradio==5.29.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'whisply=whisply.cli:run',
        ],
    },
)
