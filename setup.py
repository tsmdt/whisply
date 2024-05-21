from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="whisply",
    author="Thomas Schmidt",
    version=0.1,
    packages=find_packages(),
    license="Apache-2.0",
    description="Transcribe, translate, diarize, annotate and subtitle audio and video files with Whisper ... fast!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/th-schmidt/whisply",
    install_requires=[
        'click==8.1.7',
        'faster-whisper==1.0.1',
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
        'yt-dlp==2024.4.9'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'whisply=whisply.cli:main',
        ],
    },
)