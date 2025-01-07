import re
import os
import json
import logging
import ffmpeg
import validators
import numpy as np

from enum import Enum
from pathlib import Path
from typing import Callable, Any, List
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TextColumn, SpinnerColumn
from whisply import download_utils

# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)


class DeviceChoice(str, Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'
    MPS = 'mps'
    
    
def get_device(device: DeviceChoice = DeviceChoice.AUTO) -> str:
    """
    Determine the computation device based on user preference and 
    availability.
    """
    import torch
    
    if device == DeviceChoice.AUTO:
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device == DeviceChoice.GPU:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print(f"[blue1]→ NVIDIA GPU not available. Using CPU.")
            device = 'cpu'
    elif device == DeviceChoice.MPS:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            print(f"[blue1]→ MPS not available. Using CPU.")
            device = 'cpu'
    elif device == DeviceChoice.CPU:
        device = 'cpu'
    else:
        device = 'cpu'
    return device


class FilePathProcessor:
    """
    Utility class for validating various filepaths.
    """
    def __init__(self, file_formats: List[str]):
        self.file_formats = [fmt.lower() for fmt in file_formats]
        self.filepaths: List[Path] = []

    def get_filepaths(self, filepath: str):
        """
        Processes the provided filepath which can be a URL, a single file, a directory,
        or a .list file containing multiple paths/URLs. It validates each input, downloads
        URLs if necessary, and accumulates valid file paths for further processing.
        """
        path = Path(filepath).expanduser().resolve()
        
        try:
            # Handle URL
            if validators.url(filepath):
                logging.info(f"Processing URL: {filepath}")
                downloaded_path = download_utils.download_url(
                    filepath, 
                    downloads_dir=Path('./downloads')
                )
                if downloaded_path:
                    self.filepaths.append(downloaded_path)
                else:
                    logging.error(f"Failed to download URL: {filepath}")
                    print(f"→ Failed to download URL: {filepath}")
                return  

            # Handle .list file
            elif path.suffix.lower() == '.list':
                if not path.is_file():
                    logging.error(f'The .list file "{path}" does not exist or is not a file.')
                    print(f'→ The .list file "{path}" does not exist or is not a file.')
                    return
                
                logging.info(f"Processing .list file: {path}")
                with path.open('r', encoding='utf-8') as file:
                    lpaths = set()
                    for line in file:
                        lpath = line.strip()
                        if not lpath:
                            continue 
                        lpaths.add(lpath)
                        
                    for lpath in lpaths:
                        if validators.url(lpath):
                            downloaded_path = download_utils.download_url(
                                lpath, 
                                downloads_dir=Path('./downloads')
                            )
                            if downloaded_path:
                                self.filepaths.append(downloaded_path)
                            else:
                                print(f'→ Failed to download URL: {lpath}')
                        else:
                            self._process_path(lpath)
                return

            # Handle single file or directory
            else:
                self._process_path(path)

        except Exception as e:
            logging.exception(f"An unexpected error occurred while processing '{filepath}': {e}")
            return

        # Remove duplicates by converting to a set of resolved absolute paths
        unique_filepaths = set(p.resolve() for p in self.filepaths)
        self.filepaths = list(unique_filepaths)

        # Filter out files that have already been converted
        self._filter_converted_files()
        
        # Final check to ensure there are files to process
        if not self.filepaths:
            logging.warning(f'No valid files found for processing. Please check the provided path: "{filepath}".')
            print(f'→ No valid files found for processing. Please check the provided path: "{filepath}".')
        else:
            logging.info(f"Total valid files to process: {len(self.filepaths)}")

    def _process_path(self, path_input: str | Path):
        """
        Processes a single path input, which can be a file or a directory.
        """
        path = Path(path_input).expanduser().resolve()

        if path.is_file():
            if path.suffix.lower() in self.file_formats:
                logging.info(f"Adding file: {path}")
                normalized_path = self._normalize_filepath(path)
                self.filepaths.append(normalized_path)
            else:
                logging.warning(f'File "{path}" has unsupported format and will be skipped.')
                print(f'→ File "{path}" has unsupported format and will be skipped.')
        elif path.is_dir():
            logging.info(f"Processing directory: {path}")
            for file_format in self.file_formats:
                for file in path.rglob(f'*{file_format}'):
                    if file.is_file():
                        logging.debug(f"Found file: {file}")
                        normalized_path = self._normalize_filepath(file)
                        self.filepaths.append(normalized_path)
        else:
            logging.error(f'Path "{path}" does not exist or is not accessible.')
            print(f'→ Path "{path}" does not exist or is not accessible.')
             
    def _normalize_filepath(self, filepath: Path) -> Path:
        """
        Normalizes the filepath by replacing non-word characters with underscores,
        collapsing multiple underscores into one, and removing leading/trailing underscores.
        """
        new_filename = re.sub(r'\W+', '_', filepath.stem)
        new_filename = new_filename.strip('_')
        new_filename = re.sub(r'_+', '_', new_filename)
        
        suffix = filepath.suffix.lower()
        
        # Construct the new path
        new_path = filepath.parent / f"{new_filename}{suffix}"
        
        # Rename the file
        filepath.rename(new_path)

        return new_path.resolve()

    def _filter_converted_files(self):
        """
        Removes files that have already been converted to avoid redundant processing.
        """
        converted_suffix = '_converted.wav'
        original_filepaths = []
        converted_filepaths = set()

        for fp in self.filepaths:            
            if fp.name.endswith(converted_suffix):
                converted_filepaths.add(fp)
            else:
                original_filepaths.append(fp)
                
        # Remove originals if their converted version exists
        filtered_filepaths = [
            fp for fp in original_filepaths
            if not (fp.with_name(fp.stem + converted_suffix) in converted_filepaths)
        ]
        
        # Extened filtered paths with converted paths
        filtered_filepaths.extend(converted_filepaths)

        removed_count = len(self.filepaths) - len(filtered_filepaths)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} files already converted.")
        self.filepaths = filtered_filepaths

def ensure_dir(dir: Path) -> None:
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir
        
def set_output_dir(filepath: Path, base_dir: Path) -> None:
    output_dir = base_dir / filepath.stem
    ensure_dir(output_dir)
    return output_dir

def return_valid_fileformats() -> list[str]:
    return [
        '.mp3',
        '.wav',
        '.m4a',
        '.aac',
        '.flac',
        '.ogg',
        '.mkv',
        '.mov',
        '.mp4',
        '.avi',
        '.mpeg',
        '.vob'
        ]

def check_file_format(
    filepath: Path, 
    del_originals: bool = True
    ) -> tuple[Path, np.ndarray]:
    """
    Checks the format of an audio file and converts it if it doesn't meet specified criteria.
    Then, loads the audio into a 1D NumPy array.

    The function uses `ffmpeg` to probe the metadata of an audio file at the given `filepath`.
    It checks if the audio stream meets the following criteria:
    - Codec name: 'pcm_s16le'
    - Sample rate: 16000 Hz
    - Number of channels: 1 (mono)

    If the audio stream does not meet these criteria, the function attempts to convert the file
    to meet the required format and saves the converted file with a '_converted.wav' suffix in the same directory.
    After successful conversion, it deletes the original file.

    Finally, it loads the audio (original or converted) as a 1D NumPy array and returns it.

    Args:
        filepath (Path): The path to the audio file to be checked and potentially converted.

    Returns:
        filepath (Path): filepath of the checked and / or converted audio file.
        np.ndarray: 1D NumPy array of the audio data.
    """ 
    import librosa
    
    # Define the converted file path
    new_filepath = filepath.with_name(f"{filepath.stem}_converted.wav")
    
    converted = False
    
    if new_filepath.exists():
        target_filepath = new_filepath
        converted = True
    else:
        try:
            # Probe the audio file for stream information
            probe = ffmpeg.probe(str(filepath))
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not audio_streams:
                raise ValueError(f"→ No audio stream found for {filepath}. Please check if the file you have provided contains audio content.")
            
            audio_stream = audio_streams[0]
            codec_name = audio_stream.get('codec_name')
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            
            # Check if the audio stream meets the criteria
            if codec_name != 'pcm_s16le' or sample_rate != 16000 or channels != 1:
                try:
                    # Convert the file and show progress
                    run_with_progress(
                        description=(
                            f"[orchid]→ Converting file to .wav: {filepath.name}"
                            ), 
                        task=lambda: convert_file_format(
                            old_filepath=filepath, 
                            new_filepath=new_filepath
                            )
                    )
                    target_filepath = new_filepath
                    converted = True
                except Exception as e:
                    raise RuntimeError(
                        f"→ An error occurred while converting {filepath}: {e}"
                        )
            else:
                # If already in correct format, use the original file
                target_filepath = filepath
            
        except ffmpeg.Error as e:
            print(f"→ Error running ffprobe: {e}")
            print(f"→ You may have provided an unsupported file type.\
Please check 'whisply --list_formats' for all supported formats.")
    
    try:
        # Load the audio file into a NumPy array
        audio, _ = librosa.load(str(target_filepath), sr=16000, mono=True)
        audio_array = audio.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {target_filepath}: {e}") from e
    
    # If conversion occurred delete the original file if del_originals
    if (converted and del_originals) and target_filepath != filepath:
        try:
            os.remove(filepath)
        except OSError as e:
            print(f"Warning: {e}")
    
    return Path(target_filepath), audio_array
       
def convert_file_format(old_filepath: str, new_filepath: str):
    """
    Converts a video file into an audio file in WAV format using the ffmpeg library.
    """
    (
        ffmpeg
        .input(str(old_filepath))
        .output(str(new_filepath), 
                acodec='pcm_s16le', # Audio codec: PCM signed 16-bit little-endian
                ar='16000',         # Sampling rate 16 KHz
                ac=1)               # Mono channel                       
        .run(quiet=True,
             overwrite_output=True)
    )

def load_config(config: json) -> dict:
    with open(config, 'r', encoding='utf-8') as file:
        return json.load(file)

def format_time(seconds, delimiter=',') -> str:
    """
    Function for time conversion.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    
    return f"{h:02}:{m:02}:{s:02}{delimiter}{ms:03}"
 
def run_with_progress(description: str, task: Callable[[], Any]) -> Any:
    """
    Helper function to run a task with a progress bar.
    """
    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}")
    ) as progress:
        progress.add_task(description, total=None)
        return task()