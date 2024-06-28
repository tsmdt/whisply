import re
import json
import logging
import ffmpeg

from pathlib import Path
from datetime import timedelta
from typing import Callable, Any
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)


def load_config(config: json) -> dict:
    with open(config, 'r', encoding='utf-8') as file:
        return json.load(file)


def ensure_dir(path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        

def set_output_dir(filepath: Path, base_dir: Path) -> None:
    output_dir = base_dir / filepath.stem
    ensure_dir(output_dir)
    return output_dir


def normalize_filepath(filepath: str) -> Path:
    """
    Renames a file at the given filepath by sanitizing its filename to a more standard format.

    The function takes an existing file path, extracts the file name (excluding the extension),
    and replaces any non-alphanumeric characters with underscores. It also ensures that the 
    sanitized filename does not start or end with an underscore. Finally, it renames the file 
    with the sanitized name while preserving the original file extension.

    Parameters:
    filepath : str or Path
        The path to the file that needs to be renamed. This can be a string or a Path object
        from the pathlib module.

    Returns:
    Path
        The new path of the renamed file as a Path object. This includes the parent directory
        and the new filename with its extension.

    Raises:
    FileNotFoundError:
        If the file at the specified `filepath` does not exist.
    PermissionError:
        If the file is open in another program or the operation does not have the necessary
        permissions to rename the file.

    Example:
    --------
    >>> from pathlib import Path
    >>> original_path = Path("/path/to/your/file/Example File.txt")
    >>> rename_file(original_path)
    PosixPath('/path/to/your/file/example_file.txt')

    """
    original_path = Path(filepath)
    
    # Normalize title
    new_filename = re.sub(r'\W+', '_', original_path.stem)
    if new_filename.startswith('_'):
        new_filename = new_filename[1:]
    if new_filename.endswith('_'):
        new_filename = new_filename[:-1]

    # Construct new path
    new_path = original_path.parent / f"{new_filename}{original_path.suffix}"
    
    # Rename file
    original_path.rename(new_path)
    return new_path


def save_transcription(result: dict, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=4)
    print(f'Saved .json transcription → {filepath}.')
    

def save_txt_transcript(transcription: dict, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(transcription['text'].strip())
    print(f'Saved .txt transcript → {filepath}.')
    
    
def save_subtitles(text: str, type: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as subtitle_file:
        subtitle_file.write(text)
    print(f'Saved .{type} subtitles → {filepath}.')
    
    
def save_rttm_annotations(diarization, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as rttm_file:
        rttm_file.write(diarization.to_rttm())
    print(f'Saved .rttm annotations → {filepath}.')
    
    
def save_results(result: dict, 
                 srt: bool = False, 
                 webvtt: bool = False, 
                 sub_length: int = None,
                 txt: bool = False, 
                 detect_speakers: bool = False) -> None:
    logger.info(f"""Saved .json transcription to {Path(f"{result['output_filepath']}.json")}""")
    save_transcription(result['transcription'], filepath=Path(f"{result['output_filepath']}.json"))
    if srt:
        logger.info(f"""Saved .srt subtitles to {Path(f"{result['output_filepath']}.srt")}""")
        for language, transcription in result['transcription']['transcriptions'].items():
            srt_text = create_subtitles(transcription, sub_length, type='srt')
            save_subtitles(srt_text, type='srt', filepath=Path(f"{result['output_filepath']}_{language}.srt"))
    if webvtt:
        logger.info(f"""Saved .webvtt subtitles to {Path(f"{result['output_filepath']}.webvtt")}""")
        for language, transcription in result['transcription']['transcriptions'].items():
            webvtt_text = create_subtitles(transcription, sub_length, type='webvtt')
            save_subtitles(webvtt_text, type='webvtt', filepath=Path(f"{result['output_filepath']}_{language}.webvtt"))
    if txt:
        logger.info(f"""Saved .txt transcript to {Path(f"{result['output_filepath']}.txt")}""")
        for language, transcription in result['transcription']['transcriptions'].items():
            save_txt_transcript(transcription, filepath=Path(f"{result['output_filepath']}_{language}.txt"))
    if detect_speakers:
        logger.info(f"""Saved .rttm to {Path(f"{result['output_filepath']}.rttm")}""")
        save_rttm_annotations(result['diarization'], filepath=Path(f"{result['output_filepath']}.rttm"))


def format_time(seconds, delimiter=',') -> str:
    # Function for time conversion
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}{delimiter}{ms:03}"

    
def create_subtitles(transcription_dict: dict, sub_length: int = None, type: str = 'srt') -> str:
    """
    Converts a transcription dictionary into subtitle format (.srt or .webvtt).

    Args:
        transcription_dict (dict): Dictionary containing transcription data with 'chunks'.
        sub_length (int, optional): Maximum duration in seconds for each subtitle block.
        type (str, optional): Subtitle format, either 'srt' or 'webvtt'. Default is 'srt'.

    Returns:
        str: Formatted subtitle text in the specified format.
    """
    subtitle_text = ''
    seg_id = 0
    merged_chunks = []
    current_block = {"start": None, "end": None, "text": ""}
    
    # Merging chunks based on sub_length
    for chunk in transcription_dict['chunks']:
        start_time = chunk['timestamp'][0]
        end_time = chunk['timestamp'][1]
        text = chunk['text']
        
        # Skip chunk if there is no start_time or end_time
        if start_time is None or end_time is None: 
            continue
        
        if sub_length is not None:
            if current_block["start"] is None:
                current_block["start"] = start_time
                current_block["end"] = end_time
                current_block["text"] = text
            else:
                current_duration = current_block["end"] - current_block["start"]
                additional_duration = end_time - start_time
                
                if current_duration + additional_duration <= sub_length:
                    current_block["end"] = end_time
                    current_block["text"] += text
                else:
                    merged_chunks.append(current_block)
                    current_block = {"start": start_time, "end": end_time, "text": text}
        else:
            merged_chunks.append({"start": start_time, "end": end_time, "text": text})

    if current_block["start"] is not None:
        merged_chunks.append(current_block)
    
    for chunk in merged_chunks:
        # Create .srt subtitles
        if type == 'srt':
            start_time_str = format_time(chunk['start'], delimiter=',')
            end_time_str = format_time(chunk['end'], delimiter=',')
            text = chunk['text'].replace('’', '\'')
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
        # Create .webvtt subtitles    
        elif type == 'webvtt':
            start_time_str = format_time(chunk['start'], delimiter='.')
            end_time_str = format_time(chunk['end'], delimiter='.')
            text = chunk['text'].replace('’', '\'')

            if seg_id == 0:
                subtitle_text += 'WEBVTT\n\n'
                
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
    return subtitle_text


def convert_video_to_wav(videofile_path, output_audio_path):
    """
    Converts a video file into an audio file in WAV format using the ffmpeg library.

    This function takes a video file and extracts its audio content, converting it to a WAV
    file with specific audio settings. The output audio will use PCM signed 16-bit little-endian
    format, a sampling rate of 16 KHz, and will be mono (single channel). The output file is
    overwritten if it already exists.

    Parameters:
    videofile_path : str
        The path to the video file from which audio will be extracted. This must be a path
        to a file that ffmpeg can read.
    output_audio_path : str
        The path where the converted audio file will be saved as a WAV file. If a file at this
        path already exists, it will be overwritten.

    Raises:
    FileNotFoundError:
        If the `videofile_path` does not exist.
    PermissionError:
        If there is no permission to read the video file or write the audio file.
    RuntimeError:
        If ffmpeg encounters an issue during the conversion process.

    Example:
    --------
    >>> convert_video_to_wav("/path/to/video.mp4", "/path/to/output/audio.wav")

    Note:
    The function requires ffmpeg to be installed and properly configured in the system path.
    """
    (
        ffmpeg
        .input(videofile_path)
        .output(output_audio_path, 
                acodec='pcm_s16le', # Audio codec: PCM signed 16-bit little-endian
                ar='16000',         # Sampling rate 16 KHz
                ac=1)               # Mono channel
        .run(overwrite_output=True)
    )
    

def run_with_progress(description: str, task: Callable[[], Any]) -> Any:
    """
    Helper function to run a task with a progress bar.

    Parameters:
    - description (str): The description to display in the progress bar.
    - task (Callable[[], Any]): The task to run, which returns a result.

    Returns:
    - Any: The result returned by the task.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="bright_yellow", pulse_style="bright_cyan"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(description, total=None)
        return task()
