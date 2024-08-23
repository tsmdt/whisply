import re
import json
import logging
import ffmpeg

from pathlib import Path
# from datetime import timedelta
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
    >>> original_path = Path("/path/to/your/file/Example File.MP3")
    >>> rename_file(original_path)
    PosixPath('/path/to/your/file/example_file.mp3')
    """
    original_path = Path(filepath)
    
    # Normalize title
    new_filename = re.sub(r'\W+', '_', original_path.stem)
    
    if new_filename.startswith('_'):
        new_filename = new_filename[1:]
        
    if new_filename.endswith('_'):
        new_filename = new_filename[:-1]

    # Construct new path
    new_path = original_path.parent / f"{new_filename}{original_path.suffix.lower()}"
    
    # Rename file
    original_path.rename(new_path)
    
    return new_path


def save_json(result: dict, filepath: Path) -> None:
    # Exclude the 'diarization' object from dict as this object is only relevant for .rttm subtitles
    filtered_result = {key: value for key, value in result.items() if key != 'diarization'}
    
    with open(filepath, 'w', encoding='utf-8') as fout:
        json.dump(filtered_result, fout, indent=4)
    print(f'Saved .json → {filepath}.')


def save_txt(transcription: dict, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(transcription['text'].strip())
    print(f'Saved .txt transcription → {filepath}.')
    
    
def save_txt_with_speaker_annotation(chunks: dict, filepath: Path) -> None:
    """
    Write .txt by combining transcription chunks with speaker_annotation chunks.
    """
    current_speaker = None

    txt = ''
    for key in chunks.keys():
        for item in chunks[key]:
            if current_speaker == None:
                txt += f"\n[{item['speakers'][0]}]\n{item['text'].strip()} "
                current_speaker = item['speakers'][0]
            elif current_speaker == item['speakers'][0]:
                txt += f"{item['text'].strip()} "
                current_speaker = item['speakers'][0]
            else:
                txt += f"\n[{item['speakers'][0]}]\n{item['text'].strip()} "
                current_speaker = item['speakers'][0]
    
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(txt.strip())
        
    print(f'Saved .txt transcription with speaker annotation → {filepath}.')
    
    
def save_subtitles(text: str, type: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as subtitle_file:
        subtitle_file.write(text)
    print(f'Saved .{type} subtitles → {filepath}.')
    
    
def save_rttm_annotations(diarization, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as rttm_file:
        rttm_file.write(diarization.to_rttm())
    print(f'Saved .rttm annotations → {filepath}.')
    
    
def save_results(result: dict, srt: bool = False, webvtt: bool = False, sub_length: int = None,
                 txt: bool = False, detect_speakers: bool = False) -> None:
    """
    Write various output formats to disk.
    """
    logger.info(f"""Saved .json to {Path(f"{result['output_filepath']}.json")}""")
    
    # Write .json
    save_json(result, filepath=Path(f"{result['output_filepath']}.json"))
    
    # Write .srt
    if srt:
        logger.info(f"""Saved .srt subtitles to {Path(f"{result['output_filepath']}.srt")}""")
        for language, transcription in result['transcription']['transcriptions'].items():
            srt_text = create_subtitles(transcription, sub_length, type='srt')
            save_subtitles(srt_text, type='srt', filepath=Path(f"{result['output_filepath']}_{language}.srt"))
            
    # Write .webvtt
    if webvtt:
        logger.info(f"""Saved .webvtt subtitles to {Path(f"{result['output_filepath']}.webvtt")}""")
        for language, transcription in result['transcription']['transcriptions'].items():
            webvtt_text = create_subtitles(transcription, sub_length, type='webvtt')
            save_subtitles(webvtt_text, type='webvtt', filepath=Path(f"{result['output_filepath']}_{language}.webvtt"))
    
    # Write .txt
    if txt:
        logger.info(f"""Saved .txt transcript to {Path(f"{result['output_filepath']}.txt")}""")
        
        # If self.detect_speakers write additional .txt with annotated speakers
        if detect_speakers:
            for language, _ in result['transcription']['transcription_and_speaker_annotation'].items():
                save_txt_with_speaker_annotation(
                    chunks=result['transcription']['transcription_and_speaker_annotation'], 
                    filepath=Path(f"{result['output_filepath']}_{language}_annotated.txt")
                    )
   
        for language, transcription in result['transcription']['transcriptions'].items():
            save_txt(transcription, filepath=Path(f"{result['output_filepath']}_{language}.txt"))
        
    # Write .rttm
    if detect_speakers:
        logger.info(f"""Saved .rttm to {Path(f"{result['output_filepath']}.rttm")}""")
        save_rttm_annotations(result['diarization'], filepath=Path(f"{result['output_filepath']}.rttm"))


def format_time(seconds, delimiter=',') -> str:
    """
    Function for time conversion.
    """
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


def check_file_format(filepath: Path) -> Path:
    """
    Checks the format of an audio file and converts it if it doesn't meet specified criteria.

    This function uses `ffmpeg` to probe the metadata of an audio file at the given `filepath`.
    It checks if the audio stream meets the following criteria:
    - Codec name: 'pcm_s16le'
    - Sample rate: 16000 Hz
    - Number of channels: 1 (mono)

    If the audio stream does not meet these criteria, the function attempts to convert the file
    to meet the required format and saves the converted file with a '_converted' suffix in the same directory.
    If the conversion is successful, the function returns the path to the converted file.

    Args:
        filepath (Path): The path to the audio file to be checked and potentially converted.

    Returns:
        Path: The path to the original file if it meets the criteria, or the path to the converted file.

    Raises:
        RuntimeError: If an error occurs during the conversion process.
        ffmpeg.Error: If there is an error running `ffmpeg.probe`.
    """ 
    try:
        probe = ffmpeg.probe(filepath)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        if not audio_streams:
            print(f"No audio stream found for {filepath}. Please check if the file you have provided contains audio content.")
            return False
        
        audio_stream = audio_streams[0]
        codec_name = audio_stream.get('codec_name')
        sample_rate = int(audio_stream.get('sample_rate', 0))
        channels = int(audio_stream.get('channels', 0))
        
        # Convert the file if its metadata do not match these criteria:
        if codec_name != 'pcm_s16le' or sample_rate != 16000 or channels != 1:
            try:
                new_filepath = f"{filepath.parent}/{filepath.stem}_converted.wav"
                
                # Convert file and show progress bar
                run_with_progress(
                    description=f"[orchid]Converting file to .wav → {filepath.name[:20]}...wav ", 
                    task=lambda: convert_file_format(old_filepath=filepath, new_filepath=new_filepath)
                    )
            
                return Path(new_filepath)
            
            except Exception as e:
                raise RuntimeError(f"An error occurred while converting {filepath}: {e}")
        else:
            print(f'No coversion for {filepath}')
            print(f'Found code_name: {codec_name}')
            print(f'Found sample_rate: {sample_rate}')
            print(f'Found channels: {channels}')
            return filepath
        
    except ffmpeg.Error as e:
        print(f"Error running ffprobe: {e}")
        print(f"You may have provided an unsupported file type. Please check 'whisply --list_formats' for all supported formats.")
    

def convert_file_format(old_filepath, new_filepath):
    """
    Converts a video file into an audio file in WAV format using the ffmpeg library.

    This function takes a video file and extracts its audio content, converting it to a WAV
    file with specific audio settings. The output audio will use PCM signed 16-bit little-endian
    format, a sampling rate of 16 KHz, and will be mono (single channel). The output file is
    overwritten if it already exists.

    Parameters:
    old_filepath : str
        The path to the video file from which audio will be extracted. This must be a path
        to a file that ffmpeg can read.
    new_filepath : str
        The path where the converted audio file will be saved as a WAV file. If a file at this
        path already exists, it will be overwritten.
    """
    (
        ffmpeg
        .input(old_filepath)
        .output(new_filepath, 
                acodec='pcm_s16le', # Audio codec: PCM signed 16-bit little-endian
                ar='16000',         # Sampling rate 16 KHz
                ac=1)               # Mono channel                       
        .run(quiet=True,
             overwrite_output=True)
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
