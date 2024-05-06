import re
import json
import logging
import ffmpeg

from pathlib import Path


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
    

def save_txt_transcript(result: dict, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(result['text'])
    print(f'Saved .txt transcript → {filepath}.')


def save_srt_subtitles(srt_text: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as srt_file:
        srt_file.write(srt_text)
    print(f'Saved .srt subtitles → {filepath}.')
    
    
def save_rttm_annotations(diarization, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as rttm_file:
        rttm_file.write(diarization.to_rttm())
    print(f'Saved .rttm annotations → {filepath}.')
    
    
def save_results(result: dict, srt: bool = False, txt: bool = False, detect_speakers: bool = False) -> None:
    logger.info(f"""Saved .json transcription to {Path(f"{result['output_filepath']}.json")}""")
    save_transcription(result['transcription'], filepath=Path(f"{result['output_filepath']}.json"))
    if srt:
        logger.info(f"""Saved .srt subtitles to {Path(f"{result['output_filepath']}.srt")}""")
        srt_text = create_srt_subtitles(result['transcription'])
        save_srt_subtitles(srt_text, filepath=Path(f"{result['output_filepath']}.srt"))
    if txt:
        logger.info(f"""Saved .txt transcript to {Path(f"{result['output_filepath']}.txt")}""")
        save_txt_transcript(result['transcription'], filepath=Path(f"{result['output_filepath']}.txt"))
    if detect_speakers:
        logger.info(f"""Saved .rttm to {Path(f"{result['output_filepath']}.rttm")}""")
        save_rttm_annotations(result['diarization'], filepath=Path(f"{result['output_filepath']}.rttm"))


def format_time(seconds) -> str:
    # Function for time conversion
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
    
    
def create_srt_subtitles(transcription_dict: dict) -> str:
    """
    Converts a transcription dictionary into SRT (SubRip Text) format for subtitles.

    The function processes each text chunk in the transcription dictionary. Each chunk is expected
    to contain a 'timestamp' key with a tuple of start and end times, and a 'text' key with the
    corresponding transcript text. Chunks without valid start or end times are skipped.

    Args:
        transcription_dict (dict): A dictionary containing transcription data, with a key 'chunks'
                                   that holds a list of dictionaries. Each dictionary in this list
                                   should have a 'timestamp' key with a tuple (start_time, end_time)
                                   and a 'text' key with the transcription text.

    Returns:
        str: A string formatted as SRT, which is composed of segments where each segment includes a
             sequence number, the timestamp interval, the transcript text, and a blank line to
             separate entries.

    Example:
        transcription_dict = {
            "chunks": [
                {"timestamp": (30, 45), "text": "Hello, world."},
                {"timestamp": (50, 60), "text": "This is an example."}
            ]
        }
        srt_result = create_srt_subtitles(transcription_dict)
        print(srt_result)
        
        Output:
        
        1
        00:00:30,000 --> 00:00:45,000
        Hello, world.
        
        2
        00:00:50,000 --> 00:00:60,000
        This is an example.
    """    
    srt_text = ''
    seg_id = 0
    
    # Creating subtitles from transcription_dict
    for chunk in transcription_dict['chunks']:
        start_time = chunk['timestamp'][0]
        end_time = chunk['timestamp'][1]
        
        # Skip chunk if there is no start_time or end_time
        if start_time is None or end_time is None: 
            continue
        
        start_time_str = format_time(start_time)
        end_time_str = format_time(end_time)
        text = chunk['text']
        seg_id += 1
        srt_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
    return srt_text


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
    