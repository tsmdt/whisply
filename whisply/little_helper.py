import re
import json
import logging
import ffmpeg

from pathlib import Path
from typing import Callable, Any
from rich import print
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)


def load_config(config: json) -> dict:
    with open(config, 'r', encoding='utf-8') as file:
        return json.load(file)


def ensure_dir(dir: Path) -> None:
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir
        

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
    with open(filepath, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=4)
    print(f'[bold]→ Saved .json: {filepath}')
    logger.info(f"Saved .json to {filepath}")


def save_txt(transcription: dict, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(transcription['text'].strip())
    print(f'[bold]→ Saved .txt: {filepath}')
    logger.info(f"Saved .txt transcript to {filepath}")
    

def save_txt_with_speaker_annotation(annotated_text: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(annotated_text)
    print(f'[bold]→ Saved .txt with speaker annotation: {filepath}')
    logger.info(f'Saved .txt transcription with speaker annotation → {filepath}')
    
    
def save_subtitles(text: str, type: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as subtitle_file:
        subtitle_file.write(text)
    print(f'[bold]→ Saved .{type} subtitles: {filepath}')
    logger.info(f'Saved .{type} subtitles → {filepath}')
    
    
def save_rttm_annotations(rttm: str, filepath: Path) -> None:
    with open(filepath, 'w', encoding='utf-8') as rttm_file:
        rttm_file.write(rttm)
    print(f'[bold]→ Saved .rttm annotations: {filepath}')
    logger.info(f'Saved .rttm annotations → {filepath}')
    
    
def save_results(result: dict, subtitle: bool = None, annotate: bool = False) -> None:
    """
    Write various output formats to disk.
    """
    # Write .json
    save_json(result, filepath=Path(f"{result['output_filepath']}.json"))
    
    # Write .txt
    for language, transcription in result['transcription'].items():
        save_txt(transcription, filepath=Path(f"{result['output_filepath']}_{language}.txt"))
    
    # Write subtitles (.srt and .webvtt)
    if subtitle:
        for language, transcription in result['transcription'].items():
            # .srt subtitles
            srt_text = create_subtitles(transcription, type='srt')
            save_subtitles(srt_text, type='srt', filepath=Path(f"{result['output_filepath']}_{language}.srt"))
            
            # .webvtt / .vtt subtitles
            for type in ['webvtt', 'vtt']:
                webvtt_text = create_subtitles(transcription, type=type, result=result)
                save_subtitles(webvtt_text, type=type, filepath=Path(f"{result['output_filepath']}_{language}.{type}"))
    
    # If self.annotate write additional .txt with annotated speakers
    if annotate:
        for language, transcription in result['transcription'].items():
            save_txt_with_speaker_annotation(
                annotated_text=transcription['text_with_speaker_annotation'], 
                filepath=Path(f"{result['output_filepath']}_{language}_annotated.txt")
                )

    # Write .rttm
    if annotate:
        # Create .rttm annotations
        rttm_dict = dict_to_rttm(result)
        
        # Save .rttm for each language found in result
        for language, rttm_annotation in rttm_dict.items():
            save_rttm_annotations(rttm=rttm_annotation,
                                  filepath=Path(f"{result['output_filepath']}_{language}.rttm"))        



def create_text_with_speakers(transcription_dict: dict) -> dict:
    """
    Iterates through all chunks of each language and creates the complete text with 
    speaker labels inserted when there is a speaker change.

    Args:
        transcription_dict (dict): The dictionary containing transcription data.

    Returns:
        dict: A dictionary mapping each language to its formatted text with speaker labels.
    """    
    transcriptions = transcription_dict.get('transcriptions', {})
    
    for lang, lang_data in transcriptions.items():
        text = ""
        current_speaker = None
        chunks = lang_data.get('chunks', [])
        
        for chunk in chunks:
            words = chunk.get('words', [])
            
            for word_info in words:
                speaker = word_info.get('speaker')
                word = word_info.get('word', '')
                start_timestamp = format_time(word_info.get('start'), delimiter='.')
                
                # Insert speaker label if a speaker change is detected
                if speaker != current_speaker:
                    text += f"\n[{start_timestamp}] [{speaker}] "
                    current_speaker = speaker
                
                # Append the word with a space
                text += word + " "
        
        transcription_dict['transcriptions'][lang]['text_with_speaker_annotation'] = text.strip()
    
    return transcription_dict


def format_time(seconds, delimiter=',') -> str:
    """
    Function for time conversion.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    
    return f"{h:02}:{m:02}:{s:02}{delimiter}{ms:03}"


def create_subtitles(transcription_dict: dict, type: str = 'srt', result: dict = None) -> str:
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
    
    for chunk in transcription_dict['chunks']:
        start_time = chunk['timestamp'][0]
        end_time = chunk['timestamp'][1]
        text = chunk['text'].replace('’', '\'')
        
        # Create .srt subtitles
        if type == 'srt':
            start_time_str = format_time(start_time, delimiter=',')
            end_time_str = format_time(end_time, delimiter=',')
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
        # Create .webvtt subtitles    
        elif type in ['webvtt', 'vtt']:
            start_time_str = format_time(start_time, delimiter='.')
            end_time_str = format_time(end_time, delimiter='.')

            if seg_id == 0:
                subtitle_text += f"WEBVTT {Path(result['output_filepath']).stem}\n\n"
                
                if type == 'vtt':
                    subtitle_text += 'NOTE transcribed with whisply\n\n'
                    subtitle_text += f"NOTE media: {Path(result['input_filepath']).absolute()}\n\n"
                
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
    return subtitle_text


def dict_to_rttm(result: dict) -> dict:
    """
    Converts a transcription dictionary to RTTM file format.
    """
    file_id = result.get('input_filepath', 'unknown_file')
    file_id = Path(file_id).stem
    rttm_dict = {}

    # Iterate over each available language
    for lang, transcription in result.get('transcription', {}).items():
        lines = []
        current_speaker = None
        speaker_start_time = None
        speaker_end_time = None

        chunks = transcription.get('chunks', [])

        # Collect all words from chunks
        all_words = []
        for chunk in chunks:
            words = chunk.get('words', [])
            all_words.extend(words)

        # Sort all words by their start time
        all_words.sort(key=lambda w: w.get('start', 0.0))

        for word_info in all_words:
            speaker = word_info.get('speaker', 'SPEAKER_00')
            word_start = word_info.get('start', 0.0)
            word_end = word_info.get('end', word_start)

            if speaker != current_speaker:
                # If there is a previous speaker segment, write it to the RTTM
                if current_speaker is not None:
                    duration = speaker_end_time - speaker_start_time
                    rttm_line = (
                        f"SPEAKER {file_id} 1 {speaker_start_time:.3f} {duration:.3f} "
                        f"<NA> <NA> {current_speaker} <NA>"
                    )
                    lines.append(rttm_line)

                # Start a new speaker segment
                current_speaker = speaker
                speaker_start_time = word_start
                speaker_end_time = word_end
            else:
                # Extend the current speaker segment
                speaker_end_time = max(speaker_end_time, word_end)

        # Write the last speaker segment to the RTTM
        if current_speaker is not None:
            duration = speaker_end_time - speaker_start_time
            rttm_line = (
                f"SPEAKER {file_id} 1 {speaker_start_time:.3f} {duration:.3f} "
                f"<NA> <NA> {current_speaker} <NA>"
            )
            lines.append(rttm_line)

        rttm_content = "\n".join(lines)
        rttm_dict[lang] = rttm_content

    return rttm_dict


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
    """ 
    # Create a new file path for .wav conversion
    new_filepath = f"{filepath.parent}/{filepath.stem}_converted.wav"
    
    if Path(new_filepath).exists():
        return Path(new_filepath)
    else:
        try:
            probe = ffmpeg.probe(filepath)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not audio_streams:
                print(f"[bold]→ No audio stream found for {filepath}. Please check if the file you have provided contains audio content.")
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
                        description=f"[orchid]→ Converting file to .wav: {filepath.name}", 
                        task=lambda: convert_file_format(old_filepath=filepath, new_filepath=new_filepath)
                        )
                
                    return Path(new_filepath)
                
                except Exception as e:
                    raise RuntimeError(f"[bold]→ An error occurred while converting {filepath}: {e}")
            else:
                return filepath
            
        except ffmpeg.Error as e:
            print(f"[bold]→ Error running ffprobe: {e}")
            print(f"[bold]→ You may have provided an unsupported file type. Please check 'whisply --list_formats' for all supported formats.")
    

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
