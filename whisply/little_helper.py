import re
import os
import json
import logging
import ffmpeg
import validators
import numpy as np

from pathlib import Path
from typing import Callable, Any, List, Dict, Tuple
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TextColumn, SpinnerColumn
from whisply import download_utils
from whisply.post_correction import Corrections

# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)

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
        Normalizes the filepath. This function can be expanded based on specific normalization needs.
        """    
        # Normalize title
        new_filename = re.sub(r'\W+', '_', filepath.stem)
        
        if new_filename.startswith('_'):
            new_filename = new_filename[1:]
        if new_filename.endswith('_'):
            new_filename = new_filename[:-1]

        # Construct new path and rename
        new_path = filepath.parent / f"{new_filename}{filepath.suffix.lower()}"
        filepath.rename(new_path)
        
        return filepath.resolve()

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


class OutputWriter:
    """
    Class for writing various output formats to disk.
    """
    def __init__(self, corrections: Corrections):
        self.cwd = Path.cwd()
        self.corrections = corrections
        self.compiled_simple_patterns = self._compile_simple_patterns()
        self.compiled_regex_patterns = self._compile_regex_patterns()
    
    def _compile_simple_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Pre-compile regex patterns for simple word corrections.
        Returns a list of tuples containing compiled patterns and their replacements.
        """
        patterns = []
        for wrong, correct in self.corrections.simple.items():
            # Wrap simple corrections with word boundaries
            pattern = re.compile(
                r'\b{}\b'.format(re.escape(wrong)), flags=re.IGNORECASE
                )
            patterns.append((pattern, correct))
            logger.debug(
                f"Compiled simple pattern: '\\b{wrong}\\b' → '{correct}'"
                )
        return patterns
    
    def _compile_regex_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Pre-compile regex patterns for pattern-based corrections.
        Returns a list of tuples containing compiled regex patterns and their replacements.
        """
        patterns = []
        for entry in self.corrections.patterns:
            original_pattern = entry['pattern']
            replacement = entry['replacement']
            # Wrap patterns with word boundaries and non-capturing group
            new_pattern = r'\b(?:' + original_pattern + r')\b'
            regex = re.compile(new_pattern, flags=re.IGNORECASE)
            patterns.append((regex, replacement))
            logger.debug(
                f"Compiled pattern-based regex: '{new_pattern}' → '{replacement}'"
                )
        return patterns
    
    def correct_transcription(self, transcription: str) -> str:
        """
        Apply both simple and pattern-based corrections to the transcription.
        """
        # Apply simple corrections
        for pattern, correct in self.compiled_simple_patterns:
            transcription = pattern.sub(
                lambda m: self.replace_match(m, correct), transcription
                )
        
        # Apply pattern-based corrections
        for regex, replacement in self.compiled_regex_patterns:
            transcription = regex.sub(replacement, transcription)
        
        return transcription
    
    @staticmethod
    def replace_match(match, correct: str) -> str:
        """
        Replace the matched word while preserving the original casing.
        """
        word = match.group()
        if word.isupper():
            return correct.upper()
        elif word[0].isupper():
            return correct.capitalize()
        else:
            return correct
        
    def _save_file(
        self, 
        content: str, 
        filepath: Path, 
        description: str, 
        log_message: str
        ) -> None:
        """
        Generic method to save content to a file.
        """
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f'[blue1]→ Saved {description}: [bold]{filepath.relative_to(self.cwd)}')
        logger.info(f'{log_message} {filepath}')

    def save_json(
        self, 
        result: dict, 
        filepath: Path
        ) -> None:
        with open(filepath, 'w', encoding='utf-8') as fout:
            json.dump(result, fout, indent=4)
        print(f'[blue1]→ Saved .json: [bold]{filepath.relative_to(self.cwd)}')
        logger.info(f"Saved .json to {filepath}")
        
    def save_txt(
        self, 
        transcription: Dict[str, str], 
        filepath: Path
        ) -> None:
        """
        Save the transcription as a TXT file after applying corrections.
        """
        original_text = transcription.get('text', '').strip()
        corrected_text = self.correct_transcription(original_text)
        self._save_file(
            content=corrected_text,
            filepath=filepath,
            description='.txt',
            log_message='Saved .txt transcript to'
        )
    
    def save_txt_with_speaker_annotation(
        self, 
        annotated_text: str, 
        filepath: Path
        ) -> None:
        """
        Save the annotated transcription as a TXT file after applying corrections.
        """
        corrected_annotated_text = self.correct_transcription(annotated_text)
        self._save_file(
            content=corrected_annotated_text,
            filepath=filepath,
            description='.txt with speaker annotation',
            log_message='Saved .txt transcription with speaker annotation →'
        )
    
    def save_subtitles(
        self, 
        text: str, 
        type: str, 
        filepath: Path
        ) -> None:
        """
        Save subtitles in the specified format after applying corrections.
        """
        corrected_text = self.correct_transcription(text)
        description = f'.{type} subtitles'
        log_message = f'Saved .{type} subtitles →'
        self._save_file(
            content=corrected_text,
            filepath=filepath,
            description=description,
            log_message=log_message
        )

    def save_rttm_annotations(
        self, 
        rttm: str, 
        filepath: Path
        ) -> None:
        self._save_file(
            content=rttm,
            filepath=filepath,
            description='.rttm annotations',
            log_message='Saved .rttm annotations →'
        )

    def save_results(
        self,
        result: dict,
        export_formats: List[str]
    ) -> List[Path]:
        """
        Write various output formats to disk based on the specified export formats.
        """
        output_filepath = Path(result['output_filepath'])
        written_filepaths = []
        
        # Apply corrections if they are provided
        if self.corrections and (
            self.corrections.simple or self.corrections.patterns
            ):
            for language, transcription in result.get('transcription', {}).items():
                # Correct the main transcription text
                original_text = transcription.get('text', '').strip()
                corrected_text = self.correct_transcription(original_text)
                result['transcription'][language]['text'] = corrected_text
                
                # Correct chunks and word dicts
                chunks = transcription.get('chunks', '')
                for c in chunks:
                    # Text chunk
                    c['text'] = self.correct_transcription(c['text'])
                    # Words
                    words = c.get('words', '')
                    for w in words:
                        w['word'] = self.correct_transcription(w['word'])

                # Correct speaker annotations if present
                if 'text_with_speaker_annotation' in transcription:
                    original_annotated = transcription['text_with_speaker_annotation']
                    corrected_annotated = self.correct_transcription(original_annotated)
                    result['transcription'][language]['text_with_speaker_annotation'] = corrected_annotated

        # Now, transcription_items reflects the corrected transcriptions
        transcription_items = result.get('transcription', {}).items()
    
        # Write .txt
        if 'txt' in export_formats:
            for language, transcription in transcription_items:
                fout = output_filepath.parent / f"{output_filepath.name}_{language}.txt"
                self.save_txt(
                    transcription,
                    filepath=fout
                )
                written_filepaths.append(str(fout))

        # Write subtitles (.srt, .vtt and .webvtt)
        subtitle_formats = {'srt', 'vtt', 'webvtt'}
        if subtitle_formats.intersection(export_formats):
            for language, transcription in transcription_items:
                # .srt subtitles
                if 'srt' in export_formats:
                    fout = output_filepath.parent / f"{output_filepath.name}_{language}.srt"
                    srt_text = create_subtitles(transcription, type='srt')
                    self.save_subtitles(srt_text, type='srt', filepath=fout)
                    written_filepaths.append(str(fout))

                # .vtt / .webvtt subtitles
                if 'vtt' in export_formats or 'webvtt' in export_formats:
                    for subtitle_type in ['webvtt', 'vtt']:
                        fout = output_filepath.parent / f"{output_filepath.name}_{language}.{subtitle_type}"
                        vtt_text = create_subtitles(transcription, type=subtitle_type, result=result)
                        self.save_subtitles(vtt_text, type=subtitle_type, filepath=fout)
                        written_filepaths.append(str(fout))

        # Write annotated .txt with speaker annotations
        has_speaker_annotation = any(
            'text_with_speaker_annotation' in transcription
            for transcription in result['transcription'].values()
        )

        if 'txt' in export_formats and has_speaker_annotation:
            for language, transcription in transcription_items:
                if 'text_with_speaker_annotation' in transcription:
                    fout = output_filepath.parent / f"{output_filepath.name}_{language}_annotated.txt"
                    self.save_txt_with_speaker_annotation(
                        annotated_text=transcription['text_with_speaker_annotation'],
                        filepath=fout
                    )
                    written_filepaths.append(str(fout))

        # Write .rttm
        if 'rttm' in export_formats:
            # Create .rttm annotations
            rttm_dict = dict_to_rttm(result)

            for language, rttm_annotation in rttm_dict.items():
                fout = output_filepath.parent / f"{output_filepath.name}_{language}.rttm"
                self.save_rttm_annotations(
                    rttm=rttm_annotation,
                    filepath=fout
                )
                written_filepaths.append(str(fout))

        # Write .json
        if 'json' in export_formats:
            fout = output_filepath.with_suffix('.json')
            written_filepaths.append(str(fout))
            result['written_files'] = written_filepaths
            self.save_json(result, filepath=fout)

        return written_filepaths


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

def create_text_with_speakers(
    transcription_dict: dict, 
    delimiter: str = '.'
    ) -> dict:
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
                start_timestamp = format_time(word_info.get('start'), delimiter)
                
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
            print(f"→ You may have provided an unsupported file type. Please check 'whisply --list_formats' for all supported formats.")
    
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