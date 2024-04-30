import re
import logging
import json
import ffmpeg
import click
import torch
import validators
import yt_dlp as url_downloader

from pathlib import Path
from datetime import datetime
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


logging.basicConfig(filename=f"whisply_{datetime.now().strftime('%Y%m%d')}.log", 
                    level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s')


class TranscriptionHandler:
    """
    Handles the transcription of audio and video files, with options for speaker diarization and generating subtitles.

    Attributes:
        base_dir (Path): Base directory where transcriptions are stored.
        model (str): Model identifier used for transcription.
        device (str): Computational device to use ('cpu', 'gpu' (= 'cuda:0') , 'mps').
        language (str, optional): Language code for transcription.
        detect_speakers (bool): Whether to detect and annotate speaker segments.
        hf_token (str, optional): HuggingFace Access token required for speaker diarization.
        srt (bool): Whether to generate subtitles in SRT format.
        filepaths (list): (List of) filepaths to be processed.
        output_dir (Path, optional): Directory for storing output files.
        output_filepath (Path, optional): Full path to the output file.

    Methods:
        ensure_dir(path): Ensures the directory at `path` exists, creating it if necessary.
        convert_video_to_wav(videofile_path, output_audio_path): Converts a video file to WAV format.
        annotate_speakers(filepath, result): Annotates speaker segments in the audio file.
        transcribe_with_insane_whisper(filepath): Performs transcription using the 'insane-whisper' model configuration.
        transcribe_with_faster_whisper(filepath): Performs transcription using a faster configuration.
        format_time(seconds): Converts seconds to a time string in the format 'HH:MM:SS,mmm'.
        create_srt_subtitles(transcription_dict): Generates SRT subtitles from transcription data.
        set_output_dir(filepath): Sets the output directory based on a file's path.
        save_transcription(result, json_filepath): Saves the transcription to a JSON file.
        save_srt_subtitles(srt_text, srt_filepath): Saves the subtitles in SRT format.
        save_rttm_annotations(diarization, rttm_filepath): Saves speaker diarization annotations in RTTM format.
        download_url(url): Downloads an audio file from a URL and returns its path.
        get_filepaths(filepath): Determines and sets the filepaths to be processed.
        process_files(files): Processes a list of files for transcription and speaker diarization.
    """
    def __init__(self, base_dir='./transcriptions', model='large-v3', device='cpu', 
                 language=None, detect_speakers=False, hf_token=None, srt=False):
        self.base_dir = Path(base_dir)
        self.ensure_dir(self.base_dir)
        self.device = device
        self.language = language
        self.model = model
        self.detect_speakers = detect_speakers
        self.hf_token = hf_token
        self.srt = srt
        self.filepaths = []
        self.output_dir = None
        self.output_filepath = None
        
        
    def ensure_dir(self, path):
        if not path.exists():
            path.mkdir(parents=True)

    def convert_video_to_wav(self, videofile_path, output_audio_path):
        (
            ffmpeg
            .input(videofile_path)
            .output(output_audio_path, 
                    acodec='pcm_s16le',     # Audio codec: PCM signed 16-bit little-endian
                    ar='16000',             # Sampling rate 16 KHz
                    ac=1)                   # Mono channel
            .run(overwrite_output=True)
        )

    def annotate_speakers(self, filepath: Path, result: dict) -> dict:
        is_video = Path(filepath).suffix in ['.mkv', '.mp4', '.mov']
        
        # Convert video to wav if neccessary
        if is_video:
            audio_path = Path(filepath).with_suffix('.wav')
            self.convert_video_to_wav(videofile_path=Path(filepath), output_audio_path=audio_path.as_posix())
            filepath = audio_path
        
        # Load pipeline for speaker detection
        try:
            diarization_pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', 
                                                            use_auth_token=self.hf_token)
            if self.device in ['cuda:0', 'cuda']:
                diarization_pipeline.to(torch.device('cuda'))
            logging.info("Diarization pipeline loaded")
        except Exception as e:
            logging.error(f"Error loading diarization pipeline: {e}")
            raise RuntimeError("Failed to load diarization pipeline")

        # Start speaker detection
        logging.info("Starting diarization process")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style = "bright_yellow", pulse_style = "bright_cyan"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task(f"[green]Annotating Speakers ({self.device.upper()}) → {filepath.name[:20]}..{filepath.suffix}", 
                              total = None)
            
            # Diarize audio
            diarization = diarization_pipeline(filepath)
                
            # Append annotations to results dict     
            segments = []
            for segment, _, label in diarization.itertracks(yield_label=True):
                temp = {"timestamp": [round(segment.start, 2), round(segment.end, 2)], "speaker": label}
                segments.append(temp)
            result['speaker_annotation'] = segments
        logging.info("Diarization completed")
        
        # Delete temp audio file
        if is_video:
            audio_path.unlink()
        return result, diarization
    
    def transcribe_with_insane_whisper(self, filepath: Path) -> dict:
        logging.info(f"Initializing *insane-whisper* pipeline for {filepath.name}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model = f"openai/whisper-{self.model}", 
            torch_dtype = torch.float32 if self.device=='cpu' else torch.float16,
            device = self.device,
            model_kwargs = {"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style = "bright_yellow", pulse_style = "bright_cyan"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task(f"[cyan]Transcribing ({self.device.upper()}) → {filepath.name[:20]}..{filepath.suffix}", 
                              total = None)
            result = pipe(
                filepath.as_posix(),
                chunk_length_s = 30,
                batch_size = 4 if self.device in ['cpu', 'mps'] else 24,
                return_timestamps = True,
                generate_kwargs = {'language': self.language} if self.language else {'language': None}
            )
        result['text'] = result['text'].strip()
        logging.info("Transcription completed")
        
        # Speaker detection and annotation 
        if self.detect_speakers:
            logging.info("Start: Speaker detection")
            result, diarization = self.annotate_speakers(filepath=filepath, result=result)
            return result, diarization
        return result, None

    def transcribe_with_faster_whisper(self, filepath: Path) -> dict:
        logging.info(f"Initializing *faster-whisper* pipeline for {filepath.name}")
        model = WhisperModel(self.model, device='cpu', compute_type="int8")

        segments, info = model.transcribe(filepath.as_posix(), beam_size=5, language=self.language)

        result = {}
        result['filename'] = filepath.name
        result['language'] = info.language
        result['language_propability'] = info.language_probability
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style = "bright_yellow", pulse_style = "bright_cyan"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task(f"[cyan]Transcribing ({self.device.upper()}) → {filepath.name[:20]}..{filepath.suffix}", 
                              total = None)
            
            segments_full = []    
            for segment in segments:
                seg = {}
                seg['id'] = segment.id
                seg['timestamp'] = (round(segment.start, 2), round(segment.end, 2))
                seg['text'] = segment.text
                segments_full.append(seg)

        result['chunks'] = segments_full
        result['text'] = ' '.join([segment['text'].strip() for segment in result['chunks']])
        logging.info("Transcription completed")
        
        # Speaker detection and annotation 
        if self.detect_speakers:
            logging.info("Start: Speaker detection")
            result, diarization = self.annotate_speakers(filepath=filepath, result=result)
            return result, diarization
        return result, None

    def format_time(self, seconds):
        # Function for time conversion
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
        
    def create_srt_subtitles(self, transcription_dict: dict) -> str:    
        srt_text = ''
        seg_id = 0
        
        # Creating subtitles from transcription_dict
        for chunk in transcription_dict['chunks']:
            start_time = chunk['timestamp'][0]
            end_time = chunk['timestamp'][1]
            
            # Skip chunk if there is no start_time or end_time
            if start_time is None or end_time is None: 
                continue
            
            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time)
            text = chunk['text']
            seg_id += 1
            srt_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        return srt_text

    def set_output_dir(self, filepath):
        self.output_dir = self.base_dir / filepath.stem
        self.ensure_dir(self.output_dir)

    def save_transcription(self, result, json_filepath) -> None:
        with open(json_filepath, 'w', encoding='utf-8') as fout:
            json.dump(result, fout, indent=4)
        print(f'Saved transcription → {json_filepath}.')

    def save_srt_subtitles(self, srt_text, srt_filepath) -> None:
        with open(srt_filepath, 'w', encoding='utf-8') as srt_file:
            srt_file.write(srt_text)
        print(f'Saved .srt subtitles → {srt_filepath}.')
        
    def save_rttm_annotations(self, diarization, rttm_filepath) -> None:
        with open(rttm_filepath, 'w', encoding='utf-8') as rttm_file:
            rttm_file.write(diarization.to_rttm())
        print(f'Saved .rttm annotations → {rttm_filepath}.')

    def download_url(self, url: str) -> Path:
        downloads_dir = Path('./downloads')
        self.ensure_dir(downloads_dir)

        temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}"
        options = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 
                                'preferredcodec': 'wav', 
                                'preferredquality': '192'}],
            'outtmpl': f'{downloads_dir}/{temp_filename}.%(ext)s'
        }
        try:
            with url_downloader.YoutubeDL(options) as ydl:
                ydl.download([url])
                video_info = ydl.extract_info(url, download=False)
                downloaded_file = list(downloads_dir.glob(f'{temp_filename}*'))[0]
                
                # Normalize title
                new_filename = re.sub(r'\W+', '_', video_info.get('title', 'downloaded_video'))
                
                # Remove trailing underscores
                if new_filename.startswith('_'):
                    new_filename = new_filename[1:]
                if new_filename.endswith('_'):
                    new_filename = new_filename[:-1]
                
                # Rename the file
                renamed_file = downloaded_file.rename(f"{downloads_dir}/{new_filename}{downloaded_file.suffix}")
                return Path(renamed_file)
        except Exception as e:
            print(f'Error downloading {url}: {e}')
            return None
    
    def get_filepaths(self, filepath: str):
        file_formats = ['.mp3', '.mp4', '.wav', '.m4a']
        self.filepaths = []  # Clear any existing list to avoid appending to old data

        if validators.url(filepath):
            downloaded_path = self.download_url(filepath)
            if downloaded_path:
                self.filepaths.append(downloaded_path)
        elif Path(filepath).suffix in file_formats:
            self.filepaths.append(Path(filepath))
        elif Path(filepath).is_dir():
            for file_format in file_formats:
                self.filepaths.extend(Path(filepath).glob(f'*{file_format}'))
        elif Path(filepath).suffix == '.list':
            with open(filepath, 'r', encoding='utf-8') as lin:
                listpaths = lin.read().split('\n')
                for path in listpaths:
                    if validators.url(path):
                        downloaded_path = self.download_url(path)
                        if downloaded_path:
                            self.filepaths.append(downloaded_path)
                    elif Path(path).is_file() and Path(path).suffix in file_formats:
                        self.filepaths.append(Path(path))
        else:
            print(f'The provided file or filetype "{filepath}" is not supported.')

    def process_files(self, files):
        # Get filepaths
        self.get_filepaths(files)
        for filepath in self.filepaths:
            self.set_output_dir(filepath)
            self.output_filepath = self.output_dir / filepath.stem

            # Transcription
            logging.debug(f"Transcribing file: {filepath.name}")
            if self.device in ['mps', 'cuda:0']:
                result, diarization = self.transcribe_with_insane_whisper(filepath=filepath)
            elif self.device == 'cpu':
                result, diarization = self.transcribe_with_faster_whisper(filepath=filepath)
            
            # Save .json transcriptions, .srt subtitles, .rttm annotations
            logging.info("Saving transcription .json")
            self.save_transcription(result=result, json_filepath=Path(f'{self.output_filepath}.json'))
            
            if self.srt:
                logging.info("Saving .srt subtitles")
                srt_text = self.create_srt_subtitles(transcription_dict=result)
                self.save_srt_subtitles(srt_text=srt_text, srt_filepath=Path(f'{self.output_filepath}.srt')) 
            if self.detect_speakers:
                logging.info("Saving .rttm annotations")
                self.save_rttm_annotations(diarization, rttm_filepath=Path(f'{self.output_filepath}.rttm'))

def load_config(config: json) -> dict:
    with open(config, 'r', encoding='utf-8') as file:
        return json.load(file)

@click.command()
@click.option('--files', type=click.Path(), help='Path to file, folder, URL or .list to process.')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'gpu', 'mps'], case_sensitive=False), 
              help='Select the computation device: CPU, GPU (nvidia CUDA), or MPS (Metal Performance Shaders).')
@click.option('--lang', default=None, type=click.Choice(['en', 'fr', 'de'], case_sensitive=False), 
              help='Specify the language of the audio for transcription.')
@click.option('--detect_speakers', default=False, is_flag=True, 
              help='Enable speaker diarization to identify and separate different speakers.')
@click.option('--hf_token', type=str, default=None, help='HuggingFace Access token required for speaker diarization.')
@click.option('--srt', default=False, is_flag=True, help='Generate SRT subtitles from the transcription.')
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='Path to configuration file.')
def main(files, device, lang, detect_speakers, hf_token, srt, config):
    # Load configuration from config.json if provided
    if config:
        config_data = load_config(Path(config))
        files = files or config_data.get('files')
        device = config_data.get('device', device)
        lang = config_data.get('lang', lang)
        detect_speakers = config_data.get('detect_speakers', detect_speakers)
        hf_token = config_data.get('hf_token', hf_token)
        srt = config_data.get('srt', srt)
    
    # Instantiate TranscriptionHandler
    service = TranscriptionHandler(device='cuda:0' if device == 'gpu' else device,
                                   language=lang, 
                                   detect_speakers=detect_speakers, 
                                   hf_token=hf_token, 
                                   srt=srt)
    
    # Process files
    service.process_files(files)

if __name__ == '__main__':
    main()
