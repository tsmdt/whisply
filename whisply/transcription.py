import logging
import time
import torch
import validators

from pathlib import Path
from datetime import datetime
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from faster_whisper import WhisperModel
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from whisply import little_helper, download_utils, speaker_detection


# Set logging configuration
logging.basicConfig(filename=f"whisply_{datetime.now().strftime('%Y%m%d')}.log", 
                    level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s')


class TranscriptionHandler:
    """
    Handles transcription and speaker diarization of audio files.

    This class provides methods to transcribe audio files using different OpenAI Whisper models,
    with support for parallel processing. It also allows for speaker diarization if enabled.

    Attributes:
    - base_dir (str): The base directory for storing transcriptions.
    - device (str): The device to use for transcription ('cpu', 'mps', or 'cuda:0').
    - language (str or None): The language of the audio files, if known.
    - model (str): The Whisper model to use for transcription.
    - detect_speakers (bool): Flag indicating whether to perform speaker diarization.
    - hf_token (str or None): Hugging Face API token for speaker diarization.
    - srt (bool): Flag indicating whether to generate SRT files.
    - metadata (dict): Metadata containing parameters used for transcription.

    Methods:
    - collect_metadata(): Collects metadata about the transcription parameters.
    - transcribe_with_insane_whisper(filepath: Path) -> dict: Transcribes an audio file using the 'insane-whisper' model.
    - transcribe_with_faster_whisper(filepath: Path, num_workers: int = 1) -> dict: Transcribes an audio file using the 'faster-whisper' model.
    - get_filepaths(filepath: str): Retrieves file paths for processing from various sources.
    - process_single_file(filepath): Processes a single audio file for transcription and/or diarization.
    - process_files_in_parallel(files): Processes multiple audio files in parallel.
    - process_files(files): Processes multiple audio files sequentially.

    Side Effects:
    - Logs various steps of the process using 'logging' at both debug and info levels.
    - Writes files to disk based on processing results if SRT generation is enabled.

    Returns:
    - None
    """
    def __init__(self, base_dir='./transcriptions', model='large-v3', device='cpu', 
                 language=None, detect_speakers=False, hf_token=None, srt=False):
        self.base_dir = Path(base_dir)
        little_helper.ensure_dir(self.base_dir)
        self.device = device
        self.language = language
        self.model = model
        self.detect_speakers = detect_speakers
        self.hf_token = hf_token
        self.srt = srt
        self.metadata = self._collect_metadata()
        self.filepaths = []
        self.output_dir = None
        self.processed_files = []


    def _collect_metadata(self):
        metadata = {'base_dir': self.base_dir,
                    'language': self.language,
                    'model': self.model,
                    'detect_speakers': self.detect_speakers,
                    'srt': self.srt}
        return metadata


    def transcribe_with_insane_whisper(self, filepath: Path) -> dict:
        """
        Transcribes an audio file using the 'insanely-fast-whisper' implementation: https://github.com/chenxwh/insanely-fast-whisper

        This method utilizes the 'insanely-fast-whisper' implemantation of OpenAI Whisper for automatic speech recognition.
        It initializes a pipeline for transcription and retrieves the result. If speaker detection is enabled,
        it also annotates speakers in the transcription result.

        Parameters:
        - filepath (Path): The path to the audio file for transcription.

        Returns:
        - dict: A dictionary containing the transcription result and, if speaker detection is enabled,
                the speaker diarization result. The transcription result includes the recognized text
                and timestamps if available.
        """
        # Start and time transcription
        logging.info(f"⭐️ Transcription started with ⏭️ insane-whisper for {filepath.name}")
        t_start = time.time()
        
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
            TimeElapsedColumn()
        ) as progress:
            progress.add_task(f"[cyan]Transcribing ({self.device.upper()}) → {filepath.name[:20]}..{filepath.suffix}", 
                              total = None)
            result = pipe(
                str(filepath),
                chunk_length_s = 30,
                batch_size = 8 if self.device in ['cpu', 'mps'] else 24,
                return_timestamps = True,
                generate_kwargs = {'language': self.language} if self.language else {'language': None}
            )
        result['text'] = result['text'].strip()
        
        # Stop timing transcription
        logging.info(f"⭐️ Transcription completed in {time.time() - t_start:.2f} sec.")
        
        # Speaker detection and annotation 
        if self.detect_speakers:
            result, diarization = speaker_detection.annotate_speakers(filepath=filepath, 
                                                                      result=result, 
                                                                      device=self.device,
                                                                      hf_token=self.hf_token)
        else:
            diarization = None
        return {'transcription': result, 'diarization': diarization}


    def transcribe_with_faster_whisper(self, filepath: Path, num_workers: int = 1) -> dict:
        """
        Transcribes an audio file using the 'faster-whisper' implementation: https://github.com/SYSTRAN/faster-whisper

        This method utilizes the 'faster-whisper' implementation of OpenAI Whisper for automatic speech recognition.
        It loads the model and sets parameters for transcription. After transcription, it formats the result
        into segments with timestamps and combines them into a single text. If speaker detection is enabled,
        it also annotates speakers in the transcription result.

        Parameters:
        - filepath (Path): The path to the audio file for transcription.
        - num_workers (int): The number of workers to use for transcription.

        Returns:
        - dict: A dictionary containing the transcription result and, if speaker detection is enabled,
                the speaker diarization result. The transcription result includes the recognized text
                and segmented chunks with timestamps if available.
        """
        # Start and time transcription
        logging.info(f"⭐️ Transcription started with ▶️ faster-whisper for {filepath.name}")
        t_start = time.time()
        
        # Load model and set parameters
        model = WhisperModel(self.model, device='cpu', num_workers=num_workers, compute_type='int8')
        segments, _ = model.transcribe(str(filepath), beam_size=5, language=self.language)
        
        result = {}
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
        
        # Stop timing transcription
        logging.info(f"⭐️ Transcription completed in {time.time() - t_start:.2f} sec.")
        
        # Speaker detection and annotation 
        if self.detect_speakers:
            result, diarization = speaker_detection.annotate_speakers(filepath=filepath, 
                                                                      result=result, 
                                                                      device=self.device,
                                                                      hf_token=self.hf_token)
        else:
            diarization = None
        return {'transcription': result, 'diarization': diarization}


    def get_filepaths(self, filepath: str):
        # Set correct file formats and clear filepaths list
        file_formats = ['.mp3', '.mp4', '.wav', '.m4a']
        self.filepaths = []  
        
        # Get single url
        if validators.url(filepath):
            downloaded_path = download_utils.download_url(filepath, downloads_dir=Path('./downloads'))
            if downloaded_path:
                self.filepaths.append(downloaded_path)
        
        # Get single file with correct file_format
        elif Path(filepath).suffix in file_formats:
            filepath = little_helper.normalize_filepath(filepath)
            self.filepaths.append(Path(filepath))
        
        # Get all files with correct file_format from folder
        elif Path(filepath).is_dir():
            for file_format in file_formats:
                filepaths = Path(filepath).glob(f'*{file_format}')
                new_filepaths = [little_helper.normalize_filepath(filepath) for filepath in filepaths]
                self.filepaths.extend(new_filepaths)
                
        # Get all files, folders, urls from .list
        elif Path(filepath).suffix == '.list':
            with open(filepath, 'r', encoding='utf-8') as file:
                listpaths = file.read().split('\n')
                for lpath in listpaths:
                    if validators.url(lpath):
                        downloaded_path = download_utils.download_url(lpath, downloads_dir=Path('./downloads'))
                        if downloaded_path:
                            self.filepaths.append(downloaded_path)
                    elif Path(lpath).is_file() and Path(lpath).suffix in file_formats:
                        newpath = little_helper.normalize_filepath(lpath)
                        self.filepaths.append(Path(newpath))
        else:
            print(f'The provided file or filetype "{filepath}" is not supported.')


    def process_files(self, files):
        """
        Processes a list of audio files for transcription and/or diarization.

        This method logs the processing parameters, extracts filepaths from the input list,
        and initializes an empty list for storing results. Each file is processed based on the
        compute device specified ('mps', 'cuda:0', or 'cpu'). Appropriate transcription method is
        chosen based on the device. Results, including file ids, paths, transcriptions, and diarizations,
        are stored in a dictionary and saved to a designated output directory. Each result is also
        appended to `self.processed_files`.

        Parameters:
        files (list of str): A list of file paths or file-like objects representing the audio files to be processed.

        Side Effects:
        - Initializes `self.filepaths` and `self.processed_files`.
        - Sets `self.output_dir` for each file.
        - Calls `self.save_results` which may write files to disk.
        - Logs various steps of the process using `logging` at both debug and info levels.

        Returns:
        None
        """
        logging.debug(f"🛠️ Provided parameters for processing: {self.metadata}")

        # Get filepaths
        self.get_filepaths(files)
        logging.debug(f"🚀 Processing files: {self.filepaths}")

        self.processed_files = []
        for idx, filepath in enumerate(self.filepaths):
            self.output_dir = little_helper.set_output_dir(filepath, self.base_dir)
            output_filepath = self.output_dir / Path(filepath).stem

            # Transcription and diarization
            logging.debug(f"Transcribing file: {filepath.name}")
            if self.device in ['mps', 'cuda:0']:
                result_data = self.transcribe_with_insane_whisper(filepath)
            elif self.device == 'cpu':
                result_data = self.transcribe_with_faster_whisper(filepath)
            
            result = {
                'id': f't_000{idx + 1}',
                'input_filepath': filepath,
                'transcription': result_data['transcription'],
                'diarization': result_data['diarization'],
                'output_filepath': output_filepath
            }

            little_helper.save_results(result=result, 
                                       srt=self.srt, 
                                       detect_speakers=self.detect_speakers)
            self.processed_files.append(result)
