import logging
import time
import validators

from pathlib import Path
from datetime import datetime
from functools import partial
from rich import print

from whisply import little_helper, download_utils, models


# Set logging configuration
logging.basicConfig(filename=f"log_whisply_{datetime.now().strftime('%Y-%m-%d')}.log", 
                    level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s')


class TranscriptionHandler:
    """
    Handles transcription, translation, and speaker annotation of audio and video files using various Whisper implementations.

    This class provides methods to process audio files for transcription using different Whisper-based models,
    including WhisperX, insanely-fast-whisper, and faster-whisper. It supports language detection, speaker diarization,
    subtitle generation, and translation. The class can handle audio and video files from local paths, directories, URLs, or lists of files.

    Parameters:
        base_dir (str, optional): The base directory to store transcriptions. Defaults to './transcriptions'.
        model (str, optional): The Whisper model to use (e.g., 'large-v2'). Defaults to 'large-v2'.
        device (str, optional): The device to use for computation ('cpu', 'cuda:0', 'mps'). Defaults to 'cpu'.
        file_language (str, optional): The language code of the audio files. If None, language detection will be performed.
        annotate (bool, optional): Whether to perform speaker annotation. Defaults to False.
        hf_token (str, optional): Hugging Face token for authentication (required for some models).
        subtitle (bool, optional): Whether to generate subtitles. Defaults to False.
        sub_length (int, optional): Maximum number of words per subtitle segment. Defaults to 10.
        translate (bool, optional): Whether to translate the transcription into English. Defaults to False.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.

    Methods:
        transcribe_with_whisperx(filepath):
            Transcribe an audio file using WhisperX implementation, with options for word-level timestamps and speaker annotation.

        transcribe_with_insane_whisper(filepath):
            Transcribe an audio file using the insanely-fast-whisper implementation. Fastest with nvidia and Apple M1-M3.

        transcribe_with_faster_whisper(filepath, num_workers=1):
            Transcribe an audio file using the faster-whisper implementation. Fastest with CPU.

        get_filepaths(filepath):
            Parse and collect file paths from a given input (file, directory, URL, or .list file).

        detect_language(file):
            Detect the language of the input audio file.

        process_files(files):
            Process a list of files for transcription, translation, and/or speaker diarization.
    """
    def __init__(self, 
                 base_dir='./transcriptions', 
                 model='large-v2', 
                 device='cpu', 
                 file_language=None, 
                 annotate=False, 
                 hf_token=None, 
                 subtitle=False, 
                 sub_length=None, 
                 translate=False, 
                 verbose=False):
        self.base_dir = little_helper.ensure_dir(Path(base_dir))
        self.file_formats = little_helper.return_valid_fileformats()
        self.device = device
        self.file_language = file_language
        self.file_language_provided = file_language is not None
        self.model = None
        self.model_provided = model
        self.annotate = annotate
        self.translate = translate
        self.hf_token = hf_token
        self.subtitle = subtitle
        self.sub_length = sub_length
        self.verbose = verbose
        self.metadata = self._collect_metadata()
        self.filepaths = []
        self.output_dir = None
        self.processed_files = []


    def _collect_metadata(self):
        metadata = {'output_dir': str(self.base_dir),
                    'file_language': self.file_language,
                    'model': self.model,
                    'device': self.device,
                    'annotate': self.annotate,
                    'translate': self.translate,
                    'subtitle': self.subtitle,
                    'sub_length': self.sub_length
                    }
        return metadata


    def transcribe_with_whisperx(self, filepath: Path) -> dict:
        """
        Transcribe a file with the whisperX implementation that returns word-level timestamps and speaker annotation:
        https://github.com/m-bain/whisperX 
        
        This implementation is used when a specific subtitle length (e.g. 5 words per individual subtitle) is needed.
        """
        import torch
        import whisperx
        import gc
        
        def fill_missing_timestamps(segments: list) -> list:
            """
            whisperX does not provide timestamps for words containing only numbers (e.g. "1.5", "2024" etc.).
            
            The function fills these missing timestamps by padding the last known 'end' timestamp for a missing
            'start' timestamp or by cutting the next known 'start' timestamp for a missing 'end' timestamp.
            """
            padding = 0.05 # in seconds
            
            for segment in segments:
                words = segment['words']
                num_words = len(words)
                
                for i, word in enumerate(words):
                    # If the 'start' key is missing
                    if 'start' not in word:
                        if i > 0 and 'end' in words[i-1]:
                            word['start'] = round(words[i-1]['end'] + padding, 2)
                        else:
                            word['start'] = segment['start']
                    
                    # If the 'end' key is missing
                    if 'end' not in word:
                        if i < num_words - 1 and 'start' in words[i+1]:
                            word['end'] = round(words[i+1]['start'] - padding, 2)
                        elif i == num_words - 1:
                            word['end'] = round(words[i]['start'] + padding, 2)
                            segment['end'] = word['end']
                        else:
                            word['end'] = round(words[i]['start'] + padding, 2)
                    
                    # If 'score' key is missing       
                    if 'score' not in word:
                        word['score'] = 0.5
        
                    # If 'speaker' key is missing
                    if self.annotate and 'speaker' not in word:
                        speaker_assigned = False
                        
                         # Case 1: If it's the first word, look forward for the next speaker
                        if i == 0:
                            for j in range(i + 1, num_words):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        # Case 2: If it's the last word, look backward for the previous speaker
                        elif i == num_words - 1:
                            for j in range(i - 1, -1, -1):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        # Case 3: For other words, prefer the previous speaker; if not found, look forward
                        if not speaker_assigned:
                            # Look backward
                            for j in range(i - 1, -1, -1):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        if not speaker_assigned:
                            # Look forward
                            for j in range(i + 1, num_words):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        if not speaker_assigned:
                            # Default speaker if none found
                            word['speaker'] = 'UNKNOWN'
                        
            return segments
        
        
        def adjust_word_chunk_length(result: dict) -> dict:
            """
            Generates text chunks based on the maximum number of words.

            Parameters:
                result (dict): The nested dictionary containing segments and words.
                max_number (int): The maximum number of words per chunk. Default is 6.

            Returns:
                dict: A dictionary containing a list of chunks, each with 'text', 'timestamp', and 'words'.
            """
            # Flatten all words from all segments
            words = [
                word_info
                for segment in result.get('segments', [])
                for word_info in segment.get('words', [])
            ]

            # Split words into chunks of size max_number
            def split_into_chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            chunks = []
            for word_chunk in split_into_chunks(words, self.sub_length):
                chunk_text = ' '.join(word_info['word'] for word_info in word_chunk)
                chunk_start = word_chunk[0]['start']
                chunk_end = word_chunk[-1]['end']
                chunk = {
                    'timestamp': [chunk_start, chunk_end],
                    'text': chunk_text,
                    'words': word_chunk
                }
                chunks.append(chunk)

            result_temp = {
                'text': ' '.join(chunk['text'].strip() for chunk in chunks),
                'chunks': chunks
            }

            return result_temp

        
        def whisperx_task(task: str = 'transcribe', language = None):
            """
            Define a transcription / translation task with whisperX
            """
            # Set parameters
            device = 'cuda' if self.device == 'cuda:0' else 'cpu'
            
            # Transcribe / translate
            model = whisperx.load_model(
                whisper_arch=self.model, 
                device=device, 
                compute_type='float16' if self.device == 'cuda:0' else 'int8', 
                language=self.file_language or None
                )
            audio = whisperx.load_audio(str(filepath), sr=16000)
            result = model.transcribe(
                audio, 
                batch_size=16 if self.device == 'cuda:0' else 8, 
                task=task)
            
            model_a, metadata = whisperx.load_align_model(device=device, language_code=language)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, 
                                    return_char_alignments=False)
            
            # Speaker annotation 
            if self.annotate:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=device)
                diarize_segments = diarize_model(str(filepath))
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
            # Empty CUDA cache
            if self.device == 'cuda:0':
                gc.collect()
                torch.cuda.empty_cache()
                del model_a
                    
            return result
        
        # Start and time transcription
        logging.info(f"👨‍💻 Transcription started with 🆇 whisperX for {filepath.name}")
        t_start = time.time()
        
        # Run the transcription
        transcription_task = partial(whisperx_task, task='transcribe', language=self.file_language)
        transcription_result = little_helper.run_with_progress(
            description=f"[cyan]→ Transcribing ({'CUDA' if self.device == 'cuda:0' else 'CPU'}) {filepath.name}",
            task=transcription_task
        )
        
        # Fill in missing timestamps and adjust word chunk length
        transcription_result['segments'] = fill_missing_timestamps(transcription_result['segments'])
        transcription_result = adjust_word_chunk_length(transcription_result)
        
        # Create result dict and append transcription to it
        result = {'transcriptions': {}}
        result['transcriptions'][self.file_language] = transcription_result
        
        # Print transcription if verbose
        if self.verbose:
            print(f"[bold]{result['transcriptions'][self.file_language]['text']}")
        
        # Translation task (to English)
        if self.translate and self.file_language != 'en':
            translation_task = partial(whisperx_task, task='translate', language='en')
            translation_result = little_helper.run_with_progress(
                description=f"[dark_blue]→ Translating ({'CUDA' if self.device == 'cuda:0' else 'CPU'}) {filepath.name}",
                task=translation_task
            )
            
            # Fill in missing timestamps and adjust word chunk length
            translation_result['segments'] = fill_missing_timestamps(translation_result['segments'])
            translation_result = adjust_word_chunk_length(translation_result)
            result['transcriptions']['en'] = translation_result
            
            if self.verbose:
                print(f"[bold]{result['transcriptions']['en']['text']}")

        # if self.annotate == 'gat2':
        #     result = enhance_annotations.transform2gat2(result)

        # Create full transcription with speaker annotation
        result = little_helper.create_text_with_speakers(result)
        
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}


    def transcribe_with_insane_whisper(self, filepath: Path) -> dict:
        """
        Transcribes a file using the 'insanely-fast-whisper' implementation: https://github.com/chenxwh/insanely-fast-whisper

        This method utilizes the 'insanely-fast-whisper' implementation of OpenAI Whisper for automatic speech recognition.
        It initializes a pipeline for transcription and retrieves the result. If speaker detection is enabled,
        it also annotates speakers in the transcription result.

        Parameters:
        - filepath (Path): The path to the audio file for transcription.

        Returns:
        - dict: A dictionary containing the transcription result and, if speaker detection is enabled,
                the speaker diarization result. The transcription result includes the recognized text
                and timestamps if available.
        """
        import torch
        from transformers import pipeline
        from transformers.utils import is_flash_attn_2_available

        # Start and time transcription
        logging.info(f"👨‍💻 Transcription started with 🚅 insane-whisper for {filepath.name}")
        t_start = time.time()
        
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model = self.model, 
                torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16,
                device = self.device,
                model_kwargs = {"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            )
            
            # Define transcription function
            def transcription_task():
                transcription_result = pipe(
                    str(filepath),
                    chunk_length_s = 30,
                    batch_size = 8 if self.device in ['cpu', 'mps'] else 24,
                    return_timestamps = True,
                    generate_kwargs = {'language': self.file_language},
                )
                return transcription_result
             
            # Add progress bar and run the transcription task
            transcription_result = little_helper.run_with_progress(
                description=f"[cyan]→ Transcribing ({self.device.upper()}) {filepath.name}",
                task=transcription_task
            )
            
            result = {'transcriptions': {}}
            result['transcriptions'] = {
                self.file_language: transcription_result
            }
            
            # If verbose Flag 
            if self.verbose:
                print(result['transcriptions'][self.file_language]['text'])
            
            # Translation
            if self.translate and self.file_language != 'en':
                # Define translation function
                def translation_task():
                    translation_result = pipe(
                        str(filepath),
                        chunk_length_s = 30,
                        batch_size = 8 if self.device in ['cpu', 'mps'] else 24,
                        return_timestamps = True,
                        generate_kwargs = {'task': 'translate',
                                           'language': self.file_language}
                    )
                    return translation_result
                
                # Add progress bar and run the translation task
                translation_result = little_helper.run_with_progress(
                    description=f"[dark_blue]→ Translating ({self.device.upper()}) {filepath.name}",
                    task=translation_task
                )
                
                result['transcriptions']['en'] = translation_result
                
                # If verbose Flag 
                if self.verbose:
                    print(result['transcriptions']['en']['text'])

        except ValueError as e:
            print(f'[bold]{e}')
        
        # Stop timing transcription
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}


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
        from faster_whisper import WhisperModel

        # Start and time transcription
        logging.info(f"👨‍💻 Transcription started with 🏃‍♀️‍➡️ faster-whisper for {filepath.name}")
        t_start = time.time()
        
        # Load model and set parameters
        model = WhisperModel(
            self.model, 
            device='cpu' if self.device in ['mps', 'cpu'] else 'cuda', 
            num_workers=num_workers, 
            compute_type='int8' if self.device in ['mps', 'cpu'] else 'float16'
        )
        
        # Define the transcription task
        def transcription_task():
            segments, _ = model.transcribe(
                str(filepath), 
                beam_size=5, 
                language=self.file_language,
                word_timestamps=True
            )
                
            chunks = []    
            for segment in segments:
                seg = {
                    'timestamp': (float(f"{segment.start:.2f}"), float(f"{segment.end:.2f}")),
                    'text': segment.text.strip(),
                    'words': [{
                        'word': i.word.strip(),
                        'start': float(f"{i.start:.2f}"),
                        'end': float(f"{i.end:.2f}"),
                        'score': float(f"{i.probability:.2f}")
                    } for i in segment.words]
                }
                chunks.append(seg)
                
                # If verbose Flag 
                if self.verbose:
                    print(seg['text'])
                    
            return chunks
        
        # Add progress bar and run the transcription task
        chunks = little_helper.run_with_progress(
            description=f"[cyan]→ Transcribing ({self.device.upper()}) {filepath.name}",
            task=transcription_task
        )
        
        # Create result dict and append transcriptions to it
        result = {'transcriptions': {}}
        result['transcriptions'][self.file_language] = {
                'text': ' '.join([segment['text'].strip() for segment in chunks]),
                'chunks': chunks
                }

        # Translation
        if self.translate and self.file_language != 'en':
            # Define the translation task
            def translation_task():
                segments, _ = model.transcribe(
                    str(filepath), 
                    beam_size=5, 
                    task='translate', 
                    language='en',
                    word_timestamps=True)
                
                translation_chunks = []    
                for segment in segments:
                    seg = {
                        'timestamp': (float(f"{segment.start:.2f}"), float(f"{segment.end:.2f}")),
                        'text': segment.text.strip(),
                        'words': [{
                            'word': i.word.strip(),
                            'start': float(f"{i.start:.2f}"),
                            'end': float(f"{i.end:.2f}"),
                            'score': float(f"{i.probability:.2f}")
                        } for i in segment.words]
                    }
                    translation_chunks.append(seg)
                    
                    # If verbose Flag 
                    if self.verbose:
                        print(seg['text'])
                        
                return translation_chunks
            
            # Add progress bar and run the translation task
            translation_chunks = little_helper.run_with_progress(
                description=f"[dark_blue]→ Translating ({self.device.upper()}) {filepath.name}",
                task=translation_task
            )

            # Add translation to result dict
            result['transcriptions']['en'] = {
                'text': ' '.join([segment['text'].strip() for segment in translation_chunks]),
                'chunks': translation_chunks
                }
            
        # if self.annotate == 'gat2':
        #     result = enhance_annotations.transform2gat2(result)
        
        # Stop timing transcription
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}
            

    def get_filepaths(self, filepath: str):
        self.filepaths = []  
        
        # Get single url
        if validators.url(filepath):
            downloaded_path = download_utils.download_url(filepath, downloads_dir=Path('./downloads'))
            if downloaded_path:
                self.filepaths.append(downloaded_path)
        
        # Get single file with correct file_format
        elif Path(filepath).suffix.lower() in self.file_formats:
            filepath = little_helper.normalize_filepath(filepath)
            self.filepaths.append(Path(filepath))
        
        # Get all files with correct file_format from folder
        elif Path(filepath).is_dir():
            for file_format in self.file_formats:
                filepaths = Path(filepath).glob(f'*{file_format}')
                new_filepaths = [little_helper.normalize_filepath(filepath) for filepath in filepaths]
                self.filepaths.extend(new_filepaths)
                
        # Get all files, folders, urls from .list
        elif Path(filepath).suffix == '.list':
            with open(filepath, 'r', encoding='utf-8') as file:
                listpaths = file.read().split('\n')
                for lpath in listpaths:
                    if validators.url(lpath):
                        downloaded_path = download_utils.download_url(
                            lpath, 
                            downloads_dir=Path('./downloads')
                            )
                        if downloaded_path:
                            self.filepaths.append(downloaded_path)
                    elif Path(lpath).is_file() and Path(lpath).suffix.lower() in self.file_formats:
                        newpath = little_helper.normalize_filepath(lpath)
                        self.filepaths.append(Path(newpath))
                    else:
                        print(f'[bold]→ Error loading "{lpath}": Check if the file exists and the filepath is correct.')
        else:
            print(f'[bold]→ The provided file or filetype "{filepath}" is not supported.')
            
        # Filter out duplicates from previous file conversions
        to_remove = []
        for filepath in self.filepaths:
            converted_filepath = filepath.with_stem(filepath.stem + '_converted').with_suffix('.wav')
            if converted_filepath in self.filepaths:
                to_remove.append(filepath)
        
        for filepath in to_remove:
            self.filepaths.remove(filepath)
   

    def detect_language(self, file: Path) -> str:   
        """
        Detects the language of the input file.
        """
        from faster_whisper import WhisperModel

        logging.debug(f"Detecting language of file: {file.name}")        
        
        def run_language_detection():
            lang_detection_model = WhisperModel(
                models.set_supported_model(self.model_provided, implementation='faster-whisper'), 
                device='cpu' if self.device in ['mps', 'cpu'] else 'cuda', 
                compute_type='int8' if self.device in ['mps', 'cpu'] else 'float16'
                )
            _, info = lang_detection_model.transcribe(str(file), beam_size=5)
            return info
        
        info = little_helper.run_with_progress(
            description=f"[dark_goldenrod]→ Detecting language for {file.name}",
            task=run_language_detection                  
        )    
        
        # Set file_language
        self.file_language = info.language    

        print(f'[bold]→ Detected language "{info.language}" with probability {info.language_probability:.2f}')
        logging.debug(f'Detected language → "{info.language}" with probability {info.language_probability:.2f}')
        

    def process_files(self, files) -> None:
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
        """
        logging.info(f"Provided parameters for processing: {self.metadata}")

        # Get filepaths
        for file in files:
            self.get_filepaths(file)
            
        logging.info(f"Processing files: {self.filepaths}")

        self.processed_files = []
        for idx, filepath in enumerate(self.filepaths):            
            # Create and set output_dir and output_filepath
            self.output_dir = little_helper.set_output_dir(filepath, self.base_dir)
            output_filepath = self.output_dir / Path(filepath).stem
            
            # Convert file format 
            filepath = little_helper.check_file_format(filepath)
            
            # Detect file language
            if not self.file_language:
                self.detect_language(file=filepath)

            # Transcription and speaker annotation
            logging.info(f"Transcribing file: {filepath.name}")
            
            # If subtitles or speaker annotation use whisperX
            if self.subtitle or self.annotate:
                self.model = models.set_supported_model(self.model_provided, implementation='whisperx')
                print(f'[bold]→ Using {self.device.upper()} and whisper🆇  with model "{self.model}"')
                result_data = self.transcribe_with_whisperx(filepath)

            # Else use faster_whisper / insanely_fast_whisper depending on self.device
            else:
                if self.device == 'mps':
                    self.model = models.set_supported_model(self.model_provided, implementation='insane-whisper')
                    print(f'[bold]→ Using {self.device.upper()} and 🚅 Insanely-Fast-Whisper with model "{self.model}"')
                    result_data = self.transcribe_with_insane_whisper(filepath)
                    
                elif self.device in ['cpu', 'cuda:0']:
                    self.model = models.set_supported_model(self.model_provided, implementation='faster-whisper')
                    print(f'[bold]→ Using {self.device.upper()} and 🏃‍♀️‍➡️ Faster-Whisper with model "{self.model}"')
                    result_data = self.transcribe_with_faster_whisper(filepath)
            
            result = {
                'id': f'file_00{idx + 1}',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_filepath': str(filepath.absolute()),
                'output_filepath': str(Path(output_filepath).absolute()),
                'device': self.device,
                'model': self.model,
                'transcription': result_data['transcription']['transcriptions'],
            }

            # Save results
            little_helper.save_results(
                result=result, 
                subtitle=self.subtitle,
                annotate=self.annotate
                )
            
            self.processed_files.append(result)
            
            if not self.file_language_provided:
                self.file_language = None
