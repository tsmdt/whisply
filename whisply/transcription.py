import logging
import time
from pathlib import Path
from datetime import datetime
from functools import partial
from rich import print

from whisply import little_helper, output_utils, models
from whisply.little_helper import FilePathProcessor
from whisply.post_correction import Corrections

# Set logging configuration
logging.basicConfig(
    filename=f"log_whisply_{datetime.now().strftime('%Y-%m-%d')}.log", 
    level=logging.INFO, format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s'
    )

class TranscriptionHandler:
    """
    Handles transcription and diarization of audio/video files using various 
    Whisper-based models.

    This class leverages different implementations of OpenAI's Whisper models 
    (whisperX, insanely-fast-whisper, faster-whisper) to transcribe audio and 
    video files. It supports features like language detection, speaker 
    diarization, translation, subtitle generation, and exporting transcriptions
    in multiple formats. It is capable of processing single files, directories, 
    URLs, and lists of files, providing flexibility for diverse transcription 
    needs.

    Args:
        base_dir (str, optional): Directory to store transcription outputs. 
            Defaults to './transcriptions'.
        model (str, optional): Whisper model variant to use (e.g., 'large-v2'). 
            Defaults to 'large-v3-turbo'.
        device (str, optional): Compute device ('cpu', 'cuda', etc.). 
            Defaults to 'cpu'.
        file_language (str, optional): Language of the input audio. 
            If not provided, language detection is performed.
        annotate (bool, optional): Enable speaker diarization. 
            Defaults to False.
        hf_token (str, optional): Hugging Face token for accessing restricted  
            models or features.
        subtitle (bool, optional): Generate subtitles with word-level timestamps. 
            Defaults to False.
        sub_length (int, optional): Maximum number of words per subtitle chunk. 
            Required if subtitle is True.
        translate (bool, optional): Translate transcription to English if the 
            original language is different. Defaults to False.
        verbose (bool, optional): Enable detailed logging and output. 
            Defaults to False.
        export_formats (str or list, optional): Formats to export transcriptions 
            (e.g., 'json', 'srt'). Defaults to 'all'.

    Attributes:
        base_dir (Path): Directory for storing transcriptions.
        device (str): Compute device in use.
        file_language (str or None): Detected or specified language of the 
            audio.
        annotate (bool): Indicates if speaker diarization is enabled.
        translate (bool): Indicates if translation is enabled.
        subtitle (bool): Indicates if subtitle generation is enabled.
        verbose (bool): Indicates if verbose mode is active.
        export_formats (str or list): Selected formats for exporting 
            transcriptions.
        processed_files (list): List of processed file information and results.

    Methods:
        get_filepaths(filepath: str):
            Retrieves and validates file paths from various input types.
        
        detect_language(file: Path, audio_array) -> str:
            Detects the language of the given audio file.
        
        process_files(files: list):
            Processes a list of audio files for transcription and diarization.
        
        transcribe_with_whisperx(filepath: Path) -> dict:
            Transcribes an audio file using the whisperX implementation.
        
        transcribe_with_insane_whisper(filepath: Path) -> dict:
            Transcribes an audio file using the insanely-fast-whisper 
            implementation.
        
        transcribe_with_faster_whisper(filepath: Path, num_workers: int = 1) 
            -> dict:
            Transcribes an audio file using the faster-whisper implementation.
        
        adjust_word_chunk_length(result: dict) -> dict:
            Splits transcription text into chunks based on a maximum word 
            count.
        
        to_transcription_dict(insanely_annotation: list[dict]) -> dict:
            Converts speaker-annotated results into a standardized dictionary.
        
        to_whisperx(transcription_result: dict) -> dict:
            Normalizes transcription results to the whisperX format.
        
        create_text_with_speakers(transcription_dict: dict, 
            delimiter: str = '.') -> dict:
            Inserts speaker labels into the transcription text upon speaker 
            changes.
    """
    def __init__(
        self, 
        base_dir='./transcriptions', 
        model='large-v3-turbo', 
        device='cpu', 
        file_language=None, 
        annotate=False,
        num_speakers=None,
        hf_token=None, 
        subtitle=False, 
        sub_length=None, 
        translate=False, 
        verbose=False,
        del_originals=False,
        corrections=Corrections,
        export_formats='all'
    ):
        self.base_dir = little_helper.ensure_dir(Path(base_dir))
        self.file_formats = little_helper.return_valid_fileformats()
        self.device = device
        self.file_language = file_language
        self.file_language_provided = file_language is not None
        self.model = None
        self.model_provided = model
        self.annotate = annotate
        self.num_speakers = num_speakers
        self.translate = translate
        self.hf_token = hf_token
        self.subtitle = subtitle
        self.sub_length = sub_length
        self.verbose = verbose
        self.del_originals = del_originals
        self.corrections = corrections
        self.export_formats = export_formats
        self.metadata = self._collect_metadata()
        self.filepaths = []
        self.output_dir = None
        self.processed_files = []

    def _collect_metadata(self):
        return {
            'output_dir': str(self.base_dir),
            'file_language': self.file_language,
            'model': self.model_provided,
            'device': self.device,
            'annotate': self.annotate,
            'num_speakers': self.num_speakers,
            'translate': self.translate,
            'subtitle': self.subtitle,
            'sub_length': self.sub_length
            }
    
    def adjust_word_chunk_length(self, result: dict) -> dict:
        """
        Generates text chunks based on the maximum number of words.

        Parameters:
            result (dict): The nested dictionary containing segments 
            and words.
            max_number (int): The maximum number of words per chunk. 
            Default is 6.

        Returns:
            dict: A dictionary containing a list of chunks, each with 
            'text', 'timestamp', and 'words'.
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

    def to_transcription_dict(self, insanely_annotation: list[dict]) -> dict:
        """
        Transform insanely-fast-whisper speaker annotation result to dict.
        """
        chunks = []
        for s in insanely_annotation:
            chunk = {
                'text': s['text'],
                'timestamp': (s['timestamp'][0], s['timestamp'][1]),
                'speaker': s['speaker']
            }
            chunks.append(chunk)
            
        result = {
            'text': ''.join([s['text'] for s in insanely_annotation]),
            'chunks': chunks
        }
        return result

    def to_whisperx(self, transcription_result: dict) -> dict:
        """
        Normalize insanely-fast-whisper transcription result to whisperX dict.
        """
        words = []
        for c in transcription_result['chunks']:
            if 'speaker' in c:
                word = {
                    'word': c['text'].strip(),
                    'start': c['timestamp'][0],
                    'end': c['timestamp'][1],
                    'speaker': c['speaker']
                }
            else:
                word = {
                    'word': c['text'].strip(),
                    'start': c['timestamp'][0],
                    'end': c['timestamp'][1]
                }
            words.append(word)

        result = {
            'segments': [
                {
                    'start': transcription_result['chunks'][0]['timestamp'][0],
                    'end': transcription_result['chunks'][-1]['timestamp'][1],
                    'text': transcription_result['text'].strip(),
                    'words': words
                }
            ]
        }
        return result
    
    def create_text_with_speakers(
        self,
        transcription_dict: dict, 
        delimiter: str = '.'
        ) -> dict:
        """
        Iterates through all chunks of each language and creates the complete 
        text with speaker labels inserted when there is a speaker change.

        Args:
            transcription_dict (dict): The dictionary containing transcription 
            data.

        Returns:
            dict: A dictionary mapping each language to its formatted text with 
            speaker labels.
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
                    start_timestamp = little_helper.format_time(
                        word_info.get('start'), 
                        delimiter
                        )
                    
                    # Insert speaker label if a speaker change is detected
                    if speaker != current_speaker:
                        text += f"\n[{start_timestamp}] [{speaker}] "
                        current_speaker = speaker
                    
                    # Append the word with a space
                    text += word + " "
            
            transcription_dict['transcriptions'][lang]['text_with_speaker_annotation'] = text.strip()
        
        return transcription_dict

    def transcribe_with_whisperx(self, filepath: Path) -> dict:
        """
        Transcribe a file with the whisperX implementation that returns word-level
        timestamps and speaker annotation:https://github.com/m-bain/whisperX 
        
        This implementation is used when a specific subtitle length (e.g. 5 words 
        per individual subtitle) is needed.
        """
        import torch
        import whisperx
        import gc
        
        def empty_cuda_cache(model):
            gc.collect()
            torch.cuda.empty_cache()
            del model
        
        def fill_missing_timestamps(segments: list) -> list:
            """
            whisperX does not provide timestamps for words containing only 
            numbers (e.g. "1.5", "2024" etc.).
            
            The function fills these missing timestamps by padding the last 
            known 'end' timestamp for a missing 'start' timestamp or by cutting 
            the next known 'start' timestamp for a missing 'end' timestamp.
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
                        
                         # Case 1: If it's the first word, look forward for the 
                         # next speaker
                        if i == 0:
                            for j in range(i + 1, num_words):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        # Case 2: If it's the last word, look backward for the 
                        # previous speaker
                        elif i == num_words - 1:
                            for j in range(i - 1, -1, -1):
                                if 'speaker' in words[j]:
                                    word['speaker'] = words[j]['speaker']
                                    speaker_assigned = True
                                    break

                        # Case 3: For other words, prefer the previous speaker; 
                        # If not found, look forward
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
        
        def whisperx_task(task: str = 'transcribe', language = None):
            """
            Define a transcription / translation task with whisperX
            """
            # Set parameters
            device = 'cuda' if self.device == 'cuda:0' else 'cpu'
            
            # Transcribe or Translate
            model = whisperx.load_model(
                whisper_arch=self.model, 
                device=device,
                compute_type='float16' if self.device == 'cuda:0' else 'int8', 
                language=self.file_language or None,
                asr_options={
                    "hotwords": None,
                    "multilingual": False
                })
            audio = whisperx.load_audio(str(filepath), sr=16000)
            result = model.transcribe(
                audio, 
                batch_size=16 if self.device == 'cuda:0' else 8, 
                task=task
                )
            model_a, metadata = whisperx.load_align_model(
                device=device, 
                language_code=language
                )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                device,
                return_char_alignments=False
                )
                
            # Empty CUDA cache
            if self.device == 'cuda:0':
                empty_cuda_cache(model_a)
                    
            return result
        
        def whisperx_annotation(transcription_result: dict) -> dict:
            # Set parameters
            device = 'cuda' if self.device == 'cuda:0' else 'cpu'
            
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token, 
                device=device
                )
            diarize_segments = diarize_model(
                str(filepath),
                max_speakers=self.num_speakers
                )
            result = whisperx.assign_word_speakers(
                diarize_segments, 
                transcription_result
                )                  
            
            # Empty CUDA cache
            if self.device == 'cuda:0':
                empty_cuda_cache(diarize_model)
              
            return result
        
        # Start and time transcription
        logging.info(f"👨‍💻 Transcription started with whisper🆇 for {filepath.name}")
        t_start = time.time()
        
        # Run the transcription
        transcription_task = partial(
            whisperx_task, 
            task='transcribe', 
            language=self.file_language
            )
        transcription_result = little_helper.run_with_progress(
            description=f"[cyan]→ Transcribing ({'CUDA' if self.device == 'cuda:0' else 'CPU'}) [bold]{filepath.name}",
            task=transcription_task
        )
        
        # Speaker annotation
        if self.annotate:
            annotation_task = partial(
                whisperx_annotation, 
                transcription_result
                )
            transcription_result = little_helper.run_with_progress(
            description=f"[purple]→ Annotating ({self.device.upper()}) [bold]{filepath.name}",
            task=annotation_task
        )
        
        # Fill in missing timestamps and adjust word chunk length
        transcription_result['segments'] = fill_missing_timestamps(
            transcription_result['segments']
            )
        transcription_result = self.adjust_word_chunk_length(
            transcription_result
            )
        
        # Create result dict and append transcription to it
        result = {'transcriptions': {}}
        result['transcriptions'][self.file_language] = transcription_result
        
        if self.verbose:
            print(f"{result['transcriptions'][self.file_language]['text']}")
        
        # Translation task (to English)
        if self.translate and self.file_language != 'en':
            translation_task = partial(
                whisperx_task, 
                task='translate', 
                language='en'
                )
            translation_result = little_helper.run_with_progress(
                description=f"[dark_blue]→ Translating ({'CUDA' if self.device == 'cuda:0' else 'CPU'}) [bold]{filepath.name}",
                task=translation_task
            )
            
            # Speaker annotation
            if self.annotate:
                annotation_task = partial(
                    whisperx_annotation, 
                    translation_result
                    )
                translation_result = little_helper.run_with_progress(
                description=f"[purple]→ Annotating ({self.device.upper()}) [bold]{filepath.name}",
                task=annotation_task
            )
            
            # Fill in missing timestamps and adjust word chunk length
            translation_result['segments'] = fill_missing_timestamps(
                translation_result['segments']
                )
            translation_result = self.adjust_word_chunk_length(
                translation_result
                )
            result['transcriptions']['en'] = translation_result
            
            if self.verbose:
                print(f"{result['transcriptions']['en']['text']}")

        # Create full transcription with speaker annotation
        if self.annotate:
            result = self.create_text_with_speakers(result)
        
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}

    def transcribe_with_insane_whisper(self, filepath: Path) -> dict:
        """
        Transcribes a file using the 'insanely-fast-whisper' implementation:
        https://github.com/Vaibhavs10/insanely-fast-whisper

        This method utilizes the 'insanely-fast-whisper' implementation of 
        OpenAI Whisper for automatic speech recognition on Mac M1-M4 devices.

        Parameters:
        - filepath (Path): The path to the audio file for transcription.

        Returns:
        - dict: A dictionary containing the transcription result and, if 
                speaker detection is enabled, the speaker diarization result. 
                The transcription result includes the recognized text and 
                timestamps if available.
        """
        import torch
        from transformers import pipeline
        from transformers import logging as hf_logger
        from whisply import diarize_utils
        
        hf_logger.set_verbosity_error()
        
        def insane_whisper_annotation(transcription_result: dict) -> dict:
            # Speaker annotation
            annotation_result = diarize_utils.diarize(
                transcription_result,
                diarization_model='pyannote/speaker-diarization-3.1',
                hf_token=self.hf_token,
                file_name=str(filepath),
                num_speakers=self.num_speakers,
                min_speakers=None,
                max_speakers=None,
            )
            # Transform annotation_result to correct dict structure                
            transcription_result = self.to_transcription_dict(annotation_result)
            return transcription_result

        # Start and time transcription
        logging.info(
            f"👨‍💻 Transcription started with 🚅 insane-whisper for {filepath.name}"
            )
        t_start = time.time()
                
        try:                       
            pipe = pipeline(
                'automatic-speech-recognition',
                model = self.model,
                torch_dtype = torch.float16,
                device = self.device,
                model_kwargs = {
                    'attn_implementation': 'eager'
                    }
            )
            
            # Define transcription function            
            def transcription_task():                
                transcription_result = pipe(
                    str(filepath),
                    batch_size=1,
                    return_timestamps='word',
                    generate_kwargs={
                        'use_cache': True,
                        'return_legacy_cache': False,
                        'language': self.file_language,
                        'task': "transcribe",
                        'forced_decoder_ids': None
                    }
                )
                return transcription_result
             
            # Transcription
            transcription_result = little_helper.run_with_progress(
                description=f"[cyan]→ Transcribing ({self.device.upper()}) [bold]{filepath.name}",
                task=transcription_task
            )
            
            # Speaker annotation
            if self.annotate:
                transcription_result = little_helper.run_with_progress(
                    description=f"[purple]→ Annotating ({self.device.upper()}) [bold]{filepath.name}",
                    task=partial(
                        insane_whisper_annotation,
                        transcription_result
                    )
                )
                            
            # Adjust word chunk length
            transcription_result = self.to_whisperx(transcription_result)
            transcription_result = self.adjust_word_chunk_length(transcription_result)
            
            # Build result dict            
            result = {'transcriptions': {}}
            result['transcriptions'] = {
                self.file_language: transcription_result
            }
    
            if self.verbose:
                print(result['transcriptions'][self.file_language]['text'])
            
            # Translation
            if self.translate and self.file_language != 'en':
                def translation_task():
                    translation_result = pipe(
                        str(filepath),
                        batch_size=1,
                        return_timestamps='word',
                        generate_kwargs={
                            'use_cache': True,
                            'return_legacy_cache': False,
                            'task': 'translate',
                            # 'forced_decoder_ids': None
                        }
                    )
                    return translation_result
                
                # Run the translation task
                translation_result = little_helper.run_with_progress(
                    description=f"[dark_blue]→ Translating ({self.device.upper()}) [bold]{filepath.name}",
                    task=translation_task
                )
                
                # Speaker annotation
                if self.annotate:
                    translation_result = little_helper.run_with_progress(
                        description=f"[purple]→ Annotating ({self.device.upper()}) [bold]{filepath.name}",
                        task=partial(
                            insane_whisper_annotation,
                            translation_result
                        )
                    )
                
                # Adjust word chunk length
                translation_result = self.to_whisperx(translation_result)
                translation_result = self.adjust_word_chunk_length(translation_result)
                
                result['transcriptions']['en'] = translation_result
                
                if self.verbose:
                    print(result['transcriptions']['en']['text'])
                    
            if self.annotate:
                # Create full transcription with speaker annotation
                result = self.create_text_with_speakers(result)

        except Exception as e:
            print(f'{e}')
        
        # Stop timing transcription
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}

    def transcribe_with_faster_whisper(
        self, 
        filepath: Path, 
        num_workers: int = 1
        ) -> dict:
        """
        Transcribes an audio file using the 'faster-whisper' implementation: 
        https://github.com/SYSTRAN/faster-whisper

        This method utilizes the 'faster-whisper' implementation of OpenAI 
        Whisper for automatic speech recognition. It loads the model and sets 
        parameters for transcription. After transcription, it formats the 
        result into segments with timestamps and combines them into a single 
        text. If speaker detection is enabled, it also annotates speakers in 
        the transcription result.

        Parameters:
        - filepath (Path): The path to the audio file for transcription.
        - num_workers (int): The number of workers to use for transcription.

        Returns:
        - dict: A dictionary containing the transcription result and, if 
                speaker detection is enabled, the speaker diarization result. 
                The transcription result includes the recognized text and 
                segmented chunks with timestamps if available.
        """
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        
        # Start and time transcription
        logging.info(f"👨‍💻 Transcription started with 🏃‍♀️‍➡️ faster-whisper for {filepath.name}")
        t_start = time.time()
        
        # Load model and set parameters
        model = BatchedInferencePipeline(
            model = WhisperModel(
            self.model, 
            device='cpu' if self.device in ['mps', 'cpu'] else 'cuda', 
            num_workers=num_workers, 
            compute_type='int8' if self.device in ['mps', 'cpu'] else 'float16'
        ))
                
        # Define the transcription task
        def transcription_task():
            segments, _ = model.transcribe(
                str(filepath), 
                beam_size=5, 
                language=self.file_language,
                word_timestamps=True,
                batch_size=16
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
            description=f"[cyan]→ Transcribing ({self.device.upper()}) [bold]{filepath.name}",
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
                    word_timestamps=True
                    )
                
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
                description=f"[dark_blue]→ Translating ({self.device.upper()}) [bold]{filepath.name}",
                task=translation_task
            )

            # Add translation to result dict
            result['transcriptions']['en'] = {
                'text': ' '.join([segment['text'].strip() for segment in translation_chunks]),
                'chunks': translation_chunks
                }
        
        # Stop timing transcription
        logging.info(f"👨‍💻 Transcription completed in {time.time() - t_start:.2f} sec.")
        
        return {'transcription': result}
   
    def detect_language(self, filepath, audio_array) -> str:   
        """
        Detects the language of the input file.
        """
        from faster_whisper import WhisperModel
        
        logging.info(f"Detecting language of file: {filepath.name}")    
        
        def run_language_detection():
            lang_detection_model = WhisperModel(
                models.set_supported_model(
                    model=self.model_provided, 
                    implementation='faster-whisper',
                    translation=self.translate
                    ), 
                device='cpu' if self.device in ['mps', 'cpu'] else 'cuda', 
                compute_type='int8' if self.device in ['mps', 'cpu'] else 'float16'
                )
            lang, score, _ = lang_detection_model.detect_language(audio_array)
            return lang, score
        
        lang, score = little_helper.run_with_progress(
            description=f"[dark_goldenrod]→ Detecting language for [bold]{filepath.name}",
            task=run_language_detection                  
        )    
        
        self.file_language = lang   

        print(f'[blue1]→ Detected language "{lang}" with probability {score:.2f}')
        logging.info(f'Detected language → "{lang}" with probability {score:.2f}')
        
    def process_files(self, files) -> None:
        """
        Processes a list of audio files for transcription and/or diarization.

        This method logs the processing parameters, extracts filepaths from the
        input list, and initializes an empty list for storing results. Each 
        file is processed based on the compute device specified ('mps', 'cuda:0',
        or 'cpu'). Appropriate transcription method is chosen based on the 
        device. Results, including file ids, paths, transcriptions, and 
        diarizations, are stored in a dictionary and saved to a designated 
        output directory. Each result is also appended to `self.processed_files`.

        Parameters:
        files (list of str): A list of file paths or file-like objects 
        representing the audio files to be processed.
        """
        logging.info(f"Provided parameters for processing: {self.metadata}")
        
        # Get filepaths
        filepath_handler = FilePathProcessor(self.file_formats)
        [filepath_handler.get_filepaths(f) for f in files]            
        self.filepaths = filepath_handler.filepaths
        
        # Process filepaths
        logging.info(f"Processing files: {self.filepaths}")
        
        self.processed_files = []
        for idx, filepath in enumerate(self.filepaths):      
                  
            # Create and set output_dir and output_filepath
            self.output_dir = little_helper.set_output_dir(filepath, self.base_dir)
            output_filepath = self.output_dir / Path(filepath).stem
            
            # Convert file format 
            filepath, audio_array = little_helper.check_file_format(
                filepath=filepath,
                del_originals=self.del_originals
                )
            
            # Detect file language
            if not self.file_language:
                self.detect_language(filepath, audio_array)

            logging.info(f"Transcribing file: {filepath.name}")
            
            # Transcription and speaker annotation
            if self.device == 'mps':
                self.model = models.set_supported_model(
                    self.model_provided, 
                    implementation='insane-whisper',
                    translation=self.translate
                )
                print(f'[blue1]→ Using {self.device.upper()} and 🚅 Insanely-Fast-Whisper with model "{self.model}"')
                result_data = self.transcribe_with_insane_whisper(filepath)
            
            elif self.device in ['cpu', 'cuda:0']:
                if self.annotate or self.subtitle:
                    # WhisperX for annotation / subtitling
                    self.model = models.set_supported_model(
                        self.model_provided, 
                        implementation='whisperx',
                        translation=self.translate
                    )
                    print(f'[blue1]→ Using {self.device.upper()} and whisper🆇  with model "{self.model}"')
                    result_data = self.transcribe_with_whisperx(filepath)
                else:
                    # Faster-Whisper for raw transcription
                    self.model = models.set_supported_model(
                        self.model_provided, 
                        implementation='faster-whisper',
                        translation=self.translate
                    )
                    print(f'[blue1]→ Using {self.device.upper()} and 🏃‍♀️‍➡️ Faster-Whisper with model "{self.model}"')
                    result_data = self.transcribe_with_faster_whisper(filepath)

            result = {
                'id': f'file_00{idx + 1}',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_filepath': str(filepath.absolute()),
                'output_filepath': str(Path(output_filepath).absolute()),
                'written_files': None,
                'device': self.device,
                'model': self.model,
                'transcription': result_data['transcription']['transcriptions'],
            }

            # Save results
            result['written_files'] = output_utils.OutputWriter(
                corrections=self.corrections
                ).save_results(
                    result=result,
                    export_formats=self.export_formats
                    )
            
            self.processed_files.append(result)
            
            if not self.file_language_provided:
                self.file_language = None
