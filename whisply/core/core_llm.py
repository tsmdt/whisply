import re
import typer
import logging
from datetime import datetime
from pathlib import Path
from rich import print
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List

from google import genai
from google.genai import types

from whisply.core import BaseService
from whisply.templates import prompts
from whisply.utils import core_utils, output_utils

@dataclass(kw_only=True)
class LLMService(BaseService):
    """
    Transcription service using cloud LLM providers.
    """
    provider: core_utils.LLMProviders
    api_key: Optional[str]
    temperature: float = 0.3
    client: Any = field(init=False, default=None)
    _api_key_internal: Optional[str] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """
        Set up logging, resolve api_keys and update metadata.
        """
        super().__post_init__()
        logging.info(f"Initializing LLM client for service: {self.provider.value}")

        # Check and resolve api_keys
        if self.api_key:
            self._api_key_internal = self._resolve_and_persist_api_key(
                cli_provided_key=self.api_key,
                env_var_name=f'{self.provider.value.upper()}_API_KEY', 
                service_name=self.provider.value.upper()
            )
            if not self._api_key_internal:
                print(f'→ {self.provider.value.upper()} API key required.')
                print('→ Please provide a token via --api_key / -k.')
                raise typer.Exit()
            
        # Instantiate the LLM client
        try:
            if self.provider.value == 'google':
                self.client = GoogleClient(api_key=self._api_key_internal)
            elif self.provider.value == 'openai':
                pass
        except Exception as e:
            logging.error(f"Failed to initialize {self.provider.value} client: {e}")
            print(f"[bold]→ Failed to initialize {self.provider.value} client.")
            raise typer.Exit(code=1)

        # Update metadata
        self.metadata.update({
            'provider': self.provider.value,
            'client': self.client
        })

    def validate_iso_code(self, llm_response: str) -> str | None:
        """
        Extracts the first standalone 2-letter lowercase string from the input
        and validates if it's a known ISO 639-1 language code.
        """
        match = re.search(r'\b([a-zA-Z]{2})\b', llm_response)
        if match:
            potential_code = match.group(1).lower()
            if potential_code in core_utils.VALID_ISO_CODES:
                self.file_language = potential_code

    def llm_response_to_transcription_dict(
            self,
            llm_response: str
        ) -> dict:
        """
        Convert a LLM transcription into a whisply transcription dict.
        """
        # Split response into textlines
        textlines = llm_response.split('\n')

        chunks = []
        annotation_text = ""

        # Pattern: [MM:SS:MS - MM:SS:MS] [Speaker Label] Transcript Segment
        mm_ss_ms = re.compile(r'\[(\d{2}:\d{2}:\d{1,3})\s*-\s*(\d{2}:\d{2}:\d{1,3})\]\s*\[(.*?)\]\s*(.*)')
        current_hour = 0
        previous_start_minute = -1

        for t in textlines:
            t = t.strip()

            if self.verbose:
                print(f'Processing transcript segment: {t}')

            if t and t.startswith('['):
                match = mm_ss_ms.match(t)
                if match:
                    # Extract the captured groups (MM:SS:MS format)
                    start_mm_ss_ms, end_mm_ss_ms, speaker, text = match.groups()
                    speaker = speaker.strip()
                    text = text.strip()

                    # Check for hour change (59:50:239 – 00:42:154)
                    try:
                        current_start_minute = int(start_mm_ss_ms.split(':')[0])

                        # Check for hour rollover (current minutes < previous minutes)
                        if current_start_minute < previous_start_minute:
                            current_hour += 1
                            print(f"Hour rollover detected. New hour: {current_hour}")

                        # Update previous minute for the next iteration
                        previous_start_minute = current_start_minute

                    except (ValueError, IndexError):
                         print(f"Warning: Could not parse minutes from start timestamp: {start_mm_ss_ms}")
                         pass

                    # Construct full timestamp in HH:MM:SS:MS pattern
                    start_hh_mm_ss_ms = f"{current_hour:02d}:{start_mm_ss_ms}"
                    end_hh_mm_ss_ms = f"{current_hour:02d}:{end_mm_ss_ms}"
                    try:
                        end_minute = int(end_mm_ss_ms.split(':')[0])
                        if end_minute < current_start_minute:
                             pass

                    except (ValueError, IndexError):
                         pass

                    # Add to annotation text using the reconstructed full timestamp
                    annotation_text += f"[{start_hh_mm_ss_ms}] [{speaker}] {text}\n"

                    try:
                        start_seconds = core_utils.convert_hms_to_seconds(start_hh_mm_ss_ms)
                        end_seconds = core_utils.convert_hms_to_seconds(end_hh_mm_ss_ms)

                        # Create chunks dict
                        chunk = ({
                            'timestamp': [start_seconds, end_seconds],
                            'text': text,
                            'speaker': speaker
                        })
                        chunks.append(chunk)

                        if self.verbose:
                            print(f'Successfully processed transcript segment: {chunk}')
                            
                    except Exception as e:
                        print(f"Error converting timestamp to seconds: {e} for line: {t}")

                else:
                    if re.match(r'\[\d{2}:\d{2}:\d{1,3}', t):
                         print(f"Warning: Could not parse line with regex: {t}")
                    else:
                         print(f"Info: Skipping non-timestamp line: {t}")
            elif t:
                 print(f"Info: Skipping non-timestamp line: {t}")


        # Build transcription dict
        result_dict = {
            'transcriptions': {
                self.file_language: {
                    'text': ' '.join(chunk['text'] for chunk in chunks),
                    'chunks': chunks,
                    'text_with_speaker_annotation': annotation_text
                }
            }
        }
        return result_dict
    
    def google_file_api_upload(self, filepath: Path):
        """
        Upload file via Google File API.
        """
        myfile = core_utils.run_with_progress(
            description=f"[deep_pink4]→ Uploading {filepath.name} via Google File API",
            task=lambda: self.client.api.files.upload(file=filepath)
        )
        return myfile
    
    def call_google_client(
            self, 
            myfile, 
            prompt: str, 
            description: str,
            return_dict: bool = False):
        """
        TODO
        """
        result = core_utils.run_with_progress(
            description=description,
            task=lambda: self.client.api.models.generate_content(
                model=self.model,
                contents=[prompt, myfile],
                config=types.GenerateContentConfig(
                    temperature=self.temperature
                    )
            )  
        )
        # Get only the text content
        result = result.text

        if return_dict:
            # Convert to transcription dict
            result_dict = self.llm_response_to_transcription_dict(
                llm_response=result
            )
            return result_dict
        else:
            return result

    def process_files(self, files) -> None:
        """
        Processes a list of audio files for transcription using LLM providers.
        """
        logging.info(f"Provided parameters for processing: {self.metadata}")

        # Get filepaths
        filepath_handler = core_utils.FilePathProcessor(self.file_formats)
        [filepath_handler.get_filepaths(f) for f in files]
        self.filepaths = filepath_handler.filepaths
        
        # Process filepaths
        logging.info(f"Processing files: {self.filepaths}")
        
        self.processed_files = []
        for idx, filepath in enumerate(self.filepaths):      
            # Create and set output_dir and output_filepath
            self.output_dir = core_utils.set_output_dir(filepath, self.base_dir)
            output_filepath = self.output_dir / Path(filepath).stem
            
            # Convert file format
            filepath, _ = core_utils.check_file_format(
                filepath=filepath,
                )
            logging.info(f"Transcribing file: {filepath.name}")  
                
            ### Google API ###
            if self.provider.value == 'google':
                # Upload file
                myfile = self.google_file_api_upload(filepath) 

                if myfile.state == 'ACTIVE':
                    # Detect file language
                    if not self.file_language:
                        result = self.call_google_client(
                            myfile,
                            prompt=prompts.DETECT_LANG,
                            description=f"[dark_goldenrod]→ Detecting language for [bold]{filepath.name}"
                        )
                        # Validate ISO language code
                        self.validate_iso_code(llm_response=result)
                        if not self.file_language:
                            self.file_language = 'en' # Default language
                    
                        print(f'[blue1]→ Detected language "{self.file_language}"')
                        logging.info(f'Detected language "{self.file_language}"')

                    # Transcription
                    result_data = self.call_google_client(
                            myfile,
                            prompt=prompts.TRANSCRIBE_ANNOTATE,
                            description=f"[purple]→ Transcribing + Annotating [bold]{filepath.name}[/] with [bold]{self.model}",
                            return_dict=True
                        )

                    # Build results
                    result = {
                        'id': f'file_00{idx + 1}',
                        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'input_filepath': str(filepath.absolute()),
                        'output_filepath': str(Path(output_filepath).absolute()),
                        'written_files': None,
                        'model': self.model,
                        'transcription': result_data['transcriptions'],
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


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMClientInitializationError(LLMClientError):
    """Error during client initialization (e.g., bad API key, missing package)."""
    pass

class LLMClientRequestError(LLMClientError):
    """Error during an API request (e.g., network issue, API error response)."""
    pass

class LLMClient(ABC):
    """
    Abstract Base Class for LLM API clients.
    """
    def __init__(self, api_key: Optional[str]):
        """
        Initializes the base client.
        """
        self._api_key: Optional[str] = api_key
        self._client: Any = None

        logging.debug(f"Initializing base client for {self.__class__.__name__}")

        try:
            self._initialize_client()
            if self._client is None:
                 raise LLMClientInitializationError(
                     f"{self.__class__.__name__}._initialize_client failed."
                 )
            logging.debug(f"Client for {self.__class__.__name__} initialized.")
        except Exception as e:
            logging.error(
                f"Failed during initialization of {self.__class__.__name__}: {e}",
                exc_info=True
                )
            typer.Exit()

    @property
    def api(self) -> Any:
        """
        Provides access to the underlying initialized provider client 
        library instance.
        """
        if self._client is None:
            raise LLMClientInitializationError("LLMClient Initialization failed.")
        return self._client

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Method to initialize the LLM provider client instance.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        Retrieve a list of available models by the LLM provider.
        """
        pass


class GoogleClient(LLMClient):
    """
    LLMClient implementation for Google Generative AI.
    """
    def _initialize_client(self) -> None:
        """
        Initializes the Google GenAI library.
        """
        if not self._api_key:
            raise LLMClientInitializationError(
                "Google GenAI API key is required but was not provided."
                )
        try:
            # Instantiate the client
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            logging.error("'google-genai' package not found.")
            raise LLMClientInitializationError(
                "'google-genai' not found. Please run 'pip install google-genai'."
            )
        except Exception as e:
            logging.error(f"Unexpected error configuring Google GenAI: {e}", exc_info=True)
            raise LLMClientInitializationError(f"Failed to configure Google GenAI: {e}") from e

    def list_models(self) -> List[str]:
        """
        Lists available Google GenAI models.
        """
        try:
            for model in self.api.models.list():
                print(f"... {model.name}")
        except Exception as e:
            print(f'Error: {e}')
