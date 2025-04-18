import re
import os
import json
import logging
import ffmpeg
import typer
import validators
import numpy as np

from enum import Enum
from pathlib import Path
from dotenv import set_key, load_dotenv
from typing import Callable, Any, List, Optional
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TextColumn, SpinnerColumn
from whisply.utils import download_utils, core_utils

# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)

VALID_ISO_CODES = {
  "aa": "Afar",
  "ab": "Abkhazian",
  "ae": "Avestan",
  "af": "Afrikaans",
  "ak": "Akan",
  "am": "Amharic",
  "an": "Aragonese",
  "ar": "Arabic",
  "as": "Assamese",
  "av": "Avaric",
  "ay": "Aymara",
  "az": "Azerbaijani",
  "ba": "Bashkir",
  "be": "Belarusian",
  "bg": "Bulgarian",
  "bi": "Bislama",
  "bm": "Bambara",
  "bn": "Bengali",
  "bo": "Tibetan",
  "br": "Breton",
  "bs": "Bosnian",
  "ca": "Catalan",
  "ce": "Chechen",
  "ch": "Chamorro",
  "co": "Corsican",
  "cr": "Cree",
  "cs": "Czech",
  "cu": "Church Slavic",
  "cv": "Chuvash",
  "cy": "Welsh",
  "da": "Danish",
  "de": "German",
  "dv": "Divehi",
  "dz": "Dzongkha",
  "ee": "Ewe",
  "el": "Greek",
  "en": "English",
  "eo": "Esperanto",
  "es": "Spanish",
  "et": "Estonian",
  "eu": "Basque",
  "fa": "Persian",
  "ff": "Fulah",
  "fi": "Finnish",
  "fj": "Fijian",
  "fo": "Faroese",
  "fr": "French",
  "fy": "Western Frisian",
  "ga": "Irish",
  "gd": "Gaelic",
  "gl": "Galician",
  "gn": "Guarani",
  "gu": "Gujarati",
  "gv": "Manx",
  "ha": "Hausa",
  "he": "Hebrew",
  "hi": "Hindi",
  "ho": "Hiri Motu",
  "hr": "Croatian",
  "ht": "Haitian",
  "hu": "Hungarian",
  "hy": "Armenian",
  "hz": "Herero",
  "ia": "Interlingua",
  "id": "Indonesian",
  "ie": "Interlingue",
  "ig": "Igbo",
  "ii": "Sichuan Yi",
  "ik": "Inupiaq",
  "io": "Ido",
  "is": "Icelandic",
  "it": "Italian",
  "iu": "Inuktitut",
  "ja": "Japanese",
  "jv": "Javanese",
  "ka": "Georgian",
  "kg": "Kongo",
  "ki": "Kikuyu",
  "kj": "Kuanyama",
  "kk": "Kazakh",
  "kl": "Kalaallisut",
  "km": "Central Khmer",
  "kn": "Kannada",
  "ko": "Korean",
  "kr": "Kanuri",
  "ks": "Kashmiri",
  "ku": "Kurdish",
  "kv": "Komi",
  "kw": "Cornish",
  "ky": "Kirghiz",
  "la": "Latin",
  "lb": "Luxembourgish",
  "lg": "Ganda",
  "li": "Limburgan",
  "ln": "Lingala",
  "lo": "Lao",
  "lt": "Lithuanian",
  "lu": "Luba-Katanga",
  "lv": "Latvian",
  "mg": "Malagasy",
  "mh": "Marshallese",
  "mi": "Maori",
  "mk": "Macedonian",
  "ml": "Malayalam",
  "mn": "Mongolian",
  "mr": "Marathi",
  "ms": "Malay",
  "mt": "Maltese",
  "my": "Burmese",
  "na": "Nauru",
  "nb": "Bokmål, Norwegian",
  "nd": "Ndebele, North",
  "ne": "Nepali",
  "ng": "Ndonga",
  "nl": "Dutch",
  "nn": "Norwegian Nynorsk",
  "no": "Norwegian",
  "nr": "Ndebele, South",
  "nv": "Navajo",
  "ny": "Chichewa",
  "oc": "Occitan",
  "oj": "Ojibwa",
  "om": "Oromo",
  "or": "Oriya",
  "os": "Ossetian",
  "pa": "Panjabi",
  "pi": "Pali",
  "pl": "Polish",
  "ps": "Pushto",
  "pt": "Portuguese",
  "qu": "Quechua",
  "rm": "Romansh",
  "rn": "Rundi",
  "ro": "Romanian",
  "ru": "Russian",
  "rw": "Kinyarwanda",
  "sa": "Sanskrit",
  "sc": "Sardinian",
  "sd": "Sindhi",
  "se": "Northern Sami",
  "sg": "Sango",
  "si": "Sinhala",
  "sk": "Slovak",
  "sl": "Slovenian",
  "sm": "Samoan",
  "sn": "Shona",
  "so": "Somali",
  "sq": "Albanian",
  "sr": "Serbian",
  "ss": "Swati",
  "st": "Sotho, Southern",
  "su": "Sundanese",
  "sv": "Swedish",
  "sw": "Swahili",
  "ta": "Tamil",
  "te": "Telugu",
  "tg": "Tajik",
  "th": "Thai",
  "ti": "Tigrinya",
  "tk": "Turkmen",
  "tl": "Tagalog",
  "tn": "Tswana",
  "to": "Tonga",
  "tr": "Turkish",
  "ts": "Tsonga",
  "tt": "Tatar",
  "tw": "Twi",
  "ty": "Tahitian",
  "ug": "Uighur",
  "uk": "Ukrainian",
  "ur": "Urdu",
  "uz": "Uzbek",
  "ve": "Venda",
  "vi": "Vietnamese",
  "vo": "Volapük",
  "wa": "Walloon",
  "wo": "Wolof",
  "xh": "Xhosa",
  "yi": "Yiddish",
  "yo": "Yoruba",
  "za": "Zhuang",
  "zh": "Chinese",
  "zu": "Zulu"
}


class DeviceChoice(str, Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'
    MPS = 'mps'


class LLMProviders(str, Enum):
    GOOGLE = 'google'
    # OPENAI = 'openai'


class FilePathProcessor:
    """
    Utility class for validating various filepaths.
    """
    def __init__(self, file_formats: List[str]):
        self.file_formats = [fmt.lower() for fmt in file_formats]
        self.filepaths: List[Path] = []

    def get_filepaths(self, filepath: str, download_urls: bool = True):
        """
        Processes the provided filepath which can be a URL, a single file,
        a directory,or a .list file containing multiple paths/URLs. It
        validates each input, downloads URLs if necessary, and accumulates
        valid file paths for further processing.
        """
        path = Path(filepath).expanduser().resolve()
        
        try:
            # Handle URL
            if validators.url(filepath):
                if download_urls:
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
                else:
                    self.filepaths.append(filepath)
                return  

            # Handle .list file
            elif path.suffix.lower() == '.list':
                if not path.is_file():
                    logging.error(
                        f'.list "{path}" does not exist or is not a file.'
                        )
                    print(
                        f'→ .list "{path}" does not exist or is not a file.'
                        )
                    return
                
                logging.info(f"Processing .list file: {path}")
                list_file_dir = path.parent
                with path.open('r', encoding='utf-8') as file:
                    lpaths = set()
                    for line in file:
                        lpath = line.strip()
                        if not lpath:
                            continue
                        lpaths.add(lpath)

                    for lpath in lpaths:
                        cleaned_lpath = lpath.replace('\\?', '?').replace('\\=', '=')

                        if validators.url(cleaned_lpath):
                            if download_urls:
                                logging.info(f"Processing URL from list: {cleaned_lpath}")
                                downloaded_path = download_utils.download_url(
                                    cleaned_lpath,
                                    downloads_dir=Path('./downloads')
                                )
                                if downloaded_path:
                                    self.filepaths.append(downloaded_path)
                                else:
                                    logging.error(f"Failed to download URL from list: {cleaned_lpath}")
                                    print(f'→ Failed to download URL: {cleaned_lpath}')
                            else:
                                logging.info(f"Skipping download for URL from list: {cleaned_lpath}")

                        else:
                            line_path = Path(lpath)
                            if not line_path.is_absolute():
                                absolute_lpath = (list_file_dir / line_path).resolve()
                            else:
                                absolute_lpath = line_path.resolve()

                            self._process_path(absolute_lpath)
                return

            # Handle single file or directory
            else:
                self._process_path(path)

        except Exception as e:
            logging.exception(f"Could not load file '{filepath}': {e}")
            return

        # Remove duplicates by converting to a set of resolved absolute paths
        unique_filepaths = set(p.resolve() for p in self.filepaths)
        self.filepaths = list(unique_filepaths)

        # Filter out files that have already been converted
        self._filter_converted_files()
        
        # Final check to ensure there are files to process
        if not self.filepaths:
            logging.warning(f'No valid files found for processing. \
Please check the provided path: "{filepath}".')
            print(f'→ No valid files found for processing. Please \
check the provided path: "{filepath}".')
        else:
            logging.info(f"Total files to process: {len(self.filepaths)}")

    def _process_path(self, path_input: str | Path):
        """
        Processes a single path input, which can be a file or a directory.
        """
        if isinstance(path_input, Path):
            path = path_input
        else:
            path = Path(path_input).expanduser().resolve()

        if path.is_file():
            if path.suffix.lower() in self.file_formats:
                logging.info(f"Adding file: {path}")
                normalized_path = self._normalize_filepath(path)
                self.filepaths.append(normalized_path)
            else:
                logging.warning(
                    f'"{path}" has unsupported format and will be skipped.'
                    )
                print(
                    f'→ "{path}" has unsupported format and will be skipped.'
                    )
        elif path.is_dir():
            logging.info(f"Processing directory: {path}")
            for file_format in self.file_formats:
                for file in path.rglob(f'*{file_format}'):
                    if file.is_file():
                        logging.debug(f"Found file: {file}")
                        normalized_path = self._normalize_filepath(file)
                        self.filepaths.append(normalized_path)
        else:
            logging.error(f'Path "{path}" does not exist / is not accessible.')
            print(f'→ Path "{path}" does not exist / is not accessible.')
             
    def _normalize_filepath(self, filepath: Path) -> Path:
        """
        Normalizes the filepath by replacing non-word characters with
        underscores, collapsing multiple underscores into one, and
        removing leading/trailing underscores.
        """
        new_filename = re.sub(r'\W+', '_', filepath.stem)
        new_filename = new_filename.strip('_')
        new_filename = re.sub(r'_+', '_', new_filename)
        
        suffix = filepath.suffix.lower()
        
        # Construct the new path
        new_path = filepath.parent / f"{new_filename}{suffix}"
        
        # Rename the file
        filepath.rename(new_path)

        return new_path.resolve()

    def _filter_converted_files(self):
        """
        Removes files that have already been converted to avoid redundant
        processing.
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
            if not (
                fp.with_name(fp.stem + converted_suffix) in converted_filepaths
                )
        ]
        
        # Extened filtered paths with converted paths
        filtered_filepaths.extend(converted_filepaths)

        removed_count = len(self.filepaths) - len(filtered_filepaths)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} files already converted.")
        self.filepaths = filtered_filepaths

def ensure_dir(dir: Path) -> None:
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir
        
def set_output_dir(filepath: Path, base_dir: Path) -> None:
    output_dir = base_dir / filepath.stem
    ensure_dir(output_dir)
    return output_dir

def set_and_validate_dotenv():
    """
    Set and validate .env config file.
    """
    dotenv_path = Path.cwd() / '.env'

    if not dotenv_path:
        print(f'→ No .env file found, creating one at: {dotenv_path}')
        try:
            dotenv_path.touch()
        except OSError as e:
            print(f"→ Error creating .env file: {e}")
            raise typer.Exit(code=1)
    else:
        dotenv_path = Path(dotenv_path)
    
    return dotenv_path

def update_dotenv_configuration(
        provider: core_utils.LLMProviders,
        model: Optional[str],
        api_key: Optional[str],
        show: bool = False
) -> bool:
    """
    Load and update .env config file to set the LLM configuration.
    Returns True if config was changed; False if it was not.
    """
    # Create config .env if it doesn't exist and load it
    dotenv_path = core_utils.set_and_validate_dotenv()

    # Load .env params
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Print config
    if show:
        print("[bold]... Current LLM Configuration:")
        active_provider = os.getenv("LLM_PROVIDER")

        if not active_provider:
            print("→ No active LLM provider set.")
            print("→ Set one using: [bold]whisply llm config <provider_name>[/bold]")
            raise typer.Exit()

        # Construct provider-specific keys
        provider_upper = active_provider.upper()
        model_key_name = f"LLM_MODEL_{provider_upper}"
        api_key_name = f"{provider_upper}_API_KEY"

        current_model = os.getenv(model_key_name)
        api_key_value = os.getenv(api_key_name)
        api_key_display = "Not set"
        if api_key_value:
            api_key_display = f"{api_key_value[:3]}...{api_key_value[-3:]}"

        print(f"[blue1]... Active Provider:[bold] {active_provider}")
        print(f"[blue1]... Model ({model_key_name}):[bold] {current_model or 'Not set'}")
        print(f"[blue1]... API Key ({api_key_name}):[bold] {api_key_display}")

        raise typer.Exit()

    # Get LLM provider
    provider_to_configure = None
    if provider:
        provider_to_configure = provider.value
    elif model or api_key:
        provider_to_configure = os.getenv("LLM_PROVIDER")
        if not provider_to_configure:
            print("→ Please specify a provider when setting a model or API key.")
            raise typer.Exit(code=1)
        print(f"→ Configuring settings for the active provider: '{provider_to_configure}'")

    # Set Config
    config_changed = False

    # 1. Set active provider
    if provider:
        key_name_active_provider = "LLM_PROVIDER"
        set_key(dotenv_path, key_name_active_provider, provider.value)
        print(f"→ Set active provider {key_name_active_provider}={provider.value}")
        config_changed = True

    # 2. Set model for the active provider
    if model:
        if not provider_to_configure:
            print("→ Error: Cannot set model without a target provider.")
            raise typer.Exit(code=1)
        key_name_model = f"LLM_MODEL_{provider_to_configure.upper()}"
        set_key(dotenv_path, key_name_model, model)
        print(f"→ Set model for {provider_to_configure}: {key_name_model}={model}")
        config_changed = True

    # 3. Set API key for the active provider
    if api_key:
        if not provider_to_configure:
            print("[bold red]Error:[/bold red] Cannot set API key without a LLM provider.")
            raise typer.Exit(code=1)
        key_name_api_key = f"{provider_to_configure.upper()}_API_KEY"
        set_key(dotenv_path, key_name_api_key, api_key)
        print(f"→ Set API key for {provider_to_configure}: {key_name_api_key}=[bold]{api_key[:3]}...{api_key[-3:]}[/]") # Masked confirmation
        config_changed = True

    return config_changed

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

def get_device(device: DeviceChoice = DeviceChoice.AUTO) -> str:
    """
    Determine the computation device based on user preference and 
    availability.
    """
    import torch
    
    if device == DeviceChoice.AUTO:
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device == DeviceChoice.GPU:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print(f"[blue1]→ NVIDIA GPU not available. Using CPU.")
            device = 'cpu'
    elif device == DeviceChoice.MPS:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            print(f"[blue1]→ MPS not available. Using CPU.")
            device = 'cpu'
    elif device == DeviceChoice.CPU:
        device = 'cpu'
    else:
        device = 'cpu'
    return device

def load_audio_ffmpeg(filepath: str) -> np.ndarray:
    try:
        out, _ = (
            ffmpeg
            .input(filepath)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16000')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error loading audio with ffmpeg: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.float32)

def check_file_format(
    filepath: Path, 
    del_originals: bool = True
    ) -> tuple[Path, np.ndarray]:
    """
    Checks the format of an audio file and converts it if it doesn't meet
    specified criteria. Then, loads the audio into a 1D NumPy array.

    The function uses `ffmpeg` to probe the metadata of an audio file at
    the given `filepath`. It checks if the audio stream meets the following
    criteria:
        - Codec name: 'pcm_s16le'
        - Sample rate: 16000 Hz
        - Number of channels: 1 (mono)

    If the audio stream does not meet these criteria, the function attempts
    to convert the file to meet the required format and saves the converted
    file with a '_converted.wav' suffix in the same directory. After successful
    conversion, it deletes the original file.

    Finally, it loads the audio (original or converted) as a 1D NumPy array
    and returns it.
    """     
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
            print(f"→ You may have provided an unsupported file type.\
Please check 'whisply --list_formats' for all supported formats.")
    
    try:
        audio_array = load_audio_ffmpeg(str(target_filepath))
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

def convert_seconds_to_hms(seconds: int, delimiter=',') -> str:
    """
    Function for formatting seconds into HH:MM:SS{,.}MS
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}{delimiter}{ms:03}"

def convert_hms_to_seconds(hms_string: str) -> float:
    """
    Converts an HH:MM:SS[.,:]MS string into total seconds.
    """
    hms_string = hms_string.strip()

    # Get index of last potential delimiter for ms
    last_comma_index = hms_string.rfind(',')
    last_period_index = hms_string.rfind('.')
    last_colon_index = hms_string.rfind(':')

    # Determine the actual delimiter index (the rightmost one )
    delimiter_idx = max(last_comma_index, last_period_index, last_colon_index)

    ms_str = '0'
    hms_part = hms_string

    # Check if any delimiter was found
    if delimiter_idx != -1:
        # Find the index of the second colon (M:S separator)
        first_colon_index = hms_string.find(':')
        second_colon_index = -1
        if first_colon_index != -1:
            second_colon_index = hms_string.find(':', first_colon_index + 1)

        # Divide HMS and MS part
        if second_colon_index != -1 and delimiter_idx > second_colon_index:
            hms_part = hms_string[:delimiter_idx]
            ms_str = hms_string[delimiter_idx + 1:]

            if ms_str and not ms_str.isdigit():
                raise ValueError(
                    f"Invalid format: MS part '{ms_str}' has non-digit chars."
                )
            # Cases with empty MS parts (e.g., "01:02:03:")
            if not ms_str:
                ms_str = '0'

    # Process the HH:MM:SS part
    time_parts = hms_part.split(':')
    if len(time_parts) != 3:
        raise ValueError(f"Invalid format: '{hms_part}' (derived from \
'{hms_string}') must be in HH:MM:SS format.")

    try:
        h = int(time_parts[0])
        m = int(time_parts[1])
        s = int(time_parts[2])
        ms = int(ms_str)

        # Limit MS to 3 digits
        ms_str = ms_str[:3]
        ms = int(ms_str) if ms_str else 0

        # Basic validation for time components
        if h < 0 or m < 0 or s < 0 or ms < 0:
             raise ValueError("Time components cannot be negative.")
        if m > 59 or s > 59:
             raise ValueError("Minutes or seconds exceed 59.")

        # Calculate total seconds
        total_seconds = (h * 3600.0) + (m * 60.0) + s + (ms / 1000.0)
        return total_seconds

    except ValueError as e:
        raise ValueError(f"Invalid hms_string format or value in '{hms_string}': {e}")
 
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
