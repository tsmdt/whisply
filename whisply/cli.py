import os
import typer
import warnings
import yaml

from pathlib import Path
from enum import Enum
from typing import Optional, List
from rich import print
from whisply.post_correction import Corrections

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer()

class DeviceChoice(str, Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'
    MPS = 'mps'
    
class ExportFormats(str, Enum):
    ALL = 'all'
    JSON = 'json'
    TXT = 'txt'
    RTTM = 'rttm'
    VTT = 'vtt'
    WEBVTT = 'webvtt'
    SRT = 'srt'

def get_device(device: DeviceChoice = DeviceChoice.AUTO) -> str:
    """
    Determine the computation device based on user preference and availability.
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
            device = 'cpu'
    elif device == DeviceChoice.MPS:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device == DeviceChoice.CPU:
        device = 'cpu'
    else:
        device = 'cpu'
    return device

def determine_export_formats(
    export_format: ExportFormats,
    annotate: bool,
    subtitle: bool
) -> List[str]:
    """
    Determine the export formats based on user options and availability.

    Returns a list of export format strings to be used.
    """
    available_formats = set()
    if export_format == ExportFormats.ALL:
        available_formats.add(ExportFormats.JSON.value)
        available_formats.add(ExportFormats.TXT.value)
        if annotate:
            available_formats.add(ExportFormats.RTTM.value)
        if subtitle:
            available_formats.add(ExportFormats.WEBVTT.value)
            available_formats.add(ExportFormats.VTT.value)
            available_formats.add(ExportFormats.SRT.value)
    else:
        if export_format in (ExportFormats.JSON, ExportFormats.TXT):
            available_formats.add(export_format.value)
        elif export_format == ExportFormats.RTTM:
            if annotate:
                available_formats.add(export_format.value)
            else:
                print("→ RTTM export format requires annotate option to be True.")
                raise typer.Exit()
        elif export_format in (
            ExportFormats.VTT,
            ExportFormats.SRT,
            ExportFormats.WEBVTT
            ):
            if subtitle:
                available_formats.add(export_format.value)
            else:
                print(f"→ {export_format.value.upper()} export format requires subtitle option to be True.")
                raise typer.Exit()
        else:
            print(f"→ Unknown export format: {export_format.value}")
            raise typer.Exit()

    return list(available_formats)

def load_correction_list(filepath: str | Path) -> Corrections:
    """
    Load the correction dictionary and patterns from a YAML file.

    :param filepath: Path to the YAML correction file.
    :return: Corrections object containing simple and pattern-based corrections.
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError("→ Correction file must contain a YAML dictionary.")

        # Extract simple corrections
        simple_corrections = {k: v for k, v in data.items() if k != 'patterns'}

        # Extract pattern-based corrections
        pattern_corrections = data.get('patterns', [])

        # Validate patterns
        for entry in pattern_corrections:
            if 'pattern' not in entry or 'replacement' not in entry:
                raise ValueError("→ Each pattern entry must contain 'pattern' and 'replacement' keys.")

        return Corrections(simple=simple_corrections, patterns=pattern_corrections)

    except FileNotFoundError:
        print(f"→ Correction file not found: {filepath}")
        return Corrections()
    except yaml.YAMLError as e:
        print(f"→ Error parsing YAML file: {e}")
        return Corrections()
    except Exception as e:
        print(f"→ Unexpected error loading correction list: {e}")
        return Corrections()

@app.command(no_args_is_help=True)
def main(
    files: Optional[List[str]] = typer.Option(
        None,
        "--files",
        "-f",
        help="Path to file, folder, URL or .list to process.",
    ),
    output_dir: Path = typer.Option(
        Path("./transcriptions"),
        "--output_dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help="Folder where transcripts should be saved.",
    ),
    device: DeviceChoice = typer.Option(
        DeviceChoice.AUTO,
        "--device",
        "-d",
        help="Select the computation device: CPU, GPU (NVIDIA), or MPS (Mac M1-M4).",
    ),
    model: str = typer.Option(
        "large-v3-turbo",
        "--model",
        "-m",
        help='Whisper model to use (List models via --list_models).',
    ),
    lang: Optional[str] = typer.Option(
        None,
        "--lang",
        "-l",
        help='Language of provided file(s) ("en", "de") (Default: auto-detection).',
    ),
    annotate: bool = typer.Option(
        False,
        "--annotate",
        "-a",
        help="Enable speaker annotation (Saves .rttm).",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf_token",
        "-hf",
        help="HuggingFace Access token required for speaker annotation.",
    ),
    translate: bool = typer.Option(
        False,
        "--translate",
        "-t",
        help="Translate transcription to English.",
    ),
    subtitle: bool = typer.Option(
        False,
        "--subtitle",
        "-s",
        help="Create subtitles (Saves .srt, .vtt and .webvtt).",
    ),
    sub_length: int = typer.Option(
        5,
        "--sub_length",
        help="Subtitle segment length in words."
    ),
    export_format: ExportFormats = typer.Option(
        ExportFormats.ALL,
        "--export",
        "-e",
        help="Choose the export format."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print text chunks during transcription.",
    ),
    del_originals: bool = typer.Option(
        False,
        "--del_originals",
        "-del",
        help="Delete original input files after file conversion. (Default: False)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration file.",
    ),
    post_correction: Optional[Path] = typer.Option(
        None,
        "--post_correction",
        "-post",
        help="Path to YAML file for post-correction.",
    ),
    list_filetypes: bool = typer.Option(
        False,
        "--list_filetypes",
        help="List supported audio and video file types.",
    ),
    list_models: bool = typer.Option(
        False,
        "--list_models",
        help="List available models.",
    ),
):
    """
    WHISPLY 💬 Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!
    """
    from whisply import little_helper, transcription, models

    # Load configuration from config.json if provided
    if config:
        config_data = little_helper.load_config(config)
        files = files or Path(config_data.get("files")) if config_data.get("files") else files
        output_dir = Path(config_data.get("output_dir")) if config_data.get("output_dir") else output_dir
        device = DeviceChoice(config_data.get("device", device.value))
        model = config_data.get("model", model)
        lang = config_data.get("lang", lang)
        annotate = config_data.get("annotate", annotate)
        translate = config_data.get("translate", translate)
        hf_token = config_data.get("hf_token", hf_token)
        subtitle = config_data.get("subtitle", subtitle)
        sub_length = config_data.get("sub_length", sub_length)
        verbose = config_data.get("verbose", verbose)
        del_originals = config_data.get("del_originals", del_originals)
        post_correction = config_data.get("post_correction", post_correction)

    # Print supported filetypes 
    if list_filetypes:
        supported_filetypes = "Supported filetypes: "
        supported_filetypes += ' '.join(little_helper.return_valid_fileformats())
        print(f"{supported_filetypes}")
        raise typer.Exit()

    # Print available models
    if list_models:
        available_models = "Available models:\n... "
        available_models += '\n... '.join(models.WHISPER_MODELS.keys())
        print(f"{available_models}")
        raise typer.Exit()

    # Check if provided model is available
    if not models.ensure_model(model):
        msg = f"""→ Model "{model}" is not available.\n→ Available models:\n... """
        msg += '\n... '.join(models.WHISPER_MODELS.keys())
        print(f"{msg}")
        raise typer.Exit()

    # Check for HuggingFace Access Token if speaker annotation is enabled
    if annotate and not hf_token:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print('→ Please provide a HuggingFace access token (--hf_token / -hf) to enable speaker annotation.')
            raise typer.Exit()

    # Determine the computation device
    device_str = get_device(device=device)
    
    # Determine the ExportFormats
    export_formats = determine_export_formats(export_format, annotate, subtitle)
    
    # Load corrections if post_correction is provided
    corrections = Corrections()
    if post_correction:
        corrections = load_correction_list(post_correction)
    
    # Instantiate TranscriptionHandler
    service = transcription.TranscriptionHandler(
        base_dir=output_dir,
        device=device_str,
        model=model,
        file_language=lang, 
        annotate=annotate, 
        translate=translate,
        hf_token=hf_token, 
        subtitle=subtitle,
        sub_length=sub_length,
        verbose=verbose,
        del_originals=del_originals,
        corrections=corrections,
        export_formats=export_formats
    )
    # Process files
    service.process_files(files)

def run():
    app()

if __name__ == "__main__":
    run()
