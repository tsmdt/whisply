import os
import typer

from pathlib import Path
from enum import Enum
from typing import Optional, List
from rich import print

app = typer.Typer()


class DeviceChoice(str, Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'
    MPS = 'mps'


def get_device(device: DeviceChoice = DeviceChoice.AUTO, exclude_mps: bool = True) -> str:
    """
    Determine the computation device based on user preference and availability.
    
    Parameters:
    device (DeviceChoice): The computation device that will be checked for availability.
    exclude_mps (bool): Flag to exclude MPS device for certain transcription tasks
        that do not allow MPS backend (e.g., whisperX)
    """
    import torch

    if device == DeviceChoice.AUTO and exclude_mps:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    elif device == DeviceChoice.AUTO:
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
    elif device == DeviceChoice.MPS and not exclude_mps:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device == DeviceChoice.CPU:
        device = 'cpu'
    else:
        device = 'cpu'
    return device


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
        help="Select the computation device: CPU, GPU (NVIDIA), or MPS (Mac M1-M3).",
    ),
    model: str = typer.Option(
        "large-v2",
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
        help="Subtitle segment length in words"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print text chunks during transcription.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration file.",
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

    # Print supported filetypes 
    if list_filetypes:
        supported_filetypes = "Supported filetypes: "
        supported_filetypes += ' '.join(little_helper.return_valid_fileformats())
        print(f"[bold]{supported_filetypes}")
        raise typer.Exit()

    # Print available models
    if list_models:
        available_models = "Available models: "
        available_models += ', '.join(models.WHISPER_MODELS.keys())
        print(f"[bold]{available_models}")
        raise typer.Exit()

    # Check if provided model is available
    if not models.ensure_model(model):
        msg = f"""[bold]→ Model "{model}" is not available.\n→ Available models: """
        msg += ', '.join(models.WHISPER_MODELS.keys())
        print(f"[bold]{msg}")
        raise typer.Exit(code=1)

    # Check for HuggingFace Access Token if speaker annotation is enabled
    if annotate and not hf_token:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print('[bold]→ Please provide a HuggingFace access token (--hf_token / -hf) to enable speaker annotation.')
            raise typer.Exit(code=1)

    # Determine the computation device
    exclude_mps = subtitle or annotate
    device_str = get_device(device=device, exclude_mps=exclude_mps)

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
        verbose=verbose
    )
    # Process files
    service.process_files(files)


def run():
    app()


if __name__ == "__main__":
    run()
