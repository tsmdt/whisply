import os
import typer
import warnings
from pathlib import Path
from typing import Optional, List
from rich import print
from whisply import output_utils
from whisply import post_correction as post
from whisply.output_utils import ExportFormats
from whisply.little_helper import DeviceChoice

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cli_app = typer.Typer(no_args_is_help=True)


@cli_app.command("run", no_args_is_help=True)
def run_cmd(
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
        help="Select your device: CPU, GPU (NVIDIA), or MPS (Mac M1-M4)",
    ),
    model: str = typer.Option(
        "large-v3-turbo",
        "--model",
        "-m",
        help='Whisper model to use (run "whisply list" to see options)',
    ),
    lang: Optional[str] = typer.Option(
        None,
        "--lang",
        "-l",
        help='Language of your file(s) ("en", "de") (Default: auto-detection)',
    ),
    annotate: bool = typer.Option(
        False,
        "--annotate",
        "-a",
        help="Enable speaker annotation (Saves .rttm | Default: False)",
    ),
    num_speakers: Optional[int] = typer.Option(
        None,
        "--num_speakers",
        "-num",
        help="Number of speakers to annotate (Default: auto-detection)",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf_token",
        "-hf",
        help="HuggingFace Access token required for speaker annotation",
    ),
    subtitle: bool = typer.Option(
        False,
        "--subtitle",
        "-s",
        help="Create subtitles (Saves .srt, .vtt & .webvtt | Default: False)",
    ),
    sub_length: int = typer.Option(
        5,
        "--sub_length",
        help="Subtitle segment length in words"
    ),
    translate: bool = typer.Option(
        False,
        "--translate",
        "-t",
        help="Translate transcription to English (Default: False)",
    ),
    export_format: ExportFormats = typer.Option(
        ExportFormats.ALL,
        "--export",
        "-e",
        help="Choose the export format"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print text chunks during transcription (Default: False)",
    ),
    del_originals: bool = typer.Option(
        False,
        "--del_originals",
        "-del",
        help="Delete input files after file conversion. (Default: False)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration file",
    ),
    post_correction: Optional[Path] = typer.Option(
        None,
        "--post_correction",
        "-post",
        help="Path to YAML file for post-correction",
    ),
):
    """
    💬 Transcribe files with whisply
    """
    from whisply import little_helper, models

    # Load configuration from config.json if provided
    if config:
        config_data = little_helper.load_config(config)
        files = (
            files or Path(config_data.get("files"))
            if config_data.get("files") else files
        )
        output_dir = (
            Path(config_data.get("output_dir"))
            if config_data.get("output_dir") else output_dir
        )
        device = DeviceChoice(config_data.get("device", device.value))
        model = config_data.get("model", model)
        lang = config_data.get("lang", lang)
        annotate = config_data.get("annotate", annotate)
        num_speakers = config_data.get(
            "num_speakers", num_speakers
        )
        translate = config_data.get("translate", translate)
        hf_token = config_data.get("hf_token", hf_token)
        subtitle = config_data.get("subtitle", subtitle)
        sub_length = config_data.get("sub_length", sub_length)
        verbose = config_data.get("verbose", verbose)
        del_originals = config_data.get("del_originals", del_originals)
        post_correction = config_data.get("post_correction", post_correction)

    # Check if provided model is available
    if not models.ensure_model(model):
        msg = f"""→ Model "{model}" not available.\n→ Available models:\n..."""
        msg += '\n... '.join(models.WHISPER_MODELS.keys())
        print(f"{msg}")
        raise typer.Exit()

    # Check for HuggingFace Access Token if speaker annotation is enabled
    if annotate and not hf_token:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print('→ Please provide a HuggingFace access token with option')
            print('  (--hf_token / -hf) to enable speaker annotation.')
            raise typer.Exit()

    # Determine the computation device
    device_str = little_helper.get_device(device=device)

    # Determine the ExportFormats
    export_formats = output_utils.determine_export_formats(
        export_format,
        annotate,
        subtitle
    )

    # Load corrections if post_correction is provided
    if post_correction:
        corrections = post.load_correction_list(post_correction)

    # Transcription
    if files:
        from whisply.transcription import TranscriptionHandler

        # Instantiate TranscriptionHandler
        service = TranscriptionHandler(
            base_dir=output_dir,
            device=device_str,
            model=model,
            file_language=lang,
            annotate=annotate,
            num_speakers=num_speakers,
            translate=translate,
            hf_token=hf_token,
            subtitle=subtitle,
            sub_length=sub_length,
            verbose=verbose,
            del_originals=del_originals,
            corrections=corrections if post_correction else None,
            export_formats=export_formats
        )
        # Process files
        service.process_files(files)
    else:
        print("[bold]→ Please provide a Path to a file, folder,")
        print("  URL or .list to start the transcription.")
        raise typer.Exit()


@cli_app.command("app")
def app_cmd():
    """
    🦜 Launch the whisply app
    """
    from whisply.app import main as run_gradio_app
    run_gradio_app()


@cli_app.command("list")
def list_cmd():
    """
    ⚙️ List available models (Note: some models might not be available for
    certain tasks like annotation or subtitling)
    """
    from whisply import models
    available_models = "[bold]Available models:\n... "
    available_models += '\n... '.join(models.WHISPER_MODELS.keys())
    print(f"{available_models}")


def run():
    cli_app()


if __name__ == "__main__":
    run()
