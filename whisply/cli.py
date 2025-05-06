import os
import typer
import warnings
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List
from rich import print
from whisply.utils import output_utils, core_utils

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Typer app
cli_app = typer.Typer(
    help="💬 Transcribe, translate, diarize, annotate and subtitle files with \
OpenAI's Whisper and multimodal LLMs on Win, Linux and Mac.",
    no_args_is_help=True
)

# Shared params for transcription services
def shared_params():
    return dict(
        files=typer.Option(
            None,
            "--files",
            "-f",
            help="Path to file, folder, URL or .list to process.",
        ),
        output_dir=typer.Option(
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
        device=typer.Option(
            core_utils.DeviceChoice.AUTO,
            "--device",
            "-d",
            help="Select AUTO, CPU, GPU (NVIDIA), or MPS (Mac M1-M4).",
        ),
        model=typer.Option(
            "large-v3-turbo",
            "--model",
            "-m",
            help='Whisper model to use (List models via --list_models).',
        ),
        lang=typer.Option(
            None,
            "--lang",
            "-l",
            help='Language of provided file(s) ("en", "de") (Default: auto-detection).',
        ),
        annotate=typer.Option(
            False,
            "--annotate",
            "-a",
            help="Enable speaker annotation (Saves .rttm | Default: False).",
        ),
        num_speakers=typer.Option(
            None,
            "--num_speakers",
            "-num",
            help="Number of speakers to annotate (Default: auto-detection).",
        ),
        hf_token=typer.Option(
            None,
            "--hf_token",
            "-hf",
            help="HuggingFace Access token required for speaker annotation.",
        ),
        subtitle=typer.Option(
            False,
            "--subtitle",
            "-s",
            help="Create subtitles (Saves .srt, .vtt and .webvtt | Default: False).",
        ),
        sub_length=typer.Option(
            5,
            "--sub_length",
            help="Subtitle segment length in words."
        ),
        translate=typer.Option(
            False,
            "--translate",
            "-t",
            help="Translate transcription to English (Default: False).",
        ),
        export_format=typer.Option(
            output_utils.ExportFormats.ALL,
            "--export",
            "-e",
            help="Choose the export format."
        ),
        verbose=typer.Option(
            False,
            "--verbose",
            "-v",
            help="Print text chunks during transcription (Default: False).",
        ),
        del_originals=typer.Option(
            False,
            "--del_originals",
            "-del",
            help="Delete original input files after file conversion. (Default: False)",
        ),
        post_correction=typer.Option(
            None,
            "--post_correction",
            "-post",
            help="Path to YAML file for post-correction.",
        ),
        list_models=typer.Option(
            False,
            "--list_models",
            help="List available models.",
        ),
    )


### Gradio App ###
@cli_app.command("app")
def app():
    """
    Start the whisply browser app.
    """
    try:
        from whisply.app import main
        print(f"[blue3]→ App startup")
        print(f"[blue3]→ Open the [bold]local URL[/] in your browser:")
        main.start_app()
        print(f"[blue3]→ App closed.")
    except ImportError:
        print('[dark_magenta]→ Please install necessary dependencies before \
running the app with: [bold]pip install "whisply\\[app]"[/bold]')
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[dark_magenta]→ Error during app startup: {e}")
        raise typer.Exit(code=1)
    

### Local Transcription ###
@cli_app.command("local", no_args_is_help=True)
def local(
    files: Optional[List[str]] = shared_params()["files"],
    output_dir: Optional[Path] = shared_params()["output_dir"],
    device: Optional[core_utils.DeviceChoice] = shared_params()["device"],
    model: Optional[str] = shared_params()["model"],
    lang: Optional[str] = shared_params()["lang"],
    annotate: Optional[bool] = shared_params()["annotate"],
    num_speakers: Optional[int] = shared_params()["num_speakers"],
    hf_token: Optional[str] = shared_params()["hf_token"],
    subtitle: Optional[bool] = shared_params()["subtitle"],
    sub_length: Optional[int] = shared_params()["sub_length"],
    translate: Optional[bool] = shared_params()["translate"],
    export_format: output_utils.ExportFormats = shared_params()["export_format"],
    verbose: Optional[bool] = shared_params()["verbose"],
    del_originals: Optional[bool] = shared_params()["del_originals"],
    post_correction: Optional[Path] = shared_params()["post_correction"],
    list_models: Optional[bool] = shared_params()["list_models"],
):
    """
    Transcribe files locally on your machine.
    """
    from whisply.models import models
    from whisply.utils import post_correction as post
    from whisply.core import LocalService

    # Print available models
    if list_models:
        try:
            available_models_dict = models.WHISPER_MODELS
            available_models_str = "Available models:\n... "
            available_models_str += '\n... '.join(available_models_dict.keys())
            print(f"{available_models_str}")
        except AttributeError:
            print("Error: Could not retrieve model list.")
        raise typer.Exit()

    # Check if provided model is available
    try:
        if not models.ensure_model(model):
            msg = f"""→ "{model}" not available.\n→ Available models:\n... """
            msg += '\n... '.join(models.WHISPER_MODELS.keys())
            print(f"{msg}")
            raise typer.Exit()
    except AttributeError:
        print(f"→ Error: Could not validate model '{model}'.")
        raise typer.Exit()

    # Determine the computation device
    device_str = core_utils.get_device(device=device)

    # Determine the ExportFormats
    export_formats = output_utils.determine_export_formats(
        export_format,
        annotate, 
        subtitle
        )

    # Load corrections if post_correction is provided
    corrections = None
    if post_correction:
        corrections = post.load_correction_list(post_correction)

    # Check if files were provided
    if files is None:
        print("[bold]→ Error: The '--files' option is required for transcription.")
        print("→ Usage: whisply local --files <path_or_url> [OPTIONS] ...")
        print("→ Or use --list_models to see models, or --launch_app to start the web UI.")
        raise typer.Exit(code=1)

    # Instantiate local transcription service
    try:
        service = LocalService(
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
            corrections=corrections,
            export_formats=export_formats
        )
    except Exception as e:
        print(f"→ Error initializing local transcription service: {e}")
        raise typer.Exit()

    # Process files
    try:
        service.process_files(files)
    except FileNotFoundError:
        print(f"→ Error: One or more input files/paths not found: {files}")
        raise typer.Exit()
    except Exception as e:
        print(f"→ An error occurred during file processing: {e}")
        raise typer.Exit()
    

### LLM Transcription ###
llm_app = typer.Typer(
    name="llm",
    help="Transcribe files using LLM providers.",
    no_args_is_help=True
    )
cli_app.add_typer(llm_app, name="llm")

@llm_app.command("config", no_args_is_help=True)
def config(
    provider: core_utils.LLMProviders = typer.Argument(
        None,
        help="Select the LLM service provider."
        ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use."
        ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api_key",
        "-k",
        help="API key for the LLM service provider."
        ),
    show: Optional[bool] = typer.Option(
        None,
        "--show",
        "-s",
        help="Show your current LLM configuration."
    ),
    list_models: Optional[bool] = shared_params()["list_models"]
    ):
    """
    Configure your LLM provider, model and api-key. 
    """
    try:
        from whisply.core import LLMService
    except ImportError:
        print('[dark_magenta]→ Please install necessary dependencies before \
running LLM transcription with: [bold]pip install "whisply\\[llm]"[/bold]')
        raise typer.Exit(code=1)

    # Load and update .env params
    config_changed = core_utils.update_dotenv_config(
        provider=provider,
        model=model,
        api_key=api_key,
        show=show
    )

    # List available models for the active provider
    if list_models:
        try:
            # Determine provider to list models for (argument > active .env)
            provider_for_list = provider.value if provider else os.getenv('LLM_PROVIDER')

            if not provider_for_list:
                print("→ No active LLM provider set.")
                print("→ Set one using: [bold]whisply llm config <provider_name>[/bold]")
                raise typer.Exit()

            # Get the API key for this specific provider
            api_key_env_name = f"{provider_for_list.upper()}_API_KEY"
            provider_api_key = os.getenv(api_key_env_name)

            if not provider_api_key:
                print(f"→ API key ({api_key_env_name}) not found in .env for provider '{provider_for_list}'.")
                print(f"→ Set it using: 'whisply llm config {provider_for_list} --api_key YOUR_KEY'")
                raise typer.Exit(code=1)

            # LLMService
            service = LLMService(
                provider=core_utils.LLMProviders(provider_for_list),
                api_key=provider_api_key,
                )
            service.client.list_models()
        except ValueError:
             valid_providers = [p.value for p in core_utils.LLMProviders]
             print(f"→ Invalid provider '{provider_for_list}'.")
             print(f"→ Available providers: {valid_providers}")
             raise typer.Exit(code=1)
        except AttributeError:
             print(f"→ Model listing not implemented or failed for provider '{provider_for_list}'.")
             raise typer.Exit(code=1)
        except Exception as e:
            print(f"→ Error retrieving provider models: {e}")
            raise typer.Exit(code=1)

        if not config_changed:
             raise typer.Exit()

    if not config_changed and not list_models:
         print("→ Use --show, --list-models, or provide config options.")
         raise typer.Exit()

@llm_app.command("run", no_args_is_help=True)
def run(
    files: Optional[List[str]] = shared_params()["files"],
    output_dir: Optional[Path] = shared_params()["output_dir"],
    lang: Optional[str] = shared_params()["lang"],
    annotate: Optional[bool] = shared_params()["annotate"],
    subtitle: Optional[bool] = shared_params()["subtitle"],
    translate: Optional[str] = typer.Option(
        None,
        "--translate",
        "-t",
        help="Translate to language (fr, de, es ...)."
        ),
    # translate: Optional[bool] = shared_params()["translate"],
    export_format: output_utils.ExportFormats = shared_params()["export_format"],
    verbose: Optional[bool] = shared_params()["verbose"]
    ):
    """
    Transcribe files using LLM providers.
    """
    try:
        from whisply.core import LLMService
    except ImportError:
        print('[dark_magenta]→ Please install necessary dependencies before \
running LLM transcription with: [bold]pip install "whisply\\[llm]"[/bold]')
        raise typer.Exit(code=1)

    # Load env
    dotenv_path = core_utils.set_and_validate_dotenv()
    load_dotenv(dotenv_path=dotenv_path, override=True)
    
    # Load and validate .env params
    if core_utils.validate_dotenv_config():
        # Get params
        provider = os.getenv('LLM_PROVIDER')
        api_key = os.getenv(f'{provider.upper()}_API_KEY')
        model = os.getenv(f'LLM_MODEL_{provider.upper()}')

        # Determine the ExportFormats
        export_formats = output_utils.determine_export_formats(
            export_format,
            annotate, 
            subtitle
            )
        
        # Check if language for translation is valid
        if translate:
            if not core_utils.VALID_ISO_CODES.get(translate):
                print(f'[bold]→ Please provide a valid language for translation: ')
                print(f"{', '.join(core_utils.VALID_ISO_CODES.keys())}")
                raise typer.Exit()
        
        # Instantiate LLM transcription service
        try:
            service = LLMService(
                base_dir=output_dir,
                provider=core_utils.LLMProviders(provider),
                model=model,
                api_key=api_key,
                file_language=lang,
                annotate=annotate,
                translate=translate,
                subtitle=subtitle,
                verbose=verbose,
                export_formats=export_formats
            )
        except Exception as e:
            print(f"→ Error initializing LLM transcription service: {e}")
            raise typer.Exit()
        
        # Process files
        # try:
        service.process_files(files)
        # except FileNotFoundError:
        #     print(f"→ Error: One or more input files/paths not found: {files}")
        #     raise typer.Exit()
        # except Exception as e:
        #     print(f"→ An error occurred during file processing: {e}")
        #     raise typer.Exit()
    else:
        print("[blue1]→ LLM service not configured. Use 'whisply llm config'")
        raise typer.Exit(code=1)

def main():
    cli_app()

if __name__ == "__main__":
    main()
