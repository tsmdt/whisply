import click
import torch
from pathlib import Path
from whisply import little_helper, transcription


def get_device(device: str = 'auto', exclude_mps: bool = True):
    """
    Determine the computation device based on user preference and availability.
    
    Parameters:
    device (str): The computation device that will be checked for availability.
    exclude_mps (bool): Flag to exclude MPS device for certain transcription tasks
        that do not allow MPS backend (e.g. whisperX)
    """
    if device == 'auto' and exclude_mps:
        if torch.cuda.is_available():
            device = 'cuda:0'
            click.echo('→ Using GPU (CUDA)')
        else:
            device = 'cpu'
            click.echo('→ No GPU (CUDA) available, using CPU')
    elif device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            click.echo('→ Using GPU (CUDA)')
        elif torch.backends.mps.is_available():
            device = 'mps'
            click.echo('→ Using MPS')
        else:
            device = 'cpu'
            click.echo('→ No GPU (CUDA) or MPS available, using CPU')
    elif device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
            click.echo('→ Using GPU (CUDA)')
        else:
            device = 'cpu'
            click.echo('→ No GPU (CUDA) available, using CPU')
    elif device == 'mps' and not exclude_mps:
        if torch.backends.mps.is_available():
            device = 'mps'
            click.echo('→ Using MPS')
        else:
            device = 'cpu'
            click.echo('→ MPS not available, using CPU')
    elif device == 'cpu':
        device = 'cpu'
        click.echo('→ Using CPU')
    else:
        device = 'cpu'
        click.echo('→ No GPU (CUDA) or MPS available, using CPU')
    return device


@click.command(no_args_is_help=True)
@click.option('--files', type=click.Path(file_okay=True, dir_okay=True), help='Path to file, folder, URL or .list to process.')
@click.option('--output_dir', default='./transcriptions', type=click.Path(file_okay=False, dir_okay=True), 
              help='Folder where transcripts should be saved. Default: "./transcriptions".')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'gpu', 'mps'], case_sensitive=False), 
              help='Select the computation device: auto (default), CPU, GPU (NVIDIA CUDA), or MPS (Mac M1-M3).')
@click.option('--model', type=str, default='large-v2', 
              help='Select the whisper model to use (Default: large-v2). Refers to whisper model size: https://huggingface.co/collections/openai')
@click.option('--lang', type=str, default=None, 
              help='Specifies the language of the file your providing (en, de, fr ... Default: auto-detection).')
@click.option('--annotate', default=False, is_flag=True, 
              help='Enable speaker detection to identify and annotate different speakers. Creates .rttm file.')
@click.option('--hf_token', type=str, default=None, help='HuggingFace Access token required for speaker detection.')
@click.option('--translate', default=False, is_flag=True, help='Translate transcription to English.')
@click.option('--subtitle', default=False, is_flag=True, help='Create .srt and .webvtt subtitles from the transcription.')
@click.option('--sub_length', default=5, type=int, help="""Subtitle length in words for each subtitle block (Default: 5);
              e.g. "10" produces subtitles where each individual subtitle block covers exactly 10 words.""")
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='Path to configuration file.')
@click.option('--filetypes', default=False, is_flag=True, help='List supported audio and video file types.')
@click.option('--verbose', default=False, is_flag=True, help='Print text chunks during transcription.')
def main(**kwargs):
    """
    WHISPLY 💬 Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!
    """
    # Load configuration from config.json if provided
    if kwargs['config']:
        config_data = little_helper.load_config(Path(kwargs['config']))
        kwargs['files'] = kwargs['files'] or config_data.get('files')
        kwargs['output_dir'] = config_data.get('output_dir') if config_data.get('output_dir') is not None else kwargs['output_dir']
        kwargs['device'] = config_data.get('device', kwargs['device'])
        kwargs['model'] = config_data.get('model', kwargs['model'])
        kwargs['lang'] = config_data.get('lang', kwargs['lang'])
        kwargs['annotate'] = config_data.get('annotate', kwargs['annotate'])
        kwargs['translate'] = config_data.get('translate', kwargs['translate'])
        kwargs['hf_token'] = config_data.get('hf_token', kwargs['hf_token'])
        kwargs['subtitle'] = config_data.get('subtitle', kwargs['subtitle'])
        kwargs['sub_length'] = config_data.get('sub_length', kwargs['sub_length'])
        kwargs['verbose'] = config_data.get('verbose', kwargs['verbose'])

    # Check if speaker detection is enabled but no HuggingFace token is provided
    if kwargs['annotate'] and not kwargs['hf_token']:
        click.echo('→ Please provide a HuggingFace access token (--hf_token) to enable speaker annotation.')
        return 
    
    if kwargs['filetypes']:
        click.echo(f"{' '.join(transcription.TranscriptionHandler().file_formats)}")
        return

    # Determine the computation device
    exclude_mps = kwargs['subtitle'] or kwargs['annotate']
    device = get_device(device=kwargs['device'], exclude_mps=exclude_mps)

    # Instantiate TranscriptionHandler
    service = transcription.TranscriptionHandler(base_dir=kwargs['output_dir'],
                                                 device=device,
                                                 model=kwargs['model'],
                                                 file_language=kwargs['lang'], 
                                                 detect_speakers=kwargs['annotate'], 
                                                 translate=kwargs['translate'],
                                                 hf_token=kwargs['hf_token'], 
                                                 subtitle=kwargs['subtitle'],
                                                 sub_length=kwargs['sub_length'],
                                                 verbose=kwargs['verbose'])
    # Process files
    service.process_files(kwargs['files'])

if __name__ == '__main__':
    main()
