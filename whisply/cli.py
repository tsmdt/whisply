import click
import os 
from pathlib import Path


def get_device(device: str = 'auto', exclude_mps: bool = True):
    """
    Determine the computation device based on user preference and availability.
    
    Parameters:
    device (str): The computation device that will be checked for availability.
    exclude_mps (bool): Flag to exclude MPS device for certain transcription tasks
        that do not allow MPS backend (e.g. whisperX)
    """
    import torch

    if device == 'auto' and exclude_mps:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    elif device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        # elif torch.backends.mps.is_available():0
        #     device = 'mps'
        else:
            device = 'cpu'
    elif device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    # elif device == 'mps' and not exclude_mps:
    #     if torch.backends.mps.is_available():
    #         device = 'mps'
    #     else:
    #         device = 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        device = 'cpu'
    return device


@click.command(no_args_is_help=True)
@click.option('--files', '-f', type=click.Path(file_okay=True, dir_okay=True), help='Path to file, folder, URL or .list to process.')
@click.option('--output_dir', '-o', default='./transcriptions', type=click.Path(file_okay=False, dir_okay=True), 
              help='Folder where transcripts should be saved. Default: ./transcriptions')
@click.option('--device', '-d', default='auto', type=click.Choice(['auto', 'cpu', 'gpu'], case_sensitive=False), 
              help='Select the computation device: auto (default), CPU, GPU (NVIDIA CUDA), or MPS (Mac M1-M3).')
@click.option('--model', '-m', type=str, default='large-v2', 
              help='Whisper model to use (Default: "large-v2"). https://huggingface.co/collections/openai')
@click.option('--lang', '-l', type=str, default=None, 
              help='Language of provided file(s) ("en", "de") (Default: auto-detection).')
# @click.option('--annotate', '-a', default=None, type=click.Choice(['default', 'gat2']), 
#               help='Enable speaker annotation. Use "gat2" for Gat 2 guideline annotations. Creates .rttm')
@click.option('--annotate', '-a', default=False, is_flag=True, help='Enable speaker annotation. Creates .rttm')
@click.option('--hf_token', '-hf', type=str, default=None, help='HuggingFace Access token required for speaker annotation.')
@click.option('--translate', '-t', default=False, is_flag=True, help='Translate transcription to English.')
@click.option('--subtitle', '-s', default=False, is_flag=True, help='Create .srt and .webvtt subtitles.')
@click.option('--sub_length', default=5, type=int, help="""Subtitle block length in words (Default: 5);
              e.g. "10" produces subtitles with subtitle blocks of exactly 10 words.""")
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='Path to configuration file.')
@click.option('--filetypes', default=False, is_flag=True, help='List supported audio and video file types.')
@click.option('--verbose', default=False, is_flag=True, help='Print text chunks during transcription.')
def main(**kwargs):
    """
    WHISPLY 💬 Transcribe, translate, annotate and subtitle audio and video files with OpenAI's Whisper ... fast!
    """
    from whisply import little_helper, transcription

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

    # Check for HuggingFace Access Token if speaker annotation is enabled
    if kwargs['annotate'] and not kwargs['hf_token']:
        kwargs['hf_token'] = os.getenv('HF_TOKEN')
        if not kwargs['hf_token']:
            click.echo('→ Please provide a HuggingFace access token (--hf_token) to enable speaker annotation.')
            return 

    if kwargs['filetypes']:
        click.echo('\n'.join(little_helper.return_valid_fileformats()))
        return

    # Determine the computation device
    exclude_mps = kwargs['subtitle'] or kwargs['annotate']
    device = get_device(device=kwargs['device'], exclude_mps=exclude_mps)

    # Instantiate TranscriptionHandler
    service = transcription.TranscriptionHandler(base_dir=kwargs['output_dir'],
                                                 device=device,
                                                 model=kwargs['model'],
                                                 file_language=kwargs['lang'], 
                                                 annotate=kwargs['annotate'], 
                                                 translate=kwargs['translate'],
                                                 hf_token=kwargs['hf_token'], 
                                                 subtitle=kwargs['subtitle'],
                                                 sub_length=kwargs['sub_length'],
                                                 verbose=kwargs['verbose'])
    # Process files
    service.process_files(kwargs['files'])

if __name__ == '__main__':
    main()
