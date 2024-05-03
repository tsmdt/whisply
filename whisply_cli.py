import click

from pathlib import Path
from whisply import little_helper, transcription


@click.command(no_args_is_help=True)
@click.option('--files', type=click.Path(), help='Path to file, folder, URL or .list to process.')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'gpu', 'mps'], case_sensitive=False), 
              help='Select the computation device: CPU, GPU (nvidia CUDA), or MPS (Metal Performance Shaders).')
@click.option('--lang', default=None, type=click.Choice(['en', 'fr', 'de'], case_sensitive=False), 
              help='Specify the language of the audio for transcription.')
@click.option('--detect_speakers', default=False, is_flag=True, 
              help='Enable speaker diarization to identify and separate different speakers.')
@click.option('--hf_token', type=str, default=None, help='HuggingFace Access token required for speaker diarization.')
@click.option('--srt', default=False, is_flag=True, help='Generate SRT subtitles from the transcription.')
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='Path to configuration file.')
def main(files, device, lang, detect_speakers, hf_token, srt, config):
    # Load configuration from config.json if provided
    if config:
        config_data = little_helper.load_config(Path(config))
        files = files or config_data.get('files')
        device = config_data.get('device', device)
        lang = config_data.get('lang', lang)
        detect_speakers = config_data.get('detect_speakers', detect_speakers)
        hf_token = config_data.get('hf_token', hf_token)
        srt = config_data.get('srt', srt)

    # Check if speaker detection is enabled but no HuggingFace token is provided
    if detect_speakers and not hf_token:
        click.echo('Error: Speaker diarization is enabled but no HuggingFace access token is provided.')
        return 

    # Instantiate TranscriptionHandler
    service = transcription.TranscriptionHandler(device='cuda:0' if device == 'gpu' else device,
                                                 language=lang, 
                                                 detect_speakers=detect_speakers, 
                                                 hf_token=hf_token, 
                                                 srt=srt)
    
    # Process files
    service.process_files(files)

if __name__ == '__main__':
    main()
