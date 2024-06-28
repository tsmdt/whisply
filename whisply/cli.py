import click
from pathlib import Path
from whisply import little_helper, transcription


@click.command(no_args_is_help=True)
@click.option('--files', type=click.Path(file_okay=True, dir_okay=True), help='Path to file, folder, URL or .list to process.')
@click.option('--output_dir', default='./transcriptions', type=click.Path(file_okay=False, dir_okay=True), 
              help='Folder where transcripts should be saved. Default: "./transcriptions"')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'gpu', 'mps'], case_sensitive=False), 
              help='Select the computation device: CPU, GPU (nvidia CUDA), or MPS (Metal Performance Shaders).')
@click.option('--lang', type=str, default=None, 
              help='Specifies the language of the file your providing (en, de, fr ...). Default: auto-detection)')
@click.option('--detect_speakers', default=False, is_flag=True, 
              help='Enable speaker diarization to identify and separate different speakers. Creates .rttm file.')
@click.option('--hf_token', type=str, default=None, help='HuggingFace Access token required for speaker diarization.')
@click.option('--translate', default=False, is_flag=True, help='Translate transcription to English.')
@click.option('--srt', default=False, is_flag=True, help='Create .srt subtitles from the transcription.')
@click.option('--webvtt', default=False, is_flag=True, help='Create .webvtt subtitles from the transcription.')
@click.option('--sub_length', default=None, type=int, help="""Maximum duration in seconds for each subtitle block (Default: auto);
              e.g. "10" produces subtitles where each individual subtitle block covers at least 10 seconds of the video.""")
@click.option('--txt', default=False, is_flag=True, help='Create .txt with the transcription.')
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), help='Path to configuration file.')
@click.option('--list_formats', default=False, is_flag=True, help='List supported audio and video formats.')
@click.option('--verbose', default=False, is_flag=True, help='Print text chunks during transcription.')
def main(**kwargs):
    """
    WHISPLY 🗿 Transcribe, translate, diarize, annotate and subtitle audio and video files with Whisper ... fast!
    """
    # Load configuration from config.json if provided
    if kwargs['config']:
        config_data = little_helper.load_config(Path(kwargs['config']))
        kwargs['files'] = kwargs['files'] or config_data.get('files')
        kwargs['output_dir'] = config_data.get('output_dir') if config_data.get('output_dir') is not None else kwargs['output_dir']
        kwargs['device'] = config_data.get('device', kwargs['device'])
        kwargs['lang'] = config_data.get('lang', kwargs['lang'])
        kwargs['detect_speakers'] = config_data.get('detect_speakers', kwargs['detect_speakers'])
        kwargs['translate'] = config_data.get('translate', kwargs['translate'])
        kwargs['hf_token'] = config_data.get('hf_token', kwargs['hf_token'])
        kwargs['txt'] = config_data.get('txt', kwargs['txt'])
        kwargs['srt'] = config_data.get('srt', kwargs['srt'])
        kwargs['webvtt'] = config_data.get('webvtt', kwargs['webvtt'])
        kwargs['sub_length'] = config_data.get('sub_length', kwargs['sub_length'])
        kwargs['verbose'] = config_data.get('verbose', kwargs['verbose'])

    # Check if speaker detection is enabled but no HuggingFace token is provided
    if kwargs['detect_speakers'] and not kwargs['hf_token']:
        click.echo('---> Speaker diarization is enabled but no HuggingFace access token is provided.')
        return 
    
    if kwargs['list_formats']:
        click.echo(f"{' '.join(transcription.TranscriptionHandler().file_formats)}")
        return

    # Instantiate TranscriptionHandler
    service = transcription.TranscriptionHandler(base_dir=kwargs['output_dir'],
                                                 device='cuda:0' if kwargs['device'] == 'gpu' else kwargs['device'],
                                                 file_language=kwargs['lang'], 
                                                 detect_speakers=kwargs['detect_speakers'], 
                                                 translate=kwargs['translate'],
                                                 hf_token=kwargs['hf_token'], 
                                                 txt=kwargs['txt'],
                                                 srt=kwargs['srt'],
                                                 webvtt=kwargs['webvtt'],
                                                 sub_length=kwargs['sub_length'],
                                                 verbose=kwargs['verbose'])
    # Process files
    service.process_files(kwargs['files'])

if __name__ == '__main__':
    main()
