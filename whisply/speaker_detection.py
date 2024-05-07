import logging
import torch
import time

from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from whisply import little_helper


# Set logging configuration
logger = logging.getLogger('speaker_detection')
logger.setLevel(logging.DEBUG)


def annotate_speakers(filepath: Path, result: dict, device: str, hf_token: str) -> tuple[dict, Annotation]:
    """
    Performs speaker diarization on an audio or video file.

    This function detects and annotates speakers in an audio or video file using a pre-trained speaker
    diarization pipeline. It loads the pipeline, processes the file for speaker diarization, and appends
    the speaker annotations to the provided transcription result dictionary.

    Parameters:
    - filepath (Path): The path to the audio or video file for speaker diarization.
    - result (dict): The transcription result dictionary to which speaker annotations will be appended.
    - device (str): The device to use for speaker diarization ('cpu', 'cuda:0', etc.).
    - hf_token (str): Hugging Face API token for loading the diarization pipeline.

    Returns:
    - tuple[dict, Annotation]: A tuple containing the updated transcription result dictionary with
                              speaker annotations and the speaker diarization object.
    """
    logger.info("🗣️ Speaker detection (diarization) started ...")
    
    # Convert video to wav if necessary
    is_video = Path(filepath).suffix in ['.mkv', '.mp4', '.mov']
    if is_video:
        audio_path = Path(filepath).with_suffix('.wav')
        little_helper.convert_video_to_wav(videofile_path=Path(filepath), 
                                           output_audio_path=audio_path.as_posix())
        filepath = audio_path
    
    # Load and time the pipeline for diarization
    p_start = time.time()
    try:
        diarization_pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', 
                                                        use_auth_token=hf_token)
        if device in ['cuda:0', 'cuda']:
            diarization_pipeline = diarization_pipeline.to(torch.device('cuda'))
    except Exception as e:
        logger.error(f"Error loading diarization pipeline: {e}")
        raise RuntimeError("Failed to load diarization pipeline")
    logger.info(f"Diarization pipeline loaded in {time.time() - p_start:.2f} sec.")

    # Start diarization and time the process
    logger.info("Starting diarization process")
    d_start = time.time()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style = "bright_yellow", pulse_style = "bright_cyan"),
        TimeElapsedColumn()
    ) as progress:
        progress.add_task(f"[green]Annotating Speakers ({device.upper()}) → {filepath.name[:20]}..{filepath.suffix}", 
                            total = None)
        
        # Diarize audio
        diarization = diarization_pipeline(filepath)
            
        # Append annotations to results dict     
        segments = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            temp = {"timestamp": [round(segment.start, 2), round(segment.end, 2)], "speaker": label}
            segments.append(temp)
        result['speaker_annotation'] = segments
        
    # Stop timing diarization   
    logger.info(f"🗣️ Speaker detection (diarization) completed in {time.time() - d_start:.2f} sec.")
    
    # Delete temporary audio file
    if is_video:
        audio_path.unlink()
        
    return result, diarization
