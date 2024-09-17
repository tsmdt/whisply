import logging
import torch
import time

from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Annotation
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
    
    # Define diarization function
    def diarize():
        diarization = diarization_pipeline(filepath)
        segments = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            temp = {"timestamp": [round(segment.start, 2), round(segment.end, 2)], "speaker": label}
            segments.append(temp)
        result['speaker_annotation'] = segments
        return segments, diarization
    
    # Add progress bar and run the diarization task
    segments, diarization = little_helper.run_with_progress(
        description=f"[orange_red1]Annotating Speakers ({device.upper()}) → {filepath.name[:20]}..{filepath.suffix}",
        task=diarize
    )
    
    # Add diarization to result dict
    result['speaker_annotation'] = segments
        
    # Stop timing diarization   
    logger.info(f"🗣️ Speaker detection (diarization) completed in {time.time() - d_start:.2f} sec.")

    return result, diarization


# def combine_transcription_with_speakers(result: dict) -> dict:
#     # Iterate through the language of the transcriptions results
#     for language in result['transcription']['transcriptions'].keys():
#         chunks = result['transcription']['transcriptions'][language]['chunks']

#         combined_data = []

#         # Iterate through all transcription chunks
#         for chunk in chunks:
#             chunk_start, chunk_end = chunk['timestamp']
#             chunk_text = chunk['text']

#             # Ensure chunk_start and chunk_end are not None
#             if chunk_start is None or chunk_end is None:
#                 logger.warning(f"Skipping chunk with None timestamp: {chunk}")
#                 continue

#             # Find the corresponding speaker(s) for the transcription chunk
#             speakers = []
#             for speaker_segment in result['transcription']['speaker_annotation']:
#                 speaker_start, speaker_end = speaker_segment['timestamp']
#                 speaker_id = speaker_segment['speaker']

#                 # Ensure speaker_start and speaker_end are not None
#                 if speaker_start is None or speaker_end is None:
#                     logger.warning(f"Skipping speaker segment with None timestamp: {speaker_segment}")
#                     continue

#                 # Check if the transcription chunk and speaker segment overlap
#                 if not (chunk_end < speaker_start or chunk_start > speaker_end):
#                     speakers.append(speaker_id)
            
#             # If a chunk overlaps multiple speakers, we can handle it by appending all relevant speakers
#             combined_data.append({
#                 'timestamp': chunk['timestamp'],
#                 'text': chunk_text,
#                 'speakers': speakers
#             })

#         # Append the combined transcription and speaker detection data
#         result['transcription']['transcription_and_speaker_annotation'] = {language: combined_data}

#     return result
