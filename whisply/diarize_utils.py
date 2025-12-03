import requests
import torch
import numpy as np
from torchaudio import functional as F
from pyannote.audio import Pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

# Code lifted from
# https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
# and from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's
            # impossible to use a local file like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets`
        # for better integration
        if not (
            "sampling_rate" in inputs
            and ("raw" in inputs or "array" in inputs)
        ):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, "
                "the dict needs to contain a "
                '"raw" key containing the numpy array representing '
                'the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with "
                "that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(
            f"We expect a numpy ndarray as input, got `{type(inputs)}`"
        )
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of
    # shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(
    diarizer_inputs,
    diarization_pipeline,
    num_speakers,
    min_speakers,
    max_speakers
) -> dict:
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # diarizer output may contain consecutive segments from the same
    # speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each
    # speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments


def post_process_segments_and_transcripts(
    new_segments,
    transcript,
    group_by_speaker
) -> list:
    segmented_preds = []
    transcript_idx = 0
    num_chunks = len(transcript)

    # Iterate through each diarization segment and assign transcript chunks
    # whose end timestamp falls within the segment
    for segment in new_segments:
        # seg_start = segment["segment"]["start"]
        seg_end = segment["segment"]["end"]
        segment_chunks = []

        # Collect transcript chunks until the chunk's end timestamp exceeds
        # the diarization segment's end
        while (
            transcript_idx < num_chunks
            and
            transcript[transcript_idx]["timestamp"][1] <= seg_end
        ):
            segment_chunks.append(transcript[transcript_idx])
            transcript_idx += 1

        # If no transcript chunks were found for this segment, continue
        # to next segment
        if not segment_chunks:
            continue

        if group_by_speaker:
            # Combine the text from all transcript chunks within this segment
            text = "".join(chunk["text"] for chunk in segment_chunks)
            segmented_preds.append({
                "speaker": segment["speaker"],
                "text": text,
                "timestamp": (
                    segment_chunks[0]["timestamp"][0],
                    segment_chunks[-1]["timestamp"][1]
                    )
            })
        else:
            # Assign the speaker label to each transcript chunk in the segment
            for chunk in segment_chunks:
                chunk_copy = chunk.copy()
                chunk_copy["speaker"] = segment["speaker"]
                segmented_preds.append(chunk_copy)

    return segmented_preds


def diarize(outputs, **kwargs):
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path=kwargs['diarization_model'],
        use_auth_token=kwargs['hf_token']
    )
    diarization_pipeline.to(torch.device("mps"))

    _, diarizer_inputs = preprocess_inputs(
        inputs=kwargs['file_name']
        )

    segments = diarize_audio(
        diarizer_inputs,
        diarization_pipeline,
        kwargs['num_speakers'],
        kwargs['min_speakers'],
        kwargs['max_speakers']
    )

    return post_process_segments_and_transcripts(
        segments, outputs["chunks"], group_by_speaker=False
    )
