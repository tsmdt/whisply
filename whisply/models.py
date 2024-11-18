from rich import print

WHISPER_MODELS = {
    # Models for faster-whisper / whisperx: https://huggingface.co/Systran
    # Models for insanely-fast-whisper: https://huggingface.co/openai
    'tiny': {
        'faster-whisper': 'tiny',
        'insane-whisper': 'openai/whisper-tiny',
        'whisperx': 'tiny'
        },
    'tine-en': {
        'faster-whisper': 'tiny.en',
        'insane-whisper': 'openai/whisper-tiny.en',
        'whisperx': 'tiny.en'
        },        
    'base': {
        'faster-whisper': 'base',
        'insane-whisper': 'openai/whisper-base',
        'whisperx': 'base'
        },
    'base-en': {
        'faster-whisper': 'base.en',
        'insane-whisper': 'openai/whisper-base.en',
        'whisperx': 'base.en'
        },
    'small': {
        'faster-whisper': 'small',
        'insane-whisper': 'openai/whisper-small',
        'whisperx': 'small'
        },
    'small-en': {
        'faster-whisper': 'small.en',
        'insane-whisper': 'openai/whisper-small.en',
        'whisperx': 'small.en'
        },
    'distil-small-en': {
        'faster-whisper': 'distil-small.en', 
        'insane-whisper': 'distil-whisper/distil-small.en',
        'whisperx': None
        },
    'medium': {
        'faster-whisper': 'medium', 
        'insane-whisper': 'openai/whisper-medium',
        'whisperx': 'medium'
        },
    'medium-en': {
        'faster-whisper': 'medium.en', 
        'insane-whisper': 'openai/whisper-medium.en',
        'whisperx': 'medium.en'
        },
    'distil-medium-en': {
        'faster-whisper': 'distil-medium.en', 
        'insane-whisper': 'distil-whisper/distil-medium.en',
        'whisperx': None
        },
    'large': {
        'faster-whisper': 'large', 
        'insane-whisper': 'openai/whisper-large',
        'whisperx': 'large'
        },
    'large-v1': {
        'faster-whisper': 'large-v1', 
        'insane-whisper': 'openai/whisper-large-v1',
        'whisperx': 'large-v1'
        },
    'large-v2': {
        'faster-whisper': 'large-v2', 
        'insane-whisper': 'openai/whisper-large-v2',
        'whisperx': 'large-v2'
        },
    'distil-large-v2': {
        'faster-whisper': 'distil-large-v2', 
        'insane-whisper': 'distil-whisper/distil-large-v2',
        'whisperx': None
        },
    'large-v3': {
        'faster-whisper': 'large-v3', 
        'insane-whisper': 'openai/whisper-large-v3',
        'whisperx': 'large-v3'
        },
    'distil-large-v3': {
        'faster-whisper': 'distil-large-v3', 
        'insane-whisper': 'distil-whisper/distil-large-v3',
        'whisperx': None
        },
    'large-v3-turbo': {
        'faster-whisper': 'deepdml/faster-whisper-large-v3-turbo-ct2', 
        'insane-whisper': 'openai/whisper-large-v3-turbo',
        'whisperx': None
        },
}

def ensure_model(model: str) -> bool:
    return model in WHISPER_MODELS

def is_model_supported(model: str, implementation: str) -> bool:
    return True if WHISPER_MODELS.get(model)[implementation] != None else False

def set_supported_model(model: str, implementation: str) -> str:
    if not is_model_supported(model, implementation):
        print(f'[bold]→ Model "{model}" is not available for this task / implementation → Using default model "large-v2".')
        return WHISPER_MODELS.get("large-v2")[implementation]
    return WHISPER_MODELS.get(model)[implementation]
