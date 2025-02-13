from rich import print

WHISPER_MODELS = {
    # Models for faster-whisper / whisperx: https://huggingface.co/Systran
    # Models for insanely-fast-whisper: https://huggingface.co/openai
    'tiny': {
        'faster-whisper': 'tiny',
        'insane-whisper': 'openai/whisper-tiny',
        'whisperx': 'tiny',
        'translation': True
        },
    'tine-en': {
        'faster-whisper': 'tiny.en',
        'insane-whisper': 'openai/whisper-tiny.en',
        'whisperx': 'tiny.en',
        'translation': False
        },        
    'base': {
        'faster-whisper': 'base',
        'insane-whisper': 'openai/whisper-base',
        'whisperx': 'base',
        'translation': True
        },
    'base-en': {
        'faster-whisper': 'base.en',
        'insane-whisper': 'openai/whisper-base.en',
        'whisperx': 'base.en',
        'translation': False
        },
    'small': {
        'faster-whisper': 'small',
        'insane-whisper': 'openai/whisper-small',
        'whisperx': 'small',
        'translation': True
        },
    'small-en': {
        'faster-whisper': 'small.en',
        'insane-whisper': 'openai/whisper-small.en',
        'whisperx': 'small.en',
        'translation': False
        },
    'distil-small-en': {
        'faster-whisper': 'distil-small.en', 
        'insane-whisper': 'distil-whisper/distil-small.en',
        'whisperx': None,
        'translation': False
        },
    'medium': {
        'faster-whisper': 'medium', 
        'insane-whisper': 'openai/whisper-medium',
        'whisperx': 'medium',
        'translation': True
        },
    'medium-en': {
        'faster-whisper': 'medium.en', 
        'insane-whisper': 'openai/whisper-medium.en',
        'whisperx': 'medium.en',
        'translation': False
        },
    'distil-medium-en': {
        'faster-whisper': 'distil-medium.en', 
        'insane-whisper': 'distil-whisper/distil-medium.en',
        'whisperx': None,
        'translation': False
        },
    'large': {
        'faster-whisper': 'large', 
        'insane-whisper': 'openai/whisper-large',
        'whisperx': 'large',
        'translation': True
        },
    'large-v2': {
        'faster-whisper': 'large-v2', 
        'insane-whisper': 'openai/whisper-large-v2',
        'whisperx': 'large-v2',
        'translation': True
        },
    'distil-large-v2': {
        'faster-whisper': 'distil-large-v2', 
        'insane-whisper': 'distil-whisper/distil-large-v2',
        'whisperx': None,
        'translation': True
        },
    'large-v3': {
        'faster-whisper': 'large-v3', 
        'insane-whisper': 'openai/whisper-large-v3',
        'whisperx': 'large-v3',
        'translation': True
        },
    'distil-large-v3': {
        'faster-whisper': 'distil-large-v3', 
        'insane-whisper': 'distil-whisper/distil-large-v3',
        'whisperx': 'distil-large-v3',
        'translation': True
        },
    'large-v3-turbo': {
        'faster-whisper': 'deepdml/faster-whisper-large-v3-turbo-ct2', 
        'insane-whisper': 'openai/whisper-large-v3-turbo',
        'whisperx': None,
        'translation': False
        },
}

def ensure_model(model: str) -> bool:
    return model in WHISPER_MODELS

def is_model_supported(model: str, implementation: str, translation: bool) -> bool:
    model_info = WHISPER_MODELS.get(model)
    if not model_info:
        return False
    if model_info.get(implementation) is None:
        return False
    if translation and not model_info.get("translation", False):
        return False
    return True

def set_supported_model(model: str, implementation: str, translation: bool) -> str:
    if not is_model_supported(model, implementation, translation):
        default_model = "large-v2"
        print(f'[blue1]→ Model "{model}" is not available for this task/implementation → Using default model "{default_model}".')
        return WHISPER_MODELS.get(default_model)[implementation]
    return WHISPER_MODELS.get(model)[implementation]
