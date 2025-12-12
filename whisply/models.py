from rich import print

WHISPER_MODELS = {
    # Models for faster-whisper / whisperx: https://huggingface.co/Systran
    # Models for insanely-fast-whisper: https://huggingface.co/openai
    # Models for mlx-whisper: https://huggingface.co/mlx-community
    'tiny': {
        'faster-whisper': 'tiny',
        'insane-whisper': 'openai/whisper-tiny',
        'whisperx': 'tiny',
        'mlx-whisper': 'mlx-community/whisper-tiny-mlx',
        'translation': True
        },
    'tine-en': {
        'faster-whisper': 'tiny.en',
        'insane-whisper': 'openai/whisper-tiny.en',
        'whisperx': 'tiny.en',
        'mlx-whisper': 'mlx-community/whisper-tiny.en-mlx',
        'translation': False
        },
    'base': {
        'faster-whisper': 'base',
        'insane-whisper': 'openai/whisper-base',
        'whisperx': 'base',
        'mlx-whisper': 'mlx-community/whisper-base-mlx',
        'translation': True
        },
    'base-en': {
        'faster-whisper': 'base.en',
        'insane-whisper': 'openai/whisper-base.en',
        'whisperx': 'base.en',
        'mlx-whisper': 'mlx-community/whisper-base.en-mlx',
        'translation': False
        },
    'small': {
        'faster-whisper': 'small',
        'insane-whisper': 'openai/whisper-small',
        'whisperx': 'small',
        'mlx-whisper': 'mlx-community/whisper-small-mlx',
        'translation': True
        },
    'small-en': {
        'faster-whisper': 'small.en',
        'insane-whisper': 'openai/whisper-small.en',
        'whisperx': 'small.en',
        'mlx-whisper': 'mlx-community/whisper-small.en-mlx',
        'translation': False
        },
    'distil-small-en': {
        'faster-whisper': 'distil-small.en',
        'insane-whisper': 'distil-whisper/distil-small.en',
        'whisperx': None,
        'mlx-whisper': None,
        'translation': False
        },
    'medium': {
        'faster-whisper': 'medium',
        'insane-whisper': 'openai/whisper-medium',
        'whisperx': 'medium',
        'mlx-whisper': 'mlx-community/whisper-medium-mlx',
        'translation': True
        },
    'medium-en': {
        'faster-whisper': 'medium.en',
        'insane-whisper': 'openai/whisper-medium.en',
        'whisperx': 'medium.en',
        'mlx-whisper': 'mlx-community/whisper-medium.en-mlx',
        'translation': False
        },
    'distil-medium-en': {
        'faster-whisper': 'distil-medium.en', 
        'insane-whisper': 'distil-whisper/distil-medium.en',
        'whisperx': None,
        'mlx-whisper': None,
        'translation': False
        },
    'large': {
        'faster-whisper': 'large',
        'insane-whisper': 'openai/whisper-large',
        'whisperx': 'large',
        'mlx-whisper': 'mlx-community/whisper-large-mlx',
        'translation': True
        },
    'large-v2': {
        'faster-whisper': 'large-v2',
        'insane-whisper': 'openai/whisper-large-v2',
        'whisperx': 'large-v2',
        'mlx-whisper': 'mlx-community/whisper-large-v2-mlx',
        'translation': True
        },
    'distil-large-v2': {
        'faster-whisper': 'distil-large-v2',
        'insane-whisper': 'distil-whisper/distil-large-v2',
        'whisperx': None,
        'mlx-whisper': None,
        'translation': True
        },
    'large-v3': {
        'faster-whisper': 'large-v3',
        'insane-whisper': 'openai/whisper-large-v3',
        'whisperx': 'large-v3',
        'mlx-whisper': 'mlx-community/whisper-large-v3-mlx',
        'translation': True
        },
    'distil-large-v3': {
        'faster-whisper': 'distil-large-v3',
        'insane-whisper': 'distil-whisper/distil-large-v3',
        'whisperx': 'distil-large-v3',
        'mlx-whisper': None,
        'translation': True
        },
    'large-v3-turbo': {
        'faster-whisper': 'deepdml/faster-whisper-large-v3-turbo-ct2',
        'insane-whisper': 'openai/whisper-large-v3-turbo',
        'whisperx': None,
        'mlx-whisper': 'mlx-community/whisper-large-v3-turbo',
        'translation': False
        },
}


def ensure_model(model: str) -> bool:
    return model in WHISPER_MODELS


def _get_default_model_for_impl(
    implementation: str,
    translation: bool
) -> str:
    """
    Pick a sensible default model for the requested implementation.
    Falls back to any model that supports the implementation if the
    preferred choice does not satisfy translation requirements.
    """
    preferred = (
        'large-v3-turbo'
        if implementation == 'mlx-whisper'
        else 'large-v2'
    )

    if is_model_supported(preferred, implementation, translation):
        return preferred

    if is_model_supported(preferred, implementation, False):
        return preferred

    for name in WHISPER_MODELS:
        if is_model_supported(name, implementation, translation):
            return name

    for name, info in WHISPER_MODELS.items():
        if info.get(implementation):
            return name

    raise ValueError(
        f'No model available for implementation "{implementation}".'
        )


def is_model_supported(
    model: str,
    implementation: str,
    translation: bool
) -> bool:
    model_info = WHISPER_MODELS.get(model)
    if not model_info:
        return False
    if model_info.get(implementation) is None:
        return False
    if translation and not model_info.get("translation", False):
        return False
    return True


def set_supported_model(
    model: str,
    implementation: str,
    translation: bool
) -> str:
    if not is_model_supported(model, implementation, translation):
        default_model = _get_default_model_for_impl(
            implementation, translation
        )
        print(
            f'[blue1]→ Model "{model}" is not available for this task/'
            f'implementation → Using default model "{default_model}".'
        )
        return WHISPER_MODELS.get(default_model)[implementation]
    return WHISPER_MODELS.get(model)[implementation]


def set_mlx_model(model: str, translation: bool) -> str:
    """
    Selects a supported mlx-whisper model, falling back to a default one
    if the requested combination is unavailable.
    """
    return set_supported_model(
        model=model,
        implementation='mlx-whisper',
        translation=translation
    )
