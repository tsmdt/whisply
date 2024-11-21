import gradio as gr
import os
import shutil
from datetime import datetime

from pathlib import Path
from whisply.transcription import TranscriptionHandler
from whisply import little_helper, models

CSS = """
h1 {
    font-size: 36px;
    font-weight: 800; 
    }
    
.svelte-1ed2p3z {
    text-align: center;
}
"""

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

def get_device() -> str:
    """
    Determine the computation device based on user preference and availability.
    """
    import torch

    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def transcribe(file, model, device, language, options, hf_token, sub_length):
    if not options:
        options = []
    annotate = 'Annotate Speakers' in options
    translate = 'Translate to English' in options
    subtitle = 'Generate Subtitles' in options

    if (annotate or subtitle) and not hf_token:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            yield 'A HuggingFace Access Token is required for annotation or subtitling: https://huggingface.co/docs/hub/security-tokens', None
            return

    if file is None:
        yield "Please upload a file.", None
        return

    # If file is not a list, make it a list
    if not isinstance(file, list):
        file = [file]

    # Start the progress bar
    progress = gr.Progress()
    progress(0)

    try:
        # Total steps calculation
        steps_per_file = 5  # Number of steps per file
        total_steps = steps_per_file * len(file)
        current_step = 0

        # Save the uploaded file to a temporary directory
        temp_dir = './app_uploads'
        os.makedirs(temp_dir, exist_ok=True)

        temp_file_paths = []
        for uploaded_file in file:
            # Get the base name of the file to avoid issues with absolute paths
            temp_file_name = os.path.basename(uploaded_file.name)
            temp_file_path = os.path.join(temp_dir, temp_file_name)

            # Copy the file from Gradio's temp directory to our local directory
            shutil.copyfile(uploaded_file.name, temp_file_path)
            temp_file_paths.append(temp_file_path)

        # Adjust the device based on user selection
        if device == 'auto':
            device_selected = get_device()
        elif device == 'gpu':
            import torch
            if torch.cuda.is_available():
                device_selected = 'cuda:0'
            else:
                print("→ CUDA is not available. Falling back to auto device selection.")
                device_selected = get_device()
        else:
            device_selected = device

        # Handle export formats
        export_formats_map = {
            'standard': ['json', 'txt'],
            'annotate': ['rttm', 'txt', 'json'],
            'subtitle': ['vtt', 'webvtt', 'srt', 'txt', 'json'],
            'translate': ['txt', 'json']
        }

        export_formats_list = set(export_formats_map['standard'])

        if annotate:
            export_formats_list.update(export_formats_map['annotate'])
        if subtitle:
            export_formats_list.update(export_formats_map['subtitle'])
        if translate:
            export_formats_list.update(export_formats_map['translate'])

        export_formats_list = list(export_formats_list)

        # Create an instance of TranscriptionHandler with the provided parameters
        handler = TranscriptionHandler(
            base_dir='./app_transcriptions',
            model=model,
            device=device_selected,
            file_language=language if language else None,
            annotate=annotate,
            translate=translate,
            subtitle=subtitle,
            sub_length=int(sub_length) if subtitle else 5,
            hf_token=hf_token,
            verbose=True,
            export_formats=export_formats_list
        )

        # Initialize processed_files list
        handler.processed_files = []
        for idx, filepath in enumerate(temp_file_paths):
            filepath = Path(filepath)
            
            # Update progress
            current_step += 1
            progress(current_step / total_steps)

            # Create and set output_dir and output_filepath
            handler.output_dir = little_helper.set_output_dir(filepath, handler.base_dir)
            output_filepath = handler.output_dir / filepath.stem

            # Convert file format
            filepath = little_helper.check_file_format(filepath)
            
            # Update progress
            current_step += 1
            progress(current_step / total_steps)

            # Detect file language
            if not handler.file_language:
                handler.detect_language(file=filepath)
                
            # Update progress
            current_step += 1
            progress(current_step / total_steps)

            # Transcription and speaker annotation
            if handler.device == 'mps':
                handler.model = models.set_supported_model(
                    handler.model_provided,
                    implementation='insane-whisper'
                )
                result_data = handler.transcribe_with_insane_whisper(filepath)

            elif handler.device in ['cpu', 'cuda:0']:
                if handler.annotate or handler.subtitle:
                    handler.model = models.set_supported_model(
                        handler.model_provided,
                        implementation='whisperx'
                    )
                    result_data = handler.transcribe_with_whisperx(filepath)
                else:
                    handler.model = models.set_supported_model(
                        handler.model_provided,
                        implementation='faster-whisper'
                    )
                    result_data = handler.transcribe_with_faster_whisper(filepath)
                    
            # Update progress
            current_step += 1
            progress(current_step / total_steps)

            result = {
                'id': f'file_00{idx + 1}',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_filepath': str(Path(filepath).absolute()),
                'output_filepath': str(Path(output_filepath).absolute()),
                'written_files': None,
                'device': handler.device,
                'model': handler.model,
                'transcription': result_data['transcription']['transcriptions'],
            }

            # Save results
            result['written_files'] = little_helper.save_results(
                result=result,
                export_formats=handler.export_formats
            )
            
            # Update progress
            current_step += 1
            progress(current_step / total_steps)

            handler.processed_files.append(result)

            if not handler.file_language_provided:
                handler.file_language = None

    except {} as e:
        print(f"→ Error during transcription: {e}")
        yield f"Transcription Error: {e}", None

    finally:
        progress(100)

    # Get the transcription results
    if handler and handler.processed_files:
        output_files_set = set()
        for processed_file in handler.processed_files:
            # Collect the paths of the generated files directly
            output_files = processed_file.get('written_files', [])
            output_files_set.update(output_files)

        output_files = sorted(list(output_files_set))

        yield output_files
    else:
        yield "Transcription Error."

# Define Gradio interface components
inputs = [
    gr.File(label="Upload File(s)", file_count='multiple'),
    gr.Dropdown(
        choices=[
            'tiny',
            'tiny-en',
            'base',
            'base-en',
            'small',
            'small-en',
            'distil-small-en',
            'medium',
            'medium-en',
            'distil-medium-en',
            'large',
            'large-v1',
            'large-v2',
            'distil-large-v2',
            'large-v3',
            'distil-large-v3',
            'large-v3-turbo'],
        label="Model",
        value='large-v3-turbo',
        info='Choose the Whisper model for the transcription. (**larger roughly equals to more accurate**)'
    ),
    gr.Radio(
        choices=['auto', 'cpu', 'gpu', 'mps'],
        label="Device",
        value='auto',
        info="**auto** = auto-detection | **gpu** = Nvidia GPUs | **mps** = Mac M1-M4"
    ),
    gr.Dropdown(
        choices=sorted(list(LANGUAGES.keys())),
        label="Language (leave blank for auto-detection)",
        value=None
    ),
    gr.CheckboxGroup(
        choices=['Annotate Speakers', 'Translate to English', 'Generate Subtitles'],
        label="Options",
        value=[]
    ),
    gr.Text(
        label='HuggingFace Access Token (for annotation and subtitling)',
        info="Refer to **README.md** to set up the Access Token correctly ...",
        value=None,
        lines=1,
        max_lines=1
    ),
    gr.Number(
        label="Subtitle Length (words)",
        value=5,
        info="""Subtitle segment length in words. \
(Example: "10" will result in subtitles where each subtitle block has \
exactly 5 words)"""
    )
]

outputs = [
    gr.Files(label="Transcriptions")
]

title = "whisply 💬"
desc = """
Transcribe, translate, annotate and subtitle audio and video files with \
OpenAI's Whisper ... fast!
"""
theme = gr.themes.Citrus(
    primary_hue="emerald",
    neutral_hue="slate",
    spacing_size=gr.themes.sizes.spacing_sm,
    text_size="md",
    radius_size="sm",
    font=[gr.themes.GoogleFont('Open Sans', 'Roboto'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['Roboto Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

# Build and launch the Gradio interface
app = gr.Interface(
    fn=transcribe,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=desc,
    theme=theme,
    flagging_mode='never',
    submit_btn='Transcribe',
    clear_btn=None,
    css=CSS
)

app.queue()
app.launch()
