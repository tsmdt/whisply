import gradio as gr
import os
import threading
import time
import shutil
import io
import contextlib

from whisply.transcription import TranscriptionHandler

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

def transcribe(file, model, device, language, options, sub_length):
    if not options:
        options = []
    annotate = 'Annotate Speakers' in options
    translate = 'Translate to English' in options
    subtitle = 'Generate Subtitles' in options

    if file is None:
        yield "Please upload a file.", None
        return

    # If file is not a list, make it a list
    if not isinstance(file, list):
        file = [file]

    # Create an io.StringIO buffer to capture stdout
    stdout_buffer = io.StringIO()

    # Define handler in the enclosing scope
    handler = None

    def run_transcription():
        nonlocal handler
        try:            
            # Redirect stdout to the buffer
            with contextlib.redirect_stdout(stdout_buffer):
                print('→ Starting transcription.')
                
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

                print(f'→ Loaded file(s): {temp_file_paths}')
                
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
                    
                print(f'→ Found device: {device_selected.upper()}')
                
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
                    verbose=True,
                    export_formats=export_formats_list
                )

                # Process the uploaded files
                handler.process_files(temp_file_paths)
                print(f'→ Finished transcription.')
                
        except Exception as e:
            print(f"→ Error during transcription: {e}")

    # Start the transcription in a separate thread
    transcription_thread = threading.Thread(target=run_transcription)
    transcription_thread.start()

    # While the thread is running, yield updated logs
    while transcription_thread.is_alive():
        stdout_output = stdout_buffer.getvalue()
        combined_logs = '\n' + stdout_output
        yield combined_logs, None
        time.sleep(0.5) 

    # After processing is done
    stdout_output = stdout_buffer.getvalue()
    combined_logs = '\n' + stdout_output

    # Get the transcription results
    if handler and handler.processed_files:
        output_files_set = set()
        for processed_file in handler.processed_files:
            # Collect the paths of the generated files directly
            output_files = processed_file.get('written_files', [])
            output_files_set.update(output_files)
        
        output_files = list(output_files_set)
    
        yield combined_logs, output_files
    else:
        yield combined_logs + "\nTranscription failed.", None

# Define Gradio interface components
inputs = [
    gr.File(label="Upload File(s)", file_count='multiple'),
    gr.Dropdown(
        choices=[
            'tiny',
            'tine-en',
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
        info='Choose the Whisper model for the transcription. (*larger roughly equals to more accurate*)'
    ),
    gr.Radio(
        choices=['auto', 'cpu', 'gpu', 'mps'],
        label="Device ('auto' will run auto-dedection first)",
        value='auto',
        info="'gpu' = Nvidia GPUs | 'mps' = Mac M1-M4"
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
    gr.Number(
        label="Subtitle Length (words)", 
        value=5,
        info="""Subtitle segment length in words. \
(Example: "10" will result in subtitles where each subtitle block has \
exactly 5 words)"""
    )
]

outputs = [
    gr.Textbox(label="Logs", lines=10, max_lines=10, interactive=False),
    gr.Files(label="Transcriptions")
]

title = "whisply 💬"
theme = gr.themes.Citrus(
    primary_hue="emerald",
    neutral_hue="slate",
    spacing_size=gr.themes.sizes.spacing_sm,
    text_size="md",
    font=[gr.themes.GoogleFont('Open Sans', 'Roboto'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

# Build and launch the Gradio interface
app = gr.Interface(
    fn=transcribe,
    inputs=inputs,
    outputs=outputs,
    title=title,
    theme=theme,
    allow_flagging='never',
    submit_btn='Transcribe',
    clear_btn=None
)

app.launch()
