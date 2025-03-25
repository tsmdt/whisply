import gradio as gr
import os
import shutil
import zipfile
import tempfile

from datetime import datetime
from pathlib import Path
from whisply import output_utils
from whisply.app_helpers import CSS, LANGUAGES


def zip_files(file_paths: list[str]) -> str:
    """
    Given a list of file paths, zip them up into a single archive
    and return the path to that archive.
    """
    if not file_paths:
        return None  

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        zip_path = tmp.name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fpath in file_paths:
            arcname = os.path.basename(fpath)
            zipf.write(fpath, arcname=arcname)

    return zip_path


def create_gradio_interface():
    """
    Main gradio interface.
    """
    def get_device() -> str:
        """
        Determine the computation device based on user preference and 
        availability.
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
        from whisply.transcription import TranscriptionHandler
        from whisply import little_helper, models

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
                'annotate': ['rttm', 'txt', 'json', 'html'],
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
                file_language=None if language == 'auto' else language,
                annotate=annotate,
                translate=translate,
                subtitle=subtitle,
                sub_length=int(sub_length) if subtitle else 5,
                hf_token=hf_token,
                verbose=False,
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
                filepath, audio_array = little_helper.check_file_format(
                    filepath=filepath,
                    del_originals=False
                    )
                
                # Update progress
                current_step += 1
                progress(current_step / total_steps)

                # Detect file language
                if not handler.file_language:
                    handler.detect_language(filepath, audio_array)
                    
                # Update progress
                current_step += 1
                progress(current_step / total_steps)

                # Transcription and speaker annotation
                if handler.device == 'mps':
                    handler.model = models.set_supported_model(
                        handler.model_provided,
                        implementation='insane-whisper',
                        translation=handler.translate
                    )
                    print(f'→ Using {handler.device.upper()} and 🚅 Insanely-Fast-Whisper with model "{handler.model}"')
                    result_data = handler.transcribe_with_insane_whisper(filepath)

                elif handler.device in ['cpu', 'cuda:0']:
                    if handler.annotate or handler.subtitle:
                        handler.model = models.set_supported_model(
                            handler.model_provided,
                            implementation='whisperx',
                            translation=handler.translate
                        )
                        print(f'→ Using {handler.device.upper()} and whisper🆇  with model "{handler.model}"')
                        result_data = handler.transcribe_with_whisperx(filepath)
                    else:
                        handler.model = models.set_supported_model(
                            handler.model_provided,
                            implementation='faster-whisper',
                            translation=handler.translate
                        )
                        print(f'→ Using {handler.device.upper()} and 🏃‍♀️‍➡️ Faster-Whisper with model "{handler.model}"')
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
                result['written_files'] = output_utils.OutputWriter().save_results(
                        result=result,
                        export_formats=handler.export_formats
                        )
                
                # Update progress
                current_step += 1
                progress(current_step / total_steps)

                handler.processed_files.append(result)

                if not handler.file_language_provided:
                    handler.file_language = None

        except Exception as e:
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

            yield output_files, output_files, gr.update(visible=True)
        else:
            yield "Transcription Error."
            
    def toggle_visibility(options):
        """
        Updates the visibility of conditional components based on selected options.
        """
        show_access_token = 'Annotate Speakers' in options
        show_subtitle_length = 'Generate Subtitles' in options
        return gr.update(visible=show_access_token), gr.update(visible=show_subtitle_length)

    # Theme
    theme = gr.themes.Citrus(
        primary_hue="emerald",
        neutral_hue="slate",
        spacing_size=gr.themes.sizes.spacing_sm,
        text_size="md",
        radius_size="sm",
        font=[gr.themes.GoogleFont('Open Sans', 'Roboto'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['Roboto Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    # Build the Gradio Blocks interface
    with gr.Blocks(theme=theme, css=CSS) as app:
        gr.Markdown("# whisply 💬")
        gr.Markdown("""
        Transcribe, translate, annotate, and subtitle audio and video files with \
        OpenAI's Whisper ... fast!
        """)

        # File Upload and Model Selection
        with gr.Row():
            with gr.Column():
                uploaded_files = gr.File(label="Upload File(s)", file_count='multiple')
                with gr.Row():
                    model_dropdown = gr.Dropdown(
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
                            'large-v2',
                            'distil-large-v2',
                            'large-v3',
                            'distil-large-v3',
                            'large-v3-turbo'],
                        label="Model",
                        value='large-v3-turbo',
                        info='Whisper model for the transcription.'
                    )
                    language_dropdown = gr.Dropdown(
                    choices=sorted(LANGUAGES.keys()),
                    label="Language",
                    value='auto',
                    info="**auto** = auto-detection"
                    )
                with gr.Row():
                    device_radio = gr.Radio(
                        choices=['auto', 'cpu', 'gpu', 'mps'],
                        label="Device",
                        value='auto',
                        info="**auto** = auto-detection | **gpu** = Nvidia GPUs | **mps** = Mac M1-M4"
                    )
                with gr.Row():
                    options_checkbox = gr.CheckboxGroup(
                        choices=['Annotate Speakers', 'Translate to English', 'Generate Subtitles'],
                        label="Options",
                        value=[]
                    )
                with gr.Row():
                    access_token_text = gr.Text(
                        label='HuggingFace Access Token (for annotation and subtitling)',
                        info="Refer to **README.md** to set up the Access Token correctly.",
                        value=None,
                        lines=1,
                        max_lines=1,
                        visible=False
                    )
                with gr.Row():
                    subtitle_length_number = gr.Number(
                        label="Subtitle Length (words)",
                        value=5,
                        info="""Subtitle segment length in words. \
                    (Example: "10" will result in subtitles where each subtitle block has \
                    exactly 10 words)""",
                        visible=False
                    )
                    
                    # Event Handler to Toggle Visibility
                    options_checkbox.change(
                        toggle_visibility,
                        inputs=options_checkbox,
                        outputs=[access_token_text, subtitle_length_number]
                    )
            
            with gr.Column():
                outputs = gr.Files(label="Transcriptions")
                transcribed_files_state = gr.State([])
                transcribe_button = gr.Button("Transcribe")
                
                # Download button
                download_all_button = gr.Button(
                    value="Download Transcripts",
                    visible=False
                    )
                download_all_button.click(
                    zip_files,
                    outputs=[outputs],
                    inputs=[transcribed_files_state],
                )
                
                # Transcribe button
                transcribe_button.click(
                    transcribe,
                    inputs=[
                        uploaded_files,
                        model_dropdown,
                        device_radio,
                        language_dropdown,
                        options_checkbox,
                        access_token_text,
                        subtitle_length_number
                    ],
                    outputs=[
                        outputs, 
                        transcribed_files_state, 
                        download_all_button # "visible=True" after transcription
                        ]
                )
                
            app.queue()
    
    return app
    
def main():
    interface = create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    main()
