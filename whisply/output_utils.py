import re
import json
import logging
import typer

from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple
from rich import print
from whisply import little_helper
from whisply.post_correction import Corrections

# Set logging configuration
logger = logging.getLogger('little_helper')
logger.setLevel(logging.INFO)


class ExportFormats(str, Enum):
    ALL = 'all'
    JSON = 'json'
    TXT = 'txt'
    RTTM = 'rttm'
    VTT = 'vtt'
    WEBVTT = 'webvtt'
    SRT = 'srt'
    HTML = 'html'
    
    
def determine_export_formats(
    export_format: ExportFormats,
    annotate: bool,
    subtitle: bool
) -> List[str]:
    """
    Determine the export formats based on user options and availability.

    Returns a list of export format strings to be used.
    """
    available_formats = set()
    if export_format == ExportFormats.ALL:
        available_formats.add(ExportFormats.JSON.value)
        available_formats.add(ExportFormats.TXT.value)
        if annotate:
            available_formats.add(ExportFormats.RTTM.value)
            available_formats.add(ExportFormats.HTML.value)
        if subtitle:
            available_formats.add(ExportFormats.WEBVTT.value)
            available_formats.add(ExportFormats.VTT.value)
            available_formats.add(ExportFormats.SRT.value)
    else:
        if export_format in (ExportFormats.JSON, ExportFormats.TXT):
            available_formats.add(export_format.value)
        elif export_format in (ExportFormats.RTTM, ExportFormats.HTML):
            if annotate:
                available_formats.add(export_format.value)
            else:
                print("→ RTTM export format requires annotate option to be True.")
                raise typer.Exit()
        elif export_format in (
            ExportFormats.VTT,
            ExportFormats.SRT,
            ExportFormats.WEBVTT
            ):
            if subtitle:
                available_formats.add(export_format.value)
            else:
                print(f"→ {export_format.value.upper()} export format requires subtitle option to be True.")
                raise typer.Exit()
        else:
            print(f"→ Unknown export format: {export_format.value}")
            raise typer.Exit()

    return list(available_formats)


class OutputWriter:
    """
    Class for writing various output formats to disk.
    """
    def __init__(
        self, 
        corrections: Corrections = None
        ):
        self.cwd = Path.cwd()
        self.corrections = corrections
        self.compiled_simple_patterns = (
            self._compile_simple_patterns() if self.corrections else {}
            )
        self.compiled_regex_patterns = (
            self._compile_regex_patterns() if self.corrections else []
            )
    
    def _compile_simple_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Pre-compile regex patterns for simple word corrections.
        Returns a list of tuples containing compiled patterns and their replacements.
        """
        patterns = []
        for wrong, correct in self.corrections.simple.items():
            # Wrap simple corrections with word boundaries
            pattern = re.compile(
                r'\b{}\b'.format(re.escape(wrong)), flags=re.IGNORECASE
                )
            patterns.append((pattern, correct))
            logger.debug(
                f"Compiled simple pattern: '\\b{wrong}\\b' → '{correct}'"
                )
        return patterns
    
    def _compile_regex_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Pre-compile regex patterns for pattern-based corrections.
        Returns a list of tuples containing compiled regex patterns and their replacements.
        """
        patterns = []
        for entry in self.corrections.patterns:
            original_pattern = entry['pattern']
            replacement = entry['replacement']
            
            # Wrap patterns with word boundaries and non-capturing group
            new_pattern = r'\b(?:' + original_pattern + r')\b'
            regex = re.compile(new_pattern, flags=re.IGNORECASE)
            
            patterns.append((regex, replacement))
            logger.debug(
                f"Compiled pattern-based regex: '{new_pattern}' → '{replacement}'"
                )
        return patterns
    
    def correct_transcription(self, transcription: str) -> str:
        """
        Apply both simple and pattern-based corrections to the transcription.
        """
        # Apply simple corrections
        for pattern, correct in self.compiled_simple_patterns:
            transcription = pattern.sub(
                lambda m: self.replace_match(m, correct), transcription
                )
        
        # Apply pattern-based corrections
        for regex, replacement in self.compiled_regex_patterns:
            transcription = regex.sub(replacement, transcription)
        
        return transcription
    
    @staticmethod
    def replace_match(match, correct: str) -> str:
        """
        Replace the matched word while preserving the original casing.
        """
        word = match.group()
        if word.isupper():
            return correct.upper()
        elif word[0].isupper():
            return correct.capitalize()
        else:
            return correct
        
    def _save_file(
        self, 
        content: str, 
        filepath: Path, 
        description: str, 
        log_message: str
        ) -> None:
        """
        Generic method to save content to a file.
        """
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f'[blue1]→ Saved {description}: [bold]{filepath.relative_to(self.cwd)}')
        logger.info(f'{log_message} {filepath}')

    def save_json(
        self, 
        result: dict, 
        filepath: Path
        ) -> None:
        with open(filepath, 'w', encoding='utf-8') as fout:
            json.dump(result, fout, indent=4)
        print(f'[blue1]→ Saved .json: [bold]{filepath.relative_to(self.cwd)}')
        logger.info(f"Saved .json to {filepath}")
        
    def save_txt(
        self, 
        transcription: Dict[str, str], 
        filepath: Path
        ) -> None:
        """
        Save the transcription as a TXT file after applying corrections.
        """
        original_text = transcription.get('text', '').strip()
        corrected_text = self.correct_transcription(original_text)
        self._save_file(
            content=corrected_text,
            filepath=filepath,
            description='.txt',
            log_message='Saved .txt transcript to'
        )
    
    def save_txt_with_speaker_annotation(
        self, 
        annotated_text: str, 
        filepath: Path
        ) -> None:
        """
        Save the annotated transcription as a TXT file after applying corrections.
        """
        corrected_annotated_text = self.correct_transcription(annotated_text)
        self._save_file(
            content=corrected_annotated_text,
            filepath=filepath,
            description='.txt with speaker annotation',
            log_message='Saved .txt transcription with speaker annotation →'
        )
    
    def save_subtitles(
        self, 
        text: str, 
        type: str, 
        filepath: Path
        ) -> None:
        """
        Save subtitles in the specified format after applying corrections.
        """
        corrected_text = self.correct_transcription(text)
        description = f'.{type} subtitles'
        log_message = f'Saved .{type} subtitles →'
        self._save_file(
            content=corrected_text,
            filepath=filepath,
            description=description,
            log_message=log_message
        )

    def save_rttm_annotations(
        self, 
        rttm: str, 
        filepath: Path
        ) -> None:
        self._save_file(
            content=rttm,
            filepath=filepath,
            description='.rttm annotations',
            log_message='Saved .rttm annotations →'
        )
        
    def save_html(
        self, 
        html: str, 
        filepath: Path
        ) -> None:
        self._save_file(
            content=html,
            filepath=filepath,
            description='.html (noScribe compatible)',
            log_message='Saved .html annotations →'
        )

    def save_results(
        self,
        result: dict,
        export_formats: List[str]
    ) -> List[Path]:
        """
        Write various output formats to disk based on the specified export formats.
        """
        output_filepath = Path(result['output_filepath'])
        written_filepaths = []
        
        # Apply corrections if they are provided
        if self.corrections and (
            self.corrections.simple or self.corrections.patterns
            ):
            for language, transcription in result.get('transcription', {}).items():
                # Correct the main transcription text
                original_text = transcription.get('text', '').strip()
                corrected_text = self.correct_transcription(original_text)
                result['transcription'][language]['text'] = corrected_text
                
                # Correct chunks and word dicts
                chunks = transcription.get('chunks', '')
                for c in chunks:
                    # Text chunk
                    c['text'] = self.correct_transcription(c['text'])
                    # Words
                    words = c.get('words', '')
                    for w in words:
                        w['word'] = self.correct_transcription(w['word'])

                # Correct speaker annotations if present
                if 'text_with_speaker_annotation' in transcription:
                    original_annotated = transcription['text_with_speaker_annotation']
                    corrected_annotated = self.correct_transcription(original_annotated)
                    result['transcription'][language]['text_with_speaker_annotation'] = corrected_annotated

        # Now, transcription_items reflects the corrected transcriptions
        transcription_items = result.get('transcription', {}).items()
    
        # Write .txt
        if 'txt' in export_formats:
            for language, transcription in transcription_items:
                fout = output_filepath.parent / f"{output_filepath.name}_{language}.txt"
                self.save_txt(
                    transcription,
                    filepath=fout
                )
                written_filepaths.append(str(fout))

        # Write subtitles (.srt, .vtt and .webvtt)
        subtitle_formats = {'srt', 'vtt', 'webvtt'}
        if subtitle_formats.intersection(export_formats):
            for language, transcription in transcription_items:
                # .srt subtitles
                if 'srt' in export_formats:
                    fout = output_filepath.parent / f"{output_filepath.name}_{language}.srt"
                    srt_text = create_subtitles(
                        transcription, 
                        type='srt'
                        )
                    self.save_subtitles(srt_text, type='srt', filepath=fout)
                    written_filepaths.append(str(fout))

                # .vtt / .webvtt subtitles
                if 'vtt' in export_formats or 'webvtt' in export_formats:
                    for subtitle_type in ['webvtt', 'vtt']:
                        fout = output_filepath.parent / f"{output_filepath.name}_{language}.{subtitle_type}"
                        vtt_text = create_subtitles(
                            transcription, 
                            type=subtitle_type, 
                            result=result
                            )
                        self.save_subtitles(
                            vtt_text, 
                            type=subtitle_type, 
                            filepath=fout
                            )
                        written_filepaths.append(str(fout))

        # Write annotated .txt with speaker annotations
        has_speaker_annotation = any(
            'text_with_speaker_annotation' in transcription
            for transcription in result['transcription'].values()
        )

        if 'txt' in export_formats and has_speaker_annotation:
            for language, transcription in transcription_items:
                if 'text_with_speaker_annotation' in transcription:
                    fout = output_filepath.parent / f"{output_filepath.name}_{language}_annotated.txt"
                    self.save_txt_with_speaker_annotation(
                        annotated_text=transcription['text_with_speaker_annotation'],
                        filepath=fout
                    )
                    written_filepaths.append(str(fout))

        # Write .rttm
        if 'rttm' in export_formats:
            # Create .rttm annotations
            rttm_dict = dict_to_rttm(result)

            for language, rttm_annotation in rttm_dict.items():
                fout = output_filepath.parent / f"{output_filepath.name}_{language}.rttm"
                self.save_rttm_annotations(
                    rttm=rttm_annotation,
                    filepath=fout
                    )
                written_filepaths.append(str(fout))

        # Write .json
        if 'json' in export_formats:
            fout = output_filepath.with_suffix('.json')
            written_filepaths.append(str(fout))
            result['written_files'] = written_filepaths
            self.save_json(result, filepath=fout)
            
        # Write noScribe compatible .html
        if 'html' in export_formats:
            audio_filepath = Path(result['input_filepath'])
            for language, transcription in transcription_items:
                fout = output_filepath.parent / f"{output_filepath.name}_{language}.html"
                html = create_html(
                    transcription,
                    audio_filepath=audio_filepath,
                    output_filepath=output_filepath
                    ) 
                self.save_html(html, filepath=fout)
                written_filepaths.append(str(fout))

        return written_filepaths

def create_subtitles(
    transcription_dict: dict, 
    type: str = 'srt', 
    result: dict = None
    ) -> str:
    """
    Converts a transcription dictionary into subtitle format (.srt or .webvtt).

    Args:
        transcription_dict (dict): Dictionary containing transcription data 
            with 'chunks'.
        sub_length (int, optional): Maximum duration in seconds for each 
            subtitle block.
        type (str, optional): Subtitle format, either 'srt' or 'webvtt'. 
            Default is 'srt'.

    Returns:
        str: Formatted subtitle text in the specified format.
    """
    subtitle_text = ''
    seg_id = 0
    
    for chunk in transcription_dict['chunks']:
        start_time = chunk['timestamp'][0]
        end_time = chunk['timestamp'][1]
        text = chunk['text'].replace('’', '\'')
        
        # Create .srt subtitles
        if type == 'srt':
            start_time_str = little_helper.format_time(
                start_time, 
                delimiter=','
                )
            end_time_str = little_helper.format_time(
                end_time, 
                delimiter=','
                )
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
        # Create .webvtt subtitles    
        elif type in ['webvtt', 'vtt']:
            start_time_str = little_helper.format_time(
                start_time, 
                delimiter='.'
                )
            end_time_str = little_helper.format_time(
                end_time, 
                delimiter='.'
                )

            if seg_id == 0:
                subtitle_text += f"WEBVTT {Path(result['output_filepath']).stem}\n\n"
                
                if type == 'vtt':
                    subtitle_text += 'NOTE transcribed with whisply\n\n'
                    subtitle_text += f"NOTE media: {Path(result['input_filepath']).absolute()}\n\n"
                
            seg_id += 1
            subtitle_text += f"""{seg_id}\n{start_time_str} --> {end_time_str}\n{text.strip()}\n\n"""
        
    return subtitle_text

def dict_to_rttm(result: dict) -> dict:
    """
    Converts a transcription dictionary to RTTM file format.
    """
    file_id = result.get('input_filepath', 'unknown_file')
    file_id = Path(file_id).stem
    rttm_dict = {}

    # Iterate over each available language
    for lang, transcription in result.get('transcription', {}).items():
        lines = []
        current_speaker = None
        speaker_start_time = None
        speaker_end_time = None

        chunks = transcription.get('chunks', [])

        # Collect all words from chunks
        all_words = []
        for chunk in chunks:
            words = chunk.get('words', [])
            all_words.extend(words)

        # Sort all words by their start time
        all_words.sort(key=lambda w: w.get('start', 0.0))

        for word_info in all_words:
            speaker = word_info.get('speaker', 'SPEAKER_00')
            word_start = word_info.get('start', 0.0)
            word_end = word_info.get('end', word_start)

            if speaker != current_speaker:
                # If there is a previous speaker segment, write it to the RTTM
                if current_speaker is not None:
                    duration = speaker_end_time - speaker_start_time
                    rttm_line = (
                        f"SPEAKER {file_id} 1 {speaker_start_time:.3f} {duration:.3f} "
                        f"<NA> <NA> {current_speaker} <NA>"
                    )
                    lines.append(rttm_line)

                # Start a new speaker segment
                current_speaker = speaker
                speaker_start_time = word_start
                speaker_end_time = word_end
            else:
                # Extend the current speaker segment
                speaker_end_time = max(speaker_end_time, word_end)

        # Write the last speaker segment to the RTTM
        if current_speaker is not None:
            duration = speaker_end_time - speaker_start_time
            rttm_line = (
                f"SPEAKER {file_id} 1 {speaker_start_time:.3f} {duration:.3f} "
                f"<NA> <NA> {current_speaker} <NA>"
            )
            lines.append(rttm_line)

        rttm_content = "\n".join(lines)
        rttm_dict[lang] = rttm_content

    return rttm_dict

def create_html(
    transcription: dict, 
    audio_filepath: Path, 
    output_filepath: Path
    ) -> None:
    """
    Save an HTML file that is compatible with noScribe's editor:
    https://github.com/kaixxx/noScribe
    """
    from whisply.output_templates import NOSCRIBE_HTML_TEMPLATE
    
    # Helper function to parse a transcript line
    def parse_line(line: str):
        pattern = r'\[(.*?)\]\s+\[(.*?)\]\s+(.*)'
        match = re.match(pattern, line)
        if match:
            timestamp, speaker, text = match.groups()
            return timestamp, speaker, text
        return None, None, line

    # Helper function to convert a timestamp to milliseconds
    def convert_timestamp_to_ms(timestamp: str) -> int:
        h, m, s = timestamp.split(':')
        s = s.split('.')[0]
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000

    # Use the annotated txt as input
    input_text = transcription.get('text_with_speaker_annotation', '')

    # Replace placeholders in the HTML template
    html_template = NOSCRIBE_HTML_TEMPLATE.replace('{transcription}', str(output_filepath))
    html_template = html_template.replace('{audio_filepath}', str(audio_filepath))
    
    # Process transcript lines to build HTML content
    body_content = ""
    transcript_lines = input_text.strip().splitlines()
    num_lines = len(transcript_lines)

    for idx, line in enumerate(transcript_lines):
        segment_start, speaker, text = parse_line(line)
        
        # If there's a following line or a single line, generate detailed HTML with an anchor
        if num_lines == 1 or idx < num_lines - 1:
            # For a single line, override segment_start with a default value
            if num_lines == 1:
                segment_start = '00:00:00'
                segment_end, _, _ = parse_line(line)
            else:
                segment_end, _, _ = parse_line(transcript_lines[idx + 1])
            
            # Convert timestamps to ms
            start = convert_timestamp_to_ms(segment_start)
            end = convert_timestamp_to_ms(segment_end)
            
            # Set speaker labels
            if 'UNKNOWN' in speaker:
                speaker_label = 'SXX'
            else:
                speaker_number = re.findall(r'\d+', speaker)[0]
                speaker_label = f'S{speaker_number}'
                
            # Build the anchor tag and HTML segment
            anchor = f"ts_{start}_{end}_{speaker_label}"
            segment_html = (f'<p><a name="{anchor}">'
                            f'<span style="color:#78909C">[{segment_start}]</span> {speaker}: {text}'
                            f'</a></p>\n')
        else:
            segment_html = f'<p>{text}</p>\n'
        
        body_content += segment_html

    html_output = html_template.replace('{body_content}', body_content)

    return html_output
