DETECT_LANG = """Detect and return the language of the provided audio file in ISO format (de, fr, en etc.)

# Example 1
en

# Example 2
fr
"""

TRANSCRIBE_ANNOTATE = """Perform the following tasks on the provided audio file:
1. Transcribe the audio file as accuratly as possible.
2. Annotate speakers. If possible annotate their names. If not use a default value (SPEAKER01, SPEAKER02, etc.)
3. Annotate the start and end timestamp for each transcribed segment in **MM:SS:MS**. Be very careful annotating **precise timestamps**!
4. **Strictly adhere** to the provided output format and example and *do not* return anything else.

# Output format
[start timestamp – end timestamp] [speaker label] Transcribed segment 1 for the audio.

# Example
[00:01:862 - 00:15:572] [Announcer] Es folgt Campus TV. Die Produktion dieses Hochschul- und Forschungsmagazins
"""
