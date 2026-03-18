import re
import logging
import yt_dlp as url_downloader
from pathlib import Path
from datetime import datetime

from whisply import little_helper


# Set logging configuration
logger = logging.getLogger('download_utils')
logger.setLevel(logging.DEBUG)


def _probe_audio_format(
    url: str,
    language: str,
) -> str | None:
    """
    Probe the URL for available audio formats and return the extension of the
    first matching format in PREFERRED_AUDIO_EXTENSIONS, or None if no match.
    """
    # Preferred audio formats
    pref_audio_formats = ['wav', 'm4a', 'mp4']

    probe_opts = {
        'format': f'bestaudio[language={language}]/bestaudio',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with url_downloader.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None

    formats = info.get('formats') or ([info] if info.get('ext') else [])
    audio_formats = [
        f for f in formats
        if f.get('acodec', 'none') != 'none'
        or f.get('vcodec', 'none') == 'none'
    ]

    for preferred_ext in pref_audio_formats:
        if any(f.get('ext') == preferred_ext for f in audio_formats):
            return preferred_ext

    return None


def download_url(
    url: str,
    downloads_dir: Path,
    language: str = "en",
) -> Path:
    """
    Downloads a media file from a specified URL, typically a YouTube URL.

    First probes for natively available audio formats (wav, m4a, mp4) and
    downloads directly if one is found. Falls back to extracting/converting
    to wav via FFmpeg otherwise.

    Args:
        url (str): The URL of the video to download.
        downloads_dir (Path): The directory path where the downloaded file
            should be stored.
        language (str): Language code for the audio track to select.
    """
    little_helper.ensure_dir(downloads_dir)

    temp_filename = (
        f"temp_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}_{language}"
    )

    native_ext = _probe_audio_format(url, language)

    if native_ext:
        logger.debug(
            f"Probed native audio format '{native_ext}' for {url}"
        )
        options = {
            'format': (
                f'bestaudio[language={language}][ext={native_ext}]/'
                f'bestaudio[ext={native_ext}]'
            ),
            'outtmpl': f'{downloads_dir}/{temp_filename}.%(ext)s',
        }
    else:
        logger.debug(
            f"No preferred native format found for {url}, "
            f"falling back to FFmpeg wav extraction"
        )
        options = {
            'format': f'bestaudio[language={language}]/bestaudio',
            'postprocessors': [{'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'wav',
                                'preferredquality': '192'}],
            'outtmpl': f'{downloads_dir}/{temp_filename}.%(ext)s',
        }

    try:
        with url_downloader.YoutubeDL(options) as ydl:
            logger.debug(f"Downloading {url}")
            ydl.download([url])

            video_info = ydl.extract_info(url, download=False)
            downloaded_file = list(downloads_dir.glob(f'{temp_filename}*'))[0]
            logger.debug(f"Download complete for {downloaded_file}")

            new_filename = re.sub(r'\W+', '_', video_info.get(
                'title',
                'downloaded_video'
            ))
            new_filename = new_filename.strip('_')

            renamed_file = downloaded_file.rename(
                f"{downloads_dir}/{new_filename}"
                f"_{language}_{downloaded_file.suffix}"
            )
            logger.debug(f"Renamed downloaded file to {renamed_file}")
            return Path(renamed_file)

    except Exception as e:
        print(f'Error downloading {url}: {e}')
        return None
