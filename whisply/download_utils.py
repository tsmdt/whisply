import re
import logging
import yt_dlp as url_downloader

from pathlib import Path
from datetime import datetime

from whisply import little_helper


# Set logging configuration
logger = logging.getLogger('download_utils')
logger.setLevel(logging.DEBUG)


def download_url(url: str, downloads_dir: Path) -> Path:
    """
    Downloads a media file from a specified URL, typically a YouTube URL, extracting audio
    in WAV format, and then renames the file based on the media title. 

    The function first ensures the downloads directory exists, then initiates a download
    using youtube-dl with specific options set for audio quality and format. After the 
    download, it extracts the video's information without downloading it again to 
    rename the file more meaningfully based on the video title. Special characters 
    in the title are replaced with underscores, and unnecessary leading or trailing 
    underscores are removed.

    Args:
        url (str): The URL of the video to download.
        downloads_dir (Path): The directory path where the downloaded file should be stored.

    Returns:
        Path: A path object pointing to the renamed downloaded file. If there is an error 
        during the download or file processing, returns None.

    Raises:
        Exception: Outputs an error message to the console if the download fails.

    Examples:
        >>> download_url("https://www.youtube.com/watch?v=example", Path("/downloads"))
        Path('/downloads/example.wav')
    """
    little_helper.ensure_dir(downloads_dir)
    
    temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}"
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 
                            'preferredcodec': 'wav', 
                            'preferredquality': '192'}],
        'outtmpl': f'{downloads_dir}/{temp_filename}.%(ext)s'
    }
    try:
        with url_downloader.YoutubeDL(options) as ydl:
            logger.debug(f"Downloading {url}")
            
            # Download url
            ydl.download([url])
            video_info = ydl.extract_info(url, download=False)
            downloaded_file = list(downloads_dir.glob(f'{temp_filename}*'))[0]
            logger.debug(f"Download complete for {downloaded_file}")
            
            # Normalize title
            new_filename = re.sub(r'\W+', '_', video_info.get('title', 'downloaded_video'))
            
            # Remove trailing underscores
            if new_filename.startswith('_'):
                new_filename = new_filename[1:]
            if new_filename.endswith('_'):
                new_filename = new_filename[:-1]
            
            # Rename the file
            renamed_file = downloaded_file.rename(f"{downloads_dir}/{new_filename}{downloaded_file.suffix}")
            logger.debug(f"Renamed downloaded file to {renamed_file}")
            return Path(renamed_file)
        
    except Exception as e:
        print(f'Error downloading {url}: {e}')
        return None
    