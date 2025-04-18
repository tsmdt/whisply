import logging
import datetime
import abc
from dotenv import load_dotenv, set_key
from dataclasses import field, dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

from whisply.utils import core_utils, output_utils
from whisply.utils.post_correction import Corrections


@dataclass(kw_only=True)
class BaseService(abc.ABC):
    """
    Base Service Class for local and LLM transcription services.
    """
    base_dir: Path = field(default_factory=lambda: Path('./transcriptions'))
    dotenv_path: Path = Path('.env').resolve()
    model_provided: str = 'large-v3-turbo'
    model: str = None
    file_language: Optional[str] = None
    annotate: bool = False
    subtitle: bool = False
    sub_length: Optional[int] = None
    translate: bool = False
    verbose: bool = False
    corrections: Optional[Corrections] = None
    export_formats: Any = output_utils.ExportFormats.TXT
    file_formats: set = field(init=False, default_factory=set)
    file_language_provided: bool = field(init=False, default=False)
    metadata: Dict[str, Any] = field(init=False, default_factory=dict)
    filepaths: List[Path] = field(init=False, default_factory=list)
    output_dir: Optional[Path] = field(init=False, default=None)
    processed_files: List[Dict] = field(init=False, default_factory=list)
    _logging_configured = False

    def __post_init__(self):
        """Set up logging and validate folders and metadata."""
        # Logging
        if not BaseService._logging_configured:
            self._setup_logging()
            BaseService._logging_configured = True
            logging.info("Logging configured for the first service instance.")
        logging.info(f"Post-Initializing {self.__class__.__name__} instance.")

        # Load .env
        if self.dotenv_path and Path(self.dotenv_path).is_file():
            load_dotenv(dotenv_path=self.dotenv_path, override=False)
            logging.debug(f".env file loaded from: {self.dotenv_path}")
        else:
            logging.debug(f".env not found; will create at {self.dotenv_path}.")

        # Validation and metadata
        self.base_dir = core_utils.ensure_dir(Path(self.base_dir))
        self.file_formats = core_utils.return_valid_fileformats()
        self.file_language_provided = self.file_language is not None
        self.metadata = self._collect_metadata()

        logging.info(f"Service Configuration: {self.metadata}")

    def _setup_logging(self):
        """Helper method to configure logging."""
        log_dir = core_utils.ensure_dir(Path('./logs'))
        log_filename = f"whisply_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        log_file = log_dir / log_filename
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s [%(name)s:%(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
            ]
        )

    def _collect_metadata(self) -> dict:
        """Collects key configuration parameters for logging or reporting."""
        return {
            'service_class': self.__class__.__name__,
            'base_dir': str(self.base_dir),
            'model': self.model,
            'file_language': self.file_language,
            'annotate': self.annotate,
            'translate': self.translate,
            'subtitle': self.subtitle,
            'sub_length': self.sub_length,
            'verbose': self.verbose,
            'export_formats': self.export_formats,
            }
    
    def _resolve_and_persist_api_key(
            self, 
            cli_provided_key: Optional[str], 
            env_var_name: str, 
            service_name: str
        ) -> Optional[str]:
        """
        Resolves API key and persists CLI key to .env file.
        """
        resolved_key = None

        # CLI
        if cli_provided_key:
            resolved_key = cli_provided_key
            logging.info(f"Using {service_name} key provided via CLI.")
            try:
                logging.info(
                    f"Saving/Updating {env_var_name} in {self.dotenv_path}"
                    )
                if not Path(self.dotenv_path).is_file():
                    Path(self.dotenv_path).touch()
                    logging.info(f"Created .env file at {self.dotenv_path}")
                # Store key
                success = set_key(
                    dotenv_path=self.dotenv_path, 
                    key_to_set=env_var_name, 
                    value_to_set=resolved_key, 
                    quote_mode='always'
                    )
                if not success:
                    logging.warning(
                    f"Failed to save {env_var_name} to {self.dotenv_path}."
                    )
            except Exception as e:
                logging.error(
                    f"Error saving {env_var_name} to {self.dotenv_path}: {e}"
                    )
        # .env
        else:
            env_key = load_dotenv(self.dotenv_path)
            if env_key:
                resolved_key = env_key
                logging.info(
                    f"Using {service_name} access token found in .env {self.dotenv_path.resolve()}."
                    )
            else:
                logging.info(
                f"{service_name} key not found."
                )
        return resolved_key

    @abc.abstractmethod
    def process_files(self, files: List[str]):
        """Processes the given list of file paths or identifiers."""
        pass
