import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Corrections:
    """
    A dataclass to encapsulate both simple and pattern-based corrections.
    """
    simple: Dict[str, str] = field(default_factory=dict)
    patterns: List[Dict[str, str]] = field(default_factory=list)
    
    
def load_correction_list(filepath: str | Path) -> Corrections:
    """
    Load the correction dictionary and patterns from a YAML file.

    :param filepath: Path to the YAML correction file.
    :return: Corrections object containing simple and pattern-based 
        corrections.
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError("→ Correction file must contain a YAML dictionary.")

        # Extract simple corrections
        simple_corrections = {k: v for k, v in data.items() if k != 'patterns'}

        # Extract pattern-based corrections
        pattern_corrections = data.get('patterns', [])

        # Validate patterns
        for entry in pattern_corrections:
            if 'pattern' not in entry or 'replacement' not in entry:
                raise ValueError("→ Each pattern entry must contain 'pattern' \
and 'replacement' keys.")

        return Corrections(
            simple=simple_corrections, 
            patterns=pattern_corrections
            )

    except FileNotFoundError:
        print(f"→ Correction file not found: {filepath}")
        return Corrections()
    except yaml.YAMLError as e:
        print(f"→ Error parsing YAML file: {e}")
        return Corrections()
    except Exception as e:
        print(f"→ Unexpected error loading correction list: {e}")
        return Corrections()
