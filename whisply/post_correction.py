from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Corrections:
    """
    A dataclass to encapsulate both simple and pattern-based corrections.
    """
    simple: Dict[str, str] = field(default_factory=dict)
    patterns: List[Dict[str, str]] = field(default_factory=list)
