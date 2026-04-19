"""
src/data/labels.py
------------------
Label definitions for the UCF-Crime dataset.
"""

# Fallback hard-coded map (used when config section is absent)
_DEFAULT_UCF_CATEGORIES = {
    0: 'Normal',    1: 'Abuse',      2: 'Arrest',
    3: 'Arson',     4: 'Assault',    5: 'Burglary',
    6: 'Explosion', 7: 'Fighting',   8: 'Robbery',
    9: 'Shooting',  10: 'Shoplifting', 11: 'Stealing',
    12: 'Vandalism', 13: 'RoadAccidents',
}

try:
    from src.config import Config
    _cfg = Config.from_yaml('configs/default.yaml')
    _raw = _cfg.get('labels', {}).get('ucf_crime_categories', _DEFAULT_UCF_CATEGORIES)
    UCF_CRIME_CATEGORIES = _raw.to_dict() if hasattr(_raw, 'to_dict') else _raw
except Exception:
    UCF_CRIME_CATEGORIES = _DEFAULT_UCF_CATEGORIES


def get_class_name(label: int) -> str:
    """Return the class name for a numeric label."""
    return UCF_CRIME_CATEGORIES.get(label, 'Unknown')


def get_label_from_name(class_name: str) -> int | None:
    """Return the numeric label for a class name (case-insensitive)."""
    for label, name in UCF_CRIME_CATEGORIES.items():
        if name.lower() == class_name.lower():
            return label
    return None
