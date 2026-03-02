"""
Label definitions for the UCF-Crime dataset.

This module exposes helper functions for working with the label taxonomy.
The actual mapping is read from configuration at import time, which allows
users to override or extend the categories by editing
`configs/default.yaml` or supplying an alternate config file.
"""

from config import Config

# load configuration once
_cfg = Config.from_yaml('configs/default.yaml')

# fallback hard-coded map in case config is missing section
_DEFAULT_UCF_CATEGORIES = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Arrest',
    3: 'Arson',
    4: 'Assault',
    5: 'Burglary',
    6: 'Explosion',
    7: 'Fighting',
    8: 'Robbery',
    9: 'Shooting',
    10: 'Shoplifting',
    11: 'Stealing',
    12: 'Vandalism',
    13: 'RoadAccidents'
}

_raw_categories = _cfg.get('labels', {}).get('ucf_crime_categories', _DEFAULT_UCF_CATEGORIES)
# convert from Namespace to plain dict if necessary
if hasattr(_raw_categories, 'to_dict'):
    UCF_CRIME_CATEGORIES = _raw_categories.to_dict()
else:
    UCF_CRIME_CATEGORIES = _raw_categories


def get_class_name(label):
    """Return the class name corresponding to a numeric label index."""
    return UCF_CRIME_CATEGORIES.get(label, 'Unknown')


def get_label_from_name(class_name):
    """Return the numeric label index for a given class name.

    Comparison is case‑insensitive. Returns ``None`` if no match is found.
    """
    for label, name in UCF_CRIME_CATEGORIES.items():
        if name.lower() == class_name.lower():
            return label
    return None
