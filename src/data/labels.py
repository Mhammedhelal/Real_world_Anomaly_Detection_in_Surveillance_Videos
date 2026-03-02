"""
Label definitions for the UCF-Crime dataset.

This module contains only the taxonomy dictionary and helper functions to
map between numeric labels and human-readable class names. It intentionally
avoids any filesystem or OpenCV dependencies.
"""

# UCF-Crime Dataset Categories
UCF_CRIME_CATEGORIES = {
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
