"""Units of length the codecs read and write, and how they convert."""

from __future__ import annotations

from typing import Any

# Millimetres per unit, under every spelling a codec reads a unit as: 3MF's
# and AMF's names, and the names DXF's $INSUNITS codes read as.
MILLIMETRES: dict[str, float] = {
    "angstrom": 1e-7,
    "nanometer": 1e-6,
    "microinch": 25.4e-6,
    "micron": 1e-3,
    "micrometer": 1e-3,
    "mil": 25.4e-3,
    "millimeter": 1.0,
    "centimeter": 10.0,
    "inch": 25.4,
    "us_survey_inch": 1e5 / 3937,
    "decimeter": 100.0,
    "foot": 304.8,
    "feet": 304.8,
    "us_survey_foot": 1.2e6 / 3937,
    "yard": 914.4,
    "us_survey_yard": 3.6e6 / 3937,
    "meter": 1e3,
    "decameter": 1e4,
    "hectometer": 1e5,
    "kilometer": 1e6,
    "mile": 1_609_344.0,
    "us_survey_mile": 6.336e9 / 3937,
    "gigameter": 1e12,
    "astronomical_unit": 1.495978707e14,
    "light_year": 9.4607304725808e18,
    "parsec": 3.0856775814913673e19,
}


def conversion(unit: Any, target: str) -> float | None:
    """Return the factor that turns lengths in ``unit`` into lengths in ``target``.

    Parameters
    ----------
    unit
        The unit the lengths are in, as a mesh's ``global_attrs["unit"]``
        holds it.
    target
        The unit to convert to, a key of ``MILLIMETRES``.

    Returns
    -------
    float or None
        The factor, or None when ``unit`` is not a length this module knows.
    """
    if not isinstance(unit, str) or unit not in MILLIMETRES:
        return None
    return MILLIMETRES[unit] / MILLIMETRES[target]
