from typing import Tuple, TypeAlias, Union

Num: TypeAlias = Union[int, float]
"""A number that can be either an integer or a float."""

Rectangle: TypeAlias = Tuple[Num, Num, Num, Num]
"""A rectangle defined by (x1, y1, x2, y2) coordinates."""

Color: TypeAlias = Tuple[Num, Num, Num]
"""An RGB color represented as a tuple (R, G, B)."""
