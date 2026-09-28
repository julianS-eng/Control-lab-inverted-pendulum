"""Shared type aliases."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Array of 64-bit floats. Shapes are documented at each use site.
FloatArray = NDArray[np.float64]

#: Array of booleans.
BoolArray = NDArray[np.bool_]

#: Array of complex numbers (poles, frequency responses).
ComplexArray = NDArray[np.complex128]
