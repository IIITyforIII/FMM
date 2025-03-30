from geolib.coordinates import Polar3D, Point3D

from typing import Self

import numpy as np


class Cell:
    center: Polar3D
    multipole: complex

    def __init__(self: Self, center: Point3D, expansion_order: int):
        self.center = center.to_Polar3D()
        self.p: int = expansion_order
        self.multipoles = np.empty((self.p + 1, 2 * self.p + 1), dtype=complex)

    def compute_multipoles(self: Self) -> None:
        pass

    def apply_multipoles(self, cells: list[Self]) -> None:
        pass

    def get_multipole(self: Self, n: int, m: int) -> complex:
        if self.p >= n >= abs(m):
            return self.multipoles[n, m]
        else:
            return 0j
