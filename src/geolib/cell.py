from coordinates import SphericalCoordinate, CartesianCoordinate

from typing_extensions import Self


class Cell:
    center: SphericalCoordinate
    multipole: complex

    def __init__(self, center: CartesianCoordinate):
        self.center = center.as_spherical_coordinates()
        self.multipole = 0

    def compute_multipoles(self, m, n) -> complex:
        pass

    def apply_multipoles(self, cells: list[Self], p: int, m: int, n: int) -> None:
        pass
