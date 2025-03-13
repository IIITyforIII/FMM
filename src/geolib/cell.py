from geolib.coordinates import Polar3D, Point3D

from typing_extensions import Self


class Cell:
    center: Polar3D
    multipole: complex

    def __init__(self, center: Point3D):
        self.center = center.to_Polar3D()
        self.multipole = 0

    def compute_multipoles(self, m, n) -> complex:
        pass

    def apply_multipoles(self, cells: list[Self], p: int, m: int, n: int) -> None:
        pass
