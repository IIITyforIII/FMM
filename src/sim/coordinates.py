from typing_extensions import Self
from dataclasses import dataclass
import math


@dataclass
class SphericalCoordinate:
    r: float
    theta: float
    phi: float

    def __add__(self: Self, other: Self) -> Self:
        return (self.as_cartesian_coordinates() + other.as_cartesian_coordinates()).as_spherical_coordinates()

    def __sub__(self: Self, other: Self) -> Self:
        return (self.as_cartesian_coordinates() - other.as_cartesian_coordinates()).as_spherical_coordinates()

    def as_cartesian_coordinates(self):
        x = self.r * math.sin(self.phi) * math.cos(self.theta)
        y = self.r * math.sin(self.phi) * math.sin(self.theta)
        z = self.r * math.cos(self.theta)
        return CartesianCoordinate(x, y, z)


@dataclass
class CartesianCoordinate:
    x: float
    y: float
    z: float

    def __add__(self: Self, other: Self) -> Self:
        return CartesianCoordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: Self) -> Self:
        return CartesianCoordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def as_spherical_coordinates(self) -> SphericalCoordinate:
        r = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = math.atan(self.y / self.x) if r != 0 else 0
        phi = math.acos(self.z / r) if r != 0 else 0
        return SphericalCoordinate(r, theta, phi)
