from dataclasses import dataclass

import numpy as np
import math
import cmath

from typing import Self
G = 6.67430e-11


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
        theta = math.atan(self.y / self.x)
        phi = math.acos(self.z / r)
        return SphericalCoordinate(r, theta, phi)


class Particle:
    position: SphericalCoordinate
    mass: float
    psi: complex

    def __init__(self, position: CartesianCoordinate, mass: float):
        self.position: position.as_spherical_coordinates()
        self.mass = mass
        self.psi = 0


class Cell:
    center: SphericalCoordinate
    multipole: complex

    def compute_multipoles(self, m, n) -> complex:
        pass

    def apply_multipoles(self, cells: list[Self], p: int, m: int, n: int) -> None:
        pass


def theta(r: SphericalCoordinate, m: int, n: int) -> complex:
    return (-1)**m * (math.factorial(n - m) / r.r**(n + 1)) * math.perm(n, m) * math.cos(r.theta) * cmath.exp(1j * m * r.phi)


def gamma(r: SphericalCoordinate, m: int, n: int) -> complex:
    return (-1)**m * (r.r**n / (math.factorial(n + m))) * math.perm(n, m) * math.cos(r.theta) * cmath.exp(1j * m * r.phi)


def p2p_kernel(particle: Particle, particle_prime: Particle) -> float:
    return (G * particle_prime.mass * particle.mass) / np.linalg.norm(particle_prime.position - particle.position)


def p2m_kernel(particle: Particle, cell: Cell, m: int, n: int) -> complex:
    return particle.mass * gamma(particle.position - cell.center, m, n)


def m2m_kernel(cell: Cell, cell_prime: Cell, m: int, n: int) -> complex:
    res = 0
    for k in range(n + 1):
        for l in range(-k, k + 1):
            res += gamma(cell.center - cell_prime.center, k, l) * cell.compute_multipoles(m, n)
    return res


def m2p_kernel(cell: Cell, particle: Particle, p: int, m: int, n: int) -> complex:
    res = 0
    for k in range(p - n + 1):
        for l in range(-k, k + 1):
            # What is l* in the paper? Only thing I can think of is complex conjugate, but this does not make any sense
            res += cell.multipole * theta(particle.position - cell.center, m + l, n + k)
    return res
