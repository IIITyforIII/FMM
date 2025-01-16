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

    def __sub__(self: Self, other: Self) -> Self:
        return self.r - other.r, self.theta - other.theta, self.phi - other.phi


@dataclass
class CartesianCoordinate:
    x: float
    y: float
    z: float


@dataclass
class Particle:
    position: SphericalCoordinate
    mass: float
    psi: complex


class Cell:
    center: SphericalCoordinate
    multipole: complex

    def compute_multipoles(self, m, n) -> complex:
        pass

    def apply_multipoles(self, cells: list[Self], p: int, m: int, n: int) -> None:
        pass


class TreeCell(Cell):
    def __init__(self):
        self.sub_cells: list[Cell] = []

    def compute_multipoles(self, m: int, n: int) -> complex:
        self.multipole = np.sum([m2m_kernel(cell, self, m, n) for cell in self.sub_cells]).item()
        return self.multipole

    def apply_multipoles(self, cells: list[Cell], p: int, m: int, n: int) -> None:
        for cell in self.sub_cells:
            cell.apply_multipoles(cells + [c for c in self.sub_cells if c != cell], p, m, n)


class LeafCell(Cell):

    def __init__(self):
        self.particles: list[Particle] = []

    def compute_multipoles(self, m: int, n: int) -> complex:
        self.multipole = np.sum([p2m_kernel(particle, self, m, n) for particle in self.particles]).item()
        return self.multipole

    def apply_multipoles(self, cells: list[Cell], p: int, m: int, n: int) -> None:
        for particle in self.particles:
            particle.psi = 0
            for cell in cells:
                particle.psi += m2p_kernel(cell, particle, p, m, n)
            for particle_prime in [p for p in self.particles if p != particle]:
                particle.psi += p2p_kernel(particle, particle_prime)


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
