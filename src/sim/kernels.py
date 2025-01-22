from coordinates import SphericalCoordinate
from particle import Particle
from cell import Cell

import numpy as np
import math
import cmath


def theta(r: SphericalCoordinate, m: int, n: int) -> complex:
    # TODO Is P actually the number of permutations or something else
    return (-1)**m * (math.factorial(n - m) / r.r**(n + 1)) * math.perm(n, m) * math.cos(r.theta) * cmath.exp(1j * m * r.phi)


def gamma(r: SphericalCoordinate, m: int, n: int) -> complex:
    # TODO Is P actually the number of permutations or something else
    return (-1)**m * (r.r**n / (math.factorial(n + m))) * math.perm(n, m) * math.cos(r.theta) * cmath.exp(1j * m * r.phi)


def p2p_kernel(particle: Particle, particle_prime: Particle) -> float:
    g = 6.67430e-11
    return (g * particle_prime.mass * particle.mass) / np.linalg.norm(particle_prime.position - particle.position)


def p2m_kernel(particle: Particle, cell: Cell, m: int, n: int) -> complex:
    return particle.mass * gamma(particle.position - cell.center, m, n)


def m2m_kernel(cell: Cell, cell_prime: Cell, m: int, n: int) -> complex:
    res = 0
    for k in range(n + 1):
        for l in range(-k, k + 1):
            res += gamma(cell.center - cell_prime.center, l, k) * cell.compute_multipoles(m - l, n - k)
    return res


def m2p_kernel(cell: Cell, particle: Particle, p: int, m: int, n: int) -> complex:
    res = 0
    for k in range(p - n + 1):
        for l in range(-k, k + 1):
            # TODO What is l* in the paper? Only thing I can think of is complex conjugate, but this does not make any sense
            res += cell.multipole * theta(particle.position - cell.center, m + l, n + k)
    return res
