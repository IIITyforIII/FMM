from geolib.coordinates import Polar3D
from geolib.cell import Cell
from physlib.entities import Particle

from scipy.special import assoc_legendre_p
import numpy as np
import math
import cmath


def theta(r: Polar3D, n: int, m: int) -> complex:
    return (-1)**m * (math.factorial(n - m) / r.r**(n + 1)) * assoc_legendre_p(n, m, math.cos(r.theta)) * cmath.exp(1j * m * r.phi)


def gamma(r: Polar3D, n: int, m: int) -> complex:
    return (-1)**m * (r.r**n / (math.factorial(n + m))) * assoc_legendre_p(n, m, math.cos(r.theta)) * cmath.exp(1j * m * r.phi)


def p2p_kernel(particle: Particle, particle_prime: Particle) -> float:
    g = 1#6.67430e-11
    return (g * particle_prime.mass * particle.mass) / np.linalg.norm((particle_prime.position - particle.position).to_Point3D().to_ndarray())


def p2m_kernel(particle: Particle, cell: Cell, n: int, m: int) -> complex:
    return particle.mass * gamma(particle.position - cell.center, n, m)


def m2m_kernel(from_cell: Cell, to_cell: Cell, n: int, m: int) -> complex:
    res = 0
    for k in range(n + 1):
        for l in range(-k, k + 1):
            res += gamma(from_cell.center - to_cell.center, k, l) * from_cell.get_multipole(n - k, m - l)
    return res


def m2p_kernel(cell: Cell, particle: Particle, p: int, m: int, n: int) -> complex:
    res = 0
    for k in range(p - n + 1):
        for l in range(-k, k + 1):
            if p >= n + k >= abs(m + l):
                res += cell.get_multipole(k, l).conjugate() * theta(particle.position - cell.center, n + k, m + l)
    return res
