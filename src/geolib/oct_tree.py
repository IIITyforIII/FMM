from fmm.kernels import p2p_kernel, p2m_kernel, m2m_kernel, m2p_kernel
from cell import Cell
from physlib.entities import Particle
from geolib.coordinates import Point3D

import numpy as np


class OctCell(Cell):

    def __init__(self, center: Point3D, width: float):
        super().__init__(center)
        self.width = width

    def extract_misfit_particles(self) -> list[Particle]:
        pass

    def insert_particles(self, particles: list[Particle]) -> None:
        pass

    def contains_particle(self, particle: Particle) -> bool:
        p = particle.position.to_Point3D()
        c = self.center.to_Point3D()
        return c.x - self.width / 2 <= p.x < c.x + self.width / 2 \
            and c.y - self.width / 2 <= p.y < c.y + self.width / 2 \
            and c.z - self.width / 2 <= p.z < c.z + self.width / 2


class OctTreeCell(OctCell):

    def __init__(self, center: Point3D, width: float, depth: int):
        super().__init__(center, width)
        if depth > 0:
            self.sub_cells: list[OctCell] = [
                OctTreeCell(center + Point3D(width / 4, width / 4, width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(-width / 4, width / 4, width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(width / 4, -width / 4, width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(-width / 4, -width / 4, width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(width / 4, width / 4, -width / 4), width / 2, depth - 1,),
                OctTreeCell(center + Point3D(-width / 4, width / 4, -width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(width / 4, -width / 4, -width / 4), width / 2, depth - 1),
                OctTreeCell(center + Point3D(-width / 4, -width / 4, -width / 4), width / 2, depth - 1)]
        else:
            self.sub_cells: list[OctCell] = [
                OctLeafCell(center + Point3D(width / 4, width / 4, width / 4), width / 2),
                OctLeafCell(center + Point3D(-width / 4, width / 4, width / 4), width / 2),
                OctLeafCell(center + Point3D(width / 4, -width / 4, width / 4), width / 2),
                OctLeafCell(center + Point3D(-width / 4, -width / 4, width / 4), width / 2),
                OctLeafCell(center + Point3D(width / 4, width / 4, -width / 4), width / 2),
                OctLeafCell(center + Point3D(-width / 4, width / 4, -width / 4), width / 2),
                OctLeafCell(center + Point3D(width / 4, -width / 4, -width / 4), width / 2),
                OctLeafCell(center + Point3D(-width / 4, -width / 4, -width / 4), width / 2)]

    def compute_multipoles(self, m: int, n: int) -> complex:
        self.multipole = np.sum([m2m_kernel(cell, self, m, n) for cell in self.sub_cells]).item()
        return self.multipole

    def apply_multipoles(self, cells: list[OctCell], p: int, m: int, n: int) -> None:
        for cell in self.sub_cells:
            cell.apply_multipoles(cells + [c for c in self.sub_cells if c != cell], p, m, n)

    def extract_misfit_particles(self) -> list[Particle]:
        misfits = []
        for cell in self.sub_cells:
            misfits += cell.extract_misfit_particles()
        return misfits

    def insert_particles(self, particles: list[Particle]) -> None:
        for cell in self.sub_cells:
            cell.insert_particles([p for p in particles if cell.contains_particle(p)])


class OctLeafCell(OctCell):

    def __init__(self, center: Point3D, width: float):
        super().__init__(center, width)
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

    def extract_misfit_particles(self) -> list[Particle]:
        fits = [p for p in self.particles if self.contains_particle(p)]
        misfits = [p for p in self.particles if not self.contains_particle(p)]
        self.particles = fits
        return misfits

    def insert_particles(self, particles: list[Particle]) -> None:
        self.particles += particles
