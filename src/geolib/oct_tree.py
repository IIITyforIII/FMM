from fmm.kernels import p2p_kernel, p2m_kernel, m2m_kernel, m2p_kernel
from geolib.cell import Cell
from physlib.entities import Particle
from geolib.coordinates import Point3D
from dataclasses import astuple

import numpy as np

from typing import Self
from typing_extensions import override

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

    def get_particle_position_list(self) ->list[np.ndarray]:
        """
        Get all particles in this OctCell

        :return: list[np.ndarray] in the form [ndarray(x1,y1,z1), ndarray(x2,y2,z2),...] containing all the cartesian
        positions for the particles contained in the OctCell
        """
        pass


class OctDynamicCell(OctCell):

    def __init__(self: Self, center: Point3D, width: float, depth_remaining: int):
        super().__init__(center, width)
        self.depth_remaining = depth_remaining
        self.is_leaf = True
        self.sub_cells : list[Self] = []
        self.particles : list[Particle] = []
        self.multipole: complex

    def is_empty(self):
        if not self.is_leaf:
            return all([cell.is_empty() for cell in self.sub_cells])
        else:
            return not self.particles

    def transform_to_leaf_cell(self: Self) -> None:
        assert(not self.is_leaf)
        self.sub_cells = []

    def transform_to_internal_cell(self: Self) -> None:
        assert(self.depth_remaining > 0 and self.is_leaf)
        self.sub_cells = [
            OctDynamicCell(self.center.to_Point3D() + Point3D(self.width / 4, self.width / 4, self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(-self.width / 4, self.width / 4, self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(self.width / 4, -self.width / 4, self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(-self.width / 4, -self.width / 4, self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(self.width / 4, self.width / 4, -self.width / 4), self.width / 2, self.depth_remaining - 1, ),
            OctDynamicCell(self.center.to_Point3D() + Point3D(-self.width / 4, self.width / 4, -self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(self.width / 4, -self.width / 4, -self.width / 4), self.width / 2, self.depth_remaining - 1),
            OctDynamicCell(self.center.to_Point3D() + Point3D(-self.width / 4, -self.width / 4, -self.width / 4), self.width / 2, self.depth_remaining - 1)]
        self.is_leaf = False

    @override
    def insert_particles(self: Self, particles: list[Particle]) -> None:
        if not particles:
            return

        if self.is_leaf and self.depth_remaining > 0:
            self.transform_to_internal_cell()

        if not self.is_leaf:
            for cell in self.sub_cells:
                cell.insert_particles([p for p in particles if cell.contains_particle(p)])
        else:
            self.particles += particles

    @override
    def extract_misfit_particles(self) -> list[Particle]:
        misfits = []

        if not self.is_leaf:
            for cell in self.sub_cells:
                misfits += cell.extract_misfit_particles()
            if self.is_empty():
                self.transform_to_leaf_cell()
        else:
            fits = [p for p in self.particles if self.contains_particle(p)]
            misfits = [p for p in self.particles if not self.contains_particle(p)]
            self.particles = fits

        return misfits

    @override
    def apply_multipoles(self, cells: list[Self], p: int, m: int, n: int) -> None:
        if not self.is_leaf:
            for cell in self.sub_cells:
                cell.apply_multipoles(cells + [c for c in self.sub_cells if c != cell], p, m, n)
        else:
            for particle in self.particles:
                particle.psi = 0
                for cell in cells:
                    particle.psi += m2p_kernel(cell, particle, p, m, n)
                for particle_prime in [p for p in self.particles if p != particle]:
                    particle.psi += p2p_kernel(particle, particle_prime)

    @override
    def compute_multipoles(self, m, n) -> complex:
        if not self.is_leaf:
            self.multipole = np.sum([m2m_kernel(cell, self, m, n) for cell in self.sub_cells]).item()
        else:
            self.multipole = np.sum([p2m_kernel(particle, self, m, n) for particle in self.particles]).item()
        return self.multipole

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

    def get_particle_position_list(self) -> list[np.ndarray]:
        particle_positions = []
        for cell in self.sub_cells:
            particle_positions += cell.get_particle_position_list()
        return particle_positions


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

    def get_particle_position_list(self) -> list[np.ndarray]:
        particle_positions = []
        for p in self.particles:
            position_point = astuple(p.position.to_Point3D())
            particle_positions.append(np.array(position_point))
        return particle_positions
