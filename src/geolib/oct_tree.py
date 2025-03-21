from fmm.kernels import p2m_kernel, m2m_kernel, m2p_kernel, p2p_kernel
from geolib.cell import Cell
from physlib.entities import Particle
from geolib.coordinates import Point3D
from dataclasses import astuple

import numpy as np

from typing import Self
from typing_extensions import override


class OctCell(Cell):

    def __init__(self, center: Point3D, width: float, expansion_order: int):
        super().__init__(center, expansion_order)
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

    def get_particle_position_list(self) -> list[np.ndarray]:
        """
        Get all particles in this OctCell

        :return: list[np.ndarray] in the form [ndarray(x1,y1,z1), ndarray(x2,y2,z2),...] containing all the cartesian
        positions for the particles contained in the OctCell
        """
        pass


class OctDynamicCell(OctCell):

    def __init__(self: Self, center: Point3D, width: float, depth_remaining: int, expansion_order: int):
        super().__init__(center, width, expansion_order)
        self.depth_remaining = depth_remaining
        self.is_leaf = True
        self.sub_cells: list[Self] = []
        self.particles: list[Particle] = []

    def is_empty(self):
        if not self.is_leaf:
            return all([cell.is_empty() for cell in self.sub_cells])
        else:
            return not self.particles

    def transform_to_leaf_cell(self: Self) -> None:
        assert not self.is_leaf
        self.sub_cells = []

    def transform_to_internal_cell(self: Self) -> None:
        assert (self.depth_remaining > 0 and self.is_leaf)
        self.is_leaf = False
        offsets = [Point3D(self.width / 4, self.width / 4, self.width / 4),
                   Point3D(-self.width / 4, self.width / 4, self.width / 4),
                   Point3D(self.width / 4, -self.width / 4, self.width / 4),
                   Point3D(-self.width / 4, -self.width / 4, self.width / 4),
                   Point3D(self.width / 4, self.width / 4, -self.width / 4),
                   Point3D(-self.width / 4, self.width / 4, -self.width / 4),
                   Point3D(self.width / 4, -self.width / 4, -self.width / 4),
                   Point3D(-self.width / 4, -self.width / 4, -self.width / 4)]
        for offset in offsets:
            self.sub_cells.append(
                OctDynamicCell(self.center.to_Point3D() + offset, self.width / 2, self.depth_remaining - 1, self.p))

    @override
    def insert_particles(self: Self, particles: list[Particle]) -> None:
        if not particles:
            return

        if self.is_leaf and self.depth_remaining > 0 and (len(self.particles) + len(particles)) > 100:
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
    def apply_multipoles(self, cells: list[Self]) -> None:
        # Apply multipoles for every well-separated cell
        if not self.is_leaf:
            for cell in self.sub_cells:
                cell.apply_multipoles(cells + [c for c in self.sub_cells if c != cell])
        else:
            for particle in self.particles:
                for cell in cells:
                    psi_1_0 = m2p_kernel(cell, particle, self.p, 1, 0)
                    psi_1_1 = m2p_kernel(cell, particle, self.p, 1, 1)
                    # TODO Would this be the correct results? How do we handle the influence of particles in the same cell?
                    particle.acc = (-psi_1_1.real, -psi_1_1.imag, -psi_1_0)
                # for particle_prime in [p for p in self.particles if p != particle]:
                #     particle.psi += p2p_kernel(particle, particle_prime)

    @override
    def compute_multipoles(self: Self) -> None:
        # Compute Multipoles of sub_cells
        if not self.is_leaf:
            for cell in self.sub_cells:
                cell.compute_multipoles()

        # Compute own multipoles
        for n in range(self.p + 1):
            for m in range(-n, n + 1):
                if not self.is_leaf:
                    # TODO: Is that correct? This now sums up all results from eq.3d (m2m kernel) from all sub_cells
                    self.multipoles[n, m] = np.sum([m2m_kernel(cell, self, n, m) for cell in self.sub_cells]).item()
                else:
                    self.multipoles[n, m] = np.sum(
                        np.array([p2m_kernel(particle, self, n, m) for particle in self.particles])).item()

    @override
    def get_particle_position_list(self) -> list[np.ndarray]:
        particle_positions = []
        if not self.is_leaf:
            for cell in self.sub_cells:
                particle_positions += cell.get_particle_position_list()
        else:
            for p in self.particles:
                position_point = astuple(p.position.to_Point3D())
                particle_positions.append(np.array(position_point))
        return particle_positions
