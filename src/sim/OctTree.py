from kernels import *

import random


class OctCell(Cell):
    width: float

    def extract_misfit_particles(self) -> list[Particle]:
        pass

    def insert_particles(self, particles: list[Particle]) -> None:
        pass

    def contains_particle(self, particle: Particle) -> bool:
        p = particle.position.as_cartesian_coordinates()
        c = self.center.as_cartesian_coordinates()
        return c.x - self.width / 2 <= p.x < c.x + self.width / 2 \
            and c.y - self.width / 2 <= p.y < c.y + self.width / 2 \
            and c.z - self.width / 2 <= p.z < c.z + self.width / 2


class OctTreeCell(OctCell):
    def __init__(self, center: CartesianCoordinate, width: float, depth: int, num_particles_per_leaf: int):
        self.center = center.as_spherical_coordinates()
        self.width = width
        self.multipole = 0

        if depth > 0:
            self.sub_cells: list[OctCell] = [
                OctTreeCell(center + CartesianCoordinate(width / 4, width / 4, width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(-width / 4, width / 4, width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(width / 4, -width / 4, width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(-width / 4, -width / 4, width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(width / 4, width / 4, -width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(-width / 4, width / 4, -width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(width / 4, -width / 4, -width / 4), width / 2, depth - 1, num_particles_per_leaf),
                OctTreeCell(center + CartesianCoordinate(-width / 4, -width / 4, -width / 4), width / 2, depth - 1, num_particles_per_leaf)]
        else:
            self.sub_cells: list[OctCell] = [
                OctLeafCell(center + CartesianCoordinate(width / 4, width / 4, width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(-width / 4, width / 4, width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(width / 4, -width / 4, width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(-width / 4, -width / 4, width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(width / 4, width / 4, -width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(-width / 4, width / 4, -width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(width / 4, -width / 4, -width / 4), width / 2, num_particles_per_leaf),
                OctLeafCell(center + CartesianCoordinate(-width / 4, -width / 4, -width / 4), width / 2, num_particles_per_leaf)]

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

    def __init__(self, center: CartesianCoordinate, width: float, num_particles_per_leaf: int):
        self.center = center.as_spherical_coordinates()
        self.width = width
        self.multipole = 0
        self.particles: list[Particle] = [Particle(center + CartesianCoordinate(random.uniform(-width / 2, width / 2), random.uniform(-width / 2, width / 2), random.uniform(-width / 2, width / 2)), 100) for i in range(num_particles_per_leaf)]

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
