# from sim.oct_tree import OctTreeCell
# from sim.coordinates import CartesianCoordinate
# from sim.particle import Particle
# from sim.utils import random_pos_in_box
from os import write
from geolib.coordinates import Point3D, Polar3D
from physlib.densityModels import UniformBox, UniformSphere, PlummerSphere
import matplotlib.pyplot as plt
import numpy as np
t = PlummerSphere()

samples = t.sample(int(2000))
# samples = [Point3D(p) for p in samples]
# samples = np.array([x.to_list() for x in samples]).transpose()
from utils.visualization import renderPointCloudInteractive, renderPointCloudDensityMap
# renderPointCloudInteractive(samples, 0.01, 1.0, zoom=40)
renderPointCloudDensityMap(samples)


# # sim settins
# num_particles = 1500
# domain_width = 1024
# domain_center = CartesianCoordinate(0, 0, 0)
#
#
# # harmonic values (?)0
# p = 4
# m = 3
# n = 3
#
#
# # init
# particles = [Particle(random_pos_in_box(domain_center, domain_width)) for i in range(num_particles)]
# oct_tree = OctTreeCell(domain_center, domain_width, 4)
# oct_tree.insert_particles(particles)
#
#
# # fmm loop
# oct_tree.compute_multipoles(m, n)
# oct_tree.apply_multipoles([], p, m, n)
# misfit_particles = oct_tree.extract_misfit_particles()
# oct_tree.insert_particles(misfit_particles)
#
# # integrate
#
# # visualize
