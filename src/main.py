# from sim.oct_tree import OctTreeCell
# from sim.coordinates import CartesianCoordinate
# from sim.particle import Particle
# from sim.utils import random_pos_in_box
from geolib.coordinates import Point3D, Polar3D
from physlib.densityModels import uniformBox, uniformSphere
import matplotlib.pyplot as plt

import numpy as np
t = uniformSphere(radius=10)
samples = t.samplePolar3D(500)
samples = [Point3D(p) for p in samples]
samples = np.array([x.to_list() for x in samples]).transpose()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(samples[0], samples[1], samples[2])
ax.set_box_aspect((1,1,1))  # pyright: ignore
plt.show()


# # sim settings
# num_particles = 1500
# domain_width = 1024
# domain_center = CartesianCoordinate(0, 0, 0)
#
#
# # harmonic values (?)
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
