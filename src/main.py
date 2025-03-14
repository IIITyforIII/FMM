from geolib.oct_tree import OctTreeCell
from geolib.coordinates import Point3D
from oct_tree import OctDynamicCell
from physlib.entities import Particle
from physlib.densityModels import UniformBox
from os import write
from geolib.coordinates import Point3D, Polar3D
from physlib.densityModels import UniformBox, UniformSphere, PlummerSphere
import matplotlib.pyplot as plt
import numpy as np
# t = PlummerSphere()
#
# samples = t.sample(int(1000000))
# # samples = [Point3D(p) for p in samples]
# # samples = np.array([x.to_list() for x in samples]).transpose()
# from utils.visualization import renderPointCloudInteractive, renderPointCloudDensityMap
# # renderPointCloudInteractive(samples, scaleFactor=0.01, zoom=40)
# renderPointCloudDensityMap(samples, radius=0.2, dimensions=(100,100,100), focalPoint=(0,0,0))

# sim settins
num_particles = 1500
domain_width = 1024
domain_center = Point3D(0, 0, 0)


# harmonic values (?)0
p = 4
m = 3
n = 3

# for k in range(p - n + 1):
#     for l in range(-k, k + 1, 1):
#         print("M: " + str(abs(l)) + "  N:" + str(k))

# init
particles = [Particle(Point3D(p), Polar3D(0, 0, 0)) for p in UniformBox(domain_center, domain_width, domain_width, domain_width).sample(num_particles)]
oct_tree = OctDynamicCell(domain_center, domain_width, 8)
oct_tree.insert_particles(particles)


# fmm loop
oct_tree.compute_multipoles(m, n)
oct_tree.apply_multipoles([], p, m, n)
misfit_particles = oct_tree.extract_misfit_particles()
oct_tree.insert_particles(misfit_particles)

# integrate

# visualize
