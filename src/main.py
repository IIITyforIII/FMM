from geolib.oct_tree import OctDynamicCell
from physlib.entities import Particle
from os import write
from geolib.coordinates import Point3D, Polar3D
from physlib.densityModels import UniformBox, UniformSphere, PlummerSphere
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.visualization as vis
import utils.heatmap as heatmap
import numpy as np

# sim settins


# harmonic values (?)0

# visualization settings
resolution = 100

# other
trace_filename = "../traceData/trace1.npy"
video_filename = "../heatmapVideos/video1.mp4"

def run_simulation(num_particles, domain_width, domain_center, time_steps, step_length, p):
    # init
    particles = [Particle(Point3D(particle), Polar3D(0, 0, 0)) for particle in
                 UniformBox(domain_center, domain_width, domain_width, domain_width).sample(num_particles)]
    oct_tree = OctDynamicCell(domain_center, domain_width, 8, p)
    oct_tree.insert_particles(particles)

    # fmm loop
    # position_data_vtk = np.zeros((time_steps, 3*num_particles), dtype=float)
    position_data = np.zeros((time_steps, num_particles, 3), dtype=float)
    for stepID in tqdm(range(time_steps), desc="Running Simulation: "):
        print("\nStarting Multipole Computation")
        oct_tree.compute_multipoles()
        print("\nApplying Multipoles")
        oct_tree.apply_multipoles([])
        print("\nRedistribute Particles")
        misfit_particles = oct_tree.extract_misfit_particles()
        oct_tree.insert_particles(misfit_particles)

        particle_position_list = oct_tree.get_particle_position_list()
        # all positions right afer each other as required by vtk to my knowledge (number of attributes needs to be 3)
        # particle_position_array1D = np.array(particle_position_list).ravel()
        # position_data_vtk[stepID] = particle_position_array1D

        position_data[stepID] = np.array(particle_position_list)

    # integrate
    return position_data


def visualize_simulation(position_data, resolution):
    # visualize
    # vis.renderPointCloudInteractive(position_data)
    heatmap.generate_video(video_filename, position_data, resolution)


if __name__ == "__main__":
    num_particles = 10000
    domain_width = 1024
    domain_center = Point3D(0, 0, 0)
    p = 4

    time_steps = 10
    step_length = 0.0001
    position_data = run_simulation(num_particles, domain_width, domain_center, time_steps, step_length, p)
    np.save(trace_filename, position_data)

    position_data = np.load(trace_filename)
    visualize_simulation(position_data, resolution)
