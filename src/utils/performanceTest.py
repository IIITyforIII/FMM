import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simlib.simulators import Simulator, nbodyDirectSimulator, fmmSimulator
from physlib.densityModels import PlummerSphere, UniformSphere
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import simulationBase


def get_avg_times_for_number_of_particles(domain_min, domain_max, expansion_order:int, min_num_particles:int,
                                          max_num_particles:int, particle_increase_step:int, core_rad=10, sample_size=10):
    number_of_measurements = int((max_num_particles-min_num_particles)/particle_increase_step)
    average_time_direct = np.zeros(number_of_measurements, dtype=float)
    average_time_fmm = np.zeros(number_of_measurements, dtype=float)
    iterationCounter = 0
    directSimStillPossible=True
    directSimMaxSeconds = 30.0
    for num_particles in tqdm(range(min_num_particles, max_num_particles, particle_increase_step)):
        # run simulation
        pos, vel = simulationBase.createInitState(num_particles, core_rad=core_rad)
        mass = np.ones(len(pos))
        direct_simulator = nbodyDirectSimulator(pos, vel, mass)
        fmm_simulator = fmmSimulator(pos, vel, domain_min, domain_max, mass, expansion_order)

        for i in range(sample_size):
            if directSimStillPossible:
                start = timer()
                direct_simulator.step(1)
                end = timer()
                average_time_direct[iterationCounter] += end - start
                #if end-start > directSimMaxSeconds:
                #    directSimStillPossible=False
                 #   print("directSim no longer done-")
                #    average_time_direct[iterationCounter] += directSimMaxSeconds
                #else:
                   # average_time_direct[iterationCounter] += end - start
            else:
                average_time_direct[iterationCounter] += directSimMaxSeconds

            start = timer()
            fmm_simulator.step(1)
            end = timer()
            average_time_fmm[iterationCounter] += end-start

        average_time_direct[iterationCounter] = average_time_direct[iterationCounter]/sample_size
        average_time_fmm[iterationCounter] = average_time_fmm[iterationCounter]/sample_size
        iterationCounter+=1

    return average_time_direct, average_time_fmm


def plot_avg_times(average_time_direct, average_time_fmm, min_num_particles, max_num_particles, particle_increase_step, path):
    number_of_measurements = int((max_num_particles - min_num_particles) / particle_increase_step)
    xTicks = np.arange(number_of_measurements)
    xTicksLabels = min_num_particles +( np.arange(0, number_of_measurements) * particle_increase_step)

    plt.clf()
    plt.plot(average_time_direct, color='b')
    plt.plot(average_time_fmm, color='g')
    plt.legend(["Direct Summation", "Fast Multipole"])
    plt.xticks(xTicks, xTicksLabels, rotation="vertical", fontsize=6)
    plt.xlabel("Nr. of Particles")
    plt.ylabel("Average step computation time")
    plt.savefig(path)


if __name__ == '__main__':
    dMax = np.array([50, 50, 50])
    dMin = -1 * dMax
    expansion_order = 4

    min_num_particles = 10000
    max_num_particles = 25000
    particle_increase_step = 1000

    average_time_direct, average_time_fmm = get_avg_times_for_number_of_particles(dMin, dMax,
                                expansion_order, min_num_particles, max_num_particles, particle_increase_step)

    #path = "../data/performance_over_particles.png"
    path = "performance_over_particles.png"
    plot_avg_times(average_time_direct, average_time_fmm, min_num_particles, max_num_particles,
                   particle_increase_step, path)