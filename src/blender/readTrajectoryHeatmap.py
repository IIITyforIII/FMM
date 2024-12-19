import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv
import cv2

def readTrajectory(filename):
    path = os.getcwd()
    file_path = path + "\\" + filename

    trajectory_data = {}  # {frame: {vertex_id: (x, y, z)}}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=' ')
        for row in reader:
            frame = int(row['frame'])
            vertex_id = int(row['vertex'])      #can be ignored for heatmaps
            position = (float(row['x']), float(row['y']), float(row['z']))

            if frame not in trajectory_data:
                trajectory_data[frame] = {}
            trajectory_data[frame][vertex_id] = position

    return trajectory_data


def generate_heatmap(particle_positions, resolution):
    xCords = particle_positions[:, 0]               # we are plotting projected to the xy plane in this case
    yCords = particle_positions[:, 1]               # no fancy camera angle projections
    heatmap, xedges, yedges = np.histogram2d(xCords, yCords, bins=resolution)

    return heatmap


def generate_video(filename, trajectory,  resolution, zoom_margin=0.01):
    plt.clf()
    fig, ax = plt.subplots()
    heatmap_img = ax.imshow(np.zeros((resolution, resolution)), origin="lower", cmap="inferno",
                            aspect="auto")
    plt.colorbar(heatmap_img, ax=ax, label="Density")
    ax.set_title("Particle Heatmap Over Time")
    ax.set_xlabel("x-Position")
    ax.set_ylabel("y-Position")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    def update(frame):
        # 2D array containing only the positions for the current frame
        if frame >= 1:
            particle_positions = np.asarray([*trajectory.get(frame).values()])

            heatmap = generate_heatmap(particle_positions, resolution)
            heatmap_img.set_data(heatmap.T) #transposed because imshow expects y first
            #if frame == 1:
            heatmap_img.set_clim(vmin=0, vmax=np.max(heatmap))  # Adjust color scale
            heatmap_img.set_extent([-1, 1, -1, 1])


            """"# zoom to impoertant region (smart zoom)
            x_mean, y_mean = particle_positions[:, 0].mean(), particle_positions[:, 1].mean()
            x_std, y_std = particle_positions[:, 0].std(), particle_positions[:, 1].std()
            x_min, x_max = x_mean - 2 * x_std - zoom_margin, x_mean + 2 * x_std + zoom_margin
            y_min, y_max = y_mean - 2 * y_std - zoom_margin, y_mean + 2 * y_std + zoom_margin
            heatmap_img.set_extent([x_min, x_max, y_min, y_max])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)"""
        return heatmap_img

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=100)
    #writervideo = animation.FFMpegWriter(fps=60)
    #ani.save(filename, writer=writervideo)
    ani.save(filename, fps=10)
    #animation.save('animation.gif', writer='PillowWriter', fps=2)



if __name__ == "__main__":
    trajectory = readTrajectory("trajectory.csv")
    frameCount = len(trajectory)

    generate_video("video.mp4", trajectory, 100)