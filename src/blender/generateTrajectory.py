import numpy as np

# Initialisiere Partikel-Daten (zum Beispiel 1000 Partikel)
num_particles = 150
positions = np.random.rand(num_particles, 3) * 1  # zufällige Positionen
velocities = np.zeros((num_particles, 3))  # Startgeschwindigkeiten
accelerations = np.zeros((num_particles, 3))  # Startbeschleunigungen

def calculate_accelerations(positions):
    # Beispiel für eine einfache Beschleunigung (Gravitation)
    G = 6.67430e-11  # Gravitationskonstante
    masses = np.ones(num_particles)*1000000
    masses[0] = 10000000000000000000000000000  # Alle Partikel haben dieselbe Masse
    accelerations = np.zeros_like(positions)
    
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r = positions[i] - positions[j]
            distance = np.linalg.norm(r)
            force = G * masses[i] * masses[j] / (distance ** 2)
            acceleration = force / masses[i]
            accelerations[i] -= acceleration * r / distance  # Beschleunigung auf Partikel i
            accelerations[j] += acceleration * r / distance  # Beschleunigung auf Partikel j
    
    return accelerations


# Simulation der Bewegung und schreibe file
import csv
file = open('trajectory.csv', 'w')
header = ['frame', 'vertex', 'x', 'y', 'z']
writer = csv.writer(file, delimiter=' ')
writer.writerow(header)


start_frame=1
end_frame=2000
frame = start_frame
while frame < end_frame:
    if frame % 50 == 0:
        print("{} of {} frames done!".format(frame, end_frame))
    # Berechne neue Beschleunigungen
    accelerations = calculate_accelerations(positions)
    
    # Aktualisiere Geschwindigkeiten und Positionen
    velocities += accelerations * 0.1 # Update mit einer kleinen Zeitschrittgröße (dt)
    positions += velocities * 0.1
    
    # Setze die neuen Positionen der Partikel
    for i, pos in enumerate(positions):
        data = [frame, i, pos[0], pos[1], pos[2]]
        writer.writerow(data)

    
    frame += 1

file.close()
