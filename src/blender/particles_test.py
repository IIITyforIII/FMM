import bpy
import numpy as np

# Lösche alle Objekte in der Szene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Initialisiere Partikel-Daten (zum Beispiel 1000 Partikel)
num_particles = 150
positions = np.random.rand(num_particles, 3) * 10  # zufällige Positionen
velocities = np.zeros((num_particles, 3))  # Startgeschwindigkeiten
accelerations = np.zeros((num_particles, 3))  # Startbeschleunigungen

# Erstelle Blender-Objekte für die Partikel
particles = []

#instance methode
# Erstelle eine einfache Kugel (nur einmal)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=positions[0])
particle_object = bpy.context.object
particle_object.name = "ParticleBase"
particles.append(particle_object)

for i in range(num_particles-1):
    # Erstelle ein Partikel (oder Objekt) an den berechneten Positionen
    #bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=positions[i])
    #particles.append(bpy.context.object)
    
    #instance of object methode
    new_instance = particle_object.copy()  # Erstelle eine Kopie der Kugel
    particles.append(new_instance)  # Positioniere sie an der berechneten Stelle
    bpy.context.collection.objects.link(new_instance)  # Füge sie der Sammlung hinzu
    
    # empty object methode
    #bpy.ops.object.empty_add(location=pos)
    #empty = bpy.context.object
    #empty.name = "ParticleEmpty"
    #particle.append(bpy.context.object)
    
    # Berechne die Beschleunigungen hier (zum Beispiel durch deine FMM-Methode)
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

# Simulation der Bewegung
start_frame=1
end_frame=2000
frame = start_frame
while frame < end_frame:
    # Berechne neue Beschleunigungen
    accelerations = calculate_accelerations(positions)
    
    # Aktualisiere Geschwindigkeiten und Positionen
    velocities += accelerations * 0.2 # Update mit einer kleinen Zeitschrittgröße (dt)
    positions += velocities * 0.2
    
    # Setze die neuen Positionen der Partikel
    for i, particle in enumerate(particles):
        particle.location = positions[i]
        # Füge einen Keyframe für das aktuelle Frame hinzu
        particle.keyframe_insert(data_path="location", frame=frame)
    
    frame += 1
    
    
# Setze die Start- und End-Frames der Animation
bpy.context.scene.frame_start = start_frame
bpy.context.scene.frame_end = end_frame