import bpy
import numpy as np
import csv

# Lösche alle Objekte in der Szene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Trajektorie-Daten aus CSV laden
trajectory_data = {}  # {frame: {vertex_id: (x, y, z)}}

file_path = "C:\\Users\\Luca_Dziarski\\Documents\\uni\\FMM\\src\\blender\\trajectory.csv"

with open(file_path, 'r') as file:
    reader = csv.DictReader(file, delimiter=' ')
    for row in reader:
        frame = int(row['frame'])
        vertex_id = int(row['vertex'])
        position = (float(row['x']), float(row['y']), float(row['z']))
        
        if frame not in trajectory_data:
            trajectory_data[frame] = {}
        trajectory_data[frame][vertex_id] = position
        
        
# Neues Mesh und Objekt erstellen
mesh = bpy.data.meshes.new("PointCloudMesh")
obj = bpy.data.objects.new("PointCloud", mesh)

# Objekt in die Szene einfügen
bpy.context.collection.objects.link(obj)

# Punkte dem Mesh hinzufügen
init = []
for vertex_id, position in trajectory_data[1].items():
    init.append(position)
mesh.from_pydata(init, [], [])
mesh.update()


# Simulation der Bewegung
end_frame = 1
for frame, vertices in trajectory_data.items():
    end_frame = frame
    for vertex_id, position in vertices.items():
        
        mesh.vertices[vertex_id].co = position
        # Füge einen Keyframe für das aktuelle Frame hinzu
        mesh.vertices[vertex_id].keyframe_insert(data_path="co", frame=frame)
    
    
# Setze die Start- und End-Frames der Animation
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = end_frame
