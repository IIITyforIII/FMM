import pyvista as pv
import numpy as np 
import os 
path = os.path.dirname(os.path.realpath(__file__))

data = np.genfromtxt(path + '/../../data/spiral_galaxy.csv', delimiter=',')
data = np.delete(data, 0,0)
data = np.delete(data, 3,1)


pl = pv.Plotter()
pl.background_color = 'black'

pdata = pv.PointSet(data)

pl.add_mesh(pdata, point_size=0.2, style='points_gaussian', render_points_as_spheres=False, emissive=False, opacity=0.2, color='white')

pl.show()





