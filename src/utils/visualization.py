from typing import Union
from numpy import ndarray

from vtkmodules.vtkCommonDataModel import vtkPolyData
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkActor, vtkPointGaussianMapper, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk

def renderPointCloudInteractive(data: Union[ndarray, vtkPolyData], scaleFactor: float = 0.2, opacity: float = 0.2, focalPoint: tuple = (0,0,0), zoom: float = 0) -> None:
    '''
    Visualize a point cloud in an interactive Window.

    Parameters
    ----------
    data: [ndarray, vtkPolyData]
        Point Cloud to visualize.
    ''' 
    
    # convert numpy array to vtk data format
    if isinstance(data, ndarray): 
        points = vtkPoints()
        points.SetData(numpy_to_vtk(data))

        data = vtkPolyData()
        data.SetPoints(points)

    # map to render primitives
    mapper = vtkPointGaussianMapper()
    mapper.SetInputData(data)
    mapper.SetScaleFactor(scaleFactor)

    # create actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('White'))
    actor.GetProperty().SetOpacity(opacity)

    # renderer and window
    ren = vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0,0,0)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(800,800)
    renWin.SetWindowName('Data')

    # make it interactive
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())

    # render! 
    ren.ResetCamera()
    ren.GetActiveCamera().SetFocalPoint(*focalPoint)
    ren.GetActiveCamera().Zoom(zoom)
    iren.Initialize()
    iren.Start()
