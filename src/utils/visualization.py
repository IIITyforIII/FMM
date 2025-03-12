from typing import Union
from numpy import ndarray

from vtkmodules.vtkCommonDataModel import vtkPolyData
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkActor, vtkPointGaussianMapper, vtkPolyDataMapper, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer, vtkVolume, vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkImagingHybrid import vtkGaussianSplatter
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

def renderPointCloudDensityMap(data: Union[ndarray, vtkPolyData], radius: float = 0.2, dimensions: tuple = (100,100,100), focalPoint: tuple = (0,0,0), zoom: float = 0):
    '''
    Visualize a point cloud in as density Volume.

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

    # map point cloud to volume
    splatter = vtkGaussianSplatter()
    splatter.SetInputData(data)
    splatter.SetSampleDimensions(*dimensions)
    splatter.SetRadius(radius)
    splatter.SetScalarWarping(False)

    #surface
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    surface = vtkContourFilter()
    surface.SetInputConnection(splatter.GetOutputPort())
    surface.SetValue(0, 0.08)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)

    # mapper
    # mapper = vtkFixedPointVolumeRayCastMapper()
    # mapper.SetInputConnection(splatter.GetOutputPort())
    #
    #
    # colorTransferFunction = vtkColorTransferFunction()
    # colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    # colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    # colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)
    #
    # opacityTransferFunction = vtkPiecewiseFunction()
    # opacityTransferFunction.AddPoint(20, 0.0)
    # opacityTransferFunction.AddPoint(255, 0.2)
    #
    # # create volume
    # volume = vtkVolume()
    # volume.SetMapper(mapper)
    # volume.GetProperty().SetColor(colorTransferFunction)
    # volume.GetProperty().SetScalarOpacity(opacityTransferFunction)

    # renderer and window
    ren = vtkRenderer()
    # ren.AddVolume(volume)
    ren.AddActor(actor)
    # ren.SetBackground(0,0,0)
    ren.SetBackground(vtkNamedColors().GetColor3d('Wheat'))

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(800,800)
    renWin.SetWindowName('VolumeData')

    # make it interactive
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())

    # render! 
    ren.ResetCamera()
    ren.ResetCameraClippingRange()
    ren.GetActiveCamera().SetFocalPoint(*focalPoint)
    ren.GetActiveCamera().Zoom(zoom)
    iren.Initialize()
    iren.Start()


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
