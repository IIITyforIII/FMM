from typing import Optional, Union
from numpy import ndarray

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2

from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkActor, vtkPointGaussianMapper, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer, vtkVolume, vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkImagingHybrid import vtkGaussianSplatter
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

def renderPointCloudDensityMap(data: Union[ndarray, vtkPolyData], radius: float = 0.2, dimensions: tuple = (100,100,100), focalPoint: tuple = (0,0,0), zoom: float = 0, fixedWindow: Optional[tuple] = None):
    '''
    Visualize a point cloud in as Volume.

    Parameters
    ----------
    data: [ndarray, vtkPolyData]
        Point Cloud to visualize.
    
    radius: float
        Radius of the gaussian splatter function.

    dimensions: tuple[int]
        Dimensions of the splatter volume.

    focalPoint: tuple[float]
        Focal point of the camera.

    zoom: float
        Zoom of the camera.

    fixedWindow: tuple[float]|None
        Optional fixed bounds of the volume.
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
    if fixedWindow is not None:
        splatter.SetModelBounds(*fixedWindow)

    # smooth
    from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
    smoother = vtkImageGaussianSmooth()
    smoother.SetInputConnection(splatter.GetOutputPort())

    smoother.Update()
    scalarRange = smoother.GetOutput().GetScalarRange()

    # mapper
    mapper = vtkSmartVolumeMapper() 
    # mapper.SetSampleDistance(0.05)
    mapper.SetInputConnection(smoother.GetOutputPort())
    mapper.SetBlendModeToComposite()

    colorTransferFunction = vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(scalarRange[0], 0.0, 0.0, 1.0)   # Blue at min
    colorTransferFunction.AddRGBPoint(scalarRange[1]/2, 1.0, 1.0, 1.0)   # White in middle
    colorTransferFunction.AddRGBPoint(scalarRange[1], 1.0, 0.0, 0.0)   # Red at max

    opacityTransferFunction = vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(scalarRange[0], 0.001)
    opacityTransferFunction.AddPoint(scalarRange[1]/2, 0.01)
    opacityTransferFunction.AddPoint(scalarRange[1], 1.0)

    # create volume
    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.GetProperty().SetColor(colorTransferFunction)
    volume.GetProperty().SetScalarOpacity(opacityTransferFunction)

    # renderer and window
    ren = vtkRenderer()
    ren.AddVolume(volume)
    ren.SetBackground(0,0,0)

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


def renderPointCloudInteractive(data: Union[ndarray, vtkPolyData], scaleFactor: float = 0.2, opacity: float = 1.0, focalPoint: tuple = (0,0,0), zoom: float = 0) -> None:
    '''
    Visualize a point cloud in an interactive Window.

    Parameters
    ----------
    data: [ndarray, vtkPolyData]
        Point Cloud to visualize.
    
    scaleFactor: float
        Scale of the gaussian blur.

    opacity: float
        Opacity of points.

    focalPoint: tuple[float]
        Focal point of the camera.

    zoom: float
        Zoom of the camera.
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
