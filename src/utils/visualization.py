from typing import Optional, Union
from numpy import ndarray
from pathlib import Path

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2

from vtkmodules.vtkIOXML import vtkXMLPolyDataReader 
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkActor, vtkGlyph3DMapper, vtkPointGaussianMapper, vtkPolyDataMapper, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderer, vtkVolume, vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkImagingHybrid import vtkGaussianSplatter
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkFiltersCore import vtkGlyph3D

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
    renWin.SetSize(900,900)
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


def renderPointCloudInteractive(data: Union[ndarray, vtkPolyData], scaleFactor: float = 0.2, opacity: float = 1.0, focalPoint: tuple = (0,0,0), zoom: float = 0, visVelocities: bool = False) -> None:
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

    if visVelocities:
        arrowSource = vtkArrowSource()
        arrowSource.SetTipRadius(0.1)
        arrowSource.SetShaftRadius(0.02)
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(arrowSource.GetOutputPort())
        glyph.SetInputData(data)
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.3)
        velMapper = vtkPolyDataMapper()
        velMapper.SetInputConnection(glyph.GetOutputPort())

    # map to render primitives
    mapper = vtkPointGaussianMapper()
    mapper.SetInputData(data)
    mapper.SetScaleFactor(scaleFactor)

    # create actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('White'))
    actor.GetProperty().SetOpacity(opacity)

    if visVelocities:
        velActor =vtkActor()
        velActor.SetMapper(velMapper)
        velActor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Grey'))
        velActor.GetProperty().SetOpacity(opacity)


    # renderer and window
    ren = vtkRenderer()
    ren.AddActor(actor)
    if visVelocities: 
        ren.AddActor(velActor)
    ren.SetBackground(0,0,0)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(900,900)
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

def animateTimeSeries(directory: str,renderAsSpheres: bool = False, scaleFactor: float = 0.2, opacity: float = 1.0, focalPoint: tuple = (0,0,0), zoom: float = 0, interactive:bool = True, animRate: int = 100, create_video: bool = False):

    #read out directory
    path = Path(directory)
    files = [str(f) for f in sorted(path.glob('frame_*.vtp'))]   

    # create reader
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(files[0])

    if renderAsSpheres:
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        sphereSource = vtkSphereSource()
        sphereSource.SetRadius(scaleFactor)
        sphereSource.SetThetaResolution(16) 
        sphereSource.SetPhiResolution(16)

        mapper = vtkGlyph3DMapper()
        mapper.SetSourceConnection(sphereSource.GetOutputPort())
    else:
        mapper = vtkPointGaussianMapper()
        mapper.SetScaleFactor(scaleFactor)
    mapper.SetInputConnection(reader.GetOutputPort())


    # create actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('White'))
    if renderAsSpheres:
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetPointSize(1)

    # renderer and window
    ren = vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0,0,0)
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(900,900)
    renWin.SetWindowName('SimulationState')

    # make it interactive
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())

    # define frame update callback
    class counter:
        def __init__(self):
            self.current_frame = 0
        def increment(self):
            self.current_frame += 1
        def decrement(self):
            self.current_frame -= 1 
        def getCurrentFrame(self):
            return self.current_frame

    class keypressCallback():
        def __init__(self, iren, counter:counter):
            self.iren = iren
            self.paused = True
            self.counter = counter
        def __call__(self, caller, event):
            key = caller.GetKeySym()
            if key == 'space':
                if self.paused:
                    self.iren.CreateRepeatingTimer(animRate)
                    self.paused = False
                else:
                    self.iren.DestroyTimer()
                    self.paused = True
            if key == 'Right' and self.counter.getCurrentFrame() < len(files) - 1:
                self.counter.increment()
                reader.SetFileName(files[self.counter.getCurrentFrame()])
                reader.Update()
                renWin.Render()
            if key == 'Left' and self.counter.getCurrentFrame()> 0:
                self.counter.decrement()
                reader.SetFileName(files[self.counter.getCurrentFrame()])
                reader.Update()
                renWin.Render()

    class timerCallback():
        def __init__(self, iren, counter:counter):
            self.iren = iren
            self.counter = counter
        def __call__(self, caller, event):
            if self.counter.getCurrentFrame() < len(files):
                print('Render Frame: ' + files[self.counter.getCurrentFrame()])
                reader.SetFileName(files[self.counter.getCurrentFrame()])
                reader.Update()
                renWin.Render()
                self.counter.increment()


    kCallback = keypressCallback(iren)
    tCallback = timerCallback(iren)
    iren.AddObserver('TimerEvent', tCallback)
    iren.AddObserver('KeyPressEvent', kCallback)
    iren.CreateRepeatingTimer(animRate)


    # start render loop
    ren.ResetCamera()
    ren.GetActiveCamera().SetFocalPoint(*focalPoint)
    ren.GetActiveCamera().Zoom(zoom)
     
    iren.Initialize()
    iren.Start()
