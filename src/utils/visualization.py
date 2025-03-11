import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkDataSetMapper, vtkPolyDataMapper, vtkRenderWindow, vtkRenderer

colors = vtkNamedColors()

cone = vtkConeSource()
cone.SetHeight(3.0)
cone.SetRadius(1.0)
cone.SetResolution(10)

coneMapper = vtkPolyDataMapper()
coneMapper.SetInputConnection(cone.GetOutputPort())

coneActor = vtkActor()
coneActor.SetMapper(coneMapper)
coneActor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

ren1 = vtkRenderer()
ren1.AddActor(coneActor)
ren1.SetBackground(colors.GetColor3d('MidnightBlue'))

renWin = vtkRenderWindow()
renWin.AddRenderer(ren1)
renWin.SetSize(300,300)
renWin.SetWindowName('Test')

#observer
class callback():
    def __init__(self, renderer):
        self.ren = renderer

    def __call__(self, caller, ev):
        pos = self.ren.GetActiveCamera().GetPosition()
        print('{},{},{}'.format(*pos))
obs = callback(ren1)
ren1.AddObserver('StartEvent', obs)





for i in range(360):
    renWin.Render()
    ren1.GetActiveCamera().Azimuth(1)
