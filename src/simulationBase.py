from jax.typing import ArrayLike
from geolib.tree import applyBoundaryCondition
from simlib.simulators import Simulator, nbodyDirectSimulator
from physlib.densityModels import PlummerSphere
from geolib.coordinates import Point3D 
from utils.dataIO import writeToVtp
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

def createInitState(num_particles: int, core_rad: float = 1, center = [0,0,0]):
    '''
    Sample dummy star cluster (Plummer Sphere) as starting point.

    Parameters
    ----------
    num_particles: int
        Number or particles. (Size of the cluster)
    core_rad: float
        Core radius a of the plummer sphere.
    center: [float,float,float]
        Center of the cluster.
    '''
    # smaple points from plummer sphere
    model = PlummerSphere(center)
    pos   = model.samplePolar3D(num_particles)   
    

    # get radii and scale with core 
    rad = np.fromiter((p.r for p in pos), dtype=float)
    rad   = core_rad * rad
    for point, r in zip(pos, rad):
        point.r = r

    # get cartesian array 
    pos = np.array([Point3D(p).to_list() for p in pos])
    # get velocities
    vel = model.sample_vel(rad)

    return pos,vel

def writeState(file: str, pos: ArrayLike, vel: ArrayLike) -> None:
    '''
    Write nBody state to a .vtp file

    Parameters
    ----------
    file: str
        Filename.
    pos: ArrayLike
        Positions of the n-bodies.
    vel: ArrayLike
        Velocities of the n-bodies.
    '''
    points = vtkPoints()
    points.SetData(numpy_to_vtk(pos))
    vectors = numpy_to_vtk(vel)
    vectors.SetName('Velocities')
    
    data = vtkPolyData()
    data.SetPoints(points)
    data.GetPointData().SetVectors(vectors)

    writeToVtp(data, file)

def writeMetaData(output: str, model: dict, units: str, total_time: float, dt: float, adaptive: bool, out_frequency: int):
    meta = {
            'Init Model': model,
            'Units': units,
            'Total simulation time': total_time,
            'Step size': dt,
            'Adaptive steps': adaptive,
            'Frame frequency': out_frequency
           }
    with open(output,'w') as file:
        yaml.dump(meta, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

def runSimulation(simulator: Simulator, directory: str, total_time: float, dt: float, adaptive: bool = False, out_frequency: int = 1) -> None: 
    '''
    Run a direct summation simulation

    Parameters
    ----------
    simulatot: nbodyDirectSimulator
        Simulator to perform the timesteps.
    directory: str
        Directory to save simulation results in.
    total_time: float
        Total time to be simulated
    dt: float
        Timestep size (Size of one block timestep if adaptive is true).
    adaptive: bool
        Flag wether to use adaptive timesteps or not.
    out_frequency: int
        Frequency to write system state.
    out_type: str
        Output filetype ('csv', 'vtp')
    '''
    # check if requested directory exists
    if Path(directory).exists() == False:
        raise NotADirectoryError('Given directory does not exists. ({})'.format(directory))

    # create simuation result folder   
    simDir = Path(directory+'/'+simulator.getName())
    if not Path(simDir).exists():
        simDir.mkdir()

    # write initial state
    _,pos,vel= simulator.getState()
    writeState('{}/frame_0000.vtp'.format(str(simDir)), pos, vel) # pyright: ignore

    # start loop
    for i in tqdm(range(1, int(total_time/dt) + 1)):
        if adaptive:
            simulator.blockStep(dt)
        else:
            simulator.step(dt)

        if i%out_frequency == 0:
            _,pos,vel= simulator.getState()
            writeState('{}/frame_{:04}.vtp'.format(str(simDir), i), pos, vel) # pyright: ignore
    

if __name__ == '__main__': 
    # model params
    num_particles = 1000
    core_rad     = 10
    # sim params
    total_time = 1   
    time_step  = 0.01 
    path = 'data/testFMM'

    sim = False

    if sim:
        # logging 
        model = {'Name': 'Plummer', 'Core radius': core_rad, 'Particles': num_particles}
        writeMetaData(path+'/simInfo.yaml', model,'Natural Units',total_time=total_time,dt=time_step,adaptive=False,out_frequency=1)


        # run simulation
        pos,vel = createInitState(num_particles, core_rad=core_rad)
        simulator = nbodyDirectSimulator(pos, vel, 1)
        runSimulation(simulator,path,total_time,time_step)

    # anim results
    from utils.visualization import animateTimeSeries
    from utils.visualization import renderPointCloudInteractive
    # animateTimeSeries(path+'/directSummation', scaleFactor=0.2, interactive=False, animRate=20)


    #TODO notes for later ---> DONT FORGET BOUNDARYCONDITION outside of simulator
    import time
    start = time.time()
    pos,vel = createInitState(1000, core_rad=10)
    end = time.time()
    print('State prep:')
    print(end - start)
    dMax = np.array([500,500,500])     
    dMin = -1*dMax
    mass = np.ones(len(pos))
    applyBoundaryCondition(dMin, dMax, pos)

    # from physlib.densityModels import UniformBox
    # pos = UniformBox([0,0,0], 100,100,100).sample(10000)
    #TODO notes plummer sphere brauch länger. dichtere verteilung in der mitte, mehr m2m -> mehr dauer
    # model = {'Name': 'Plummer', 'Core radius': 10, 'Particles': 5000}
    # writeMetaData(path+'/simInfo.yaml', model,'Natural Units',total_time=total_time,dt=time_step,adaptive=False,out_frequency=1)

    from simlib.simulators import fmmSimulator
    from geolib.expansionCentres import SmallestEnclosingSphere, GeometricCenter, CenterOfMass
    from simlib.acceptanceCriterion import AdvancedAcceptanceCriterion, FixedAcceptanceCriterion
    test = fmmSimulator(pos,vel,dMin,dMax,mass,expansionOrder=8, nCrit=32, acceptCrit=FixedAcceptanceCriterion(0.4), nThreads=1)
    # test = nbodyDirectSimulator(pos,vel,mass)
    # runSimulation(test, path,total_time=1, dt=0.01)

    animateTimeSeries(path+'/FastMultipoleMethod', scaleFactor=0.2, interactive=False, animRate=200)
    # animateTimeSeries(path+'/directSummation', scaleFactor=0.2, interactive=False, animRate=200)


    # directTest = nbodyDirectSimulator(pos,vel,mass)
    # start = time.time()
    # directTest.step(0.1)
    # end = time.time()
    # print('direct sum step:')
    # print(end - start)
