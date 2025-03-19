from simlib.directSummation import nbodyDirectSimulator
from physlib.densityModels import PlummerSphere, densityModel

def creatInitState(model: densityModel, num_particles: int):
    pass 


def runSimulation(simulator: nbodyDirectSimulator, directory: str, total_time: float, dt: float, adaptive: bool = False, out_frequency: int = 1, out_type: str = 'vtp') -> None: 
    '''
    Run a direct summation simulation

    Parameters
    ----------
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
    pass
