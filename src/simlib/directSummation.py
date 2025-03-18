import importlib
from typing import Tuple
from numpy.typing import ArrayLike

class nbodyDirectSimulator():
    '''
    Simulator for a n-body particle simulation using direct summation.
    Excpects natural units (G=1) in default, else use setG(). Calculations are performed unit agnostic.
    '''
    def __init__(self, initPos: ArrayLike, initVel: ArrayLike, masses: ArrayLike, use_jax: bool = False) -> None:
        '''
        Parameters
        ----------
        initPos: ArrayLike
            Initial positional state of the N-Body system.
        initVel: ArrayLike
            Initial velocity state of the N-Body system.
        use_gpu: bool
            Flag if to use jax gpu acceleration.
        '''
        # lazy loading of required modules
        # TODO
        import numpy as np
        self.np = np
        # self.use_jax= use_jax
        # if use_jax:
        #     self.np = importlib.import_module('jax.numpy')
        # else:
        #     self.np = importlib.import_module('numpy')

        # set state
        self.pos = self.np.array(initPos)
        self.num_particles = len(self.pos)
        self.vel = self.np.array(initVel)
        self.masses = self.np.array(masses)

        self.t = 0

        # units
        self.G = 1


        if self.num_particles != len(self.vel):
            raise AssertionError('Size of position array does not match velocity array. ({} != {})'.format(self.num_particles, len(self.vel)))

    def getState(self) -> Tuple[float, ArrayLike, ArrayLike]:
        '''
        Returns the current state of the system.

        Returns
        -------
        state: Tuple[float, ArrayLike, ArrayLike]
            State in form (time, positions, velocities).
        '''
        return (self.t, self.pos, self.vel)

    def resetState(self, pos: ArrayLike, vel: ArrayLike, t: float = 0) -> None:   
        '''
        Resets the state of the system to a given state.

        Parameters
        ----------
        pos: ArrayLike
            Array of particle positions.
        vel: ArrayLike
            Array of particle velocities.

        '''
        self.pos = pos
        self.vel = vel
        self.t   = t

    def setG(self, g: float):
        '''
        Set gravitational constant G. Simulator is agnostic of used units. 
        '''
        self.G = g

    def step(self, step_size: float = 0.1):
        '''
        Perform a simulation step in a given step size.
        '''
        #TODO
        pass
