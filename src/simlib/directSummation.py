from typing import Union, Tuple
import jax.numpy as jnp
from jax.typing import ArrayLike
from simlib.kernels import p2p

class nbodyDirectSimulator():
    '''
    Simulator for a n-body particle simulation using direct summation.
    Excpects natural units (G=1) in default, else use setG(). Calculations are performed unit agnostic.
    '''
    def __init__(self, initPos: ArrayLike, initVel: ArrayLike, masses: Union[ArrayLike,float] = 1) -> None:
        '''
        Parameters
        ----------
        initPos: ArrayLike
            Initial positional state of the N-Body system.
        initVel: ArrayLike
            Initial velocity state of the N-Body system.
        masses: ArrayLike
            Masses of the particles or single mass for equally heavy particles.
        '''
        # set state
        self.pos = jnp.array(initPos, dtype= float).reshape(-1,3)
        self.num_particles = len(self.pos)
        self.vel = jnp.array(initVel, dtype = float).reshape(-1,3)
        self.masses = jnp.array(masses, dtype=float).reshape(-1,1)

        self.t = 0

        # units
        self.G = 1

        if self.num_particles != len(self.vel):
            raise AssertionError('Size of position array does not match velocity array. ({} != {})'.format(self.num_particles, len(self.vel)))

    def getState(self) -> Tuple[float, list, list]:
        '''
        Returns the current state of the system.

        Returns
        -------
        state: Tuple[float, list, list]
            State in form (time, positions, velocities).
        '''
        return self.t, self.pos.tolist(), self.vel.tolist()

    def resetState(self, pos: ArrayLike, vel: ArrayLike, masses: Union[ArrayLike,float] = 1) -> None:   
        '''
        Resets the state of the system to a given state.

        Parameters
        ----------
        pos: ArrayLike
            Array of particle positions.
        vel: ArrayLike
            Array of particle velocities.
        masses: [ArrayLike, float]
            Array of masses.
        '''
        self.__init__(pos, vel, masses)

    def setG(self, g: float) -> None:
        '''
        Set gravitational constant G. Simulator is agnostic of used units. 
        '''
        self.G = g

    def step(self, step_size: float = 0.1):
        '''
        Perform a simulation step in a given step size.

        Parameters
        ----------
        step_size: float
            Step size for integration.
        '''
        # calculate resulting accelerations using P2P kernel.
        acc = p2p.acceleration(self.pos, self.pos, self.G, self.masses)
        
        # integrate motion formula and update positions
        self.vel = self.vel + step_size * acc 
        self.pos = self.pos + step_size * self.vel

    #TODO do adaptive time steps ,using jerk and snap
    def blockStep(self, step_size: float = 0.1):
        '''
        Perform a block step (adaptive stesize) of given total time.
        '''
        pass
