from typing import Union, Tuple
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from geolib.coordinates import mapCartToPolar
from geolib.tree import applyBoundaryCondition, buildTree, getMortonSortedPermutation, insertParticle
from simlib.kernels import p2p
from jax import config
from abc import ABC, abstractmethod

class Simulator(ABC):
    @abstractmethod
    def getState(self) -> Tuple[float, list, list]:
        '''
        Returns the current state of the system.

        Returns
        -------
        state: Tuple[float, list, list]
            State in form (time, positions, velocities).
        '''
        pass

    @abstractmethod
    def resetState(self, pos: ArrayLike, vel: ArrayLike, masses: ArrayLike) -> None:   
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
        pass

    @abstractmethod
    def setG(self, g: float) -> None:
        '''
        Set gravitational constant G. Simulator is agnostic of used units. 
        '''
        pass

    @abstractmethod
    def step(self, dt: float = 0.1):
        '''
        Perform a simulation step in a given step size.

        Parameters
        ----------
        dt: float
            Step size for integration.
        '''
        pass

    @abstractmethod
    def blockStep(self, dt: float = 0.1):
        '''
        Perform a block step (adaptive stesize) of given total time.

        Parameters
        ----------
        dt: float
            Total time step to update simulation. 
        '''
        pass

    @abstractmethod
    def getName(self) -> str:
        pass

    @abstractmethod
    def getNumParticles(self) -> int:
        pass

class nbodyDirectSimulator(Simulator):
    '''
    Simulator for a n-body particle simulation using direct summation.
    Excpects natural units (G=1) in default, else use setG(). Calculations are performed unit agnostic.
    '''
    def __init__(self, initPos: ArrayLike, initVel: ArrayLike, masses: ArrayLike) -> None:
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
        config.update("jax_enable_x64", True)
        self.pos = jnp.array(initPos, dtype=jnp.float64).reshape(-1,3)
        self.num_particles = len(self.pos)
        self.vel = jnp.array(initVel, dtype = jnp.float64).reshape(-1,3)
        self.masses = jnp.array(masses, dtype= float).reshape(-1,1)

        self.t = 0

        # units
        self.G = 1

        if self.num_particles != len(self.vel):
            raise AssertionError('Size of position array does not match velocity array. ({} != {})'.format(self.num_particles, len(self.vel)))

    def getState(self) -> Tuple[float, list, list]:
        '''{}'''.format(Simulator.getState.__doc__)
        return self.t, self.pos.tolist(), self.vel.tolist()

    def resetState(self, pos: ArrayLike, vel: ArrayLike, masses: ArrayLike) -> None:   
        '''{}'''.format(Simulator.resetState.__doc__)
        self.__init__(pos, vel, masses)

    def setG(self, g: float) -> None:
        '''{}'''.format(Simulator.setG.__doc__)
        self.G = g

    def step(self, dt: float = 0.1):
        '''{}'''.format(Simulator.step.__doc__)
        # calculate resulting accelerations using P2P kernel.
        acc = p2p.acceleration(self.pos, self.pos, self.G, self.masses)
        
        # integrate motion formula and update positions
        self.vel = self.vel + dt * acc 
        self.pos = self.pos + dt * self.vel

        # update time
        self.t = self.t + dt

    #TODO do adaptive time steps ,using jerk and snap
    def blockStep(self, dt: float = 0.1):
        '''
        Perform a block step (adaptive stesize) of given total time.

        Parameters
        ----------
        dt: float
            Total time step to update simulation. 
        '''
        pass

    def getName(self) -> str:
        return 'directSummation'

    def getNumParticles(self) -> int:
        return self.num_particles


class fmmSimulator(Simulator):
    '''
    Simulator for a n-body particle simulation using fast multipole method.
    Excpects natural units (G=1) in default, else use setG(). Calculations are performed unit agnostic.
    '''
    def __init__(self, initPos: ArrayLike, initVel: ArrayLike, domainMin, domainMax, masses: ArrayLike, expansionOrder: int, nCrit: int = 32, nThreads: int = 1) -> None:
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
        config.update("jax_enable_x64", True)
        self.pos = jnp.array(initPos, dtype=jnp.float64).reshape(-1,3)
        self.num_particles = len(self.pos)
        self.vel = jnp.array(initVel, dtype = jnp.float64).reshape(-1,3)
        self.masses = jnp.array(masses, dtype= float).reshape(-1,1)

        # fmm related data
        self.expansionOrder = expansionOrder
        self.polar = jnp.apply_along_axis(mapCartToPolar, 1, self.pos) # needed for spherical harmonics
        self.leafs = [None] * len(self.pos) # leafs[idx] corresponds to the leaf node of particle idx -> use for misfit calc
        
        # tree
        perm = jnp.array(getMortonSortedPermutation(np.asarray(self.pos)))  # morton sort the positions for spatial correlation especially for multithreading
        self.pos = self.pos[perm]
        self.root = buildTree(self.pos, self.leafs, np.array(domainMin), np.array(domainMax), nCrit=nCrit, nThreads=nThreads)
        self.nCrit = nCrit
        self.nThreads = nThreads
        self.multiThreaded = nThreads > 1

        self.t = 0.
        # units
        self.G = 1

        if self.num_particles != len(self.vel):
            raise AssertionError('Size of position array does not match velocity array. ({} != {})'.format(self.num_particles, len(self.vel)))

    def step(self, dt: float = 0.1):
        '''{}'''.format(Simulator.step.__doc__)

        ### compute multipoles
        # comupteMultipoles(root)

        ### compute fieldtensors/ accelerations -> traverse tree
        acc = jnp.array(self.num_particles)
        # traverseTreeSimple(root) / traverseTreeDual(root)

        ### update
        # integrate motion formula and update positions
        self.vel = self.vel + dt * acc 
        self.pos = self.pos + dt * self.vel

        # update time
        self.t = self.t + dt

        ### compute misfits
        # mfits = computeMisfits(self.pos, self.leafs)
        # for m in mfits:
        #    insertParticle(self.pos, m, self.root, self.nCrit, self.multiThreaded) 

    def getState(self) -> Tuple[float, list, list]:
        '''{}'''.format(Simulator.getState.__doc__)
        return self.t, self.pos.tolist(), self.vel.tolist() # pyright: ignore

    def resetState(self, pos: ArrayLike, vel: ArrayLike, masses: ArrayLike) -> None:   
        '''{}'''.format(Simulator.resetState.__doc__)
        self.pos = jnp.array(pos).reshape(-1,3)
        self.vel = jnp.array(vel).reshape(-1,3)
        self.masses = jnp.array(masses).reshape(-1,1)

    def setG(self, g: float) -> None:
        '''{}'''.format(Simulator.setG.__doc__)
        self.G = g

    #TODO do adaptive time steps ,using jerk and snap
    def blockStep(self, dt: float = 0.1):
        pass

    def getName(self) -> str:
        return 'Fast Multipole Method'

    def getNumParticles(self) -> int:
        return self.num_particles
