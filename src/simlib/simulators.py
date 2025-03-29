from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.special import warnings
from geolib.expansionCentres import CenterOfMass, SmallestEnclosingSphere
from geolib.tree import applyBoundaryCondition, buildTree, getMortonSortedPermutation, Node, insertParticle
from simlib.acceptanceCriterion import AcceptanceCriterion, AdvancedAcceptanceCriterion, FixedAcceptanceCriterion
import simlib.kernels as kernels
from jax import config
from abc import ABC, abstractmethod
import time

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
        acc = kernels.p2p.acceleration(self.pos, self.pos, self.masses, self.G, use_jax=True)
        
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
    def __init__(self, initPos: ArrayLike, initVel: ArrayLike, domainMin, domainMax, masses: ArrayLike, expansionOrder: int, nCrit: int = 32, acceptCrit: AcceptanceCriterion = FixedAcceptanceCriterion(), nThreads: int = 1) -> None:
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
        self.pos = np.array(initPos, dtype=np.float64).reshape(-1,3)
        self.num_particles = len(self.pos)
        self.vel = np.array(initVel, dtype =np.float64).reshape(-1,3)
        self.masses = np.array(masses, dtype=np.float64).reshape(-1,1)
        self.acc = np.zeros_like(self.pos)

        # fmm related data
        self.expansionOrder = expansionOrder
        self.harmonics = kernels.SphericalHarmonics(self.expansionOrder, self.expansionOrder)
        self.potentialExpandCenter = SmallestEnclosingSphere(multipole=False) # defines computation method of potential/force expansion center
        self.multipoleExpandCenter = None if expansionOrder >= 8 else CenterOfMass(multipole=True) # defines computation method of multipole expansion center
        self.MAC = acceptCrit
        if isinstance(self.MAC, AdvancedAcceptanceCriterion):
            self.estimator = fmmSimulator(self.pos,self.vel,domainMin,domainMax,self.masses,expansionOrder=1, nCrit=nCrit, acceptCrit=FixedAcceptanceCriterion(1.), nThreads=nThreads)

        # tree
        self.domainMin = np.array(domainMin)
        self.domainMax = np.array(domainMax)
        self.leafs = np.array([None] * len(self.pos)) # leafs[idx] corresponds to the leaf node of particle idx -> use for misfit calc
        perm = getMortonSortedPermutation(self.pos)  # morton sort the positions for spatial correlation especially for multithreading
        self.pos = self.pos[perm]
        self.root = buildTree(self.pos, self.leafs, self.domainMin, self.domainMax, nCrit=nCrit, nThreads=nThreads)
        self.nCrit = nCrit
        self.nThreads = nThreads
        self.multiThreaded = nThreads > 1

        self.t = 0.
        # units
        self.G = 1.

        if self.num_particles != len(self.vel):
            raise AssertionError('Size of position array does not match velocity array. ({} != {})'.format(self.num_particles, len(self.vel)))

    def step(self, dt: float = 0.1):
        '''{}'''.format(Simulator.step.__doc__)

        ### prepare AdvancedAcceptanceCriterion if needed
        if isinstance(self.MAC, AdvancedAcceptanceCriterion):
            print('estimation step ------------')
            start = time.time()
            self.estimator.resetState(self.pos,self.vel,self.masses)
            self.estimator.step()
            a = np.linalg.norm(self.estimator.acc, axis=1)
            self.MAC.setAorF(a)
            print(time.time()-start)
            print('----------------------------')

        ### compute multipoles
        print('multipole step')
        start = time.time()
        self.computeCentersAndMultipoles(self.root)
        print(time.time() - start)

        ### reset acceleration array
        self.acc.fill(0.)

        ### compute fieldTensors and pass down
        print('acc step')
        start = time.time()
        self.dualTreeWalk(self.root,self.root, self.expansionOrder >= 8)
        self.downpassField(self.root)
        print(time.time() - start)

        ### update state (integrate motion formula and update positions)
        print('state update')
        start = time.time()
        self.vel = self.vel + dt * self.acc 
        self.pos = self.pos + dt * self.vel
        self.t = self.t + dt
        print(time.time() - start)

        ### compute misfits
        print('misfit step')
        start = time.time()
        mfits = self.computeMisfitsAndRemove()
        mfitp = self.pos[mfits]
        applyBoundaryCondition(self.domainMin,self.domainMax, mfitp)
        self.pos[mfits] = mfitp
        for idx in mfits:
            insertParticle(self.pos,self.leafs,idx,self.root,self.nCrit,self.multiThreaded)
        print(time.time() - start)

    def getState(self) -> Tuple[float, list, list]:
        '''{}'''.format(Simulator.getState.__doc__)
        return self.t, self.pos.tolist(), self.vel.tolist() # pyright: ignore

    def resetState(self, pos: ArrayLike, vel: ArrayLike, masses: ArrayLike) -> None:   
        '''{}'''.format(Simulator.resetState.__doc__)
        self.pos = np.array(pos).reshape(-1,3)
        self.vel = np.array(vel).reshape(-1,3)
        self.masses = np.array(masses).reshape(-1,1)
        self.acc.fill(0.)

    def setG(self, g: float) -> None:
        '''{}'''.format(Simulator.setG.__doc__)
        self.G = g

    #TODO do adaptive time steps ,using jerk and snap
    def blockStep(self, dt: float = 0.1):
        pass

    def getName(self) -> str:
        return 'FastMultipoleMethod'

    def getNumParticles(self) -> int:
        return self.num_particles

    def computeCentersAndMultipoles(self, root: Node):
        '''Traverse over tree and compute expansion centers and multipole, preparion potential computation.'''
        if root.isLeaf:
            #update expansion centers
            root.potentialCenter = self.potentialExpandCenter.computeExpCenter(self.pos,root,self.masses)
            root.multipoleCenter = self.multipoleExpandCenter.computeExpCenter(self.pos,root,self.masses) if self.multipoleExpandCenter is not None else root.potentialCenter
            
            #compute multipole and reset field tensor
            if len(root.particleIds) > 0:
                root.multipoleExpansion = kernels.p2m(self.pos, root.particleIds, root.multipoleCenter[0], self.masses, self.harmonics)
                root.fieldTensor = np.zeros_like(self.harmonics.n_arr).astype(complex)

                # compute multipole Powers if needed for MAC
                if isinstance(self.MAC, AdvancedAcceptanceCriterion):
                    root.multipolePower = self.MAC.computeMultipolePower(root)
            else:
                root.multipoleExpansion = 0.

        else:
            # traverse down the tree
            root.particleIds = [] # might have changed after misfit calculation
            for c in root.children:
                self.computeCentersAndMultipoles(c)
                # propagate particle id information up the tree
                root.particleIds += c.particleIds


            # update expansion center
            root.potentialCenter = self.potentialExpandCenter.computeExpCenter(self.pos,root,self.masses)
            root.multipoleCenter = self.multipoleExpandCenter.computeExpCenter(self.pos,root,self.masses) if self.multipoleExpandCenter is not None else root.potentialCenter

            # compute multipole
            root.multipoleExpansion = np.zeros_like(self.harmonics.n_arr).astype(complex) # pyright: ignore
            for c in root.children:
                root.multipoleExpansion += kernels.m2m(c, root, self.harmonics)
            # reset field tensor
            root.fieldTensor = np.zeros_like(self.harmonics.n_arr).astype(complex)

            # compute multipole powers if needed for MAC
            if isinstance(self.MAC, AdvancedAcceptanceCriterion):
                root.multipolePower = self.MAC.computeMultipolePower(root)

    def approximate(self, A:Node, B:Node, mutual:bool):
        '''Compute interaction between cells and pass it down the tree.'''

        # check if direct summation is more efficient 
        ids  = np.hstack((A.particleIds,B.particleIds))
        idsA = ids if mutual else A.particleIds
        idsB = ids if mutual else B.particleIds
        if A.isLeaf:
            if len(B.particleIds) < self.expansionOrder**2 or (mutual and len(A.particleIds) < 4*self.expansionOrder**2): # test if direct summation is faster
                self.acc[idsB] += kernels.p2p.acceleration(self.pos[idsA],self.pos[idsB], self.masses[idsA], G=self.G, use_jax=False)
                return
        if B.isLeaf:
            if len(A.particleIds) < 4*self.expansionOrder**2 or (mutual and len(B.particleIds) < self.expansionOrder**2): # test if direct summation is faster
                self.acc[idsB] += kernels.p2p.acceleration(self.pos[idsA],self.pos[idsB], self.masses[idsA], G=self.G, use_jax=False)
                return
        # if no node is leaf
        if len(ids) < self.expansionOrder**3:
            self.acc[idsB] += kernels.p2p.acceleration(self.pos[idsA], self.pos[idsB], self.masses[idsA], G=self.G, use_jax=False)
            return 

        # compute A->B approx.
        if A.isLeaf:    # apply P2L
            for p in A.particleIds:
                B.fieldTensor += kernels.p2l(self.pos[p], self.masses[p], B, self.harmonics)  
        elif B.isLeaf:  # apple M2P
            for p in B.particleIds:
                pot = kernels.m2p(A, self.pos[p], self.harmonics)
                # add test if imaginary part is small enough as pot[1,0] analytically should be real
                if np.abs(pot[1,0].imag) > 1e-15:
                    warnings.warn('Imaginart part > 1e-15 detected in potential. Psi[1,0] should be real.', UserWarning)
                self.acc[p] += -1*np.array([pot[1,1].real, pot[1,1].imag, pot[1,0].real])
        else:           # apply M2L
            B.fieldTensor += kernels.m2l(A,B,self.harmonics)

        # compute B->A approx. if mutual
        if mutual:
            if B.isLeaf:
                for p in B.particleIds:
                    A.fieldTensor += kernels.p2l(self.pos[p], self.masses[p], A, self.harmonics)
            elif A.isLeaf:
                for p in A.particleIds:
                    pot = kernels.m2p(B, self.pos[p], self.harmonics)
                    # add test if imaginary part is small enough as pot[1,0] analytically should be real
                    if np.abs(pot[1,0].imag) > 1e-15:
                        warnings.warn('Imaginart part > 1e-15 detected in potential. Psi[1,0] should be real.', UserWarning)
                    self.acc[p] += -1 * np.array([pot[1,1].real, pot[1,1].imag, pot[1,0].real])
            else:
                A.fieldTensor += kernels.m2l(B,A,self.harmonics)
        

    def dualTreeWalk(self, A:Node, B:Node, mutual: bool):
        '''Do the tree walk and compute all interactions between cells.'''
        # if one cell is empy stop here
        if len(A.particleIds) == 0 or len(B.particleIds) == 0: return

        # approximate cell <-> cell if MAC is met (compute field tensors and pass down the tree)
        if self.MAC.eval(A,B):
            self.approximate(A,B, mutual)
        
        # Do direct summation if we end up in 2 leafs
        elif (A.isLeaf and B.isLeaf):
            ids = np.hstack((A.particleIds,B.particleIds))
            self.acc[ids] += kernels.p2p.acceleration(self.pos[ids], self.pos[ids], self.masses[ids], G=self.G, use_jax=False)

        # split internal node if one is leaf
        elif (A.isLeaf):
            for b in B.children:
                self.dualTreeWalk(A,b, mutual)
                if not mutual:
                    self.dualTreeWalk(b,A, mutual)
        elif (B.isLeaf):
            for a in A.children:
                self.dualTreeWalk(a, B, mutual)
                if not mutual:
                    self.dualTreeWalk(B, a, mutual)

        # we have to catch the case of A==B
        # perform tree walk for every child pair (have to be called twice for every pair in a non mutual version)
        elif A == B:
            for a in range(len(A.children)):
                start = a if mutual else 0
                for b in range(start, len(B.children)):
                    self.dualTreeWalk(A.children[a],B.children[b], mutual)

        # otherwise we open the larger cell 
        elif A.potentialCenter[1] < B.potentialCenter[1]:
            for b in B.children:
                self.dualTreeWalk(A,b,mutual)
                if not mutual:
                    self.dualTreeWalk(b,A,mutual)
        else:
            for a in A.children:
                self.dualTreeWalk(a,B,mutual)
                if not mutual:
                    self.dualTreeWalk(B,a,mutual)

    def downpassField(self, root:Node):
        '''Pass down the fieldTensord down the tree to sinks.'''
        if len(root.particleIds) == 0:
            return
        if not root.isLeaf:
            for c in root.children:
                if len(c.particleIds) > 0:
                    c.fieldTensor += kernels.l2l(root,c,self.harmonics) 
                self.downpassField(c)
        else: 
            for p in root.particleIds:
                pot = kernels.l2p(root,self.pos[p], self.harmonics)
                # add test if imaginary part is small enough as pot[1,0] analytically should be real
                if np.abs(pot[1,0].imag) > 1e-15:
                    warnings.warn('Imaginart part > 1e-15 detected in potential. Psi[1,0] should be real.', UserWarning)
                self.acc[p] += -1 * np.array([pot[1,1].real, pot[1,1].imag, pot[1,0].real])

    def computeMisfitsAndRemove(self):
        msfts = []
        for idx in range(len(self.pos)):
            if np.any((self.leafs[idx].domainMin > self.pos[idx]) | (self.leafs[idx].domainMax < self.pos[idx])):
                msfts.append(idx)
                self.leafs[idx].particleIds.remove(idx)
        return msfts


