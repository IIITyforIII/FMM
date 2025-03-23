from typing import Union
import jax.numpy as jnp
import numpy as np
import morton 
import threading

def getMortonSortedPermutation(positions: np.ndarray, scaleFactor: int = int(10e6)) -> np.ndarray:
    '''
    Get the sorting according to the morton order of an 3D array.

    Parameters
    ----------
    particles: np.ndarray
        Positions to be sorted accoridng to morton-order.
    scaleFactor: int
        Scale factor to deal with decimal numbers. Will determine the accuracy of the sorted array (10e6 -> Removes all decimal digits after the sixth decimal place)

    Returns
    -------
    sortPerm: np.ndarray
        Permutation of indices according to morton sorted list.
    '''
    # scale points as morton order cant handle floating points
    scaled = (positions * scaleFactor).astype(int)
    offset = np.min(scaled)
    scaled = scaled - offset

    # sort in morton order
    m = morton.Morton(dimensions=3, bits=128)
    convert = lambda p: m.pack(*p.tolist())
    zOrder = np.argsort(np.apply_along_axis(convert, 1, scaled))
    return zOrder

class Node():
    '''Class representing a node of the oct-tree.'''
    def __init__(self, domMin: np.ndarray, domMax: np.ndarray, multiThreading: bool = False) -> None:
        self.isLeaf = True
        self.particleIds = []
        self.multipoleExpansion = []
        self.domainMin = domMin
        self.domainMax = domMax
        self.children = []

        #threading
        if multiThreading:
            self.lock = threading.Lock()


def determineChildDomain(node:Node, idx: int) -> tuple:
    '''Returns the domain a child node with index 0 <= idx < 8'''
    center = (node.domainMin + node.domainMax)/2 

    new_min = np.array([node.domainMin[i] if ((idx >> i) & 1) == 0 else center[i] for i in range(3)])
    new_max = np.array([center[i] if ((idx >> i) & 1) == 0 else node.domainMax[i] for i in range(3)])

    return new_min, new_max

def splitNode(positions: Union[np.ndarray,jnp.ndarray], leafs: Union[np.ndarray, jnp.ndarray], node: Node, nCrit: int, multiThreading):
    ''' Splits a node into 8 childs, and distributes all particle ids it contains to its childs.'''
    node.children = [Node(*determineChildDomain(node,i)) for i in range(8)]
    buf = node.particleIds
    node.particleIds = []
    node.isLeaf = False
    for idx in buf:
        insertParticle(positions, leafs, idx, node, nCrit, multiThreading)
    if multiThreading:
        del node.lock

def determineOctantIdx(node: Node, pos: Union[np.ndarray, jnp.ndarray]) -> int:
    '''Determine to which child a particle should be assigned.'''
    center = (node.domainMin + node.domainMax)/2 

    octant_idx = 0
    for i in range(3):
        if pos[i] >= center[i]:  
            octant_idx |= (1 << i)
    return octant_idx

def insertParticle(positions: Union[np.ndarray, jnp.ndarray], leafs: Union[np.ndarray, jnp.ndarray], partIdx: int, root:Node, nCrit: int, multiThreading: bool):
    '''Insert the particle positions[partIdx] into the given root node.'''
    node = root
    while(node.isLeaf == False):
        idx = determineOctantIdx(node, positions[partIdx])
        node = node.children[idx]
    if multiThreading:
        node.lock.acquire_lock()
    node.particleIds.append(partIdx)
    leafs[partIdx] = node
    if len(node.particleIds) > nCrit: 
        splitNode(positions, leafs, node, nCrit, multiThreading)
    if multiThreading:
        if node.lock:
            node.lock.release()

def applyBoundaryCondition(domMin: Union[np.ndarray, jnp.ndarray], domMax: Union[np.ndarray, jnp.ndarray], positions: Union[np.ndarray,jnp.ndarray]):
    '''
    Aplly periodic boundary condition to a given domain. 

    Parameters
    ----------
    domMin: ndarray
        Minimum of the simulation domain.
    domMax: ndarray
        Maximum of the simulation domain.
    positions: numpy.ndarray | jax.numpy.array
        Positions that have to be in the domain.
    '''
    positions -= domMin
    positions %= domMax - domMin
    positions += domMin


def buildTree(positions: Union[np.ndarray,jnp.ndarray], leafs: Union[np.ndarray, jnp.ndarray], domainMin: np.ndarray, domainMax: np.ndarray, nCrit: int = 32, nThreads: int = 4) -> Node:
    '''
    Build a oct-tree in a given domain.

    Parameters
    ----------
    positions: np.ndarray
        Positions of particles to be placed inside the tree.
    domainMin: np.ndarray
        Minimal point [x_min,y_min,z_min] of simulation domain.
    domainMax: np.ndarray
        Maximal point [x_max,y_max,z_max] of simulation domain.   
    nCrit: int
        Maximal number of particles per cell. If particles > nCrit -> split cell.
    nThreads: int
        Number of threads in multithreaded execution.
    '''
    # create threads if needed
    if nThreads > 1:
        root = Node(domainMin, domainMax, True)
        partBlock = int(len(positions)/nThreads)
        remainder = len(positions) % nThreads
        def worker(threadId):
            if threadId <= nThreads:
                print(20)

        threads = []
    else:
        root = Node(domainMin, domainMax)
        for i in range(len(positions)):
            insertParticle(positions, leafs, i, root, nCrit, False)

    return root
