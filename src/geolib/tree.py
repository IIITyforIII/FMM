from typing import Union
import jax.numpy as jnp
import numpy as np
import morton 
import threading

def getMortonSortedPermutation(positions: np.ndarray, scaleFactor: int = int(10e6)) -> np.ndarray:
    '''
    Convert a List of points to their morton-order representation.

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
    def __init__(self, domMin: np.ndarray, domMax: np.ndarray, multiThreading: bool = False) -> None:
        self.isLeaf = True
        self.particleIds = []
        self.multipole = 0
        self.domainMin = domMin
        self.domainMax = domMax
        self.children = []

        #threading
        if multiThreading:
            self.lock = threading.Lock()


def determineChildDomain(node:Node, idx: int) -> tuple:
    center = (node.domainMin + node.domainMax)/2 

    new_min = np.array([node.domainMin[i] if ((idx >> i) & 1) == 0 else center[i] for i in range(3)])
    new_max = np.array([center[i] if ((idx >> i) & 1) == 0 else node.domainMax[i] for i in range(3)])

    return new_min, new_max

def splitNode(positions: Union[np.ndarray,jnp.ndarray], node: Node, nCrit: int, multiThreading):
    node.children = [Node(*determineChildDomain(node,i)) for i in range(8)]
    buf = node.particleIds
    node.particleIds = []
    node.isLeaf = False
    for idx in buf:
        insertParticle(positions, idx, node, nCrit, multiThreading)
    del node.lock

def determineOctantIdx(node: Node, pos: Union[np.ndarray, jnp.ndarray]) -> int:
    center = (node.domainMin + node.domainMax)/2 

    octant_idx = 0
    for i in range(3):
        if pos[i] >= center[i]:  
            octant_idx |= (1 << i)
    return octant_idx

def insertParticle(positions: Union[np.ndarray, jnp.ndarray], partIdx: int, root:Node, nCrit: int, multiThreading: bool):
    node = root
    while(node.isLeaf == False):
        idx = determineOctantIdx(node, positions[partIdx])
        node = node.children[idx]
    if multiThreading:
        node.lock.acquire_lock()
    node.particleIds.append(partIdx)
    if len(node.particleIds) > nCrit: 
        splitNode(positions, node, nCrit, multiThreading)
    if multiThreading:
        if node.lock:
            node.lock.release()

def buildTree(positions: Union[np.ndarray,jnp.ndarray], domainMin: np.ndarray, domainMax: np.ndarray, nCrit: int = 32, nThreads: int = 4) -> Node:
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
        partBlock = int(positions.size/nThreads)
        remainder = positions.size % nThreads
        def worker(threadId):
            if threadId <= nThreads:
                print(20)

        threads = []
    else:
        root = Node(domainMin, domainMax)
        for i in range(positions.size):
            insertParticle(positions, i, root, nCrit, False)

    return root
