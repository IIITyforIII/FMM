from abc import ABC, abstractmethod
from ctypes import ArgumentError
from typing import Optional

from geolib.tree import Node
import numpy as np
import miniball

class ExpansionCenter(ABC):
    @abstractmethod
    def computeExpCenter(self, positions: np.ndarray, node: Node, masses: np.ndarray) -> tuple:
        pass

class GeometricCenter(ExpansionCenter):
    '''Class computing an expansion center as the geometric cell center.'''

    def computeExpCenter(self, positions: np.ndarray, node: Node, masses: Optional[np.ndarray] = None) -> tuple:
        return (node.domainMin + node.domainMax)/2 , 0

class SmallesEnclosingSphere(ExpansionCenter):
    '''Class computing an expansion center as the smalles enclosing sphere.'''
    def __init__(self, multipole: bool) -> None:
        self.multipole = multipole
    
    def computeExpCenter(self, positions: np.ndarray, node: Node, masses: np.ndarray) -> tuple:
        if node.isLeaf: 
            if len(node.particleIds)== 0:
                return None, None
            parts = positions[node.particleIds]
            if len(parts)>1:
                center, radius = miniball.get_bounding_ball(parts)
            else:
                center, radius = parts[0], 0
            return center, np.sqrt(radius), np.sum(masses[node.particleIds])            
        else:
            if self.multipole:
                merged = node.children[0].multipoleCenter
                for c in node.children[1:]:
                    merged = self.mergeSpheres(merged, c.multipoleCenter)
            else:
                merged = node.children[0].potentialCenter
                for c in node.children[1:]:
                    merged = self.mergeSpheres(merged, c.potentialCenter)
            return merged

    @staticmethod
    def mergeSpheres(first: tuple, other:tuple):
        '''Merge two spheres (center, radius) together into output. They are assumed to be located in different cells.'''
        if first[0] is None:
            return other
        if other[0] is None:
            return first
        dist = np.linalg.norm(first[0] - other[0])

        new_r = (first[1] + other[1] + dist)/2
        direc = (other[0] - first[0]) / dist if dist != 0 else np.zeros_like(first[0])
        new_center = first[0] + direc * (new_r - first[1])

        return new_center, new_r, first[2]+other[2]

class CenterOfMass(ExpansionCenter):
    '''Class computing a expansion center as the center of mass.'''
    def __init__(self, multipole: bool) -> None:
        self.multipole = multipole

    def computeExpCenter(self, positions: np.ndarray, node: Node, masses: np.ndarray) -> tuple: 
        if masses is None:
            raise ArgumentError('Computation of center of mass, requires the masses.')
        if node.isLeaf:
            if len(node.particleIds) == 0:
                return None, None, None
            p = positions[node.particleIds]
            m = masses[node.particleIds]
            p = p * m
            m = m.sum()
            center = p.sum(axis=0)/m
            radius = np.max(np.linalg.norm(p-center, axis=1))
            # compared to smallest enclosing sphere also compute radius
            return p.sum(axis=0)/m, radius, m 
        else:
            # runAvg = np.zeros(3)               
            # count  = 0
            cs = []
            rs = []
            ms = []
            if self.multipole:
                for c in node.children:
                    if not (c.multipoleCenter[0] is None):
                        cs.append(c.multipoleCenter[0])
                        rs.append(c.multipoleCenter[1])
                        ms.append(c.multipoleCenter[2])
                        # runAvg += c.multipoleCenter[0] * c.multipoleCenter[1]
                        # count  += c.multipoleCenter[1]
            else:
                for c in node.children:
                    if not (c.potentialCenter[0] is None):
                        cs.append(c.potentialCenter[0])
                        rs.append(c.potentialCenter[1])
                        ms.append(c.potentialCenter[2])
                        # runAvg += c.potentialCenter[0] * c.potentialCenter[1]
                        # count  += c.potentialCenter[1]
            if(len(cs) == 0): return None,None,None
            cs = np.array(cs)
            rs = np.array(rs)
            ms = np.array(ms).reshape(-1,1)
            c = np.sum(cs*ms, axis=0)            
            m = np.sum(ms)
            c = c/m
            radius = np.max(rs + np.linalg.norm(c - cs,axis=1))

            return c, radius, m
