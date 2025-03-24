from abc import ABC, abstractmethod
from geolib.tree import Node
import numpy as np
import jax.numpy as jnp
import miniball

class ExpansionCenter(ABC):
    @abstractmethod
    def computeExpCenter(self, positions: jnp.ndarray, node: Node) -> tuple:
        pass

class GeometricCenter(ExpansionCenter):
    '''Class computing an expansion center as the geometric cell center.'''

    def computeExpCenter(self, positions: jnp.ndarray, node: Node) -> tuple:
        return (node.domainMin + node.domainMax)/2 , 0

class SmallesEnclosingSphere(ExpansionCenter):
    '''Class computing an expansion center as the smalles enclosing sphere.'''
    def __init__(self, multipole: bool) -> None:
        self.multipole = multipole
    
    def computeExpCenter(self, positions: jnp.ndarray, node: Node) -> tuple:
        if node.isLeaf: 
            if len(node.particleIds)== 0:
                return None, None
            parts = np.asarray(positions[jnp.array(node.particleIds)])
            if len(parts)>1:
                center, radius = miniball.get_bounding_ball(parts)
            else:
                center, radius = parts[0], 0
            return center, np.sqrt(radius)            
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

        return new_center, new_r

class CenterOfMass(ExpansionCenter):
    '''Class computing a expansion center as the center of mass.'''
    def __init__(self, multipole: bool) -> None:
        self.multipole = multipole

    def computeExpCenter(self, positions: jnp.ndarray, masses: jnp.array, node: Node) -> tuple: # pyright: ignore
        if node.isLeaf:
            if len(node.particleIds) == 0:
                return None, None
            p = np.asarray(positions[jnp.array(node.particleIds)])
            m = np.asarray(masses[jnp.array(node.particleIds)])
            p = p * m
            m = m.sum()
            return p.sum(axis=0)/m, m
        else:
            runAvg = np.zeros(3)               
            count  = 0
            if self.multipole:
                for c in node.children:
                    if not (c.multipoleCenter[0] is None):
                        runAvg += c.multipoleCenter[0] * c.multipoleCenter[1]
                        count  += c.multipoleCenter[1]
            else:
                for c in node.children:
                    if not (c.potentialCenter[0] is None):
                        runAvg += c.potentialCenter[0] * c.potentialCenter[1]
                        count  += c.potentialCenter[1]
            return runAvg/count, count
    
