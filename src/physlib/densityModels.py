"""
Module to sample particles according to a density model.
Sampling is unitless!!! (G=1, M=1, a,r = 1)
Convert to required units if needed. 

TODO: Right now only assuming particles of equal mass. (numerical and implementation simplicisity)
"""
from typing import Union, List
from jax.typing import ArrayLike
from typing_extensions import override

from geolib.coordinates import Point3D, Polar3D
import numpy as np

from abc import ABC, abstractmethod
class densityModel(ABC):
    @abstractmethod
    def sample(self):
        pass
    

class UniformBox(densityModel):
    """
    A bounding Box where the mass is uniformly distributed -> random sampling (assuming equal mass bodies)
    """
    def __init__(self, center: Union[Point3D, Polar3D, List[float]]= Point3D(0.,0.,0.), x_length: float = 1., y_length: float = 1., z_length: float = 1.) -> None:
        self.center = center if isinstance(center, Point3D) else Point3D(center)
        self.x_length = x_length 
        self.y_length = y_length 
        self.z_length = z_length
        # variables for sampling from the distribution
        self.__rng = np.random.default_rng()
        self.__ubound = np.array([x_length/2, y_length/2, z_length/2])
        self.__lbound = -1 * self.__ubound

    def samplePoint3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        return np.array([Point3D(p) for p in self.sample(n)])

    def samplePolar3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        return np.array([Point3D(p).to_Polar3D() for p in self.sample(n)])

    def sample(self, n: int = 1) -> np.ndarray: # pyright: ignore
        """Sample points according to density model."""
        return self.center.to_list() + self.__rng.uniform(self.__lbound, self.__ubound, (n,3))

class UniformSphere(densityModel):
    """
    A bounding Sphere where the mass is unformly distributed -> random sampling (assuming equally heavy bodies)
    """
    def __init__(self, center: Union[Point3D,Polar3D, List[float]] = Point3D(0.,0.,0.), radius: float = 1) -> None:
        self.center = center if isinstance(center, Point3D) else Point3D(center)
        self.radius = radius
        # variabless for sampling from the distribution
        self._rng = np.random.default_rng()

    def samplePoint3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        lbound = [0,-1]
        ubound = [2*np.pi, 1]
        samples = self._rng.uniform(lbound,ubound,(n,2))
        samplest = samples.transpose()

        fx = lambda theta,u: np.sqrt(1-u**2) * np.cos(theta)
        fy = lambda theta,u: np.sqrt(1-u**2) * np.sin(theta)

        samples = self.center.to_list() + (self._sample_r(n).reshape(n,1) * np.array([fx(samplest[0],samplest[1]),fy(samplest[0],samplest[1]), samplest[1]]).transpose())
        return np.array([Point3D(p) for p in samples])


    def samplePolar3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        lbound = [-1,0]
        ubound = [1, 2*np.pi]
        samples = self._rng.uniform(lbound, ubound, (n,2)).transpose()

        samples = np.array([self._sample_r(n), np.arccos(samples[0]), samples[1]]).transpose()
        return np.array([Polar3D(x) + self.center for x in samples])

    def sample(self, n: int = 1) -> np.ndarray: # pyright: ignore
        """Sample points according to density model."""
        return np.array([x.to_list() for x in self.samplePoint3D(n)])

    def _sample_r(self, n: int = 1) -> np.ndarray:
        """Sample radii according to density model"""
        samples = self._rng.uniform(0, 1, n)
        fr = lambda u: self.radius * np.cbrt(u)
        return fr(samples)
        
class PlummerSphere(UniformSphere):
    """
    A Sphere following the Plummer Model (G=1, M=1, a=1).
    Samples can be scaled manually.
    """
    def __init__(self, center: Union[Point3D, Polar3D, List[float]] = Point3D(0, 0, 0)) -> None:
        super().__init__(center, 1)

    @override
    def _sample_r(self, n: int = 1) -> np.ndarray:
        samples = self._rng.uniform(0,1,n)
        return  np.power(np.power(samples, -2/3) - 1, -1/2)

    def sample_vel(self, radius: ArrayLike):
        '''Sample velocities according to radii (Neumann's rejection method).'''
        escape_vel = lambda r: np.sqrt(2)*np.power((1+np.power(r,2)), -1/4)
        g = lambda q: np.power(q,2)*np.power((1 - np.power(q,2)),7/2)

        radius = np.array(radius)
        vel = []

        for r in radius:
            sample = self._rng.uniform([0,0], [1,1], 1)
            while 0.1 * sample[1] >= g(sample[0]):
                sample = self._rng.uniform([0,0], [1,1], 1)
            q = sample[0]
            v = q*escape_vel(r)

            sample = self._rng.uniform([0,0], [1,1], 1)
            w = (1-2*sample[0]) * v
            u = np.sqrt(np.power(v,2) - np.power(w,2)) * np.cos(2*np.pi*sample[1])
            v = np.sqrt(np.power(v,2) - np.power(w,2)) * np.sin(2*np.pi*sample[1])

            vel.append([u,v,w])

        return np.array(vel)
