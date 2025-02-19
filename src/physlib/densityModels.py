"""
Module to sample particles according to a density model.
Sampling is unitless! (G=1)

TODO: Right now only assuming particles of equal mass. (numerical and implementation simplicisity)
"""
from typing import Union, List

from geolib.coordinates import Point3D, Polar3D
import numpy as np



class uniformBox():
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

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        return self.center.to_list() + self.__rng.uniform(self.__lbound, self.__ubound, (n,3))

class uniformSphere():
    """
    A bounding Sphere where the mass is unformly distributed -> random sampling (assuming equally heavy bodies)
    """
    def __init__(self, center: Union[Point3D,Polar3D, List[float]] = Point3D(0.,0.,0.), radius: float = 1) -> None:
        self.center = center if isinstance(center, Point3D) else Point3D(center)
        self.radius = radius
        # variabless for sampling from the distribution
        self.__rng = np.random.default_rng()

    def samplePoint3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        lbound = [0,-1]
        ubound = [2*np.pi, 1]
        samples = self.__rng.uniform(lbound,ubound,(n,2))
        samplest = samples.transpose()

        fx = lambda theta,u: np.sqrt(1-u**2) * np.cos(theta)
        fy = lambda theta,u: np.sqrt(1-u**2) * np.sin(theta)

        samples = self.center.to_list() + (self.__sample_r(n).reshape(n,1) * np.array([fx(samplest[0],samplest[1]),fy(samplest[0],samplest[1]), samplest[1]]).transpose())
        return np.array([Point3D(p) for p in samples])


    def samplePolar3D(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        lbound = [-1,0]
        ubound = [1, 2*np.pi]
        samples = self.__rng.uniform(lbound, ubound, (n,2)).transpose()

        samples = np.array([self.__sample_r(n), np.arccos(samples[0]), samples[1]]).transpose()
        return np.array([Polar3D(x) + self.center for x in samples])

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample points according to density model."""
        return np.array([x.to_list() for x in self.samplePoint3D(n)])

    def __sample_r(self, n: int = 1) -> np.ndarray:
        """Sample radii according to density model"""
        samples = self.__rng.uniform(0, 1, n)
        fr = lambda u: self.radius * np.cbrt(u)
        return fr(samples)
        
        
