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
    def __init__(self, center: Point3D = Point3D(0.,0.,0.), x_length: float = 1., y_length: float = 1., z_length: float = 1.) -> None:
        self.center = center
        self.x_length = x_length 
        self.y_length = y_length 
        self.z_length = z_length
        # variables for sampling from the distribution
        self.__rng = np.random.default_rng()
        self.__ubound = np.array([x_length/2, y_length/2, z_length/2])
        self.__lbound = -1 * self.__ubound

    def samplePoint3D(self, n: int = 1) -> List[Point3D]:
        samples = self.__rng.uniform(self.__lbound, self.__ubound, (n,3))
        return [Point3D(p.tolist()) for p in samples]

    def samplePolar3D(self, n: int = 1) -> list[Polar3D]:
        return [Polar3D(x) for x in self.samplePoint3D(n)]

class uniformSphere():
    """
    A bounding Sphere where the mass is unformly distributed -> random sampling (assuming equally heavy bodies)
    """
    def __init__(self, center: Union[Point3D,Polar3D] = Point3D(0.,0.,0.), radius: float = 1) -> None:
        pass
