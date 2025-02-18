from typing import Union
from geolib.coordinates import Point3D, Polar3D


class Particle:
    """
    Basic class representing a point mass
    """
    def __init__(self, position: Union[Point3D, Polar3D], velocity: Polar3D = Polar3D(0.,0.,0.), mass: float = 1) -> None:
        self.position:     Polar3D = Polar3D(position) if isinstance(position, Point3D) else velocity
        self.velocity:     Polar3D = velocity
        self.acceleration: Polar3D = Polar3D(0., 0., 0.)
        self.mass:         float   = mass 


class Star(Particle):
    """
    TODO: Maybe go to collision handling, using particles with spatial extension
    """
    def __init__(self, position: Point3D, velocity:Polar3D = Polar3D(0.,0.,0.), mass: float = 1, radius: float = 0.5) -> None:
        super().__init__(position, velocity, mass)
        self.radius = radius

if __name__ == '__main__':
    print('test')
