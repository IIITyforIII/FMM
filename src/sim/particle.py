from coordinates import SphericalCoordinate, CartesianCoordinate


class Particle:

    def __init__(self, position: CartesianCoordinate):
        self.position: SphericalCoordinate = position.as_spherical_coordinates()
        self.velocity: SphericalCoordinate = SphericalCoordinate(0, 0, 0)
        self.acceleration: SphericalCoordinate = SphericalCoordinate(0, 0, 0)
        self.mass: complex = 100
        self.psi: float = 0
