"""
Module contains dataclasses to store coordiantes
"""
from __future__ import annotations
from typing import Optional, Union, List, Self
from dataclasses import dataclass, astuple
import numpy as np


@dataclass
class Polar3D:
    """
    Dataclass to save a point in space as polar coordinates (r, theta, phi).
    """
    r:     float
    theta: float
    phi:   float  

    def __init__(self: Self, point_or_r: Union[float, Point3D, List[Union[float,int]], np.ndarray], theta:Optional[float] = None, phi:Optional[float] = None) -> None:
        if isinstance(point_or_r, Point3D):
           self.r, self.theta, self.phi = astuple(point_or_r.toPolar3D()) 
        elif isinstance(point_or_r, (float, int)) and isinstance(theta, (float, int)) and isinstance(phi, (float, int)):
            self.r = point_or_r 
            self.theta = theta if theta != None else 0.
            self.phi = phi if phi != None else 0.
        elif isinstance(point_or_r, (list, np.ndarray)) and all(isinstance(x, (float,int)) for x in point_or_r) and len(point_or_r) == 3:
            self.r = point_or_r[0]
            self.theta = point_or_r[1]
            self.phi = point_or_r[2]
        else:
            raise TypeError('Invalid arguments: (float, float, float), (list[float|int]) of size 3 or (Point3D) expected.')

    def __add__(self: Self, other: Polar3D) -> Polar3D:
        return (self.to_Point3D() + other.to_Point3D()).to_Polar3D()

    def __sub__(self: Self, other: Polar3D) -> Polar3D:
        return (self.to_Point3D() - other.to_Point3D()).to_Polar3D()

    def to_Point3D(self: Self) -> Point3D:
        """
        Convert point to cartesian coordinates.
        """
        x = self.r * np.sin(self.theta) * np.cos(self.phi)
        y = self.r * np.sin(self.theta) * np.sin(self.phi)
        z = self.r * np.cos(self.theta)
        return Point3D(x, y, z)

    def to_list(self):
        """
        Write attributes as a list.
        """
        return [self.r, self.theta, self.phi]

    def to_ndarray(self):
        """
        Write attributes as a np.ndarray
        """
        return np.array(self.to_list())

@dataclass
class Point3D:
    """
    Dataclass to save a point in space as cartesian coordinates (x,y,z).
    """
    x: float = 0.
    y: float = 0.
    z: float = 0.

    def __init__(self, polar_or_x: Union[float, Polar3D, List[Union[float, int]]], y: Optional[float] = None, z: Optional[float] = None) -> None:
        if isinstance(polar_or_x, Polar3D):
            self.x, self.y, self.z = astuple(polar_or_x.to_Point3D())
        elif isinstance(polar_or_x, (float, int)) and isinstance(y, (float, int)) and isinstance(z, (float, int)):
            self.x = polar_or_x
            self.y = y
            self.z = z
        elif isinstance(polar_or_x, (list, np.ndarray)) and all(isinstance(x, (float,int)) for x in polar_or_x) and len(polar_or_x) == 3:
            self.x = polar_or_x[0]
            self.y = polar_or_x[1]
            self.z = polar_or_x[2]
        else:
            raise TypeError('Invalid arguments: (float, float, float), (list[float|int]) of size 3 or (Polar3D) expected.')

    def __add__(self: Self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_Polar3D(self) -> Polar3D:
        """
        Convert point to polar coordinates.
        """
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arctan(self.y / self.x) if r != 0 else 0
        phi = np.arccos(self.z / r) if r != 0 else 0
        return Polar3D(r, theta, phi)
    
    def to_list(self):
        """
        Write attributes as a list.
        """
        return [self.x, self.y, self.z]

    def to_ndarray(self):
        """
        Write attributes as a np.ndarray.
        """
        return np.array(self.to_list())

