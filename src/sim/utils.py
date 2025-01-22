from coordinates import CartesianCoordinate

import random


def random_pos_in_box(center: CartesianCoordinate, width: float):
    return center + CartesianCoordinate(random.uniform(-width / 2, width / 2),
                                        random.uniform(-width / 2, width / 2),
                                        random.uniform(-width / 2, width / 2))
