from abc import ABC, abstractmethod
from ctypes import ArgumentError
from typing import Optional
import numpy as np
from scipy.special import binom, factorial
from geolib.tree import Node

class AcceptanceCriterion(ABC):
    @abstractmethod
    def eval(self, A: Node, B:Node) -> bool:
        '''Evaluate the Multipole Acceptance criterion for potential in B due to A.'''
        pass

class FixedAcceptanceCriterion(AcceptanceCriterion):
    '''Test if the acceptance criterian for a fixed opening angle is met.'''

    def __init__(self, angle: float = 1.) -> None:
        self.openAngle = angle

    def eval(self, A: Node, B: Node) -> bool:
        if (A == B): return False
        return (A.multipoleCenter[1] + B.potentialCenter[1])/np.linalg.norm(B.potentialCenter[0] - A.multipoleCenter[0]) < self.openAngle


class AdvancedAcceptanceCriterion(AcceptanceCriterion):
    '''Test for the advanced acceptance criterion for aiming to obtain errors < a given relative force error epsilon.'''

    def __init__(self, epsilon: float = 10**(-6.25)):
        self.eps = epsilon
        self.a_f = None

    def eval(self, A: Node, B:Node) -> bool:
        if (A==B): return False
        if self.a_f is None:
            raise ArgumentError('a or f needed.')
        else:
            a_f = self.a_f

        dist = np.linalg.norm(B.potentialCenter[0] - A.multipoleCenter[0])
        # compute opening angle
        a = (A.multipoleCenter[1] + B.potentialCenter[1])/dist

        # compute relative error term
        e = self.errorBound(A,B,dist)
        e = 8*np.max((A.multipoleCenter[1], B.potentialCenter[1]))/(A.multipoleCenter[1] + B.potentialCenter[1]) * e
        e = e * A.multipoleCenter[2]/dist**2
        return a < 1 and e < self.eps*np.min(a_f[B.particleIds])

    def errorBound(self, A:Node, B:Node, dist:float):
        order = A.multipoleExpansion.shape[0]
        res = 0.
        for k in np.arange(order):
            res += binom(order-1, k) * A.multipolePower[k] * np.power(B.potentialCenter[1], order-1-k) / np.power(dist,order-1)
        return res/A.multipoleCenter[2]

    def setAorF(self, a_f):
        self.a_f = a_f

        

    @staticmethod
    def computeMultipolePower(node: Node):
        '''Compute the multipole Power (it is already the square root).'''
        if len(node.particleIds) == 0:
            return 0
        expOrder = node.multipoleExpansion.shape[0]
        res = np.zeros(expOrder)
        for n in np.arange(expOrder):
            for m in range(-n, n+1):
                res[n] = np.sqrt(factorial(n-m) * factorial(n+m)) * np.abs(node.multipoleExpansion[n,m])
        return res


