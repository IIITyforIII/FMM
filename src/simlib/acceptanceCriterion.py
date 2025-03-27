from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.special import factorial
from geolib.tree import Node

class AcceptanceCriterion(ABC):
    @abstractmethod
    def eval(self, A: Node, B:Node, a_f: Optional[np.ndarray]) -> bool:
        '''Evaluate the Multipole Acceptance criterion for potential in B due to A.'''
        pass

class FixedAcceptanceCriterion(AcceptanceCriterion):
    '''Test if the acceptance criterian for a fixed opening angle is met.'''

    def __init__(self, angle: float = 1.) -> None:
        self.openAngle = angle

    def eval(self, A: Node, B: Node, a_f: Optional[np.ndarray] = None) -> bool:
        return (A.multipoleCenter[1] + B.potentialCenter[1])/np.linalg.norm(B.potentialCenter[0] - A.multipoleCenter[0]) < self.openAngle


class AdvancedAcceptanceCriterion(AcceptanceCriterion):
    '''Test for the advanced acceptance criterion for aiming to obtain errors < a given relative force error epsilon.'''

    def __init__(self, epsilon: float = 10**(-6.25)):
        self.eps = epsilon

    def eval(self, A: Node, B:Node, a_f: Optional[np.ndarray]) -> bool:
        #TODO implement advanced criterion 
        return False

    @staticmethod
    def computeMultipolePower(node: Node):
        '''Compute the multipole Power (it is already squared).'''
        expOrder = node.multipoleExpansion.shape[0]
        res = np.zeros(expOrder)
        for n in np.arange(expOrder):
            for m in range(-n, n+1):
                res[n] = np.sqrt(factorial(n-m) * factorial(n+m)) * np.abs(node.multipoleExpansion[n,m])
        return res


