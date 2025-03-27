from geolib.tree import Node


class FixedSESAcceptanceCriterion():
    '''Test if the acceptance criterian for a fixed opening angle is met.'''

    def __init__(self, angle: float = 1.) -> None:
        self.openAngle = angle

    def eval(self, A: Node, B: Node) -> bool:
        return (A.potentialCenter[1] + B.potentialCenter[1])/np.norm(A.multipoleCenter[0]-B.potentialCenter[0]) < self.openAngle


class AdvancedAcceptanceCriterion():
    '''Test for the advanced acceptance criterion for aiming to obtain errors < a given relative force error epsilon.'''

    def __init__(self, epsilon: float = 10**(-6.25)):
        self.eps = epsilon

    def eval(self, A: Node, B:Node) -> bool:
        angle = dist < 1        
        #TODO implement advanced criterion 
        return False

