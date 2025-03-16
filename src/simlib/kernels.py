from numpy.typing import ArrayLike


class p2p():
    '''Particle-to-Particle (P2P) kernel.'''

    @staticmethod
    def pot(source: ArrayLike, sink: ArrayLike, G: float):
        '''
        Calculate resulting potentials using pairwise Particle-to-Particle interactions.
        '''
        pass

    @staticmethod
    def acc(source: ArrayLike, sink: ArrayLike, G: float):
        '''
        Calculate resulting accelerations using pairwise Particle-to-Particle interactions.
        '''
        pass

