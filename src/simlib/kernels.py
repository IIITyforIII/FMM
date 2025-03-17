'''
Module contains kernel functions used in physical nBody simulations.
'''
import jax.numpy as jnp

class p2p():
    '''Particle-to-Particle (P2P) kernel.'''

    @staticmethod
    def apply(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Calculate the (negative) potentials of sinks using pairwise Particle-to-Particle interactions to sources.
        '''
        a1, a2 = source.reshape(-1,1,3), sink.reshape(1,-1,3)
        r = jnp.linalg.norm(a1 - a2, axis=2)
        greens_func = lambda r: jnp.where(r==0, 0, 1/r)
        pot = greens_func(r)
        return G*M*jnp.sum(pot, axis=1) # pyright: ignore
    potential = apply

    @staticmethod
    def grad(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Calculate accelerations (gradient of (negative) potential) using pairwise Particle-to-Particle interactions.
        '''
        a1, a2 = source.reshape(-1,1,3), sink.reshape(1,-1,3)
        r = a1-a2
        r_abs = jnp.linalg.norm(r, axis=2)
        greens_grad = lambda r: jnp.where(r==0, 0, 1/r**3)
        acc = greens_grad(r_abs)
        acc = acc[:,:,jnp.newaxis] * r       # pyright: ignore
        return G*M*jnp.sum(acc, axis=1)      # pyright: ignore
    acceleration = grad

    @staticmethod
    def hessian(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Caculate hessians using pairwise Particle-to-Particle interactions.
        '''
    jerk = hessian

