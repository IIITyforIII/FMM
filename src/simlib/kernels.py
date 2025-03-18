'''
Module contains kernel functions used in physical nBody simulations.
'''
from operator import matmul
import jax.numpy as jnp
from jax import jacfwd, vmap

class p2p():
    '''Particle-to-Particle (P2P) kernel.'''

    @staticmethod
    def apply(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Calculate the (negative) potentials of sinks using pairwise Particle-to-Particle interactions to sources.
        '''
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = jnp.linalg.norm(a1 - a2, axis=-1)
        greens_func = lambda r: jnp.where(r==0, 0, 1/r)
        pot = greens_func(r)
        return G*M*jnp.sum(pot, axis=-1) # pyright: ignore
    potential = apply

    @staticmethod
    def grad(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Calculate accelerations (gradient of (negative) potential) using pairwise Particle-to-Particle interactions.
        '''
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = a1-a2
        r_norm= jnp.linalg.norm(r, axis=-1)
        greens_grad = lambda r: jnp.where(r==0, 0, 1/r**3)
        acc = greens_grad(r_norm)
        acc = acc[:,:,jnp.newaxis] * r       # pyright: ignore
        return -1*G*M*jnp.sum(acc, axis=1)   # pyright: ignore
    acceleration = grad

    @staticmethod
    def jacobian(sink: jnp.ndarray, source:jnp.ndarray, G: float = 1, M: float = 1):
        ''' Calculate the jacobian (∇(∇ϕ)) for all sinks using pairwise P2P kernel'''
        sink = sink.reshape(-1, 3)
        g = jacfwd(lambda x: p2p.grad(x, source, G, M)[0])
        return jnp.apply_along_axis(g, 1, sink) 

    @staticmethod
    def jerk(sink: jnp.ndarray, sink_velocity: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
        '''
        Caculate jerk using pairwise P2P kernel.
        jerk = d(∇ϕ)/dr * dr/dt = Jv = da/dt
        '''
        vel = sink_velocity.reshape(-1,3)
        return vmap(jnp.matmul)(p2p.jacobian(sink, source, G, M), vel)
    da_dt = jerk


    @staticmethod
    def snap(sink: jnp.ndarray, sink_velocity: jnp.ndarray, sink_acceleration: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: float = 1):
       '''
       Caclulate snap (d^2a/dt^2) using parwise P2P kernel.
       snap = d(Jv)/dt = dJ/dt*v + J*dv/dt = (v @ ∇)Jv + Ja
       '''
       mul = vmap(matmul)
       sink = sink.reshape(-1,3)
       vel = sink_velocity.reshape(-1,3)
       acc = sink_acceleration.reshape(-1,3)
       Ja = mul(p2p.jacobian(sink, source, G, M), acc)
       g = jacfwd(lambda x: p2p.jacobian(x, source, G, M)[0])
       advection = mul(vel, jnp.apply_along_axis(g, 1, sink))
       advection = mul(advection, vel)
       return advection + Ja        
    da_dtdt = snap
