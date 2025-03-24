'''
Module contains kernel functions used in physical nBody simulations.
'''
from typing import Union
import jax.numpy as jnp
import numpy as np
from scipy.special import factorial, assoc_legendre_p_all
from jax import jacfwd, vmap

from geolib.coordinates import mapCartToPolar

class p2p():
    '''Particle-to-Particle (P2P) kernel.'''

    @staticmethod
    def apply(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: Union[jnp.ndarray, float] = jnp.array([1])) -> jnp.ndarray:
        '''
        Calculate the (negative) potentials of sinks using pairwise Particle-to-Particle interactions to sources.
        '''
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = jnp.linalg.norm(a1 - a2, axis=-1)
        greens_func = lambda r: jnp.where(r==0, 0, 1/r)
        pot = greens_func(r)
        masses = jnp.array(M, dtype=float).reshape(1, -1)
        return G*jnp.sum(masses * pot, axis=-1) # pyright: ignore
    potential = apply

    @staticmethod
    def grad(sink: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: Union[jnp.ndarray, float]= jnp.array([1])) -> jnp.ndarray:
        '''
        Calculate accelerations (gradient of (negative) potential) using pairwise Particle-to-Particle interactions.
        '''
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = a1-a2
        r_norm= jnp.linalg.norm(r, axis=-1)
        greens_grad = lambda r: jnp.where(r==0, 0, 1/r**3)
        acc = greens_grad(r_norm)
        masses = jnp.array(M, dtype=float).reshape(1, -1)
        acc = masses * acc
        acc = acc[:,:,jnp.newaxis] * r       # pyright: ignore
        return -1*G*jnp.sum(acc, axis=1)   # pyright: ignore
    acceleration = grad

    @staticmethod
    def jacobian(sink: jnp.ndarray, source:jnp.ndarray, G: float = 1, M: Union[jnp.ndarray, float] = jnp.array([1])) -> jnp.ndarray:
        ''' Calculate the jacobian (∇(∇ϕ)) for all sinks using pairwise P2P kernel'''
        sink = sink.reshape(-1, 3)
        g = jacfwd(lambda x: p2p.grad(x, source, G, M)[0])
        return jnp.apply_along_axis(g, 1, sink) 

    @staticmethod
    def jerk(sink: jnp.ndarray, sink_velocity: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: Union[jnp.ndarray,float] = jnp.array([1])) -> jnp.ndarray:
        '''
        Caculate jerk using pairwise P2P kernel.
        jerk = d(∇ϕ)/dr * dr/dt = Jv = da/dt
        '''
        vel = sink_velocity.reshape(-1,3)
        return vmap(jnp.matmul)(p2p.jacobian(sink, source, G, M), vel)
    da_dt = jerk


    @staticmethod
    def snap(sink: jnp.ndarray, sink_velocity: jnp.ndarray, sink_acceleration: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: Union[jnp.ndarray,float] = jnp.array([1])):
       '''
       Caclulate snap (d^2a/dt^2) using parwise P2P kernel.
       snap = d(Jv)/dt = dJ/dt*v + J*dv/dt = (v @ ∇)Jv + Ja
       '''
       mul = vmap(jnp.matmul)
       sink = sink.reshape(-1,3)
       vel = sink_velocity.reshape(-1,3)
       acc = sink_acceleration.reshape(-1,3)
       Ja = mul(p2p.jacobian(sink, source, G, M), acc)
       g = jacfwd(lambda x: p2p.jacobian(x, source, G, M)[0])
       advection = mul(vel, jnp.apply_along_axis(g, 1, sink))
       advection = mul(advection, vel)
       return advection + Ja        
    da_dtdt = snap

class SphericalHarmonics():
    '''
    Compute spherical harmonics used in FMM for a given expansion order p.
    Computes for all up to degree n and order m , see assoc_legendre_p_all()
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m
        #condon shortney phase
        self.condonPhase = np.ones(2*m+1)
        self.condonPhase[1:m+1:2] = -1 
        self.condonPhase[-1:-m-1:-2] = -1
        self.condonPhase = np.tile(self.condonPhase, (n+1,1))
        # create index matrizes
        self.m_arr = np.hstack((range(m+1), range(-m,0)))
        self.m_arr = np.tile(self.m_arr, (n+1,1))
        self.n_arr = np.tile(range(n+1), (2*m + 1,1)).transpose()

    @staticmethod
    def theta(particle: jnp.ndarray):
        '''Perform θ(r) on a given particel.'''
        pass

    def ypsilon(self, particle: jnp.ndarray):
        '''Perform Υ(r) on a given particle.'''
        #polar coordinates
        pol = mapCartToPolar(particle)

        #normalization
        r_n = pol[0]**self.n_arr
        norm = factorial(self.n_arr + self.m_arr)
        norm = np.divide(r_n, norm, where=norm!=0, out=np.zeros_like(norm))
        norm.shape

        #legendre polynomials
        legendre = assoc_legendre_p_all(self.n,self.m,np.cos(pol[1]))[0]

        #azimuthal term
        azim = np.exp(1j * self.m_arr * pol[2])        

        return self.condonPhase * norm * legendre * azim
