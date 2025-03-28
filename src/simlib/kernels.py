'''
Module contains kernel functions used in physical nBody simulations.
'''
from typing import Union
import jax.numpy as jnp
import numpy as np
from scipy.special import factorial, assoc_legendre_p_all
from jax import jacfwd, vmap

from geolib.coordinates import mapCartToPolar
from geolib.tree import Node

class p2p():
    '''Particle-to-Particle (P2P) kernel.'''

    @staticmethod
    def apply(sink: Union[np.ndarray,jnp.ndarray], source: Union[np.ndarray,jnp.ndarray], G: float = 1, M: Union[np.ndarray, jnp.ndarray, float] = 1., use_jax: bool = False) -> jnp.ndarray:
        '''
        Calculate the (negative) potentials of sinks using pairwise Particle-to-Particle interactions to sources.
        '''
        mod = jnp if use_jax==True else np
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = mod.linalg.norm(a1 - a2, axis=-1)
        greens_func = lambda r: jnp.where(r==0, 0, 1/r) if use_jax else np.divide(1, r, where=r!=0, out=np.zeros_like(r))
        pot = greens_func(r)
        masses = mod.array(M, dtype=float).reshape(1, -1)
        return G*mod.sum(masses * pot, axis=-1) # pyright: ignore
    potential = apply

    @staticmethod
    def grad(sink: Union[np.ndarray,jnp.ndarray], source: Union[np.ndarray,jnp.ndarray], G: float = 1, M: Union[np.ndarray,jnp.ndarray,float]= 1., use_jax:bool = False) -> jnp.ndarray:
        '''
        Calculate accelerations (gradient of (negative) potential) using pairwise Particle-to-Particle interactions.
        '''
        mod = jnp if use_jax==True else np
        a1, a2 = sink.reshape(-1,1,3), source.reshape(1,-1,3)
        r = a1-a2
        r_norm= mod.linalg.norm(r, axis=-1)
        greens_grad = lambda r: jnp.where(r==0, 0, 1/r**3) if use_jax else np.divide(1, r**3, where=r!=0, out=np.zeros_like(r))
        acc = greens_grad(r_norm)
        masses = mod.array(M, dtype=float).reshape(1, -1)
        acc = masses * acc
        acc = acc[:,:,mod.newaxis] * r       # pyright: ignore
        return -1*G*mod.sum(acc, axis=1)   # pyright: ignore
    acceleration = grad

    @staticmethod
    def jacobian(sink: jnp.ndarray, source:jnp.ndarray, G: float = 1, M: Union[jnp.ndarray, float] = jnp.array([1.])) -> jnp.ndarray:
        ''' Calculate the jacobian (∇(∇ϕ)) for all sinks using pairwise P2P kernel'''
        sink = sink.reshape(-1, 3)
        g = jacfwd(lambda x: p2p.grad(x, source, G, M, use_jax=True)[0])
        return jnp.apply_along_axis(g, 1, sink) 

    @staticmethod
    def jerk(sink: jnp.ndarray, sink_velocity: jnp.ndarray, source: jnp.ndarray, G: float = 1, M: Union[jnp.ndarray,float] = jnp.array([1.])) -> jnp.ndarray:
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

    def theta(self, pos: np.ndarray):
        '''Perform θ(r) on a given particel.'''
        #polar coordinates
        pol = mapCartToPolar(pos)

        #normalization
        r_n1 = pol[0]**(self.n_arr + 1)
        norm = factorial(self.n_arr - self.m_arr)
        norm = np.divide(norm, r_n1, where=r_n1!=0, out=np.zeros_like(norm))

        #legendre polynomials
        legendre = assoc_legendre_p_all(self.n,self.m,np.cos(pol[1]))[0]

        #azimuthal term
        azim = np.exp(1j * self.m_arr * pol[2])        

        return self.condonPhase * norm * legendre * azim

    def ypsilon(self, pos: np.ndarray):
        '''Perform Υ(r) on a given position.'''
        #polar coordinates
        pol = mapCartToPolar(pos)

        #normalization
        r_n = pol[0]**self.n_arr
        norm = factorial(self.n_arr + self.m_arr)
        norm = np.divide(r_n, norm, where=norm!=0, out=np.zeros_like(norm))

        #legendre polynomials
        legendre = assoc_legendre_p_all(self.n,self.m,np.cos(pol[1]))[0]

        #azimuthal term
        azim = np.exp(1j * self.m_arr * pol[2])        

        return self.condonPhase * norm * legendre * azim

def p2m(particles: np.ndarray, partIds: np.ndarray, center: np.ndarray, masses: np.ndarray, harmonic: SphericalHarmonics): 
    '''Compute the Particle-to-Multipole kernel.'''
    res = particles[partIds] - center
    res = np.apply_along_axis(harmonic.ypsilon, 1, res)
    res = masses[partIds].reshape(-1,1,1) * res
    return res.sum(axis=0)

def m2m(child: Node, parent: Node, harmonic: SphericalHarmonics):
    '''Compute the Multipole-to-Multipole kernel. Passing multipole expansion of child to parent.'''
    shape = harmonic.n_arr.shape
    res = np.zeros(shape).astype(complex)

    # only compute if node contains particles
    if not child.multipoleCenter[0] is None:
        # precomputation
        dist = child.multipoleCenter[0] - parent.multipoleCenter[0]
        harm = harmonic.ypsilon(dist)
        mult = child.multipoleExpansion
        expOrder = shape[0]
        # funtion to compute a single index
        # TODO: vectorization is hard... smh replace loops
        def computeMnm(n, m):
            res = 0.
            for k in np.arange(n+1):
                for l in np.arange(-k, k+1):
                    n_k = n - k
                    m_l = m - l
                    # filter out orders out of bounds
                    if np.abs(m_l) <= n_k:
                        res += harm[k,l] * mult[n_k,m_l]
            return res

        # compute multipole
        # m_ids = np.hstack((np.arange(expOrder), np.arange(-1* expOrder + 1, 0)))
        m_ids = np.arange(-expOrder+1, expOrder)
        n_ids = np.arange(expOrder)
        for n in n_ids:
            for m in m_ids:
                res[n,m] = computeMnm(n,m)

    return res

def m2l(A: Node, B: Node, harmonic: SphericalHarmonics, accelerated: bool = False):
    '''Computes the field tensor at B due to A.'''
    shape = harmonic.n_arr.shape
    res = np.zeros(shape).astype(complex)

    # precomputation
    dist = B.potentialCenter[0] - A.multipoleCenter[0]
    harm = harmonic.theta(dist)
    mult = A.multipoleExpansion
    expOrder = shape[0]
    # funtion to compute a single index
    # TODO: vectorization is hard... smh replace loops
    def computeFnm(n, m):
        res = 0.
        for k in np.arange(expOrder-n):
            for l in np.arange(-k, k+1):
                n_k = n + k
                m_l = m + l
                # filter out orders out of bounds
                if np.abs(m_l) <= n_k:
                    res += np.conjugate(mult[k,l]) * harm[n_k,m_l]
        return res
    def computeFnmAccelerated(n,m):
        # TODO: how to implement x and z swapping
        # compute rotation angles
        az = np.arctan(dist[1]/dist[0])
        ax = np.arctan(np.sqrt(dist[2]**2 + dist[1]**2)/dist[0]) 

        # perform multipole computation
        res = 0.
        for k in np.arange(np.abs(m), expOrder-n):
            res += (-1.)**m * np.exp(-1j * m * ax) * np.exp(-1j*m*az) * mult[k,m] * (factorial(n+k)/np.linalg.norm(dist)**(n+k+1))
        return np.exp(1j*m*az) * np.exp(1j*m*ax) * res
 

    # compute fieldtensor 
    # m_ids = np.hstack((np.arange(expOrder), np.arange(-1* expOrder + 1, 0)))
    m_ids = np.arange(-expOrder+1, expOrder)
    n_ids = np.arange(expOrder)
    for n in n_ids:
        for m in m_ids:
            res[n,m] = computeFnm(n,m) if not accelerated else computeFnmAccelerated(n,m)

    return res

def l2l(child: Node, potentialCenter, fieldtensor, harmonic:SphericalHarmonics):
    '''Compute the fieldtensor w.r.t. another expansion center. Passes fieldtensor down to child.'''
    shape = harmonic.n_arr.shape
    res = np.zeros(shape).astype(complex)

    # precomputation
    dist = potentialCenter[0] - child.potentialCenter[0]
    harm = harmonic.ypsilon(dist)
    field= fieldtensor
    expOrder = shape[0]
    # funtion to compute a single index
    # TODO: vectorization is hard... smh replace loops
    def computeFnm(n, m):
        res = 0.
        for k in np.arange(expOrder-n):
            for l in np.arange(-k, k+1):
                n_k = n + k
                m_l = m + l
                # filter out orders out of bounds
                if np.abs(m_l) <= n_k:
                    res += np.conjugate(harm[k,l]) * field[n_k,m_l]
        return res

    # compute fieldtensor
    # m_ids = np.hstack((np.arange(expOrder), np.arange(-1* expOrder + 1, 0)))
    m_ids = np.arange(-expOrder+1, expOrder)
    n_ids = np.arange(expOrder)
    for n in n_ids:
        for m in m_ids:
            res[n,m] = computeFnm(n,m)

    return res

def l2p(leaf: Node, sink: np.ndarray, fieldtensor, harmonic:SphericalHarmonics):
    '''Computes the the potential field at sinkposition, due to all other cells''' 
    expOrder = harmonic.n_arr.shape[0] 
    #check if p was set to 0 , else max order 1 is needed 
    expOrder = expOrder if expOrder <= 1  else 2
    # no negative orders needed for acceleration
    res = np.zeros((expOrder,expOrder)).astype(complex)

    # precomputation
    dist = leaf.potentialCenter[0] - sink
    harm = harmonic.ypsilon(dist)
    field= fieldtensor
    # funtion to compute a single index
    # TODO: vectorization is hard... smh replace loops
    def computePsi(n, m):
        res = 0.
        for k in np.arange(expOrder-n):
            for l in np.arange(-k, k+1):
                n_k = n + k
                m_l = m + l
                # filter out orders out of bounds
                if np.abs(m_l) <= n_k:
                    res += np.conjugate(harm[k,l]) * field[n_k,m_l]
        return res

    # compute fieldtensor
    # m_ids = np.hstack((np.arange(expOrder), np.arange(-1* expOrder + 1, 0)))
    m_ids = np.arange(-expOrder+1, expOrder)
    n_ids = np.arange(expOrder)
    for n in n_ids:
        for m in m_ids:
            res[n,m] = computePsi(n,m)

    return res
