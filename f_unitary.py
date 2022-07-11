### Creation of unitary transformations ###
from numba import  jit, njit, vectorize, prange
import 	numpy as 		np
from 	numpy import 	linalg

def is_unitary(MM):

	m = np.matrix(MM)
	a = np.allclose(np.eye(m.shape[0]), m.H * m)

	return a

# matrix from GINIBRE ENSEMBLE
@njit
def gin(n):

	q   = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2*n)
	
	return q

# matrix from CUE ENSEMBLE
@njit
def cue(n):

	z   = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
	q,r = linalg.qr(z)
	d   = np.diag(r)
	ph  = d/np.absolute(d)
	q   = np.multiply(q,ph,q)
	
	return q

# rectangular reduction
@njit
def mat_rect(q,m,a):

	n   = int(a)
	n2  = int((n-m)/4)
	mat = q[:,n2:n2+m]

	return mat

# collection of n_U rectangular CUE MATRIX
def big_cue(ll,lbig,n_U):
	
	U = np.zeros((n_U,lbig,ll), dtype=complex)
	
	for x in range(n_U):
		U[x] = mat_rect(cue(lbig),ll,lbig)

	return U

# generation of CUE matrix using HURWITH decomposition
@njit
def cue_HUR(L):

	mul = E_N(1,L)
	for N in range(2,L,1):
		mul = np.dot(mul,E_N(N,L)) 
	
	return mul

@njit
def E_N(N,L):

	mul = U_j(0,L)

	for j in range(1,N,1):
		mul = np.dot(U_j(j,L),mul) 	

	return mul

@njit
def U_j(j,L):
	
	mat = np.zeros((L,L), dtype=np.complex128)
	
	for x in range(L):
		mat[x,x] = 1

	UU = U_mat_2x2(j+1)

	mat[j,j]     = UU[0,0]
	mat[j,j+1]   = UU[0,1]
	mat[j+1,j]   = UU[1,0]
	mat[j+1,j+1] = UU[1,1]

	return mat 

@njit
def U_mat_2x2(x):

	psi    = 1.0j*np.random.rand()*2*np.pi
	alpha  = 0
	phi    = np.arcsin(np.random.rand()**(1/(2*x-1)))

	if x==1:
		alpha  = 1.0j*np.random.rand()*2*np.pi

	a =  np.exp( alpha)*np.cos(phi)
	b =  np.exp( psi  )*np.sin(phi)
	c = -np.exp(-psi  )*np.sin(phi)
	d =  np.exp(-alpha)*np.cos(phi)

	mat = np.array([[a, b], [c, d]], dtype=np.complex128)

	return mat 

# collection of n_U rectangular CUE MATRIX
def big_cue_HUR(ll,lbig,n_U):
	
	U = np.zeros((n_U,lbig,ll), dtype=complex)
	
	for x in range(n_U):
		U[x] = mat_rect(cue_HUR(ll),ll,lbig)

	return U


