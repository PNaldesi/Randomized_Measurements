import numpy        as  np
import scipy        as  sp
from numba import  jit, njit, vectorize, prange


def U_single(J, V, Op2, Op4_com, par):

	ll 	  = Op2.shape[0]
	K_op 	= Op2[0,1] + Op2[1,0]
	for i in range(1,ll-2):
		K_op += Op2[i,i+1] + Op2[i+1,i]


	V_op 	= Op4_com[0,0,1,1]
	for i in range(1,ll-2):
		V_op += Op4_com[i,i,i+1,i+1]


	N_vec 	= [ Op2[i,i] for i in range(ll) ]
	N_op 	= N_vec[0]*par[0]		
	for i in range(1,ll-1):
		N_op 	= N_vec[i]*par[i]

	HAM = J*K_op + V*V_op + N_op

	U 	= sp.linalg.expm(-1j*HAM)

	return U 


def U_par(J, Op2, V, Op4_com, par_0 ):

	ll 	  = Op2.shape[0]
	
	depth = int(par_0.shape[0]/(ll))
	par   = np.reshape(par_0, (depth,ll))

	U     = U_single(J, V, Op2, Op4_com, par[0])

	for x in range(1,depth):

		U_x  = U_single(J, V, Op2, Op4_com, par[x])
		U    = np.einsum( 'xy, yj -> xj', U_x, U, optimize=True)	

	return U


def	cost_f(par_0, psi_0, J, V, Op2, Op4_com, HAM, Econst, E_0s, state):

	U       = U_par(J, Op2, V, Op4_com, par_0)
	
	psi_U 	= np.einsum( 'xy,y -> x', U, psi_0, optimize=True)
	psi_U_h = np.conjugate(psi_U)

	fid 	= np.abs(np.dot(state,psi_U))**2

	E_par 	= np.einsum( 'x,xy,y',  psi_U_h, HAM, psi_U, optimize=True).real

	print('E', E_par, E_0s, fid)

	return E_par