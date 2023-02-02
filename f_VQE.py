import numpy        as  np
import scipy        as  sp
from 	numpy	   import 	linalg

def rand_mat(n_l):
	
	# here we generate a n_l x n_l random matrix form the Circular unitary ensamble
	# algorithm is taken from https://www.ams.org/notices/200705/fea-mezzadri-web.pdf
	
	z   = (np.random.randn(n_l,n_l) + 1j*np.random.randn(n_l,n_l))
	z 	*= 1/np.sqrt(2.0)
	q,r = linalg.qr(z)
	d   = np.diag(r)
	ph  	= d/np.absolute(d)
	mat_cue = np.multiply(q,ph,q)
	
	return mat_cue

def U_sym(ll,Op2):

	O = np.random.randn(ll,ll)
	O = (O + O.T)/2

	KK	= Op2[0][0]*0
	
	for i in range(ll):
		for j in range(ll):
			KK += Op2[i][j]*O[i,j]

	U    = sp.linalg.expm(-1j*KK)

	return U


def U_1b(ll,Op2):

	O = 1j*sp.linalg.logm(rand_mat(ll))
	KK	= Op2[0][0]*0
	
	for i in range(ll):
		for j in range(ll):
			KK += Op2[i][j]*O[i,j]

	U    = sp.linalg.expm(-1j*KK)

	return U

def U_single(Op2, Op4, par, ll):

	pot  = par[:int(ll)]

	K_op = (Op2[0][1] + Op2[1][0])
	for i in range(1,ll-1):
		K_op += (Op2[i][i+1] + Op2[i+1][i])

	V_op = Op4[0][0][1][1]
	for i in range(1,ll-1):
		V_op += Op4[i][i][i+1][i+1]

	N_vec 	= [ Op2[i][i] for i in range(ll) ]
	N_op 	= N_vec[0]*pot[0]		
	for i in range(1,ll):
		N_op 	+= N_vec[i]*pot[i]

	V0     = np.abs(par[-2])	
	DtK    = np.abs(par[-1])

	HAMK    = -K_op + N_op + V_op*V0
	U1     = sp.linalg.expm(-1j*HAMK*DtK)

	return U1


def U_par(Op2, Op4, par_0, ll):
	
	par_0 = np.array(par_0)

	depth = int(par_0.shape[0]/(int(ll/1)+2))
	par   = np.reshape(par_0, (depth,(int(ll/1)+2)))

	U     = U_single(Op2, Op4, par[0], ll)
	
	for x in range(1,depth):
		U_x  = U_single(Op2, Op4, par[x], ll)
		U    = U_x.dot(U)

	return U


def	cost_f(par_0, psi_0, Op2, Op4, HAM, H2, H4, E_0s, KK0, VV0, state, ll):

	U       = U_par(Op2, Op4, par_0, ll)

	psi_U   = U.dot(psi_0)
	psi_U_h = np.conjugate(psi_U)

	fid 	= np.abs(np.dot(state,psi_U))**2

	KK    =	psi_U_h.dot(H2).dot(psi_U)[0,0].real
	VV 	  = psi_U_h.dot(H4).dot(psi_U)[0,0].real
	E_par = psi_U_h.dot(HAM).dot(psi_U)[0,0].real

	print('1/fid - E_vqe - E_exact', 1/fid, E_par, E_0s)

	return E_par


def EV_GS(HAM):
	
	A,B = sp.linalg.eigh(HAM)

	idx = A.argsort()[::1]   

	E = A[idx]
	V = B[:,idx]

	return E[0],V[0]

def EV(HAM):
	
	A,B = sp.linalg.eigh(HAM)

	idx = A.argsort()[::1]   

	E = A[idx]
	V = B[:,idx]

	return E,V


def H_24(Op2, Op4, h2, h4):

	ll = h2.shape[0]

	H2 = Op2[0][0]*0

	for i1 in range(ll):
		for i2 in range(ll):
			if np.abs(h2[i1,i2])<10**-20:
				continue
			H2 += Op2[i1][i2]*h2[i1,i2]

	H4 = Op4[0][0][0][0]*0

	for i1 in range(ll):
		for i2 in range(ll):
			for i3 in range(ll):
				for i4 in range(ll):
					if np.abs(h4[i1,i2,i3,i4])<10**-20:
						continue
					H4 += Op4[i1][i2][i3][i4]*h4[i1,i2,i3,i4]

	return H2.todense(), H4.todense()








