import numpy as np
import scipy        as sp
from numba import  jit, njit, vectorize, prange
from opt_einsum import contract


def normalize(v):

	norm = np.linalg.norm(v)
	
	if norm == 0: 
		print('zero norm')
		return v

	return v / norm

def ham_coef_uu(mol_ham):

	E0 		= 0
	h2      = []
	h4      = []
	
	for term in mol_ham:
	
		print(term)

		xx = np.asarray(term, dtype=int)
		a  = np.shape(xx)

		if   a[0] == 0:
			E0 = mol_ham[term]

		elif a[0] == 2:
			i = xx[0,0]
			j = xx[1,0]
			A = mol_ham[term]
			h2.append([i,j,A])

		elif a[0] == 4:
			i = xx[0,0]
			j = xx[1,0]
			k = xx[2,0]
			l = xx[3,0]		

			if i == j:
				continue
			if k == l:
				continue

			A = mol_ham[term]
			h4.append([i,j,k,l,A])

	ll = int(np.max(np.asarray(h2)[:,0]))+1
	
	H2 = np.zeros((ll,ll))
	H4 = np.zeros((ll,ll,ll,ll))

	for x in h2:
		i = int(x[0])
		j = int(x[1])
		A = x[2]
		H2[i,j] = A

	for x in h4:
		i = int(x[0])
		j = int(x[1])
		k = int(x[2])
		l = int(x[3])
		A = x[4]
		H4[i,j,k,l] = A

	return E0, H2, H4


def ham_coef(mol_ham):

	E0 		= 0
	h2      = []
	h4      = []
	
	for term in mol_ham:

		xx = np.asarray(term, dtype=int)
		a  = np.shape(xx)

		if   a[0] == 0:
			E0 = mol_ham[term]

		elif a[0] == 2:
			i = xx[0,0]
			j = xx[1,0]
			A = mol_ham[term]
			h2.append([i,j,A])

		elif a[0] == 4:
			i = xx[0,0]
			j = xx[1,0]
			k = xx[2,0]
			l = xx[3,0]		
			A = mol_ham[term]
			
			h4.append([i,k,j,l,-A])

			if  j==k:
				h2.append([i,l,A])

	ll = int(np.max(np.asarray(h2)[:,0]))+1
	
	H2 = np.zeros((ll,ll))
	H4 = np.zeros((ll,ll,ll,ll))

	for x in h2:
		i = int(x[0])
		j = int(x[1])
		A = x[2]
		H2[i,j] += A

	for x in h4:
		i = int(x[0])
		j = int(x[1])
		k = int(x[2])
		l = int(x[3])
		A = x[4]
		H4[i,j,k,l] += A

	return E0, H2, H4


def chem_HAM(E0, H2, H4, Op2, Op4):

	HAM_2   = chem_HAM_K(H2, Op2)
	HAM_4   = chem_HAM_V(H4, Op4)

	ll 		= np.shape(HAM_2)[0]
	HAM_0 	= np.identity(ll)*E0

	HAM 	= HAM_0 + HAM_2 + HAM_4 

	A,B = sp.linalg.eigh(HAM)

	idx = A.argsort()[::1]   

	E = A[idx]
	V = B[:,idx]

	E0 = E[0]
	E1 = E[1]
	
	V0  =  V.T[0]
	V0  =  normalize(V0)
	V0h =  np.conjugate(V0)

	E_0   = np.einsum( 'x,xy,y',  V0h, HAM_0, V0, optimize=True)
	E_2   = np.einsum( 'x,xy,y',  V0h, HAM_2, V0, optimize=True)
	E_4   = np.einsum( 'x,xy,y',  V0h, HAM_4, V0, optimize=True)
	
	return E0.real, E1.real, E_0.real, E_2.real, E_4.real, V0, HAM


def chem_HAM_K(H2, Op2):

	HAM_2   = contract( 'ij,ijxy->xy',  	 H2, Op2, optimize=True)
	
	return HAM_2

def chem_HAM_V(H4, Op4):

	HAM_4   = contract( 'ijkl,ijklxy->xy',  H4, Op4, optimize=True)

	return HAM_4


def RECO_chem_K(H2, M2):
	
	E  = contract( 'ij,ij', H2, M2, optimize=True)

	return E.real

def CORRECT_4_point(M2, M4, cor_4_TT):

	ll     = np.shape(M2)[0] 

	Id     = np.identity(ll)
	M2_co  = contract('il,kj -> ikjl ', M2, Id)

	M4_NO  = M2_co - M4

	for i in range(ll):
		for j in range(ll):
			for k in range(ll):
				for l in range(ll):
					if cor_4_TT[i,j,k,l] == 0:
						M4_NO[i,j,k,l] = 0

	return M4_NO

def RECO_chem_V(H4, M4):

	E  = contract( 'ijkl,ijkl', H4, M4, optimize=True)

	return E.real




