import numpy as np
from numba import  jit, njit, vectorize, prange

def Fermi_CC_full(L):

	ll  = int(L)

	DIM_H = 2**L

	Base_BIN  = [TO_con(xx,L) for xx in range(DIM_H)]
	Base_FOCK = BaseNumRes_creation_FERMI(L, Base_BIN)

	CDC 		= Fermi_Cc  (Base_FOCK) # + - 
	CCDCC 		= Fermi_CCcc(Base_FOCK) # + + - - 
	CCDCC_NO 	= Fermi_CcCc(Base_FOCK) # + + - - 
	
	return CDC, CCDCC, CCDCC_NO, Base_FOCK


#..................................functions to generate the operators

#..................................from bin number to configuration
def TO_con(x,L):

	x1=np.int64(x)
	L1=np.int64(L)
	
	return np.binary_repr(x1, width=L1)

def BaseNumRes_creation_FERMI(LL,B):

	Dim = len(B)
	A   = np.zeros((Dim,LL), dtype=np.int)

	for i in range(Dim):

		k=0
		for j in list(B[i]):
			A[i,k] = int(j)
			k+=1

	return A





def Fermi_Cc(Base_FOCK):

	DIM_H 	= np.shape(Base_FOCK)[0]
	ll 		= np.shape(Base_FOCK)[1]

	CDC = np.zeros((ll,ll,DIM_H,DIM_H))

	for n_psi in range(DIM_H):
		psi = Base_FOCK[n_psi]	
		nn  = np.sum(psi)

		for i in range(ll):
			for j in range(ll):

				n_phi, fact = ff_2(psi,i,j,ll)

				if n_phi < 0:
					continue

				CDC[i,j,n_phi,n_psi] = fact
		
	return CDC	

@njit
def ff_2(psi,i,j,ll):

	if psi[j]-1 < 0:
		return -1, -1

	vec_j 	 = np.zeros(ll)
	vec_j[j] = -1
	phi_j 	 = psi+vec_j
	
	n_j 	 = np.sum(psi[:j])
	fact_j 	 = (-1)**(n_j)

	if phi_j[i]+1 > 1:
		return -1, -1

	vec_i 	 = np.zeros(ll)
	vec_i[i] = +1
	phi_i 	 = phi_j+vec_i

	n_i 	 = np.sum(phi_j[:i])
	fact_i 	 = (-1)**(n_i)
	
	n_phi 	 = np.int(np.sum(np.array([ phi_i[x]*2**(ll-1-x) for x in range(ll)])))
	
	fact 	 = fact_i*fact_j

	return n_phi, fact


def Fermi_CCcc(Base_FOCK):

	DIM_H 	= np.shape(Base_FOCK)[0]
	ll 		= np.shape(Base_FOCK)[1]	

	CDC 	= np.zeros((ll,ll,ll,ll,DIM_H,DIM_H))#, dtype=float)


	for n_psi in range(DIM_H):
		psi = Base_FOCK[n_psi]	
		nn  = np.sum(psi)

		for i in range(ll):
			for j in range(ll):
				for k in range(ll):
					for l in range(ll):	

						n_phi, fact = ff_4(psi,i,j,k,l,ll)

						if n_phi < 0:
							continue

						CDC[i,j,k,l,n_phi,n_psi] = fact

	return CDC	



@njit
def ff_4(psi,i,j,k,l,ll):

	if psi[l]-1 < 0:
		return -1, -1

	vec_l 	 = np.zeros(ll)
	vec_l[l] = -1
	phi_l 	 = psi+vec_l
	
	n_l 	 = np.sum(psi[:l])
	fact_l 	 = (-1)**(n_l)

	if phi_l[k]-1 < 0:
		return -1, -1

	vec_k 	 = np.zeros(ll)
	vec_k[k] = -1
	phi_k 	 = phi_l+vec_k
	
	n_k 	 = np.sum(phi_l[:k])
	fact_k 	 = (-1)**(n_k)

	if phi_k[j]+1 > 1:
		return -1, -1

	vec_j 	 = np.zeros(ll)
	vec_j[j] = +1
	phi_j 	 = phi_k+vec_j

	n_j      = np.sum(phi_k[:j])
	fact_j 	 = (-1)**(n_j)
	
	if phi_j[i]+1 > 1:
		return -1, -1

	vec_i 	 = np.zeros(ll)
	vec_i[i] = +1
	phi_i 	 = phi_j+vec_i

	n_i      = np.sum(phi_j[:i])
	fact_i 	 = (-1)**(n_i)

	n_phi 	 = np.int(np.sum(np.array([ phi_i[x]*2**(ll-1-x) for x in range(ll)])))

	fact = fact_i*fact_j*fact_k*fact_l

	return n_phi, fact



def Fermi_CcCc(Base_FOCK):

	DIM_H 	= np.shape(Base_FOCK)[0]
	ll 		= np.shape(Base_FOCK)[1]	

	CDC 	= np.zeros((ll,ll,ll,ll,DIM_H,DIM_H))#, dtype=float)


	for n_psi in range(DIM_H):
		psi = Base_FOCK[n_psi]	
		nn  = np.sum(psi)

		for i in range(ll):
			for j in range(ll):
				for k in range(ll):
					for l in range(ll):	

						n_phi, fact = ff_4_bis(psi,i,j,k,l,ll)

						if n_phi < 0:
							continue

						CDC[i,j,k,l,n_phi,n_psi] = fact

	return CDC	



@njit
def ff_4_bis(psi,i,j,k,l,ll):

	if psi[l]-1 < 0:
		return -1, -1
	vec_l 	 = np.zeros(ll)
	vec_l[l] = -1
	phi_l 	 = psi+vec_l
	
	if phi_l[k]+1 > 1:
		return -1, -1
	vec_k 	 = np.zeros(ll)
	vec_k[k] = +1
	phi_k 	 = phi_l+vec_k

	if phi_k[j]-1 < 0:
		return -1, -1
	vec_j 	 = np.zeros(ll)
	vec_j[j] = -1
	phi_j 	 = phi_k+vec_j

	if phi_j[i]+1 > 1:
		return -1, -1
	vec_i 	 = np.zeros(ll)
	vec_i[i] = +1
	phi_i 	 = phi_j+vec_i

	n_l 	 = np.sum(psi[:l])
	fact_l 	 = (-1)**(n_l)
	
	n_k 	 = np.sum(phi_l[:k])
	fact_k 	 = (-1)**(n_k)

	n_j      = np.sum(phi_k[:j])
	fact_j 	 = (-1)**(n_j)

	n_i      = np.sum(phi_j[:i])
	fact_i 	 = (-1)**(n_i)

	n_phi 	 = np.int(np.sum(np.array([ phi_i[x]*2**(ll-1-x) for x in range(ll)])))

	fact = fact_i*fact_j*fact_k*fact_l

	return n_phi, fact
















































































