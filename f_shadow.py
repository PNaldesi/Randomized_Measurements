import 	numpy  as np
from numba import  jit, njit, vectorize, prange
from opt_einsum import contract

import f_unitary as uu


def shadow(C2, C4, lbig, n_U, n_M):

	ll    	= C2.shape[0]

	U 		= uu.big_cue(ll,lbig,n_U)
	U_h		= np.conjugate(U)

	nk 		= contract('Usl,Usm,lm -> Us', U_h, U, C2,  optimize=True).real
	nknk    = contract('Usl,Usm,Upx,Upy,lmxy -> Usp', U_h, U, U_h, U, C4,  optimize=True).real

	if n_M > 0:
		nk0, nknk0 = get_nnn(n_U, n_M, lbig, nk, nknk)
	else:
		nk0 	= nk
		nknk0	= nknk

	mas_2   = mask_1(lbig)
	mas_4   = mask_2(lbig)
	Id_2    = np.identity(lbig)

	mat2 	= contract('Up, pt,  Uti,Utj         -> Uij'  , nk0,   mas_2, U, U_h,         optimize='auto-hq')
	mat4 	= contract('Upq,pqrs,Uri,Urj,Usk,Usl -> Uijkl', nknk0, mas_4, U, U_h, U, U_h, optimize='auto-hq')

	return	 mat2, mat4


def shad_en(C2, C4, lbig, n_U, n_M, H2, H4):
	
	ll    	= C2.shape[0]
	mat2,mat4 = shadow(C2, C4, lbig, n_U, n_M)

	for X in range(n_U):	
		for i in range(ll):
			mat2[X,i,i] = C2[i,i]
			
			for j in range(ll):
				mat4[X,i,i,j,j] = C4[i,i,j,j]

	ene     = np.zeros((n_U,2))

	for x_r in range(n_U):

		E_2   = cm.RECO_chem_K(H2, mat2[x_r]) 
		E_4   = cm.RECO_chem_V(H4, mat4[x_r])

		ene[x_r] = [E_2, E_4]

	return	 ene



# here we simulate finite number of measurements

@njit(parallel=True)
def get_nnn(n_U, n_M, lbig, nk, nknk):
	
	nk0 	= np.zeros((n_U,lbig))
	nknk0 	= np.zeros((n_U,lbig,lbig))

	for g in prange(n_U):

		prob     = nk[g]
		nk0[g]   = get_prob(prob, n_M)

		prob   	 = nknk[g].flatten()
		nknk0[g] = np.reshape(get_prob(prob, n_M),(lbig,lbig))

	return nk0, nknk0

@njit(parallel=True)
def get_prob(nk0,n_M):

	nk   = np.real(nk0)
	n_s  = len(nk)

	arr  = np.arange(n_s)
	prob = nk/np.sum(nk)
	n_p  = np.int(round(np.sum(nk)))

	nk0   = np.zeros(n_s)

	for x in range(n_M):

		dat    = rand_choice_nb(arr, prob, n_p)
		nk0   += dat/n_M

	return nk0

@njit(parallel=True)
def rand_choice_nb(arr, prob, n_p):

	"""
	:param arr: 	A 1D numpy array of values to sample from.
	:param prob: 	A 1D numpy array of probabilities for the given samples.
	:return: 		A random sample from the given array with a given probability.
	"""

	res   = np.zeros(n_p)-1
	state = np.zeros(len(arr))
	j   = 0

	while (j<n_p):
		pp = arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
		
		if pp in res:
			continue

		res[j]    = pp
		state[pp] = 1
		j += 1

	if np.abs(np.sum(state)-n_p)>0.1:
		print('achtung error', np.sum(state), n_p)

	return state




# here we calculate matrices of coefficients needed in the reconstructions

@njit
def mask_1(ll):

	L0 = range(ll)

	mask = np.array([[ -1 for i in L0] for j in L0 ])
	for i in L0:
		mask[i,i] = ll

	return mask

@njit
def mask_2(ll):

	L0  = range(ll)	
	a   = np.array([[[[	el_av(i,j,p,q,ll)	for q in L0] for p in L0] for j in L0] for i in L0])*(ll)**4

	return a

@njit
def el_av(s1,s2,s3,s4,ll):

	d1 = +1/(ll*(ll-1))
	d2 = -1/(ll*(ll-1)*(ll-2))
	d3 = +1/(ll*(ll-1)*(ll-2)*(ll-3))


	if (s1-s2)*(s1-s4)*(s3-s2)*(s3-s4)==0: 
		a = 0

	elif (s1!=s3) and (s2!=s4):
		a = d3

	elif (s1==s3) and (s2!=s4):
		a = d2

	elif (s1!=s3) ==  (s2==s4):
		a = d2

	elif (s1==s3) and (s2==s4):
		a = d1		

	return a










