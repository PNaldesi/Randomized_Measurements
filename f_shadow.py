import 	numpy  as np
from numba import  jit, njit, vectorize, prange
from opt_einsum import contract

import f_unitary as uu
import f_molchem as cm



def shadow(C2, C4, lbig, n_U, n_M, psi=0, Op2=0):

	ll    	= C2.shape[0]

	U 		= uu.big_cue(ll,lbig,n_U)
	U_h		= np.conjugate(U)

	nk 		= contract('Usl,Usm,lm -> Us', U_h, U, C2,  optimize=True).real
	nknk    = contract('Usl,Usm,Upx,Upy,lmxy -> Usp', U_h, U, U_h, U, C4,  optimize=True).real

	if n_M > 0:
		if psi==0:
			print('please pass Psi to the function shadow()')

		Dim_h  = len(psi)
		Fock   = np.array([ np.diag(Op2[x][x].todense()) for x in range(lbig)], dtype=np.int64).T
		OP2_ar = np.array([[Op2[i][j].todense() for i in range(lbig) ] for j in range(lbig)])
		
		Dict = {'psi'	: psi,
				'U'  	: U,
				'Dim_h'	: Dim_h,		        
		        'n_M'	: n_M, 
		        'n_U'	: n_U, 
		        'Ns' 	: lbig,		       
		        'Fock'  : Fock, 		       
		        'OP2_ar': OP2_ar,
		        }

		nk0   = ff.get_nnn()
		nknk0 = contract('Ui,Uj->Uij', nk0, nk0)

	else:
		nk0 	= nk
		nknk0	= nknk

	mas_2   = mask_1(lbig)
	mas_4   = mask_2(lbig)
	Id_2    = np.identity(lbig)

	mat2 	= contract('Up, pt,  Uti,Utj         -> Uij'  , nk0,   mas_2, U, U_h,         optimize='auto-hq')
	mat4 	= contract('Upq,pqrs,Uri,Urj,Usk,Usl -> Uijkl', nknk0, mas_4, U, U_h, U, U_h, optimize='auto-hq')

	return	 mat2, mat4


def shad_en(C2, C4, lbig, n_U, n_M, H2, H4, psi=0, Op2=0):
	
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



#### .... PROJECTIVE MEASUREMENTS

def get_n(**args):

    psi   = args.get('psi')
    U     = args.get('U')
    Dim_h = args.get('Dim_h')
    n_M   = args.get('n_M')
    n_U   = args.get('n_U')
    Ns    = args.get('Ns')
    Fock  = args.get('Fock')
    OP2_ar= args.get('OP2_ar')

    nk0     = np.zeros((n_U,Ns))

    for g in tqdm(range(n_U)):
        
        hpq     = 1.j*sp.linalg.logm(U[g])
        Hpq     = contract(' ij, ijxy -> xy', hpq, OP2_ar, optimize=True)
        
        Hpq_sp  = sparse.csr_matrix(-1j*Hpq)
        psiU    = exp_m(Hpq_sp,psi)
        
        nk0[g]  = get_measurements(psiU, Dim_h, n_M, Fock)

        nk0     = np.mean(nk0, axis=0)

    return nk0

def get_measurements(psi, Dim_h, n_M, Fock):

    prob  = np.abs(psi)**2
    ind_M = tuple([rand_choice_nb(np.arange(Dim_h), prob, n_M)])
    occ   = Fock[ind_M]
    
    if n_M == 1:
        occ = occ[0]
    
    return occ


@njit(parallel=True)
def rand_choice_nb(arr, prob, num_M):

    """
    :param arr:     A 1D numpy array of values to sample from.
    :param prob:    A 1D numpy array of probabilities for the given samples.
    :return:        A random sample from the given array with a given probability.
    """

    choice = []

    for i in prange(num_M):
        pp = arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
        choice.append(pp)
        
    return choice







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










