import numpy as np

from openfermion.ops.operators import FermionOperator
from openfermion.linalg        import get_sparse_operator, get_number_preserving_sparse_operator, get_ground_state

def Fermi_op(ll,nn):

	CDC 	= Fe_op_Cc  (ll,nn) # + - 
	CCDCC 	= Fe_op_CcCc(ll,nn) # + - + - 
	
	return CDC, CCDCC

def Fe_op_Cc(ll,nn):

	L0  = range(ll)
	h2  = [[ Op2(i,j,ll,nn) for i in L0] for j in L0]

	return h2


def Fe_op_CcCc(ll,nn):

	L0  = range(ll)
	h4c = [[[[ Op4(i,j,k,l,ll,nn) for i in L0] for j in L0] for k in L0] for l in L0]

	return h4c


def Op2(i,j,ll,nn):
	
	OP = FermionOperator(str(i)+'^ '+str(j),   1.)
	sp_OP = get_number_preserving_sparse_operator(OP, num_qubits=ll, num_electrons=nn)

	return sp_OP


def Op4(i,j,k,l,ll,nn):
	
	OP = FermionOperator(str(i)+'^ '+str(j)+' '+str(k)+'^ '+str(l),   1.)
	sp_OP = get_number_preserving_sparse_operator(OP, num_qubits=ll, num_electrons=nn)
	
	return sp_OP


def get_N(fermion_hamiltonian,ll):
	
	sparse_hamiltonian  = get_sparse_operator(fermion_hamiltonian, n_qubits=ll)
	E_0, psi            = get_ground_state(sparse_hamiltonian)
	psih        	    = np.conjugate(psi) 

	N_op = FermionOperator.zero()
	for x in range(0,ll):
		N_op  += FermionOperator(str(x)+'^ '+str(x), 1.)
	Nop = get_sparse_operator(N_op, n_qubits=ll)

	psi_N  = Nop.dot(psi)
	nn     = np.dot(psih,psi_N)
	nn     = np.rint(nn.real)

	return int(nn)
