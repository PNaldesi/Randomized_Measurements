import numpy        as  np
import os

import scipy.optimize as opt

import f_fermi        as  fm
import f_molchem      as  cm
import f_VQE		  as  vqe_f


from openfermion.chem       import MolecularData
from openfermion.transforms import get_fermion_operator
from openfermion.linalg     import get_ground_state, get_number_preserving_sparse_operator
from openfermionpyscf       import run_pyscf


np.set_printoptions(precision=5,suppress=True)
np.set_printoptions(linewidth=np.inf)


LOCAL   = os.path.abspath('.')

# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1

# Set calculation parameters.
run_scf  = 1
run_mp2  = 1
run_cisd = 0
run_ccsd = 0
run_fci  = 1
delete_input = True
delete_output = True


bond_length = 1.5

geometry = [('H', (0., 0., -bond_length)), ('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 2*bond_length))]
molecule = MolecularData(geometry, basis, multiplicity, description=str(round(bond_length, 2)))

# Run pyscf.
molecule = run_pyscf(molecule,
					 run_scf=run_scf,
					 run_mp2=run_mp2,
					 run_cisd=run_cisd,
					 run_ccsd=run_ccsd,
					 run_fci=run_fci)

molecular_hamiltonian = molecule.get_molecular_hamiltonian()
E_0, h2, h4 	= cm.ham_coef(molecular_hamiltonian)
ll = int(h2.shape[0])
L0 = range(ll)

F_Ham = get_fermion_operator(molecular_hamiltonian)
F_Ham.compress()

nn    = fm.get_N(F_Ham,ll)

Op2, Op4 	= fm.Fermi_op(ll,nn)

sp_ham      = get_number_preserving_sparse_operator(F_Ham, num_qubits=ll, num_electrons=nn)
E_GS_old, psiGS 	= get_ground_state(sp_ham)
psiGS_h  	= np.conjugate(psiGS)

H2, H4      = vqe_f.H_24(Op2, Op4, h2, h4)

KK0 = np.einsum('i,ij,j', psiGS_h, H2, psiGS)
VV0 = np.einsum('i,ij,j', psiGS_h, H4, psiGS)

# choosing an initial state
psi_0 	 = np.zeros(psiGS.shape[0])
psi_0[37]= 1

psi_0_h = np.conjugate(psi_0)	
ni = [ np.einsum('i,ij,j', psi_0_h, Op2[x][x].todense(), psi_0).real for x in range(ll)]

depth 	 = 2

UU       = [vqe_f.U_sym(ll,Op2) for x in range(depth)]

par_0	 = np.random.uniform(low=-1, high=1, size=(depth*(int(ll/1)+2)))
R 		 = opt.minimize( vqe_f.cost_f, par_0, args=(psi_0, Op2, Op4, sp_ham, H2, H4, E_0, KK0, VV0, psiGS, ll), method='COBYLA', options={'maxiter':5*10**6}, tol = 1e-20)

quit()




