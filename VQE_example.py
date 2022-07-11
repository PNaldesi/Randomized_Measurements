import numpy        as  np
import os
import scipy.optimize as opt

import f_VQE as vqe
import f_shadow as sh
import f_fermi        as  fm
import f_molchem     as  cm

from openfermion.ops.operators import FermionOperator
from openfermion.transforms import get_fermion_operator
from openfermion.chem       import MolecularData
from openfermion.linalg     import get_ground_state, get_sparse_operator
from openfermionpyscf       import run_pyscf


np.set_printoptions(precision=6,suppress=True)
np.set_printoptions(linewidth=np.inf)

# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1
n_points = 1
bond_length_interval = 3.5 / n_points

# Set calculation parameters.
run_scf  = 1
run_mp2  = 1
run_cisd = 0
run_ccsd = 0
run_fci  = 1
delete_input = True
delete_output = True

# Generate molecule at different bond lengths.
hf_energies = []
fci_energies = []
fermion_energies = []
bond_lengths = []
for point in range(0, n_points):
	bond_length = bond_length_interval * float(point) + 1.5
	bond_lengths += [bond_length]
	geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., bond_length))]
	molecule = MolecularData(
		geometry, basis, multiplicity,
		description=str(round(bond_length, 2)))

	# Run pyscf.
	molecule = run_pyscf(molecule,
						 run_scf=run_scf,
						 run_mp2=run_mp2,
						 run_cisd=run_cisd,
						 run_ccsd=run_ccsd,
						 run_fci=run_fci)
	active_space_start = 1
	active_space_stop  = 3


	# Get the Hamiltonian in an active space.
	molecular_hamiltonian = molecule.get_molecular_hamiltonian(
		occupied_indices=range(active_space_start),
		active_indices=range(active_space_start, active_space_stop))
	
	fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
	fermion_hamiltonian.compress()
	sparse_hamiltonian  = get_sparse_operator(fermion_hamiltonian)
	energy, state       = get_ground_state(sparse_hamiltonian)

###### hamiltonian creation and check

	E0, H2, H4 	= cm.ham_coef(molecular_hamiltonian)
	Econst, H2, H4 	= cm.ham_coef(molecular_hamiltonian)
	
	ll = int(H2.shape[0])
	L0 = range(ll)

	Op2, Op4, Op4_com, Base_FOCK  	= fm.Fermi_CC_full(ll)


####### hamiltonian check

	def Op2_OF(i,j,ll):
		OP = FermionOperator(str(i)+'^ '+str(j),   1.)
		OP = get_sparse_operator( OP, n_qubits=ll ).todense()
		return OP

	def Op4_OF(i,j,k,l,ll):
		OP = FermionOperator(str(i)+'^ '+str(j)+'^ '+str(k)+' '+str(l),   1.)
		OP = get_sparse_operator( OP, n_qubits=ll ).todense()
		return OP

	h2  = np.array([[   np.sum(np.abs(Op2[i,j]-Op2_OF(i,j,ll))) for i in L0] for j in L0])
	h4  = np.array([[[[ np.sum(np.abs(Op4[i,j,k,l]-Op4_OF(i,j,k,l,ll))) for i in L0] for j in L0] for k in L0] for l in L0])

	if np.sum(np.abs(h2))>10**-6:
		print('wrong OPerators')
		quit()

	if np.sum(np.abs(h4))>10**-6:
		print('wrong OPerators')
		quit()

#####     ..........        EVERYTHING IS IN POSITION SPACE NOW     ..........        #####
	
##### GS energy, 1st state, kinetic, potential, psi, ham 
	E_0s,  E_1s,  Eex_0,  Eex_2,  Eex_4,  psi,  HAM 	= cm.chem_HAM(E0, H2,  H4,  Op2, Op4_com)
	psih        = np.conjugate(psi)

	cor_2   	= np.einsum( 'x,ijxy  ,y->ij',  psih, Op2,     psi, optimize=True)
	nn 			= np.trace(cor_2)
	
	cor_4   	= np.einsum( 'x,ijklxy,y->ijkl',psih, Op4_com, psi, optimize=True)
	cor_4_TT 	= np.einsum( 'x,ijklxy,y->ijkl',psih, Op4,     psi, optimize=True)


	# CHECKS

	if np.abs(energy-E_0s)>10**-6:
		print('wrong Hamiltonian')
		quit()

	if np.abs(E0-Eex_0)>10**-6:
		print('wrong Hamiltonian')
		quit()

	if np.abs(energy-(Eex_0+Eex_2+Eex_4))>10**-6:
		print(energy, Eex_0+Eex_2+Eex_4,'wrong Hamiltonian')
		quit()

	a = np.sum(np.array([[ cor_4 [i,i,j,j] - cor_2[i,i]*cor_2[j,j] for i in L0] for j in L0] ))
	if np.abs(a)>10**-6:
		print(a, 'no part cons')
		quit()		


	psi_0 	 = np.zeros(2**ll)
	psi_0[10]= 1   
	
	J 		 = -1
	V 		 = 1	
	depth 	 = 2
	
	par_0	 = np.random.uniform(low=-1.0, high=1.0, size=depth*ll)

	R 		 = opt.minimize( vqe.cost_f, par_0, args=(psi_0, J, V, Op2, Op4_com, HAM, Econst, E_0s,state), method='COBYLA', tol = 1e-8) #, options = {'disp':True})

quit()



