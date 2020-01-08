import sys
sys.path.insert(0,'/home/mohit/Downloads/code_notebooks/deep_boltzmann')
sys.path.insert(0,'/home/mohit/Downloads/code_notebooks/Scripts')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import keras
import tensorflow as tf
from QScore import NativeContacts, DistMatrix
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from deep_boltzmann.models import ParticleDimer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet, create_RealNVPNet
from deep_boltzmann.sampling import GaussianPriorMCMC
from deep_boltzmann.networks.plot import test_xz_projection, test_generate_x
from deep_boltzmann.util import count_transitions
from deep_boltzmann.sampling.analysis import free_energy_bootstrap, mean_finite, std_finite
from deep_boltzmann.networks.training import MLTrainer, FlexibleTrainer
from deep_boltzmann.util import save_obj, load_obj
from deep_boltzmann.sampling.analysis import free_energy_bootstrap, mean_finite, std_finite
from openmmtools.integrators import *
from deep_boltzmann.models.openmm import OpenMMEnergy
from deep_boltzmann import openmmutils
import simtk
from simtk import *
from simtk.openmm import *
from simtk.unit import *
from simtk.openmm.app import *
import mdtraj as md
'''
filename = '/home/mohit/JoinedLongTraj.xtc'
top = '/home/mohit/system.pdb'
nativeContacts = NativeContacts(filename,top)
disMatrix = DistMatrix(filename,top)
'''
import pickle
disMatrix = np.load("DistMatrix.npz.npy", allow_pickle=True)
nativeContacts = np.load('NativeContacts.npz.npy', allow_pickle=True)
top = '/home/mohit/system.pdb'
filename = '/home/mohit/FoldedTRPCage.xtc'
traj = md.load(filename,top = top)
import ast
#params = ast.literal_eval(str(trajdict['params']))
x = traj[:].xyz
topology = traj.topology

index = topology.select('backbone')
PDBObject = PDBFile(top)
cartesian = ['CA', 'C', 'N']
index = topology.select(' '.join(["name " + s for s in cartesian]))
CA_Index= [atom.index for atom in topology.atoms if atom.name == "CA" ]
IC_Index = []
for i in range(272):
    if(i not in index):
        IC_Index.append(i)

print(IC_Index)
from deep_boltzmann.models.proteins import mdtraj2Z
Z_ = np.array(mdtraj2Z(topology))
batchsize_ML =  256
batchsize_KL = 1000
modeller = Modeller(PDBObject.topology, PDBObject.positions)
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
system = forcefield.createSystem(modeller.topology, nonbondedMethod= CutoffNonPeriodic, nonbondedCutoff=1.0*nanometers, constraints= AllBonds)

# setup BPTI
INTEGRATOR_ARGS = (system, 300* kelvin,
                   1.0/ picoseconds,
                    0.002*picoseconds, 10, 5,5 )
EnergyModel = OpenMMEnergy(system,  NoseHooverChainVelocityVerletIntegrator , nanometers, n_atoms= 272, openmm_integrator_args=INTEGRATOR_ARGS )

new=[]
for snapshot in x:
    new.append(snapshot.flatten('F'))

contactMap = nativeContacts
NumberOfContacts =0
for row in contactMap:
    NumberOfContacts +=len(row)
def rcGc(inputSnapshot):
    return inputSnapshot[:,0] - inputSnapshot[:,0]
    global contactMap
    global disMatrix
    global NumberOfContacts
    global CA_Index
    import numpy as np
    from math import sqrt
    print("Separate")
    print(inputSnapshot)
    atoms = []
    for index in CA_Index:
        print(index,inputSnapshot[3*index : (3*index)+ 3])
        atoms.append(3*index)
        atoms.append(3*index+1)
        atoms.append(3*index+2)
        #atoms= atoms + inputSnapshot[3*index : (3*index)+ 3]
    atoms = np.array(atoms)
    print(atoms )
    Q_score =0
    for atom_ind in range(int(len(atoms)/3)):
        for second_atom in range(atom_ind+7, int(len(atoms)/3) ):
            if(second_atom in contactMap[atom_ind]):
                #print(atom_ind,second_atom,contactMap[atom_ind])
                print(second_atom)
                print(atoms[second_atom*3: second_atom*3+3])
                dif = atoms[atom_ind*3:atom_ind*3+3]- atoms[second_atom*3: second_atom*3+3]
                print(atoms[atom_ind*3:atom_ind*3+3])
                dist = dif[0]*dif[0] + dif[1]* dif[1] + dif[2] * dif[2]
                dist = sqrt(dist)
                #dist = np.linalg.norm(atoms[atom_ind]- atoms[second_atom])
                Q_score += 1/(1+np.exp( (dist- (disMatrix[atom_ind][second_atom]+0.1) )*100 ) )

    Q_score /= NumberOfContacts
    return  tf.Variable(Q_score)

#print(rc_coordinates(new[0]))
bg = invnet(3*272, 'RRRRRRRR', energy_model=EnergyModel, nl_layers=4, nl_hidden=20, #100
            nl_activation='relu' )
hist_bg_ML = bg.train_ML(np.array(new), xval = np.array(new) , epochs=10, lr=0.0001, batch_size=batchsize_ML,
                         std=1.0, verbose=1, return_test_energies=True)

Eschedule = [[200,  0.00001, 1e6, 1e3,  0.0, 20.0],
             [100,  0.0001, 1e6,  300,  0.0, 20.0],
             [100,  0.0001, 1e5,  100,  0.0, 20.0],
             [100,  0.0001, 5e4,   50,  0.0, 20.0],
             [100,  0.0001, 5e4,   20,  0.0, 20.0],
             [200,  0.0001, 5e4,    5,  0.0, 20.0]]
temperature=1.0
hists_bg_KL = []
for i, s in enumerate(Eschedule):
    print(s)#'high_energy =', s[0], 'weight_ML =', s[1], 'epochs =', s[2])
    sys.stdout.flush()
    #hist_bg_KL = bg.train_KL(high_energy = 200)

    hist_bg_KL = bg.train_flexible(np.array(new) , epochs=s[0], lr=s[1], batch_size=batchsize_KL,
                                   verbose=1, high_energy=s[2], max_energy=1e10,
                                   weight_ML=s[3], weight_KL=1.0, temperature=temperature, weight_MC=0.0, weight_W2=s[4],
                                   weight_RCEnt=s[5], rc_func=rcGc, rc_min=0.0, rc_max=1.0,
                                   std=1.0, reg_Jxz=0.0, return_test_energies=True)
    hists_bg_KL.append(hist_bg_KL)
