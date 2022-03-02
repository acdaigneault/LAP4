"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""


#----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        TPP2 Convection-Diffusion                           #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim               #
#----------------------------------------------------------------------------#

#%% NOTES D'UTILISATION
"""

Classe Main pour gérer les différentes classes

"""

#%% IMPORTATION DES LIBRAIRIES

import numpy as np
import pyvista as pv
import pyvistaqt as pvQt
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter
import solver as Solver
from case import Case
import matplotlib.pyplot as plt
import sympy as sp
from processing import Processing
import postProcessing as PostProcessing

#%% Données du problème
# Propriétés physiques
rho = 1 # kg/m³
Cp = 1  # J/kg-K
#k = 1   # W/m-K

# Longueur du domaine
L = 1     # m

x, y, k = sp.symbols('x y k')

# Champ de vitesse
u =  (2*x**2-x**4-1.0)*(y-y**3)
v = -(2*y**2-y**4-1.0)*(x-x**3)
f_U_MMS = sp.lambdify([x, y], u, "numpy")
f_V_MMS = sp.lambdify([x, y], v, "numpy")
def flow_velocity(x, y):
    return np.array([f_U_MMS(x, y), f_V_MMS(x, y)])

# Champ de température (MMS)
T0, Tx, Txy = 400, 45, 27.5
T_MMS = T0 + Tx*sp.cos(np.pi*x)+Txy*sp.sin(np.pi*x*y)
f_T_MMS = sp.lambdify([x, y], T_MMS, "numpy")

def MMS(x,y):
    return f_T_MMS(x, y)

# Terme source dérivé de la MMS
source = (rho*Cp*sp.diff(u*T_MMS, x, 1) +
          rho*Cp*sp.diff(v*T_MMS, y, 1) -
          k*(sp.diff(T_MMS, x, 2) + sp.diff(T_MMS, y, 2)))
f_source = sp.lambdify([x, y, k], source, "numpy")

def q(x, y, k):
    return f_source(x, y, k)

# Conditions limites dérivées de la MMS
# Cas si MMS appliquée en dirichlet à droite (dT_MMS/dx)
f_dT_MSS_dx = sp.lambdify([x, y], sp.diff(T_MMS, x, 1), "numpy")
def MMS_X_droite(x, y):
    return f_dT_MSS_dx(x, y)

# Cas si MMS appliquée en dirichlet à gauche (-dT_MMS/dx)
def MMS_X_gauche(x, y):
    return -MMS_X_droite(x, y)

# Cas MMS appliquée en bas ou en haut du domaine (dT_MMS/y)
f_dT_MSS_dy = sp.lambdify([x, y], sp.diff(T_MMS, y, 1), "numpy")
def MMS_Y_haut(x, y):
    return f_dT_MSS_dy(x, y)

def MMS_Y_bas(x, y):
    return -f_dT_MSS_dy(x, y)

# Conditions

# %% Cas à tester
#bcdata = (['NEUMANN', MMS_X_gauche], ['DIRICHLET', MMS], ['NEUMANN', MMS_X_droite], ['DIRICHLET', MMS])
#bcdata = (['DIRICHLET', MMS], ['NEUMANN', MMS_Y_bas], ['DIRICHLET', MMS], ['NEUMANN', MMS_Y_haut])
bcdata = (['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS])
domain = [-L, L, -L, L]

case = Case(k=1, rho=rho, Cp=Cp, flow_velocity=flow_velocity, source_term=q, domain=domain)


# %% Maillage à tester
processing = Processing(case, bcdata)
processing.set_analytical_function(MMS)

# Selon le maillagef maillage
"""simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 11, 'Ny': 11, 'method': 'CENTRE'},
                          {'mesh_type': 'QUAD', 'Nx': 21, 'Ny': 21, 'method': 'CENTRE'},
                          {'mesh_type': 'QUAD', 'Nx': 41, 'Ny': 41, 'method': 'CENTRE'},
                          {'mesh_type': 'QUAD', 'Nx': 50, 'Ny': 50, 'method': 'CENTRE'}]
postprocessing_parameters = ['solutions',
                              ('plans', {'x': L-L/50, 'y': L-L/50}),
                              'error']

processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
processing.execute_simulations()"""

# Selon centre vs upwind
processing = Processing(case, bcdata)
processing.set_analytical_function(MMS)
simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 11, 'Ny': 11, 'method': 'CENTRE', 'Pe': 10000},
                          {'mesh_type': 'QUAD', 'Nx': 11, 'Ny': 11, 'method': 'UPWIND', 'Pe': 10000}]
postprocessing_parameters = ['solutions',
                             ('plans', {'x': 0, 'y': 0}),
                             ('comparison', {'mesh': [0, 1], 'diff': True})]

processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
processing.execute_simulations()


