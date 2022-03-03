"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""


#----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        LAP4 Équations du momentum                          #
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
rho = 1 # masse volumique [kg/m³]
mu = 1  # viscosité dynamique [Pa*s]
U = 1   # Vitesse de la paroi mobile [m/s]

# Dimensions du domaine
b = 1  # Distance entre 2 plaques [m]
L = 1  # Longueur des plaques [m]

# Terme source de pression & champ de vitesse
x, y, P = sp.symbols('x y P')

f_dpdx= sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0


f_u = sp.lambdify([x, y, P], y*(1-P*(1-y)), "numpy")
def u(x, y, P):
    return f_u(x, y, P)
def v(x, y, P):
    return 0

f_dudy = sp.lambdify([x, y, P], sp.diff(y*(1-P*(1-y)), y, 1), "numpy")
def dudy(x, y, P):
    return f_dudy(x, y, P)
def dvdx(x, y, P):
    return 0

# Solution analytique
def couette_flow(x, y, P):
    return U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)

#%% Conditions frontières et domaine
# Conditions
bcdata = (['NEUMANN', (dudy, dvdx)], ['DIRICHLET', (u, v)], ['NEUMANN', (dudy, dvdx)], ['DIRICHLET', (u, v)])

# Domaine
domain = [0, L, 0, b]

#
case = Case(rho, mu, flow_velocities=(u, v), source_terms=(dpdx, dpdy), domain=domain)

#%% Parametres de simulation et de post-traitement

simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 0},
                          {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 1},
                          {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': -3}]
postprocessing_parameters = ['solutions',
                             ('plans', {'x': 0, 'y': 0}),
                             ('comparison', {'mesh': [0, 1], 'diff': False})]

processing = Processing(case, bcdata)
processing.set_analytical_function((couette_flow, 0))
processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
processing.execute_simulations()


