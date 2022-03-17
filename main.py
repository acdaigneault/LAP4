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

#%% IMPORTATION DES LIBRAIRIES
import numpy as np
from case import Case
import matplotlib.pyplot as plt
import sympy as sp
from processing import Processing
import os

#%% Données du problème
# Propriétés physiques
rho = 1 # masse volumique [kg/m³]
mu = 1  # viscosité dynamique [Pa*s]
U = 1   # Vitesse de la paroi mobile [m/s]

# Dimensions du domaine
H = 5  # Distance entre 2 plaques [m]
L = 10  # Longueur des plaques [m]
R = 2

def execute(processing, simulations_parameters, postprocessing_parameters, sim_name):
    print("   • En execution")
    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations(sim_name)
    print("   • Simulation terminée")

def ask_P():
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    while float(P) < -3. or float(P) > 3.:
        print("   Erreur")
        P = input("   Choix du paramètre P (entre -3 et 3): ")

    return float(P)

#%% Créer le fichier pour les images s'il n'existe pas

#Verifier si le fichier images n'existe pas
dirName="images"

if not os.path.exists(dirName):
    os.mkdir(dirName)

#%% ------------------------------------  Cas écoulement de Couette classique ------------------------------------ %%#
print(" -------- Cas d'écoulement de Couette classique --------")
exe_case = input("Tester ce cas? (Y ou N): ")

if exe_case == "Y" or exe_case == "y":

    # %% Données du problème
    # Propriétés physiques
    rho = 1  # kg/m³
    Cp = 1  # J/kg-K
    k = 1   # W/m-K
    x, y, P = sp.symbols('x y P')

    # Champ de vitesse
    u = U + U*R*R/(x**2+y**2)*(1.-2.*x*x/(x**2+y**2))
    v = U*R*R/(x**2+y**2)*(-2.*x*y/(x**2+y**2))
    f_U_MMS = sp.lambdify([x, y, P], u, "numpy")
    f_V_MMS = sp.lambdify([x, y, P], v, "numpy")


    def flow_velocity(x, y, P):
        return np.array([f_U_MMS(x, y, P), f_V_MMS(x, y, P)])


    # Champ de température (MMS)
    T_MMS = sp.tanh(-2.*(y-1.))-sp.tanh(-2.*(y+1.))
    f_T_MMS = sp.lambdify([x, y, P], T_MMS, "numpy")


    def MMS(x, y, P):
        return f_T_MMS(x, y, P)


    def q(x, y, P):
        return 0.

    def NULL(x, y, P):
        return 0.

    # Conditions

    # %% Cas à tester
    bcdata = (['DIRICHLET', MMS], ['NEUMANN', NULL],
              ['NEUMANN', NULL], ['NEUMANN', NULL],
              ['NEUMANN', NULL])
    domain = {'type': 'cercle',
              'points': [[-L, -H], [L, -H], [L, H], [-L, H]],
              'radius': R}

    #%% Initialisation du cas et du processing
    case_classic = Case(rho, gamma=k/Cp, source_terms=q, velocity_field=flow_velocity, domain=domain)

    processing = Processing(case_classic, bcdata)
    processing.set_analytical_function(MMS)

    #%% Simulations
    # Simulation pour la convergence de l'erreur
    print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'TRI', 'Nx': 10, 'Ny': 10, 'Nc': 10, 'method': 'CENTRE', 'P': P},
                                  {'mesh_type': 'TRI', 'Nx': 20, 'Ny': 20, 'Nc': 20, 'method': 'CENTRE', 'P': P},
                                  {'mesh_type': 'TRI', 'Nx': 40, 'Ny': 40, 'Nc': 40, 'method': 'CENTRE', 'P': P}]
        postprocessing_parameters = {'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couetteclassic_convergence_quad")

