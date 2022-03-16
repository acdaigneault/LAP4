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
b = 1  # Distance entre 2 plaques [m]
L = 1  # Longueur des plaques [m]

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
    # k = 1   # W/m-K

    # Longueur du domaine
    L = 1  # m

    x, y, P = sp.symbols('x y P')

    # Champ de vitesse
    u = (2 * x ** 2 - x ** 4 - 1.0) * (y - y ** 3)
    v = -(2 * y ** 2 - y ** 4 - 1.0) * (x - x ** 3)
    f_U_MMS = sp.lambdify([x, y, P], u, "numpy")
    f_V_MMS = sp.lambdify([x, y, P], v, "numpy")


    def flow_velocity(x, y, P):
        return np.array([f_U_MMS(x, y, P), f_V_MMS(x, y, P)])


    # Champ de température (MMS)
    T0, Tx, Txy = 400, 45, 27.5
    T_MMS = T0 + Tx * sp.cos(np.pi * x) + Txy * sp.sin(np.pi * x * y)
    f_T_MMS = sp.lambdify([x, y, P], T_MMS, "numpy")


    def MMS(x, y, P):
        return f_T_MMS(x, y, P)


    # Terme source dérivé de la MMS
    source = (rho * Cp * sp.diff(u * T_MMS, x, 1) +
              rho * Cp * sp.diff(v * T_MMS, y, 1) -
              P * (sp.diff(T_MMS, x, 2) + sp.diff(T_MMS, y, 2)))
    f_source = sp.lambdify([x, y, P], source, "numpy")


    def q(x, y, P):
        return f_source(x, y, P)


    # Conditions limites dérivées de la MMS
    # Cas si MMS appliquée en dirichlet à droite (dT_MMS/dx)
    f_dT_MSS_dx = sp.lambdify([x, y, P], sp.diff(T_MMS, x, 1), "numpy")


    def MMS_X_droite(x, y, P):
        return f_dT_MSS_dx(x, y, P)


    # Cas si MMS appliquée en dirichlet à gauche (-dT_MMS/dx)
    def MMS_X_gauche(x, y, P):
        return -MMS_X_droite(x, y, P)


    # Cas MMS appliquée en bas ou en haut du domaine (dT_MMS/y)
    f_dT_MSS_dy = sp.lambdify([x, y, P], sp.diff(T_MMS, y, 1), "numpy")


    def MMS_Y_haut(x, y, P):
        return f_dT_MSS_dy(x, y, P)


    def MMS_Y_bas(x, y, P):
        return -f_dT_MSS_dy(x, y, P)




    # Conditions

    # %% Cas à tester
    bcdata = (['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS])
    domain = [[-b, -L], [-b, L], [b, L], [b, -L]]

    #%% Initialisation du cas et du processing
    case_classic = Case(rho, 1, source_terms=q, velocity_field=flow_velocity, domain=domain)
    processing = Processing(case_classic, bcdata)
    processing.set_analytical_function(MMS)

    #%% Simulations
    # Simulation pour la convergence de l'erreur
    print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 10, 'Ny': 10, 'method': 'CENTRE', 'P': P},
                                  {'mesh_type': 'QUAD', 'Nx': 20, 'Ny': 20, 'method': 'CENTRE', 'P': P},
                                  {'mesh_type': 'QUAD', 'Nx': 40, 'Ny': 40, 'method': 'CENTRE', 'P': P}]
        postprocessing_parameters = {'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couetteclassic_convergence_quad")

