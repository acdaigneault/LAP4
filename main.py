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
    f_U_MMS = sp.lambdify([x, y], u, "numpy")
    f_V_MMS = sp.lambdify([x, y], v, "numpy")


    def flow_velocity(x, y):
        return np.array([f_U_MMS(x, y), f_V_MMS(x, y)])


    # Champ de température (MMS)
    T0, Tx, Txy = 400, 45, 27.5
    T_MMS = T0 + Tx * sp.cos(np.pi * x) + Txy * sp.sin(np.pi * x * y)
    f_T_MMS = sp.lambdify([x, y], T_MMS, "numpy")


    def MMS(x, y):
        return f_T_MMS(x, y)


    # Terme source dérivé de la MMS
    source = (rho * Cp * sp.diff(u * T_MMS, x, 1) +
              rho * Cp * sp.diff(v * T_MMS, y, 1) -
              P * (sp.diff(T_MMS, x, 2) + sp.diff(T_MMS, y, 2)))
    f_source = sp.lambdify([x, y, P], source, "numpy")


    def q(x, y, P):
        return f_source(x, y, P)


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
    bcdata = (['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS], ['DIRICHLET', MMS])
    domain = [-L, L, -L, L]

    #%% Initialisation du cas et du processing
    case_classic = Case(rho, mu, source_terms=MMS, domain=domain)
    processing = Processing(case_classic, bcdata)
    processing.set_analytical_function(MMS)

    #%% Simulations
    # Simulation avec P = 0, 1 et -3
    print("1. Simulation avec P = 0, 1 et -3")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 0, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 1, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': -3, 'alpha': 0.75}]
        postprocessing_parameters = {'plans': {'simulation': [0, 1, 2], 'x': 0.5, 'y': 0.5},
                                     'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters, sim_name="couetteclassic_paramP")


    # Simulation pour la convergence de l'erreur
    print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 10, 'Ny': 10, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 20, 'Ny': 20, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 40, 'Ny': 40, 'method': 'CENTRE', 'P': P, 'alpha': 0.75}]
        postprocessing_parameters = {'error': 'NA',
                                     'solutions':  {'simulation': [0, 1, 2]},
                                     'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couetteclassic_convergence_quad")



    # Simulation pour la convergence de l'erreur
    print("3. Simulations pour la convergence de l'erreur en maillage 'TRI'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'TRI', 'Nx': 5, 'Ny': 5, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'TRI', 'Nx': 10, 'Ny': 10, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'TRI', 'Nx': 20, 'Ny': 20, 'method': 'CENTRE', 'P': P, 'alpha': 0.75}]
        postprocessing_parameters = {'error': 'NA',
                                     'solutions': {'simulation': [0, 1, 2]},
                                     'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couetteclassic_convergence_tri")



#%% --------------------------------------  Cas écoulement de Couette tourné -------------------------------------- %%#
print(" -------- Cas d'écoulement de Couette tourné --------")
exe_case = input("Tester ce cas? (Y ou N): ")

if exe_case == "Y" or exe_case == "y":
    # Angle de rotation et matrice de rotation
    theta = np.pi/8
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    #%% Terme source de pression, champ de vitesse & solution analytique
    x, y, P = sp.symbols('x y P')

    # Pression
    f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
    def dpdx(x, y, P):
        return f_dpdx(x, y, P)*np.cos(theta)
    def dpdy(x, y, P):
        return f_dpdx(x, y, P)*np.sin(theta)

    # Vitesse et solution analytique
    couette_flow = U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
    f_u = sp.lambdify([x, y, P], couette_flow, "numpy")

    def u(x, y, P):
        x_turned, y_turned = np.linalg.solve(rotate, np.array([x, y]))
        return f_u(x_turned, y_turned, P)*np.cos(theta)

    def v(x, y, P):
        x_turned, y_turned = np.linalg.solve(rotate, np.array([x, y]))
        return f_u(x_turned, y_turned, P)*np.sin(theta)

    def dudn(x, y, P):
        return 0
    def dvdn(x, y, P):
        return 0

    #%% Conditions frontières et domaine
    # Conditions frontières (Neumann en entrée et en sortie & Dirichlet aux parois)
    bcdata = (['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)],
              ['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)])

    # Domaine
    domain = [np.dot(rotate, [0, 0]), np.dot(rotate, [L, 0]),
              np.dot(rotate, [L, b]), np.dot(rotate, [0, b])]

    #%% Initialisation du cas et du processing
    case_turned = Case(rho, mu, source_terms=(dpdx, dpdy), domain=domain)
    processing = Processing(case_turned, bcdata)
    processing.set_analytical_function((u, v))

    #%% Simulations
    # Simulation avec P = 0, 1 et -3
    print("1. Simulation avec P = 0, 1 et -3")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 0, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 1, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': -3, 'alpha': 0.75}]
        postprocessing_parameters = {'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couettetourne_paramP")


    # Simulation pour la convergence de l'erreur
    print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 20, 'Ny': 20, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 30, 'Ny': 30, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'QUAD', 'Nx': 40, 'Ny': 40, 'method': 'CENTRE', 'P': P, 'alpha': 0.75}]
        postprocessing_parameters = {'error': 'NA',
                                     'solutions': {'simulation': [0, 1, 2]},
                                     'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couettetourne_convergence_quad")


    # Simulation pour la convergence de l'erreur
    print("3. Simulations pour la convergence de l'erreur en maillage 'TRI'")
    rep = input("   Exécuter? (Y ou N): ")
    if rep == "Y" or rep == "y":
        P = ask_P()
        simulations_parameters = [{'mesh_type': 'TRI', 'Nx': 5, 'Ny': 5, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'TRI', 'Nx': 10, 'Ny': 10, 'method': 'CENTRE', 'P': P, 'alpha': 0.75},
                                  {'mesh_type': 'TRI', 'Nx': 20, 'Ny': 20, 'method': 'CENTRE', 'P': P, 'alpha': 0.75}]
        postprocessing_parameters = {'error': 'NA',
                                     'solutions': {'simulation': [0, 1, 2]},
                                     'pyvista': {'simulation': [0, 1, 2]}}
        execute(processing, simulations_parameters, postprocessing_parameters,
                sim_name="couettetourne_convergence_tri")


