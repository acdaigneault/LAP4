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




#%% ------------------------------------  Cas écoulement de Couette classique ------------------------------------ %%#
print(" -------- Cas d'écoulement de Couette classique --------")

#%% Terme source de pression, champ de vitesse & solution analytique
x, y, P = sp.symbols('x y P')

# Pression
f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0

# Vitesse et solution analytique
couette_flow = U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
f_u = sp.lambdify([x, y, P], couette_flow, "numpy")
def u(x, y, P):
    return f_u(x, y, P)
def v(x, y, P):
    return 0

def dudn(x, y, P):
    return 0
def dvdn(x, y, P):
    return 0

def null(x, y, P): return 0

#%% Conditions frontières et domaine
# Conditions frontières (Neumann en entrée et en sortie & Dirichlet aux parois)
bcdata = (['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)],
          ['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)])

# Domaine
domain = [[0, 0], [L, 0], [L, b], [0, b]]

#%% Initialisation du cas et du processing
case_classic = Case(rho, mu, source_terms=(dpdx, dpdy), domain=domain)
processing = Processing(case_classic, bcdata)
processing.set_analytical_function((u, null))

#%% Simulations
# Simulation avec P = 0, 1 et -3
print("1. Simulation avec P = 0, 1 et -3")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 0, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': 1, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'CENTRE', 'P': -3, 'alpha': 0.75}]
    postprocessing_parameters = {'plans': {'simulation': [2], 'x': 0.5, 'y': 0.5},
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
# Angle de rotation et matrice de rotation
theta = np.pi/8
rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Terme source de pression, champ de vitesse & solution analytique
x, y, P = sp.symbols('x y P')

# Pression
f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0

# Vitesse et solution analytique
couette_flow = U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
f_u = sp.lambdify([x, y, P], couette_flow, "numpy")

def u(x, y, P):
    x_turned, y_turned = np.linalg.solve(rotate, np.array([x, y]))
    return f_u(x_turned, y_turned, P)

def v(x, y, P):
    return 0

def dudn(x, y, P):
    return 0
def dvdn(x, y, P):
    return 0

def null(x, y, P): return 0


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
processing.set_analytical_function((u, null))

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


