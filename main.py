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

#%% Cas écoulement de Couette classique
print(" --- Cas d'écoulement de Couette classique ---")

# Terme source de pression, champ de vitesse & solution analytique
x, y, P = sp.symbols('x y P')

f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0

couette_flow = U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
f_u = sp.lambdify([x, y, P], couette_flow, "numpy")
def u(x, y, P):
    return f_u(x, y, P)
def v(x, y, P):
    return 0

def dudx(x, y, P):
    return 0
def dvdx(x, y, P):
    return 0

def null(x, y, P): return 0


#%% Conditions frontières et domaine
# Conditions frontières (Neumann en entrée et en sortie & Dirichlet aux parois)
bcdata = (['NEUMANN', (dudx, dvdx)], ['DIRICHLET', (u, v)],
          ['NEUMANN', (dudx, dvdx)], ['DIRICHLET', (u, v)])

# Domaine
domain = [0, L, 0, b]

case_classic = Case(rho, mu, source_terms=(dpdx, dpdy), domain=domain)
processing = Processing(case_classic, bcdata)
processing.set_analytical_function((u, null))

# Simulation avec P = 0, 1 et -3
print("1. Simulation avec P = 0, 1 et -3")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    print("   • En execution")
    simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': 0, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': 1, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': -3, 'alpha': 0.75}]
    postprocessing_parameters = {'plans': {'x': 0.5, 'y': 0.5},
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")


# Simulation pour la convergence de l'erreur
print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    print("   • En execution")
    simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 10, 'Ny': 10, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 20, 'Ny': 20, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 40, 'Ny': 40, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75}]
    postprocessing_parameters = {'error': 'NA',
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")


# Simulation pour la convergence de l'erreur
print("3. Simulations pour la convergence de l'erreur en maillage 'TRI'")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    print("   • En execution (peu être long)")
    simulations_parameters = [{'mesh_type': 'TRI', 'Nx': 5, 'Ny': 5, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9},
                              {'mesh_type': 'TRI', 'Nx': 10, 'Ny': 10, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9},
                              {'mesh_type': 'TRI', 'Nx': 20, 'Ny': 20, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9}]
    postprocessing_parameters = {'error': 'NA',
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")

"""
#%% Cas écoulement de Couette tourné
print(" --- Cas d'écoulement de Couette tourné ---")
theta = np.pi/8

# Terme source de pression, champ de vitesse & solution analytique
x, y, P = sp.symbols('x y P')

f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0

couette_flow= U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
f_u = sp.lambdify([x, y, P], couette_flow*np.cos(theta), "numpy")
f_v = sp.lambdify([x, y, P], couette_flow*np.sin(theta), "numpy")
def u(x, y, P):
    return f_u(x, y, P)
def v(x, y, P):
    return f_v(x, y, P)

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
rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cois(theta)]])
domain = [np.dot(rotate, [0, 0]), np.dot(rotate, [L, 0]), np.dot(rotate, [L, b]), np.dot(rotate, [0, b])]


case_turned = Case(rho, mu, source_terms=(dpdx, dpdy), domain=domain)
processing = Processing(case_turned, bcdata)
processing.set_analytical_function((u, null))

# Simulation avec P = 0, 1 et -3
print("1. Simulation avec P = 0, 1 et -3")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    print("   • En execution")
    simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': 0, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': 1, 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 25, 'Ny': 25, 'method': 'UPWIND', 'P': -3, 'alpha': 0.75}]
    postprocessing_parameters = {'plans': {'x': 0.5, 'y': 0.5},
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")


# Simulation pour la convergence de l'erreur
print("2. Simulations pour la convergence de l'erreur en maillage 'QUAD'")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    print("   • En execution")
    simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 10, 'Ny': 10, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 20, 'Ny': 20, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75},
                              {'mesh_type': 'QUAD', 'Nx': 40, 'Ny': 40, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.75}]
    postprocessing_parameters = {'error': 'NA',
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")


# Simulation pour la convergence de l'erreur
print("3. Simulations pour la convergence de l'erreur en maillage 'TRI'")
rep = input("   Exécuter? (Y ou N): ")
if rep == "Y" or rep == "y":
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    print("   • En execution (peu être long)")
    simulations_parameters = [{'mesh_type': 'TRI', 'Nx': 5, 'Ny': 5, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9},
                              {'mesh_type': 'TRI', 'Nx': 10, 'Ny': 10, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9},
                              {'mesh_type': 'TRI', 'Nx': 20, 'Ny': 20, 'method': 'UPWIND', 'P': float(P), 'alpha': 0.9}]
    postprocessing_parameters = {'error': 'NA',
                                 'pyvista': {'mesh': [0, 1, 2]}}

    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations()
    print("   • Simulation terminée")"""


plt.show()

