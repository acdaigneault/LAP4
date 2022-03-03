"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""

# ----------------------------------------------------------------------------#
#                                 MEC6616                                     #
#                        LAP4 Équations du momentum                           #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim                #
# ----------------------------------------------------------------------------#

# %% NOTES D'UTILISATION
"""

Classe de solver VF Convection Diffusion

"""

# %% IMPORTATION DES LIBRAIRIES

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import GradientLS as GLS
import sys


# %% Fonctions Internes
# Calcule les distances et vecteurs nécessaires selon les coordonnées fournies
def compute_lengths_and_unit_vectors(pta, ptb, ptA, ptP):
    (xa, ya), (xb, yb), (xA, yA), (xP, yP) = pta, ptb, ptA, ptP

    # Détermination des distances
    dx, dy = (xb - xa), (yb - ya)
    dA = np.sqrt(dx ** 2 + dy ** 2)
    dKSI = np.sqrt((xA - xP) ** 2 + (yA - yP) ** 2)

    # Détermination des vecteurs
    n = np.array([dy / dA, -dx / dA])
    eKSI = np.array([(xA - xP) / dKSI, (yA - yP) / dKSI])
    eETA = np.array([dx / dA, dy / dA])

    return dA, dKSI, n, eKSI, eETA

# Calcule le terme source dû au gradient de pression
def compute_source(P, dpdx, dpdy, volumes, centroids):
    SGPXp, SGPYp = np.zeros(len(volumes)), np.zeros(len(volumes))

    # Calcule les gradients de pression aux centroids des éléments * volume
    for i in range(len(volumes)):
        SGPXp[i] = dpdx(x=centroids[i][0], y=centroids[i][1], P=P) * volumes[i]
        SGPYp[i] = dpdy(x=centroids[i][0], y=centroids[i][1], P=P) * volumes[i]

    return SGPXp, SGPYp



# %% Classe MethodesVolumesFinisConvDiff
# Solveur utilisant la méthode des volumes finis Convection-Diffusion
class FVMMomentum:
    """
    Méthode de volumes finis pour un problème de diffusion.

    Parmeters
    ---------
    exemple: self.case
        L'exemple en cours de traitement
    
    cross_diffusion: bool
        pour activer ou désactiver la considération des termes Sdc

    Attributes
    ----------
    mesh_obj: mesh
        Maillage du problème
    
    bcdata: liste de float et str.
        Type + valeur des conditions aux limites imposées. 
    
        
    time: dictionnaire str/float.
        Temps de calcul correspondant à la méthode utilisé

    """

    def __init__(self, case, mesh_obj, bcdata, preprocessing_data):
        self.case = case  # Cas à résoudre
        self.mesh_obj = mesh_obj  # Maillage du cas
        self.bcdata = bcdata  # Conditions frontières
        self.volumes = preprocessing_data[0]
        self.centroids = preprocessing_data[1]

    def set_analytical_function(self, analytical_function):
        self.analytical_function = analytical_function

    # Accesseurs
    def get_case(self):
        return self.case

    def get_mesh(self):
        return self.mesh_obj

    def get_bcdata(self):
        return self.bcdata

    def get_volumes(self):
        return self.volumes

    def get_centroids(self):
        return self.centroids

    def get_cross_diffusion(self):
        return self.get_cross_diffusion

    def get_analytical_function(self):
        return self.analytical_function

    def get_P(self):
         return self.P

    # Modificateurs
    def set_P(self, new):
        self.P = new

    # Solveur VF
    def solve(self, method="CENTRE"):
        """
        Solveur Volumes Finis 
        
        Returns
        -------
        Solutions
        
        """
        # Chercher les différentes variables
        mesh = self.get_mesh()          # Maillage utilisé
        case = self.get_case()          # cas en cours
        centroids = self.centroids      # Chercher les centres des éléments
        volumes = self.get_volumes()    # surfaces des éléments
        bcdata = self.get_bcdata()      # Conditions limites
        P = self.get_P()                # Paramètre modifié

        # Initialisation des matrices et des vecteurs pour u et v
        NELEM = self.mesh_obj.get_number_of_elements()
        Au = np.zeros((NELEM, NELEM))
        Bu, Bv = np.zeros(NELEM), np.zeros(NELEM)
        PHIu, PHIv = np.zeros(NELEM), np.zeros(NELEM)
        PHI_EXu, PHI_EXv = np.zeros(NELEM), np.zeros(NELEM)
        GRADu, GRADv = np.zeros((NELEM, 2)), np.zeros((NELEM, 2))

        # Variables locales
        rho, mu = case.get_physical_properties()
        dpdx, dpdy = case.get_sources()
        analytical_function = self.get_analytical_function()

        # GLS (on peut modifier ce solver pour momentum!! donc pour 2 variables)
        solver_moindrescarresu = GLS.GradientMoindresCarres(case, mesh, bcdata, (volumes, centroids))
        solver_moindrescarresv = GLS.GradientMoindresCarres(case, mesh, bcdata, (volumes, centroids))

        # Calcule les termes sources reliés au gradient de pression
        SGPXp, SGPYp = compute_source(P, dpdx, dpdy, volumes, centroids)

        for i in range(3):

            # Parcours les faces sur les conditions frontières et remplis la matrice A et le vecteur B
            for i_face in range(mesh.get_number_of_boundary_faces()):
                tag = mesh.get_boundary_face_to_tag(i_face)  # Numéro de la frontière de la face
                bc_type, bc_value = bcdata[tag]  # Condition frontière (Dirichlet ou Neumann)
                nodes = mesh.get_face_to_nodes(i_face)  # Noeuds de la face
                element = mesh.get_face_to_elements(i_face)[0]  # Élément de la face

                # Détermine la position du centre de la face
                (xa, ya) = ((mesh.get_node_to_xycoord(nodes[0])[0] + mesh.get_node_to_xycoord(nodes[1])[0]) / 2,
                            (mesh.get_node_to_xycoord(nodes[0])[1] + mesh.get_node_to_xycoord(nodes[1])[1]) / 2)

                dA, dKSI, n, eKSI, eETA = \
                    compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                     ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                     ptA=(xa, ya),
                                                     ptP=centroids[element])

                # Calcule les projections de vecteurs unitaires
                dETA = dA  # Équivalent, mais noté pour éviter la confusion
                PNKSI = np.dot(n, eKSI)  # Projection de n sur ξ
                PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η

                # Patrie Convection
                F = rho * np.dot(n, U(xa, ya)) * dA

                if bc_type == "DIRICHLET":
                    D = (1 / PNKSI) * gamma * (dA / dKSI)  # Direct gradient term

                    # Calcule le terme correction de cross-diffusion si activé

                    # Évaluation des phi aux noeuds de la face frontière
                    phi0 = bc_value(mesh.get_node_to_xycoord(nodes[0])[0],
                                    mesh.get_node_to_xycoord(nodes[0])[1])
                    phi1 = bc_value(mesh.get_node_to_xycoord(nodes[1])[0],
                                    mesh.get_node_to_xycoord(nodes[1])[1])
                    Sdc = -gamma * (PKSIETA / PNKSI) * (phi1 - phi0) / dETA * dA

                    if method == "CENTRE":
                        A[element, element] += D
                        B[element] += D * bc_value(xa, ya) + Sdc - F * bc_value(xa, ya)
                    elif method == "UPWIND":
                        A[element, element] += D + max(F, 0)
                        B[element] += Sdc + max(0, -F) * bc_value(xa, ya) + D * bc_value(xa, ya)

                elif bc_type == "NEUMANN":
                    B[element] += gamma * bc_value(xa, ya) * dA

                    if method == "UPWIND":
                        A[element, element] += F
                        B[element] += -F * bc_value(xa, ya) * PNKSI * dETA

            # Parcours les faces internes et remplis la matrice A et le vecteur B
            for i_face in range(mesh.get_number_of_boundary_faces(), mesh.get_number_of_faces()):
                nodes = mesh.get_face_to_nodes(i_face)
                elements = mesh.get_face_to_elements(i_face)

                dA, dKSI, n, eKSI, eETA = \
                    compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                     ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                     ptA=centroids[elements[1]],
                                                     ptP=centroids[elements[0]])

                # Détermine la position du centre de la face
                (xa, ya) = (
                    (mesh.get_node_to_xycoord(nodes[0])[0] + mesh.get_node_to_xycoord(nodes[1])[0]) / 2,
                    (mesh.get_node_to_xycoord(nodes[0])[1] + mesh.get_node_to_xycoord(nodes[1])[1]) / 2)

                # Calcule les projections de vecteurs unitaires
                PNKSI = np.dot(n, eKSI)  # Projection de n sur ξ
                PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η

                D = (1 / PNKSI) * gamma * (dA / dKSI)  # Direct gradient term

                # Calcule le terme correction de cross-diffusion si activé
                Sdc = -gamma * (PKSIETA / PNKSI) * np.dot((GRAD[elements[1]] + GRAD[elements[0]]) / 2, eETA) * dA

                # Partie Convection
                Fi = rho * np.dot(n, U(xa, ya)) * dA

                if method == "CENTRE":
                    # Remplissage de la matrice et du vecteur
                    A[elements[0], elements[0]] += D + Fi / 2.0
                    A[elements[1], elements[1]] += D - Fi / 2.0
                    A[elements[0], elements[1]] += -D + Fi / 2.0
                    A[elements[1], elements[0]] += -D - Fi / 2.0

                elif method == "UPWIND":
                    # Remplissage de la matrice et du vecteur
                    A[elements[0], elements[0]] += (D + max(Fi, 0))
                    A[elements[1], elements[1]] += (D + max(0, -Fi))
                    A[elements[0], elements[1]] += (-D - max(0, -Fi))
                    A[elements[1], elements[0]] += (-D - max(Fi, 0))

                else:
                    print("La méthode choisie n'est pas convenable, veuillez choisir Centre ou Upwind")
                    sys.exit()

                B[elements[0]] += Sdc
                B[elements[1]] -= Sdc

            # Ajout de la contribution du terme source sur les éléments et calcul de la solution analytique
            for i_elem in range(mesh.get_number_of_elements()):
                B[i_elem] += volumes[i_elem] * source(centroids[i_elem][0], centroids[i_elem][1], gamma)
                PHI_EX[i_elem] = analytical_function(centroids[i_elem][0], centroids[i_elem][1])

            PHI = linsolve.spsolve(sps.csr_matrix(A, dtype=np.float64), B)

            solver_moindrescarres.set_phi(PHI)
            solver_moindrescarres.solve()
            GRAD = solver_moindrescarres.get_gradient()

        return PHI, PHI_EX
