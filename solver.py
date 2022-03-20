"""
MEC6616 Aérodynamique Numérique


@author: Adrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""


# %% IMPORTATION DES LIBRAIRIES

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import gradientLS as GLS
import sys

print("Les figures supplémentaires se retrouvent dans le dossier images")

# %% Fonctions Internes
def compute_lengths_and_unit_vectors(pta, ptb, ptA, ptP):
    """
    Calcule les distances et vecteurs unitaires nécessaires selon les coordonnées fournies

    Parameters
    ----------
    pta: Tuple[numpy.float64, numpy.float64]
    Coordonnée du noeud 1 de la face

    ptb: Tuple[numpy.float64, numpy.float64]
    Coordonnée du noeud 2 de la face

    ptA: numpy.ndarray
    Coordonnée du centroide de l'élément de droite

    ptP: numpy.ndarray) -> Tuple[numpy.float64, numpy.float64, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Coordonnée du centroide de l'élément de gauche

    Returns
    -------
    (dA, dKSI, n, eKSI, eETA): Tuple[numpy.float64, numpy.float64, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    dA   -> Aire de la face
    dKSI -> Distance entre les deux centroides
    n    -> Vecteur normal de la face
    eKSI -> Vecteur unitaire du centroide de gauche vers celui de droite
    eETA -> Vecteur unitaire du noeud 1 vers 2

    """

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

# %% Classe
class FVM:
    """
    Méthode de volumes finis pour un problème de transfert de momentum en 2D

    Parameters
    ---------
    case: Case
    Cas traité qui a les informations sur la physique du problème

    mesh_obj: Mesh
    Maillage de la sigammalation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    preprocessing_data: Tuple[numpy.ndarray, numpy.ndarray]
    Arrays storant les volumes des éléments et la position du centroide

    Attributes
    ----------
    case: Case
    Cas traité qui a les informations sur la physique du problème

    mesh_obj: Mesh
    Maillage de la sigammalation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    volumes: numpy.ndarray
    Array storant les volumes des éléments

    centroids: numpy.ndarray
    Array storant les coordonnées des centroides des éléments

    """

    def __init__(self, case, mesh_obj, bcdata, preprocessing_data):
        self.case = case  # Cas à résoudre
        self.mesh_obj = mesh_obj  # Maillage du cas
        self.bcdata = bcdata  # Conditions frontières
        self.volumes = preprocessing_data[0]
        self.centroids = preprocessing_data[1]

    def set_analytical_function(self, analytical_function):
        """
        Ajoute une solution analytique au problème sigammalé lorsque disponible et/ou nécessaire

        Parameters
        ----------
        analytical_function: Tuple[function, function]
        Fonction analytique du problème (u(x,y) et v(x,y))
        """
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
    def solve(self, method="CENTRE", alpha=0.75):
        """
        Effectue les calculs relatifs au maillage préalablement à l'utilisation du solver

        Parameters
        ----------
        method: str = "CENTRE"
        Méthode pour la sigammalation en convection (CENTRE ou UPWIND)

        alpha: float = 0.75)
        Facteur de relaxation

        Returns
        -------
        (u, v), (PHI_EXu, PHI_EXv): Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]
        (u, v) -> Solution numérique en x et y
        (PHI_EXu, PHI_EXv) -> Solution analytique en x et y

        """
        # Initiation des matrices
        NELEM = self.mesh_obj.get_number_of_elements()
        A = np.zeros([NELEM, NELEM])
        B = np.zeros(NELEM)
        PHI = np.zeros(NELEM)
        PHI_EX = np.zeros(NELEM)
        GRAD = np.zeros([NELEM, 2])



        # Chercher les différentes variables
        mesh = self.get_mesh()          # Maillage utilisé
        case = self.get_case()          # cas en cours
        centroids = self.centroids      # Chercher les centres des éléments
        volumes = self.get_volumes()    # surfaces des éléments
        bcdata = self.get_bcdata()      # Conditions limites
        P = self.get_P()                # Paramètre modifié

        # Variables locales
        rho, gamma = case.get_physical_properties()
        gamma=P
        source_term = case.get_sources()
        U = case.get_velocity_field()
        analytical_function = self.get_analytical_function()

        solver_GLS = GLS.GradientLeastSquares(mesh, bcdata, centroids)
        solver_GLS.set_P(P)

        # Boucle pour le cross-diffusion
        for i in range(3):
            # Parcours les faces internes et remplis la matrice A et le vecteur B
            for i_face in range(mesh.get_number_of_boundary_faces(), mesh.get_number_of_faces()):
                # Listes des noeuds et des éléments reliés à la face
                nodes = mesh.get_face_to_nodes(i_face)
                left_elem, right_elem = mesh.get_face_to_elements(i_face)

                # Calcule des grandeurs et vecteurs géométriques pertinents
                dA, dKSI, n, eKSI, eETA = \
                    compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                     ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                     ptA=centroids[right_elem],
                                                     ptP=centroids[left_elem])


                # Détermine la position du centre de la face
                pt0, pt1 = mesh.get_node_to_xycoord(nodes[0]), mesh.get_node_to_xycoord(nodes[1])
                xa, ya = 0.5*(pt0[0] + pt1[0]), 0.5*(pt0[1] + pt1[1])

                # Convection
                F = np.dot(U(xa, ya, P), n) * dA  # Débit massique qui traverse la face
                # Calcule les projections de vecteurs unitaires
                PNKSI = np.dot(n, eKSI)       # Projection de n sur ξ
                PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η

                D = (1/PNKSI) * gamma * (dA / dKSI)  # Direct gradient term

                # Calcule le terme correction de cross-diffusion
                Sdc = -gamma * (PKSIETA/PNKSI) * 0.5*np.dot((GRAD[right_elem] + GRAD[left_elem]), eETA) * dA
                
                # Ajoute la contribution de la convection à la matrice A
                if method == "CENTRE":
                    # Remplissage de la matrice et du vecteur
                    A[left_elem, left_elem]   +=  D + 0.5*F
                    A[right_elem, right_elem] +=  D - 0.5*F
                    A[left_elem, right_elem]  += -D + 0.5*F
                    A[right_elem, left_elem]  += -D - 0.5*F
                elif method == "UPWIND":
                    # Remplissage de la matrice et du vecteur
                    A[left_elem, left_elem]   +=  (D + max(F, 0))
                    A[right_elem, right_elem] +=  (D + max(0, -F))
                    A[left_elem, right_elem]  += (-D - max(0, -F))
                    A[right_elem, left_elem]  += (-D - max(F, 0))

                else:
                    print("La méthode choisie n'est pas convenable, veuillez choisir Centre ou Upwind")
                    sys.exit()

                B[left_elem]  += Sdc
                B[right_elem] -= Sdc


            # Parcours les faces sur les conditions frontières et remplis la matrice A et le vecteur B
            for i_face in range(mesh.get_number_of_boundary_faces()):
                # Détermine le numéro de la frontière et les conditions associées
                tag = mesh.get_boundary_face_to_tag(i_face)
                bc_type, bc_value = bcdata[tag]

                # Listes des noeuds et des éléments reliés à la face
                nodes = mesh.get_face_to_nodes(i_face)
                element = mesh.get_face_to_elements(i_face)[0]

                # Détermine la position du centre de la face
                pt0, pt1 = mesh.get_node_to_xycoord(nodes[0]), mesh.get_node_to_xycoord(nodes[1])
                xa, ya = 0.5*(pt0[0] + pt1[0]), 0.5*(pt0[1] + pt1[1])

                dA, dKSI, n, eKSI, eETA = \
                    compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                     ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                     ptA=(xa, ya),
                                                     ptP=centroids[element])

                # Calcule les projections de vecteurs unitaires
                dETA = dA  # Équivalent, mais noté pour éviter la confusion
                PNKSI = np.dot(n, eKSI)  # Projection de n sur ξ
                PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η

                F = np.dot(U(xa, ya, P), n) * dA

                if bc_type == "DIRICHLET":
                    # Détermine le terme du gradient direct et le flux massique au centre de la frontière
                    D = (1/PNKSI) * gamma * (dA/dKSI)

                    # Calcule du terme de cross-diffusion selon les phi aux noeuds de l'arête en x et y
                    phi0, phi1 = bc_value(pt0[0], pt0[1], P), bc_value(pt1[0], pt1[1], P)
                    Sdc = -gamma * (PKSIETA/PNKSI) * rho * (phi1 - phi0)/dETA * dA

                    if method == "CENTRE":
                        A[element, element] += D
                        B[element] += (D - F) * bc_value(xa, ya, P) + Sdc
                    elif method == "UPWIND":
                        A[element, element] += D + max(F, 0)
                        B[element] += (D + max(0, -F)) * bc_value(xa, ya, P) + Sdc
                    else:
                        print("La méthode choisie n'est pas convenable, veuillez choisir CENTRE ou UPWIND")
                        sys.exit()

                elif bc_type == "NEUMANN":
                    A[element, element] += F
                    B[element] += (gamma * dA) * bc_value(xa, ya, P)
                    if method == "UPWIND":
                        B[element] += - F * PNKSI * dKSI * bc_value(xa, ya, P)


            # Ajout de la contribution du terme source sur les éléments et calcul de la solution analytique
            for i_elem in range(mesh.get_number_of_elements()):
                B[i_elem] += source_term(centroids[i_elem][0], centroids[i_elem][1], P) * volumes[i_elem]
                PHI_EX[i_elem] = analytical_function(centroids[i_elem][0], centroids[i_elem][1], P)

            # Résolution pour l'itération
            PHI = linsolve.spsolve(sps.csr_matrix(A, dtype=np.float64), B)

            # Calcule des gradients pour le cross-diffusion
            GRAD = solver_GLS.solve(PHI)


        return PHI, PHI_EX
