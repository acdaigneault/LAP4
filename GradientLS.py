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

Classe pour calculer le gradient d'un champ donné avec la méthode des moindres carrés. 

"""

#%% IMPORTATION DES LIBRAIRIES

import numpy as np


#%% Classe Gradient LS

class GradientMoindresCarres:
    """
    Solveur utilisant la méthode des moindres carrés pour reconstruire le gradient
.

    Parmeters
    ---------
    exempmle: case
        L'exemple en cours de traitement
    

    Attributes
    ----------
    mesh_obj: mesh
        Maillage du problème
    
    bcdata: liste de float et str.
        Type + valeur des conditions aux limites imposées. 
        
    phi: ndarray float.
        Champ sur notre domaine
        
    gradient: ndarray float
        Gradient du champ phi

    """
    def __init__(self, case, mesh_obj, bcdata, preprocessing_data):
        self.case = case  # Cas à résoudre
        self.mesh_obj = mesh_obj  # Maillage du cas
        self.bcdata = bcdata # Conditions frontières
        self.volumes = preprocessing_data[0]
        self.centroids = preprocessing_data[1]
        self.phi=0                      # Champ sur le maillage
        self.gradient=0                  #Gradient du champ phi
     
    #Accesseurs
    def get_case(self):
        return self.case
    
    def get_mesh(self):
        return self.mesh_obj
    
    def get_bcdata(self):
        return self.bcdata
    
    def get_phi(self):
        return self.phi
    
    def get_gradient(self):
        return self.gradient
    
    #Modificateurs
    def set_case(self,new):
        self.case=new
    
    def set_mesh(self,new):
        self.mesh_obj=new
    
    def set_bcdata(self,new):
        self.bcdata=new
    
    def set_phi(self,new):
        self.phi=new
    
    def set_gradient(self,new):
        self.gradient=new

    # Calcule le gradient du cas étudié
    def solve(self):
        
        """
        Calcule le gradient du cas étudié
        
        Parameters
        ----------
        None
            
        
        Returns
        -------
        None
        
        """
        
        # Initialisation des matrices et des données

        NTRI = self.mesh_obj.get_number_of_elements() #Nombre d'éléments
        ATA = np.zeros((NTRI, 2, 2))
        B = np.zeros((NTRI, 2))

        # Remplissage des matrices pour le cas d'une condition frontière (Dirichlet ou Neumann)
        for i_face in range(self.mesh_obj.get_number_of_boundary_faces()):
            tag = self.mesh_obj.get_boundary_face_to_tag(i_face)  # Numéro de la frontière de la face
            bc_type, bc_value = self.bcdata[tag]  # Condition frontière (Dirichlet ou Neumann)
            element = self.mesh_obj.get_face_to_elements(i_face)[0]  # Élément de la face

            # Détermination des positions des points et de la distance
            nodes = self.mesh_obj.get_face_to_nodes(i_face)
            xa = (self.mesh_obj.get_node_to_xycoord(nodes[0])[0] + self.mesh_obj.get_node_to_xycoord(nodes[1])[0]) / 2.
            ya = (self.mesh_obj.get_node_to_xycoord(nodes[0])[1] + self.mesh_obj.get_node_to_xycoord(nodes[1])[1]) / 2.
            xb, yb = self.centroids[element][0], self.centroids[element][1]
            dx, dy = xb - xa, yb - ya

            if bc_type == 'DIRICHLET':
                # Calcul la différence des phi entre le point au centre de la face et au centre de l'élément
                dphi = bc_value(xa, ya) - self.phi[element]

            if bc_type == 'NEUMANN':
                # Modification de la position du point sur la face si Neumann
                (xa, ya), (xb, yb) = self.mesh_obj.get_node_to_xycoord(nodes[0]), self.mesh_obj.get_node_to_xycoord(nodes[1])
                dA = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
                n = np.array([(yb - ya) / dA, -(xb - xa) / dA])
                dx, dy = np.dot([dx, dy], n) * n

                # Application de la condition frontière au point sur la face perpendiculaire au point central
                dphi = np.dot([dx, dy], n) * bc_value(xa, ya)

            # Remplissage de la matrice ATA
            ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            ATA[element] += ALS

            # Remplisage du membre de droite
            B[element] += (np.array([dx, dy]) * dphi)

        # Remplissage des matrices pour les faces internes
        for i_face in range(self.mesh_obj.get_number_of_boundary_faces(), self.mesh_obj.get_number_of_faces()):
            elements = self.mesh_obj.get_face_to_elements(i_face)
            dx, dy = self.centroids[elements[1]] - self.centroids[elements[0]]

            # Remplissage de la matrice ATA pour l'arête interne
            ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            ATA[elements[0]] += ALS
            ATA[elements[1]] += ALS

            # Remplisage du membre de droite
            dphi = self.phi[elements[0]] - self.phi[elements[1]]
            B[elements[0]] += (np.array([dx, dy]) * dphi)
            B[elements[1]] += (np.array([dx, dy]) * dphi)

        # Résolution des systèmes matriciels pour tous les éléments
        ATAI = np.array([np.linalg.inv(ATA[i_tri]) for i_tri in range(NTRI)])
        GRAD = np.array([np.dot(ATAI[i_tri], B[i_tri]) for i_tri in range(NTRI)])

        self.set_gradient(GRAD)