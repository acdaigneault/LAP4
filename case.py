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

Classe pour préparer les données d'entrée

"""

#%% IMPORTATION DES LIBRAIRIES

from meshConnectivity import MeshConnectivity


#%% Classe Case


# Cas étudié regroupant le maillage, les conditions frontières, l'utilisation d'un solver
# et la solution calculée.
class Case:
    """
    Preparer les données d'entrée pour l'exemple à traiter.

    Parmeters
    ---------
    mesh_obj: mesh
        Maillage du problème
    
    gamma: float
        Coefficient de diffusion
    
    source_term: function
        Fonction qui calcule le terme source sur le maillage

    """
    
    def __init__(self, k, rho, Cp, flow_velocity, source_term, domain):
        self.k = k
        self.rho = rho
        self.Cp = Cp
        self.flow_velocity = flow_velocity
        self.source_term = source_term
        self.domain = domain
    
    #Accesseurs
    def get_source(self):
        return self.source_term

    def get_physical_properties(self):
        return self.k, self.rho, self.Cp

    def get_flow_velocity(self):
        return self.flow_velocity

    def set_source(self, source_term):
        self.source_term = source_term

    def set_Pe(self, Pe):
        self.k = 1/Pe

    def get_domain(self):
        return self.domain

