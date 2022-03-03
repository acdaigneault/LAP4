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

Classe pour préparer les données d'entrée

"""



#%% Classe Case
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
    
    def __init__(self, rho, mu, flow_velocities, source_terms, domain):
        self.rho = rho
        self.mu = mu
        self.flow_velocities = flow_velocities
        self.source_terms = source_terms
        self.domain = domain
    
    #Accesseurs
    def get_sources(self):
        return self.source_terms

    def get_physical_properties(self):
        return self.rho, self.mu

    def get_flow_velocities(self):
        return self.flow_velocities

    def set_sources(self, source_terms):
        self.source_terms = source_terms

    def get_domain(self):
        return self.domain

