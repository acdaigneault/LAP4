"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374


@author: Mohamad Karim ZAYNI
Matricule: 2167132


"""

# ----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        LAP4 Équations du momentum                          #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim               #
# ----------------------------------------------------------------------------#

# %% NOTES D'UTILISATION
"""

Classe pour traiter au préable le maillage donné. 

"""

# %% IMPORTATION DES LIBRAIRIES

import numpy as np
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from solver import FVMMomentum
from postProcessing import PostProcessing


# %% Classe Preprocessing
class Processing:
    def __init__(self, case, bcdata):
        self.case = case
        self.bcdata = bcdata
        self.simulations_parameters = None
        self.postprocessing_parameters = None

    def set_analytical_function(self, analytical_function):
        self.analytical_function = analytical_function

    def set_simulations_and_postprocessing_parameters(self, simulations_parameters, postprocessing_parameters):
        self.simulations_parameters = simulations_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def execute_simulations(self):
        postprocessing = PostProcessing()

        # Excécute plusieurs simulations selon l'ensemble de parametres d'un simulation
        for sim_param in self.simulations_parameters:
            # Crée le mesh et calcule les informations extraites à partir du maillage
            mesh_obj = self.compute_mesh_and_connectivity(sim_param)
            preprocessing_data = self.execute_preprocessing(mesh_obj)

            # Initiale le solver et paramètre la solution analytique et execute la solution selon la méthode
            solver = FVMMomentum(self.case, mesh_obj, self.bcdata, preprocessing_data)
            solver.set_analytical_function(self.analytical_function)
            solver.set_P(sim_param['P'])
            solutions = solver.solve(sim_param['method'])

            # Store les résultats de la simulation et des infos pertinentes pour le post-traitement
            postprocessing.set_data(mesh_obj, solutions, preprocessing_data, sim_param)

        # S'ils y a du post-traitement, il appelle la fonction qui les exécutes
        if self.postprocessing_parameters is not None:
            self.execute_postprocessing(postprocessing)

    def execute_postprocessing(self, postprocessing):
        for postproc_param in self.postprocessing_parameters:
            if postproc_param == 'solutions':
                postprocessing.show_solutions(i_mesh=-1, title='Solution numérique et MMS',
                                              save_path='images/solutions.png')
            elif postproc_param[0] == "plans":
                postprocessing.show_plan_solutions(i_mesh=-1, title='Solutions selon des coupes',
                                                   save_path='images/plans', X_Coupe=postproc_param[1]['x'],
                                                   Y_Coupe=postproc_param[1]['y'])
            elif postproc_param == "error":
                postprocessing.show_error()
            elif postproc_param[0] == 'comparison':
                postprocessing.show_mesh_differences(postproc_param[1]['mesh'][0], postproc_param[1]['mesh'][1],
                                                     title='Comparaison de 2 simulations', save_path='images/diff',
                                                     diff=postproc_param[1]['diff'])
            elif postproc_param[0] == "pyvista":
                for mesh in postproc_param[1]['mesh']:
                    print(mesh)
                    postprocessing.show_pyvista(mesh)
            else:
                print(f'Demande de post traitement {postproc_param} invalide.')

    # Exécute la connectivité avec le maillage généré.
    def compute_mesh_and_connectivity(self, mesh_parameters):
        """
        Exécute la connectivité avec le maillage généré.

        Parameters
        ----------
        None

        Returns
        -------
        mesh_obj

        """
        mesher = MeshGenerator()
        mesh_obj = mesher.rectangle(self.case.get_domain(), mesh_parameters)
        conec = MeshConnectivity(mesh_obj, verbose=False)
        conec.compute_connectivity()

        return mesh_obj

    def execute_preprocessing(self, mesh_obj):

        """
        Effectue les calculs relatifs au maillage préalablement à l'utilisation du solver

        Parameters
        ----------
        None

        Returns
        -------
        (volumes, centroids)

        """
        n_elem = mesh_obj.get_number_of_elements()  # Nombre d'éléments dans notre maillage
        volumes = np.zeros(n_elem)  # surface des éléments
        centroids = np.zeros((n_elem, 2))  # coordonnees des centroides

        # Détermine les centroides et l'aire de l'élément par les déterminants
        for i_elem in range(n_elem):
            nodes = mesh_obj.get_element_to_nodes(i_elem)
            area_matrices = [np.zeros([2, 2]) for i in range(len(nodes))]
            for i in range(len(nodes)):
                x, y = mesh_obj.get_node_to_xycoord(nodes[i])[0], mesh_obj.get_node_to_xycoord(nodes[i])[1]
                area_matrices[i][:, 0] = [x, y]
                area_matrices[i - 1][:, 1] = [x, y]

            # Calcule l'aire de l'élément
            volumes[i_elem] = np.sum([np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) / 2

            # Calcule du position des centroides
            cx = (np.sum(
                [np.sum(area_matrices[i][0, :]) * np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) /
                  (6 * volumes[i_elem]))
            cy = (np.sum(
                [np.sum(area_matrices[i][1, :]) * np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) /
                  (6 * volumes[i_elem]))

            centroids[i_elem] = [cx, cy]

        return volumes, centroids

