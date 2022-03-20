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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
import pyvista as pv
import pyvistaqt as pvQt
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter


#%% Fonction interne
def Coupe_X(Coordonnees, X, Solution, Analytique, Plan,only_num,plan_sortie):
    Elements_ds_coupe = []
    Solution_coupe = []
    Analytique_coupe = []
    eps = 1e-6  # Précision
    #Si on veut le plan de sortie
    if plan_sortie is True:
        X=max(Coordonnees[:,0]) #Plan de sortie
        Plan=0
            
    for i in range(len(Coordonnees)):
        if np.abs(Coordonnees[i, Plan] - X) < eps:
            Elements_ds_coupe.append(Coordonnees[i, :])
            Solution_coupe.append(Solution[i])
            if only_num is False:
                Analytique_coupe.append(Analytique[i])
        
    Elements_ds_coupe = np.array(Elements_ds_coupe)
    Solution_coupe = np.array(Solution_coupe)
    return Elements_ds_coupe, Solution_coupe, Analytique_coupe,X


#%% Classe
class PostProcessing:
    """
    Effectuer le post_traitement après la résolution du problème selon une ou plusieurs simulations
    Parmeters
    ---------
    sim_name: str
    Nom d'un groupe de simulation

    Attributes
    ----------
    sim_name: str
    Nom d'un groupe de simulation

    data: Dict
    Dictionnaire qui comporte toutes les données de nécessaire pour effectuer le post-traitement.

    """
    def __init__(self,  sim_name):
        self.sim_name = sim_name
        self.data = []  # Initialisation du dictionnaire de données

    def get_sim_name(self):
        return self.sim_name

    def set_data(self, mesh, solutions, preprocessing_data, simulation_paramaters):
        """
        Ajouter des données après une simulation à un ensemble de simulation à étudier

        Parameters
        ----------
        mesh: mesh.Mesh
        Maillage de la simulation

        solutions: Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]
        Solutions numérique (u, v) et solutions analytiques (phi_exact, v_exact)

        preprocessing_data: Tuple[numpy.ndarray, numpy.ndarray],
        Arrays contenant les données de preprocessing (volumes et position du centroide des éléments)

        simulation_paramaters: Dict[str, str]
        Paramètres de simulation pour permettre leur affichage dans les titres ou les noms de sauvegarde

        Returns
        -------
        None
        """

        # Stockage de données de la simulation
        # n: int                        -> Nombre d'éléments dans le maillage
        # mesh: Mesh                    -> maillage (utilise pour PyVista)
        # phi_num, v_num: np.ndarray      -> Solutions numériques pour les 2 directions
        # phi_exact, v_exact: np.ndarray  -> Solutions analytiques pour les 2 directions
        # phi_num: np.ndarray           -> Norme de la solution numérique
        # phi_exact: np.ndarray         -> Norme de la solution analytique
        # area: np.ndarray              -> Volume des éléments
        # position: np.ndarray          -> Position du centroides des éléments
        # P: float                      -> Valeur du paramètre P
        # method: str                   -> Méthode utilisée (correction pour la convection) CENTRE ou UPWIND
        self.data.append({'n': mesh.get_number_of_elements(),
                          'mesh': mesh,
                          'phi_num': solutions[0], 'phi_exact': solutions[1],
                          'area': preprocessing_data[0],
                          'position': preprocessing_data[1],
                          'P': simulation_paramaters['P'],
                          'method': simulation_paramaters['method']})

#%% Affichage Solutions avec TricontourF: OK
    def show_solutions(self, i_sim,only_num=False):
        """
        Affichage des graphiques qui montrent la différence entre la solution numérique et la solution analytique
        à l'aide de fonctions tricontour dans matplotlib

        Parameters
        ----------
        i_sim: int
        Numéro de la simulation sélectionnée pour le post-processing de solution

        Returns
        -------
        None
        """
        sim_name = self.get_sim_name()
        
        if only_num is False:
            
            figure, (num, ex) = plt.subplots(1, 2, figsize=(20, 6))
            cmap = cm.get_cmap('jet')
    
            # Titre de la figure
            title = f'Solution avec contours de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                    f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
            figure.suptitle(title)
    
            # Mise à l'échelle de la colorbar pour les 2 graphiques
            levels = np.linspace(np.min([self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact']]),
                                 np.max([self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact']]), num=20)
    
            # Solution numérique            
            c = num.tricontourf(self.data[i_sim]['position'][:, 0],
                                self.data[i_sim]['position'][:, 1],
                                self.data[i_sim]['phi_num'], levels=levels, cmap=cmap)
            plt.colorbar(c, ax=num)
            num.tricontour(self.data[i_sim]['position'][:, 0],
                           self.data[i_sim]['position'][:, 1],
                           self.data[i_sim]['phi_num'], '--', levels=levels, colors='k')
            num.set_xlabel("L (m)")
            num.set_ylabel("b (m)")
            num.set_title("Solution numérique")
    
            # Solution analytique
            c = ex.tricontourf(self.data[i_sim]['position'][:, 0],
                               self.data[i_sim]['position'][:, 1],
                               self.data[i_sim]['phi_exact'], levels=levels, cmap=cmap)
            plt.colorbar(c, ax=ex)
            ex.tricontour(self.data[i_sim]['position'][:, 0],
                          self.data[i_sim]['position'][:, 1],
                          self.data[i_sim]['phi_exact'], '--', levels=levels, colors='k')
            ex.set_xlabel("L (m)")
            ex.set_ylabel("b (m)")
            ex.set_title("Solution analytique analytique")
        
        else:
            figure, (num) = plt.subplots(1, 1, figsize=(20, 6))
            cmap = cm.get_cmap('jet')
    
            # Titre de la figure
            title = f'Solution avec contours de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                    f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
            figure.suptitle(title)
    
            # Mise à l'échelle de la colorbar pour les 2 graphiques
            levels = np.linspace(np.min([self.data[i_sim]['phi_num']]),
                                 np.max([self.data[i_sim]['phi_num']]), num=20)
    
            # Solution numérique            
            c = num.tricontourf(self.data[i_sim]['position'][:, 0],
                                self.data[i_sim]['position'][:, 1],
                                self.data[i_sim]['phi_num'], levels=levels, cmap=cmap)
            plt.colorbar(c, ax=num)
            num.tricontour(self.data[i_sim]['position'][:, 0],
                           self.data[i_sim]['position'][:, 1],
                           self.data[i_sim]['phi_num'], '--', levels=levels, colors='k')
            num.set_xlabel("L (m)")
            num.set_ylabel("b (m)")
            num.set_title("Solution numérique")
    

        save_path = f"images/{sim_name}_sim{i_sim}_contour.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()
        
#%% Affichage Solutions dans le plan: OK
    def show_plan_solutions(self, i_sim, x_coupe, y_coupe,only_num=False,plan_sortie=False):
        """
        Affiche les résultats selon un plan et x et un plan en y

        Parameters
        ----------
        i_sim: int
        Numéro de la simulation sélectionnée pour le post-processing de solution

        x_couple: float
        Position de la coupe en x

        y_coupe: float
        Position de la coupe en y

        Returns
        -------
        None
        """
        # Chercher l'indice des éléments à un X ou Y donné
        centres = self.data[i_sim]['position']
        elem_ds_coupeX, solution_coupeX, solutionEX_coupeX,x_coupe = \
            Coupe_X(centres, x_coupe, self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact'], 0,only_num,plan_sortie)
        elem_ds_coupeY, solution_coupeY, solutionEX_coupeY,y_coupe = \
            Coupe_X(centres, y_coupe, self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact'], 1,only_num,plan_sortie)
        
        sim_name = self.get_sim_name()
        
        if only_num is False:
            figure, (COUPEX, COUPEY) = plt.subplots(1, 2, figsize=(20, 6))
    
            # Titre de la figure
            title = f'Solution de plans de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                    f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
            figure.suptitle(title)
    
            COUPEX.plot(solution_coupeX, elem_ds_coupeX[:, 1], label="Solution numérique")
            COUPEX.plot(solutionEX_coupeX, elem_ds_coupeX[:, 1], '--', label="Solution analytique")
            COUPEX.set_xlabel("Vitesse")
            COUPEX.set_ylabel("Y (m)")
            COUPEX.set_title(f"Solution dans une coupe à X = {x_coupe}")
            COUPEX.legend()
    
            COUPEY.plot(elem_ds_coupeY[:, 0], solution_coupeY, label="Solution numérique")
            COUPEY.plot(elem_ds_coupeY[:, 0], solutionEX_coupeY, '--', label="Solution analytique")
            COUPEY.set_xlabel("X (m)")
            COUPEY.set_ylabel("Vitesse")
            COUPEY.set_title(f"Solution dans une coupe à Y = {y_coupe}")
            COUPEY.legend()
            
        else: 
            figure, (COUPEX, COUPEY) = plt.subplots(1, 2, figsize=(20, 6))
    
            # Titre de la figure
            title = f'Solution de plans de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                    f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
            figure.suptitle(title)
    
            COUPEX.plot(solution_coupeX, elem_ds_coupeX[:, 1], label="Solution numérique")
            COUPEX.set_xlabel("Vitesse")
            COUPEX.set_ylabel("Y (m)")
            COUPEX.set_title(f"Solution dans une coupe à X = {x_coupe}")
            COUPEX.legend()
    
            COUPEY.plot(elem_ds_coupeY[:, 0], solution_coupeY, label="Solution numérique")
            COUPEY.set_xlabel("X (m)")
            COUPEY.set_ylabel("Vitesse")
            COUPEY.set_title(f"Solution dans une coupe à Y = {y_coupe}")
            COUPEY.legend()

        # Enregistrer
        save_path = f"images/{sim_name}_sim{i_sim}_plans.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()
#%% Solutions sur Pyvista: Manque les contours !!!!
    def show_pyvista(self, i_sim, norm=True,only_num=False,isovalues=False):
        pv.set_plot_theme("document")
        
        isovalues_num=isovalues
        # Préparation du maillage
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self.data[i_sim]['mesh'])
        
        pv_mesh_num = pv.PolyData(nodes, elements)
        pv_num_nodes = pv_mesh_num.cell_data_to_point_data()
        
        pv_mesh_ex = pv.PolyData(nodes, elements)
        pv_ex_nodes = pv_mesh_ex.cell_data_to_point_data()


        pv_mesh_num['Vitesse numérique'] = self.data[i_sim]['phi_num']
        pv_mesh_ex['Vitesse analytique'] = self.data[i_sim]['phi_exact']
        
        #Pour tracer la solution numérique et Analytique
        if only_num is False:
            levels = [np.min(np.append(self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact'])),
                      np.max(np.append(self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact']))]
    
            # Création des graphiques
            pl = pv.Plotter(shape=(1, 2))  # Avant pvQt.BackgroundPlotter()
    
            # Solution numérique
            pl.add_text(f"Solution numérique (norme de la vitesse) \n {self.data[i_sim]['n']} éléments\n "
                        f"P = {self.data[i_sim]['P']}\n Méthode = {self.data[i_sim]['method']}", font_size=15)
            pl.add_mesh(pv_mesh_num, show_edges=True, scalars='Vitesse numérique', cmap="jet", clim=levels)
            
            #Contours Lines Num
            if isovalues is True:
                
                pv_num_nodes['field_num'] = self.data[i_sim]['phi_num']
                pv_num_nodes = pv_num_nodes.cell_data_to_point_data()
                contours_num = pv_num_nodes.contour(isosurfaces=15, scalars="field_num")
                pl.add_mesh(contours_num,color='white',show_scalar_bar=False, line_width=2)
        
            pl.camera_position = 'xy'
            pl.show_bounds()
    
            # Solution analytique
            pl.subplot(0, 1)
            pl.add_text(f"Solution analytique (norme de la vitesse) \n {self.data[i_sim]['n']} éléments\n "
                        f"P = {self.data[i_sim]['P']}\n Méthode = {self.data[i_sim]['method']}", font_size=15)
            pl.add_mesh(pv_mesh_ex, show_edges=True, scalars='Vitesse analytique', cmap="jet", clim=levels)
            
            #Contours Lines Ex
            if isovalues is True:
            
                pv_ex_nodes['field_exact'] = self.data[i_sim]['phi_exact']
                pv_ex_nodes = pv_ex_nodes.cell_data_to_point_data()
                contours_ex = pv_ex_nodes.contour(isosurfaces=15, scalars="field_exact")
                pl.add_mesh(contours_ex,color='white',show_scalar_bar=False, line_width=2)
        
            pl.camera_position = 'xy'
            pl.show_bounds()
            pl.link_views()
        
        #Pour tracer la solution numérique UNIQUEMENT
        else:
            levels = [np.min(self.data[i_sim]['phi_num']),np.max(self.data[i_sim]['phi_num'])]
    
            # Création des graphiques
            pl = pv.Plotter(shape=(1, 1))  # Avant pvQt.BackgroundPlotter()
    
            # Solution numérique
            pl.add_text(f"Solution numérique (norme de la vitesse) \n {self.data[i_sim]['n']} éléments\n "
                        f"P = {self.data[i_sim]['P']}\n Méthode = {self.data[i_sim]['method']}", font_size=15)
            pl.add_mesh(pv_mesh_num, show_edges=True, scalars='Vitesse numérique', cmap="jet", clim=levels)
            
            #Contours Lines Num
            if isovalues_num is True:
                pv_num_nodes['field_num'] = self.data[i_sim]['phi_num']
                pv_num_nodes = pv_num_nodes.cell_data_to_point_data()
                contours_num = pv_num_nodes.contour(isosurfaces=15, scalars="field_num")
                pl.add_mesh(contours_num,color='white',show_scalar_bar=False, line_width=2)
            
            pl.camera_position = 'xy'
            pl.show_bounds()
    
        sim_name = self.get_sim_name()
        save_path = f"images/{sim_name}_sim{i_sim}_pyvista.png"
        pl.show(screenshot=save_path)
        pl.clear()
        
#%% Affichage Erreur : Verification ou Validation de code (sans ou avec sol ana)
    def show_error(self,method):
        """
        Affichage des graphiques d'ordre de convergence et calcul de l'erreur par rapport au solution exacte.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if method=="VERIFICATION":
            # Calcul l'erreur l'ajoute aux données et détermine l'ordre de convergence
            for i in range(len(self.data)):
                total_area = np.sum(self.data[i]['area'])
                area, n = self.data[i]['area'], self.data[i]['n']
                phi_num, phi_exact = self.data[i]['phi_num'], self.data[i]['phi_exact']
                E_L2 = np.sqrt(np.sum(area*(phi_num - phi_exact)**2)/total_area)
                self.data[i]['err_L2'] = E_L2
                self.data[i]['h'] = np.sqrt(total_area/n)
        
        elif method=="VALIDATION":
            # Calcul l'erreur l'ajoute aux données et détermine l'ordre de convergence
            for i in range(len(self.data)-1):
                centres_gro = self.data[i]['position']
                centres_fin = self.data[i+1]['position']
                
                
                area, n = self.data[i]['area'], self.data[i]['n']
                
                elem_ds_sortie,phi_gro, areas_gro,x_coupe = \
                    Coupe_X(centres_gro, 0, self.data[i]['phi_num'], area, 0,False,True)
                
                total_area = np.sum(areas_gro)
                
                elem_ds_sortie,phi_fin, phi_useless,x_coupe = \
                    Coupe_X(centres_fin, 0, self.data[i+1]['phi_num'], self.data[i+1]['phi_num'], 0,False,True)
                

                phi_fin=phi_fin[::2] #On skip un élément sur deux

                E_L2 = np.sqrt(np.sum(areas_gro*(phi_fin - phi_gro)**2)/total_area)
                self.data[i]['err_L2'] = E_L2
                self.data[i]['h'] = np.sqrt(total_area/n)

        p = np.polyfit(np.log([self.data[i]['h'] for i in range(len(self.data)-1)]),
                  np.log([self.data[i]['err_L2'] for i in range(len(self.data)-1)]), 1)

        # Graphique de l'erreur
        fig_E, ax_E = plt.subplots(figsize=(15, 10))
        fig_E.suptitle("Normes de l'erreur L² des solutions numériques sur une échelle de logarithmique, Méthode: "+method +" de code", y=0.925)
        text = AnchoredText('Ordre de convergence: ' + str(round(p[0], 2)), loc='upper left')

        ax_E.loglog([self.data[i]['h'] for i in range(len(self.data)-1)],
                  [self.data[i]['err_L2'] for i in range(len(self.data)-1)], '.-')
        ax_E.minorticks_on()
        ax_E.grid(True, which="both", axis="both", ls="-")
        ax_E.set_xlabel('Grandeur (h)')
        ax_E.set_ylabel('Erreur (E)')
        ax_E.add_artist(text)

        # Enregistrer
        save_path = f"images/{self.sim_name}_error.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()

