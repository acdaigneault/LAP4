"""
Date :    8 février 2022
Auteurs : Audrey Collard-Daigneault (1920374) & Mohamad Karim Zayni (2167132)
Utilité : Effectuer le post-traitement des données
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pyvista as pv
import pyvistaqt as pvQt
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter


def Coupe_X(Coordonnees, X, Solution, Analytique, Plan):
    Elements_ds_coupe = []
    Solution_coupe = []
    Analytique_coupe = []
    eps = 1e-6  # Précision
    for i in range(len(Coordonnees)):
        if np.abs(Coordonnees[i, Plan] - X) < eps:
            Elements_ds_coupe.append(Coordonnees[i, :])
            Solution_coupe.append(Solution[i])
            Analytique_coupe.append(Analytique[i])
    Elements_ds_coupe = np.array(Elements_ds_coupe)
    Solution_coupe = np.array(Solution_coupe)
    return Elements_ds_coupe, Solution_coupe, Analytique_coupe

class PostProcessing:
    """
    Effectuer le post_traitement après la résolution du problème.
    Parmeters
    ---------
    exempmle: case
        L'exemple en cours de traitement
    Attributes
    ----------
    data: dictionniare
        Dictionnaire qui comporte toutes les données de nécessaire
        pour effectuer le post-traitement.
    """
    def __init__(self):
        self.data = []          # Initialisation du dictionnaire de données

    # Ajoute les données selon un nombre d'éléments'
    def set_data(self, mesh, solutions, preprocessing_data, simulation_paramaters):
        """
        Modifier les données de l'exemple traité.
        Parameters
        ----------
        case: case
            Exemple Traité.
        Returns
        -------
        None
        """

        # Calcule les normes de vitesse
        phi_num, phi_ex = np.zeros(len(solutions[0][0])), np.zeros(len(solutions[0][0]))
        for i in range(len(solutions[0][0])):
            phi_num[i] = np.sqrt(solutions[0][0][i]**2 + solutions[0][1][i]**2)
            phi_ex[i] = np.sqrt(solutions[1][0][i]**2 + solutions[1][1][i]**2)

        self.data.append({'n': mesh.get_number_of_elements(),
                          'mesh': mesh,
                          'u_num': solutions[0][0], 'u_exact': solutions[1][0],
                          'v_num': solutions[0][1], 'v_exact': solutions[1][1],
                          'phi_num': phi_num,
                          'phi_exact': phi_ex,
                          'area': preprocessing_data[0],
                          'position': preprocessing_data[1],
                          'P': simulation_paramaters['P'],
                          'method': simulation_paramaters['method']})




    # Génère les graphiques des solutions numérique et analytique
    def show_solutions(self, i_mesh, title, save_path):
        """
        Affichage des graphiques qui montrent la différence entre la solution
        numérique et la solution analytique
        Parameters
        ----------
        i_mesh: int
            Maillage de l'exemple traité.

        title: str
            Nom du document pour le sauvegarder

        save_path: str
            Nom du fichier de sauvegarde
        Returns
        -------
        None
        """
        Figure1, (NUM, EX) = plt.subplots(1, 2, figsize=(20, 8))

        Figure1.suptitle(f"{title} avec {self.data[i_mesh]['n']} éléments pour P = {self.data[i_mesh]['P']} utilisant une méthode {self.data[i_mesh]['method']}")

        # Set levels of color for the colorbar
        levels = np.linspace(np.min([self.data[i_mesh]['u_num'], self.data[i_mesh]['u_exact']]),
                             np.max([self.data[i_mesh]['u_num'], self.data[i_mesh]['u_exact']]), num=30)

        # Solution numérique
        c = NUM.tricontourf(self.data[i_mesh]['position'][:, 0],
                            self.data[i_mesh]['position'][:, 1],
                            self.data[i_mesh]['u_num'], levels=levels)
        plt.colorbar(c, ax=NUM)
        NUM.set_xlabel("L (m)")
        NUM.set_ylabel("H (m)")
        NUM.set_title("Solution numérique")

        # Solution analytique/MMS
        c = EX.tricontourf(self.data[i_mesh]['position'][:, 0],
                           self.data[i_mesh]['position'][:, 1],
                           self.data[i_mesh]['u_exact'], levels=levels)
        plt.colorbar(c, ax=EX)
        EX.set_xlabel("L (m)")
        EX.set_ylabel("H (m)")
        EX.set_title("Solution analytique MMS/analytique")

        plt.savefig(save_path, dpi=200)

    def show_plan_solutions(self, i_mesh, title, save_path, X_Coupe, Y_Coupe):

        """
        Affichage des graphiques qui montrent les résultats dans des coupes
        en X ou en Y
        Parameters
        ----------
        i_mesh: mesh
            Maillage de l'exemple traité.

        title: str
            Nom du document pour le sauvegarder

        save_path: str
            Nom du fichier de sauvegarde

        X_coupe: float
            L'endroit de la coupe suivant la droite X=X_coupe
        Y_coupe: float
            L'endroit de la coupe suivant la droite Y=Y_coupe
        Returns
        -------
        None
        """
        # Chercher l'indice des éléments à un X ou Y donné


        Figure1, (COUPEX, COUPEY) = plt.subplots(1, 2, figsize=(20, 6))

        Figure1.suptitle(f"{title} avec {self.data[i_mesh]['n']} éléments pour P = {self.data[i_mesh]['P']} utilisant une méthode {self.data[i_mesh]['method']}")

        Centres = self.data[i_mesh]['position']

        Elem_ds_coupeX, Solution_coupeX, SolutionEX_coupeX = \
            Coupe_X(Centres, X_Coupe, self.data[i_mesh]['u_num'], self.data[i_mesh]['u_exact'], 0)
        Elem_ds_coupeY, Solution_coupeY, SolutionEX_coupeY = \
            Coupe_X(Centres, Y_Coupe, self.data[i_mesh]['u_num'], self.data[i_mesh]['u_exact'], 1)

        COUPEX.plot(Solution_coupeX, Elem_ds_coupeX[:, 1], label="Solution Numérique")
        COUPEX.plot(SolutionEX_coupeX, Elem_ds_coupeX[:, 1], '--', label="Solution MMS")
        COUPEX.set_xlabel("Vitesse")
        COUPEX.set_ylabel("Y (m)")
        COUPEX.set_title(f"Solution dans une coupe à X = {X_Coupe}")
        COUPEX.legend()

        COUPEY.plot(Elem_ds_coupeY[:, 0], Solution_coupeY, label="Solution Numérique")
        COUPEY.plot(Elem_ds_coupeY[:, 0], SolutionEX_coupeY, '--', label="Solution MMS/analytique")
        COUPEY.set_xlabel("X (m)")
        COUPEY.set_ylabel("Vitesse")
        COUPEY.set_title(f"Solution dans une coupe à Y = {Y_Coupe}")
        COUPEY.legend()
        plt.show(block=False)

        # Enregistrer
        plt.savefig(save_path, dpi=200)

    def show_pyvista(self, i_mesh):
        # Préparation du maillage
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self.data[i_mesh]['mesh'])
        pv_mesh = pv.PolyData(nodes, elements)

        # Solutions numériques et analytiques
        pv_mesh['Vitesse u numérique'] = self.data[i_mesh]['u_num']
        pv_mesh['Vitesse u analytique'] = self.data[i_mesh]['u_exact']

        # Création des graphiques
        pl = pv.Plotter(shape=(1, 2))  # Avant pvQt.BackgroundPlotter()
        pl.subplot(0, 0)
        pl.add_text(f"Solution numérique\n {self.data[i_mesh]['n']} éléments\n "
                    f"P = {self.data[i_mesh]['P']}\n Méthode = {self.data[i_mesh]['method']}", font_size=15)
        pl.add_mesh(pv_mesh, show_edges=True, scalars='Vitesse u numérique', cmap="RdBu")
        pl.camera_position = 'xy'
        pl.show_bounds()

        pl.subplot(0, 1)
        pl.add_text(f"Solution analytique\n {self.data[i_mesh]['n']} éléments\n "
                    f"P = {self.data[i_mesh]['P']}\n Méthode = {self.data[i_mesh]['method']}", font_size=15)
        pl.add_mesh(pv_mesh, show_edges=True, scalars='Vitesse u analytique', cmap="RdBu")
        pl.camera_position = 'xy'
        pl.show_bounds()

        pl.link_views()
        pl.show()

    def show_mesh_differences(self, i_mesh1, i_mesh2, title, save_path, diff=False):
        """
        Affichage des graphiques qui montrent les résultats entre deux types
        de maillage
        Parameters
        ----------
        i_mesh1: mesh
            Maillage 1 de l'exemple traité.

        i_mesh2: mesh
            Maillage 2 de l'exemple traité.

        title: str
            Nom du document pour le sauvegarder

        save_path: str
            Nom du fichier de sauvegarde

        diff: Bool
            Pour décider si on trace la différence (Erreur)  entre les deux maillages.
        Returns
        -------
        None
        """
        if diff is True:
            figure, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(28, 6))
        else:
            figure, (plot1, plot2) = plt.subplots(1, 2, figsize=(20, 6))

        figure.suptitle(title)
        # Set levels of color for the colorbar
        levels = np.linspace(np.min(np.append(self.data[i_mesh1]['phi_num'], self.data[i_mesh2]['phi_num'])),
                             np.max(np.append(self.data[i_mesh1]['phi_num'], self.data[i_mesh2]['phi_num'])), num=40)

        center1 = self.data[i_mesh1]['position']
        c = plot1.tricontourf(center1[:, 0], center1[:, 1], self.data[i_mesh1]['phi_num'], levels=levels)
        plot1.set_xlabel("L (m)")
        plot1.set_ylabel("H (m)")
        plot1.set_title(f"Mesh à {self.data[i_mesh1]['n']} éléments, P = {self.data[i_mesh1]['P']}, méthode = {self.data[i_mesh1]['method']}")
        plt.colorbar(c, ax=plot1)

        center2 = self.data[i_mesh2]['position']
        c = plot2.tricontourf(center2[:, 0], center2[:, 1], self.data[i_mesh2]['phi_num'], levels=levels)
        plot2.set_xlabel("L (m)")
        plot2.set_ylabel("H (m)")
        plot2.set_title(f"Mesh à {self.data[i_mesh2]['n']} éléments, P = {self.data[i_mesh2]['P']}, méthode = {self.data[i_mesh2]['method']}")
        plt.colorbar(c, ax=plot2)

        if diff is True:
            err = np.abs(self.data[i_mesh1]['phi_num'] - self.data[i_mesh2]['phi_num'])

            levels = np.linspace(np.min(err), np.max(err), num=40)

            c = plot3.tricontourf(center1[:, 0], center1[:, 1], err, levels=levels)
            plot3.set_xlabel("L (m)")
            plot3.set_ylabel("H (m)")
            plot3.set_title(f"Erreur absolue entre les maillages à {self.data[i_mesh1]['n']} éléments")
            plt.colorbar(c, ax=plot3)


        # Enregistrer
        plt.savefig(save_path, dpi=200)

    def show_error(self):
        """
        Affichage des graphiques d'ordre de convergence et calcul de l'erreur
        par rapport au solution exacte.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calcul l'erreur (en x), l'ajoute aux données et détermine l'ordre de convergence
        for i in range(len(self.data)):
            total_area = np.sum(self.data[i]['area'])
            area, n = self.data[i]['area'], self.data[i]['n']
            phi_num, phi_exact = self.data[i]['phi_num'], self.data[i]['phi_exact']
            E_L2 = np.sqrt(np.sum(area*(phi_num - phi_exact)**2)/total_area)
            self.data[i]['err_L2'] = E_L2
            self.data[i]['h'] = np.sqrt(total_area/n)

        p = np.polyfit(np.log([self.data[i]['h'] for i in range(len(self.data))]),
                  np.log([self.data[i]['err_L2'] for i in range(len(self.data))]), 1)

        # Graphique de l'erreur
        fig_E, ax_E = plt.subplots(figsize=(15, 10))
        fig_E.suptitle("Normes de l'erreur L² des solutions numériques sur une échelle de logarithmique", y=0.925)
        text = AnchoredText('Ordre de convergence: ' + str(round(p[0], 2)), loc='upper left')

        ax_E.loglog([self.data[i]['h'] for i in range(len(self.data))],
                  [self.data[i]['err_L2'] for i in range(len(self.data))], '.-')
        ax_E.minorticks_on()
        ax_E.grid(True, which="both", axis="both", ls="-")
        ax_E.set_xlabel('Grandeur (h)')
        ax_E.set_ylabel('Erreur (E)')
        ax_E.add_artist(text)

        plt.show()