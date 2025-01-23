#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np


class GradientDescent():
    """Descent gradient class with regularize technique

    Parameters
    ----------
    regularize : bool
        If True, the regularization is used.
    bias : bool
        If the True, a bias is added to the features.
    alpha : float > 0
        Coefficient for the step when updating the parameters.

    Notes
    -----
    This class aims at computing the parameters of a linear model using
    a descent gradient method with or without regularization.
    """

    def __init__(self, regularize=True, bias=True, alpha=3e-9, regularization='none', lmb=0.1, alpha_net=0.5):
        """
        Parameters
        ----------
        regularize : bool
            Si True, active la régularisation.
        bias : bool
            Si True, ajoute un biais aux caractéristiques.
        alpha : float
            Taux d'apprentissage.
        regularization : str
            Type de régularisation ('none', 'ridge', 'lasso', 'elasticnet').
        lmb : float
            Coefficient de régularisation.
        alpha_net : float
            Poids pour Elastic Net entre L1 et L2.
        """
        self.bias = bias
        self.alpha = alpha
        self.regularize = regularize
        self.regularization = regularization
        self.lmb = lmb
        self.alpha_net = alpha_net
        self.epsilon = 1e-10 if regularize else 1e-5


    def predict(self, new_features):
        """Make predictions using the result of the gradient descent

        Parameters
        ----------
        new_features : 2d sequence of float
            The feature for which to predict the labels.

        Returns
        -------
        predicted_labels : 2d sequence of float
            The predicted labels

        Notes
        -----
        The method fit must be called first.
        """

        if self.bias:
            new_features = self._add_bias(new_features)
        return self.hypothesis(new_features, self.parameters_)

    def fit(self, features, label, parameters=None):
        """Find the optimal parameters

        Parameters
        ----------
        features : 2d sequence of float
            The input parameters.
        label : 2d sequence of float
            The output parameters
        parameters : 2d sequence of float
            The initial guess for the descent gradient.
        """
        
        # Ajouter le biais si nécessaire
        if self.bias:
            features = self._add_bias(features)

        # Initialisation des paramètres aléatoires si non fournis
        if parameters is None:
            n = features.shape[1]
            parameters = np.random.rand(n, 1)

        # Calcul initial des prédictions
        predictions = self.hypothesis(features, parameters)

        # Choisir la méthode d'entraînement
        if self.regularize:
            self.parameters_ = self._regularize_fit(features, label, parameters, predictions)
        else:
            self.parameters_ = self._classic_fit(features, label, parameters, predictions)

            

    def _classic_fit(self, features, label, parameters, predictions):
        """Find the optimal parameters with classical method
        """

        costFct = 0
        costFctEvol = []
        count = 0
        # On utilise une boucle while
        while self.testCostFct(predictions, label, costFct, self.epsilon):
            count += 1
            costFct = self.costFunction(predictions, label)
            grads = self.gradients(predictions, label, features)
            parameters = self.updateParameters(parameters, grads, self.alpha)
            predictions = self.hypothesis(features, parameters)
            if count % 1000 == 0:
                print('%3i : cost function = {}'.format(costFct) % count)
            costFctEvol.append(costFct)
        print("\nFinish: {} steps, cost function = {}".format(count, costFct))
        return parameters

    def _regularize_fit(self, features, label, parameters, predictions, max_iterations=10000):
        """Find the optimal parameters with regularized method."""
        costFct = 0
        costFctEvol = []
        count = 0

        while self.testRegCostFct(predictions, label, self.lmb, parameters, costFct, self.epsilon) and count < max_iterations:
            count += 1

            # Sélection de la régularisation
            if self.regularization == 'ridge':
                costFct = self.regCostFunction_Ridge(predictions, label, self.lmb, parameters)
            elif self.regularization == 'lasso':
                costFct = self.regCostFunction_Lasso(predictions, label, self.lmb, parameters)
            elif self.regularization == 'elasticnet':
                costFct = self.regCostFunction_ElasticNet(predictions, label, self.lmb, parameters, self.alpha_net)
            else:  # Pas de régularisation
                costFct = self.costFunction(predictions, label)

            # Calcul des gradients
            grads = self.regGradients(predictions, label, features, self.lmb, parameters)
            parameters = self.updateParameters(parameters, grads, self.alpha)
            predictions = self.hypothesis(features, parameters)

            if count % 1000 == 0:
                print(f'{count} : cost function = {costFct}')
            costFctEvol.append(costFct)

        print(f"\nFinish: {count} steps, cost function = {costFct}")
        self.costFctEvol = costFctEvol  # Pour traçage du graphe
        return parameters



    def _add_bias(self, features):
        """Add bias column (1 vector)
        """
        bias = np.ones(features.shape[0])
        return np.column_stack([features, bias])

    def hypothesis(self, x, theta):
        """Compute our hypothesis model (linear regression), use a fonction:
        """
        return np.dot(x, theta)

    def costFunction(self, yhat, y):
        """Fonction de coût
        """
        return np.square(yhat - y).sum() / (2*y.shape[0])

    def regCostFunction(self, yhat, y, lmb, theta):
        """Fonction de coût régularisée
        """
        return self.costFunction(yhat, y) + lmb/(2*y.shape[0]) * np.square(theta).sum()
    
    def regCostFunction_Ridge(self, yhat, y, lmb, theta):
        """Fonction de coût avec régularisation Ridge (L2).
        
        Parameters
        ----------
        yhat : ndarray
            Prédictions du modèle.
        y : ndarray
            Valeurs réelles.
        lmb : float
            Coefficient de régularisation.
        theta : ndarray
            Coefficients du modèle.
            
        Returns
        -------
        float
            Valeur de la fonction coût régularisée.
        """
        regularization_term = lmb / (2 * y.shape[0]) * np.sum(theta ** 2)
        return self.costFunction(yhat, y) + regularization_term
    
    def regCostFunction_ElasticNet(self, yhat, y, lmb, theta, alpha=0.5):
        """Fonction de coût avec régularisation Elastic Net (combinaison L1 et L2).
        
        Parameters
        ----------
        yhat : ndarray
            Prédictions du modèle.
        y : ndarray
            Valeurs réelles.
        lmb : float
            Coefficient de régularisation.
        theta : ndarray
            Coefficients du modèle.
        alpha : float
            Poids entre L1 et L2 (0 <= alpha <= 1).
            
        Returns
        -------
        float
            Valeur de la fonction coût régularisée.
        """
        l1_term = alpha * lmb / y.shape[0] * np.sum(np.abs(theta))
        l2_term = (1 - alpha) * lmb / (2 * y.shape[0]) * np.sum(theta ** 2)
        return self.costFunction(yhat, y) + l1_term + l2_term
    
    
    
    def regCostFunction_Lasso(self, yhat, y, lmb, theta):
            """
        Calcule la fonction de coût avec régularisation Lasso (L1).

        

        Parameters
        ----------
        yhat : ndarray
            Prédictions du modèle pour les échantillons donnés.
            Taille : (m, 1), où `m` est le nombre d'échantillons.

        y : ndarray
            Valeurs réelles correspondant aux prédictions `yhat`.
            Taille : (m, 1), où `m` est le nombre d'échantillons.

        lmb : float
            Coefficient de régularisation, contrôle l'importance du terme 
            de régularisation par rapport à l'erreur quadratique.
            Plus `lmb` est grand, plus les coefficients sont pénalisés.

        theta : ndarray
            Vecteur des paramètres du modèle (poids) à pénaliser.
            Taille : (n, 1), où `n` est le nombre de caractéristiques.

        Returns
        -------
        float
            Valeur de la fonction coût régularisée, calculée comme la 
            somme de l'erreur quadratique moyenne et du terme de 
            régularisation L1."""

            regularization_term = lmb / y.shape[0] * np.sum(np.abs(theta))
            return self.costFunction(yhat, y) + regularization_term




    def gradients(self, yhat, y, x):
        """Dérivée de la fonction de coût == gradients
        """
        return (((yhat - y) * x).sum(axis=0) / x.shape[0]).reshape(x.shape[1],1)

    def regGradients(self, yhat, y, x, lmb, theta):
        """Dérivée de la fonction de coût regularisée
        """
        return (((yhat - y) * x).sum(axis=0) / x.shape[0]).reshape(x.shape[1],1) + lmb/x.shape[0]*theta

    def updateParameters(self, parameters, grads, alpha):
        """Gradient descent: mise à jour des paramètres
        """
        return parameters - alpha * grads

    def testCostFct(self, yhat, y, prevCostFct, epsilon):
        """ Fonction pour tester l'évolution de la fonction de coût: vrai = continuer la descente de gradient
        """
        return np.abs(self.costFunction(yhat, y) - prevCostFct) >= epsilon*prevCostFct

    def testRegCostFct(self, yhat, y, lmb, theta, prevCostFct, epsilon):
        """ Fonction pour tester l'évolution de la fonction de coût régularisée

            Returns
            -------
            test : bool
                vrai = continuer la descente de gradient
        """
        return np.abs(self.regCostFunction(yhat, y, lmb, theta) - prevCostFct) >= epsilon*prevCostFct
    
    def score(self, y_true, y_pred):
        """Calcul du coefficient de détermination R^2

            Parameters
        ----------
            y_true : array-like
            Valeurs réelles.
        y_pred : array-like
            Valeurs prédites par le modèle.

        Returns
        -------
        r2_score : float
            Le coefficient de détermination.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score
    
    import matplotlib.pyplot as plt

    def plot(self, costFctEvol):
        """Affiche le graphe de convergence de la fonction coût.

        Parameters
        ----------
        costFctEvol : list of float
            Liste contenant l'évolution de la fonction coût au cours des itérations.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(costFctEvol, label='Fonction Coût')
        plt.xlabel('Itérations')
        plt.ylabel('Valeur de la Fonction Coût')
        plt.title('Convergence de la Fonction Coût')
        plt.legend()
        plt.grid(True)
        plt.show()

    import matplotlib.pyplot as plt
    
    def get_weights(self):
            """
            Retourne les poids (paramètres) du modèle après l'entraînement.

            Returns
            -------
            ndarray
                Les poids du modèle (y compris le biais s'il est activé).
            """
            return self.parameters_


    def plot_weights(self, feature_names=None):
        """
        Affiche un graphe des poids des caractéristiques après l'entraînement.

        Parameters
        ----------
        feature_names : list of str, optional
            Liste des noms des caractéristiques (colonnes de X). Si None, utilise des indices numériques.

        Notes
        -----
        Cette méthode suppose que le modèle a déjà été entraîné.
        """
        # Vérifier que les paramètres sont définis
        if not hasattr(self, 'parameters_'):
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez la méthode `fit` avant d'utiliser `plot_weights`.")

        # Exclure le biais (le dernier poids)
        weights = self.parameters_.flatten()
        if self.bias:
            weights = weights[:-1]  # Exclure le biais

        # Générer des noms pour les caractéristiques si non fournis
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(weights))]

        # Créer le graphe en barre
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, weights, color='blue', alpha=0.7)
        plt.xlabel("Caractéristiques")
        plt.ylabel("Poids")
        plt.title("Graphe des Poids des Caractéristiques")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

  




    
    