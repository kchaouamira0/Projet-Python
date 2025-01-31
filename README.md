# Projet-Python

GradientPy - Régression linéaire par descente de gradient
GradientPy est un projet implémentant un algorithme de descente de gradient en Python avec support de la régularisation (Lasso, Ridge, ElasticNet). Ce projet comprend :

🔹 Module Python (GradientDescent)
Réalise une régression linéaire via descente de gradient.
Supporte plusieurs méthodes de régularisation pour améliorer la robustesse et limiter le sur-apprentissage.
Offre des fonctionnalités analytiques : score R², visualisation de la fonction coût, graphe des poids.
🔹 Prétraitement des données (DataPreprocessor)
Normalisation des données pour garantir une échelle cohérente.
Gestion des valeurs aberrantes (outliers) en limitant leur impact.
🔹 Application sur un Dataset de Salaire
Utilisation du module sur un jeu de données "Years of Experience vs Salary".
Comparaison des performances avec et sans régularisation.
Visualisations : pairplot, courbe de convergence, analyse des résultats.
🔹 Documentation Automatique (Sphinx)
Chaque classe et méthode est documentée via des docstrings détaillées.
Génération d'une documentation HTML facilitant l'utilisation du module.
