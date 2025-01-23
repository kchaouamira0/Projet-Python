import numpy as np

from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Classe pour gérer la normalisation et le traitement des valeurs aberrantes en utilisant StandardScaler.
    """

    def __init__(self):
        # Initialisation des scalers pour les caractéristiques et les cibles
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()

    def normalize(self, features, target=None, is_training=True):
        """
        Normalise les caractéristiques et la cible en utilisant StandardScaler.

        Parameters
        ----------
        features : ndarray
            Matrice des caractéristiques.
        target : ndarray, optional
            Vecteur cible (peut être None si seulement les caractéristiques doivent être normalisées).
        is_training : bool
            Indique si les scalers doivent être ajustés (pour l'entraînement) ou réutilisés (pour le test).

        Returns
        -------
        normalized_features : ndarray
            Matrice des caractéristiques normalisées.
        normalized_target : ndarray, optional
            Vecteur cible normalisé (si la cible est fournie).
        """
        if is_training:
            # Ajuste et transforme les données d'entraînement
            normalized_features = self.scaler_features.fit_transform(features)
            if target is not None:
                # Reshape de la cible pour être compatible avec StandardScaler
                target = target.reshape(-1, 1)
                normalized_target = self.scaler_target.fit_transform(target)
                return normalized_features, normalized_target
            return normalized_features
        else:
            # Transforme uniquement (utilisé pour les données de test)
            normalized_features = self.scaler_features.transform(features)
            if target is not None:
                target = target.reshape(-1, 1)
                normalized_target = self.scaler_target.transform(target)
                return normalized_features, normalized_target
            return normalized_features

    def handle_outliers(self, features, threshold=3):
        """
        Gère les valeurs aberrantes en limitant les valeurs à un seuil.
        
        Parameters
        ----------
        features : ndarray
            Matrice des caractéristiques.
        threshold : float, optional
            Seuil (en écart-types).
        Returns
        -------
        ndarray
            Matrice traitée.
        """
        z_scores = (features - features.mean(axis=0)) / features.std(axis=0)
        return np.clip(z_scores, -threshold, threshold)
