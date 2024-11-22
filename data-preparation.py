#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Pour sauvegarder le scaler

# Charger les jeux de données
features_file = './alt_acsincome_ca_features_85(1).csv'
labels_file = './alt_acsincome_ca_labels_85.csv'

try:
    # Lecture des données
    df = pd.read_csv(features_file)
    dl = pd.read_csv(labels_file)
    print("Chargement des données réussi !")
except FileNotFoundError as e:
    print(f"Erreur : {e}")
    exit()

# Affichage des aperçus des données
print("\nAperçu des données (features) :")
print(df.head())
print("\nAperçu des données (labels) :")
print(dl.head())

# Informations générales
print("\nInformations générales sur les données (features) :")
print(df.info())
print("\nInformations générales sur les données (labels) :")
print(dl.info())

# Statistiques descriptives
print("\nStatistiques descriptives (features) :")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes dans les features :")
print(df.isnull().sum())

print("\nValeurs manquantes dans les labels :")
print(dl.isnull().sum())

# Identifier les colonnes numériques
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("\nColonnes numériques à standardiser :")
print(numerical_columns)

# Partitionner le jeu de données
# Associer les labels à `y` et les features à `X`
X = df.copy()  # Features
y = dl.iloc[:, 0]  # Cible (labels)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("\nPartitionnement réussi !")
print(f"Ensemble d'entraînement : {X_train.shape[0]} exemples")
print(f"Ensemble de test : {X_test.shape[0]} exemples")

# Standardiser les colonnes numériques
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Vérifier les données standardisées
print("\nAperçu des données standardisées (Ensemble d'entraînement) :")
print(X_train_scaled.head())

print("\nAperçu des données standardisées (Ensemble de test) :")
print(X_test_scaled.head())

# Sauvegarder le scaler pour réutilisation
scaler_file = 'scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"\nScaler sauvegardé dans le fichier '{scaler_file}'.")

# Sauvegarder les ensembles préparés
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nDonnées préparées et sauvegardées avec succès !")
