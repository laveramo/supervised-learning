#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger le jeu de données
df = pd.read_csv('./alt_acsincome_ca_features_85(1).csv')
dl = pd.read_csv('./alt_acsincome_ca_labels_85.csv')
print(df.head())
print("donnes de labels")
print(dl.head())

# Afficher des informations générales sur le jeu de données
print("\nInformations générales sur le jeu de données (features) :")
print(df.info())

print("\nInformations générales sur le jeu de données (labels) :")
print(dl.info())


# Afficher des statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Afficher la distribution des types de colonnes
print("\nDistribution des types de colonnes :")
print(df.dtypes.value_counts())

# Partitionner le jeu de données en ensemble d'entraînement et de test
# Supposons que la dernière colonne soit la cible (y)
X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière (features)
y = dl.iloc[:, -1]   # Dernière colonne (target)

# Mélanger et diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Afficher les tailles des ensembles
print("\nTailles des ensembles après partitionnement :")
print(f"Ensemble d'entraînement : {X_train.shape[0]} exemples")
print(f"Ensemble de test : {X_test.shape[0]} exemples")

# Vérifier un aperçu des ensembles
print("\nAperçu de l'ensemble d'entraînement (features) :")
print(X_train.head())
print("\nAperçu de l'ensemble de test (features) :")
print(X_test.head())
