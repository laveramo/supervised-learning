#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.inspection import permutation_importance
import numpy as np
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



# Modèles par défaut
models = {
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm="SAMME"),
    'GradientBoosting': GradientBoostingClassifier()
}


# Évaluation avec validation croisée
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f'{name} - Accuracy: {np.mean(scores):.4f}')

# Recherche des meilleurs hyperparamètres avec GridSearchCV pour RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Enregistrer le meilleur modèle
joblib.dump(grid_search.best_estimator_, 'RandomForest_BestModel.joblib')

# Rapport de classification
y_pred = grid_search.best_estimator_.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Matrice de confusion
print(confusion_matrix(y_test, y_pred))


# Importance des attributs avec permutation importance
result = permutation_importance(grid_search.best_estimator_, X_test_scaled, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean
for i, val in enumerate(importance):
    print(f'Attribut {i} - Importance: {val:.4f}')

# Équité : Comparaison des taux de prédictions positives
y_pred = grid_search.best_estimator_.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matrice de confusion :\n', conf_matrix)

# Analyser les biais selon le genre
# Supposer que la colonne 'gender' existe et contient les valeurs 'homme' et 'femme'
gender = X_test['SEX']
# Analyse des corrélations initiales
correlations = df.corr()
print("\nCorrélations initiales :")
print(correlations)

# Corrélations après entraînement (importance des attributs)
model_importance = grid_search.best_estimator_.feature_importances_
for feature, importance in zip(X.columns, model_importance):
    print(f"Importance de {feature}: {importance:.4f}")

# Ajout des métriques d'équité
def calculate_equity_metrics(y_true, y_pred, sensitive_feature):
    equity_metrics = {}
    for group in sensitive_feature.unique():
        group_indices = sensitive_feature[sensitive_feature == group].index
        group_y_true = y_true.loc[group_indices]
        group_y_pred = y_pred[group_indices]

        tp = sum((group_y_true == 1) & (group_y_pred == 1))
        fp = sum((group_y_true == 0) & (group_y_pred == 1))
        tn = sum((group_y_true == 0) & (group_y_pred == 0))
        fn = sum((group_y_true == 1) & (group_y_pred == 0))
        
        equity_metrics[group] = {
            "Statistical Parity": (tp + fp) / len(group_y_true) if len(group_y_true) > 0 else 0,
            "Equal Opportunity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Predictive Equality": fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
    return equity_metrics


# Supposer que la colonne 'gender' est présente
gender_column = 'SEX'
y_test_pred = grid_search.best_estimator_.predict(X_test_scaled)
equity_results = calculate_equity_metrics(y_test, y_test_pred, X_test[gender_column])
print("\nMétriques d'équité par genre :")
print(equity_results)

# Réentraînement sans la colonne "genre"
X_train_no_gender = X_train_scaled.drop(columns=[gender_column])
X_test_no_gender = X_test_scaled.drop(columns=[gender_column])
grid_search_no_gender = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search_no_gender.fit(X_train_no_gender, y_train)

# Évaluer les biais après suppression de la colonne "genre"
y_test_pred_no_gender = grid_search_no_gender.best_estimator_.predict(X_test_no_gender)
equity_results_no_gender = calculate_equity_metrics(y_test, y_test_pred_no_gender, X_test[gender_column])
print("\nMétriques d'équité après suppression du genre :")
print(equity_results_no_gender)


# Charger les données des états voisins
nevada_features = pd.read_csv('./nevada_features.csv')
nevada_labels = pd.read_csv('./nevada_labels.csv')
colorado_features = pd.read_csv('./colorado_features.csv')
colorado_labels = pd.read_csv('./colorado_labels.csv')

# Standardiser les données
nevada_features_scaled = scaler.transform(nevada_features)
colorado_features_scaled = scaler.transform(colorado_features)

# Prédictions sur les états voisins
nevada_predictions = grid_search.best_estimator_.predict(nevada_features_scaled)
colorado_predictions = grid_search.best_estimator_.predict(colorado_features_scaled)

# Analyse des résultats
print("\nPrédictions pour le Nevada :")
print(classification_report(nevada_labels, nevada_predictions))
print("\nPrédictions pour le Colorado :")
print(classification_report(colorado_labels, colorado_predictions))


