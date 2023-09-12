import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, r2_score

# Exemple de données (en utilisant une partie de vos données d'origine)

df = pd.read_csv('Datasets/DataTP2.csv', sep=";")

ColumnX = ['APP_Libelle_etablissement','filtre','Code_postal','ods_type_activite']

# Nétoyage et transformation des données
#df = df.dropna()
df['Code_postal'] = df['Code_postal'].str.slice(0, 2)
df['APP_Libelle_etablissement'] = df['APP_Libelle_etablissement'].fillna('')
df['filtre'] = df['filtre'].fillna('')
df['filtre'] = df['filtre'].fillna('ods_type_activite')
df['Code_postal'] = df['Code_postal'].fillna(34)

print(f"Taille de l'échantillon: {df.shape[0]}")


# Séparation des caractéristiques (X) et de la variable cible (y)
X = df[ColumnX]
y = df['Synthese_eval_sanit']

# Encodage des variables catégorielles en utilisant one-hot encoding
X = pd.get_dummies(X, drop_first=True)


# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Création du modèle de classification multinomiale (Naïve Bayes)
model = MultinomialNB()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=0)
print(f"Précision du modèle : {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
