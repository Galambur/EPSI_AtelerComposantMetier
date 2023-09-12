import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

## TODO : Changer pour notre chemin
sanitary_path = "/content/drive/MyDrive/EPSI/_I1/AtelierComposantMetier/sanitary.csv"

data = pd.read_csv(sanitary_path, sep=";")

data = data[data['Code_postal'].str.isnumeric()]
data['Departement'] = data['Code_postal'].str[:2]

synthese_eval_mapping = {
    "A améliorer": 0,
    "A corriger de manière urgente": 1,
    "Satisfaisant": 2,
    "Très satisfaisant": 3
}
data['Synthese_eval_sanit'] = data['Synthese_eval_sanit'].map(synthese_eval_mapping)

X = data[['Departement']]
y = data['Synthese_eval_sanit']

########### Diagramme note moyenne par département
data = pd.read_csv(sanitary_path, sep=";")

# Supprimer les lignes avec des codes postaux non numériques
data = data[data['Code_postal'].str.isnumeric()]
data['Departement'] = data['Code_postal'].str[:2]

# Mapping de la colonne Synthese_eval_sanit
synthese_eval_mapping = {
    "A améliorer": 0,
    "A corriger de manière urgente": 1,
    "Satisfaisant": 2,
    "Très satisfaisant": 3
}
data['Synthese_eval_sanit'] = data['Synthese_eval_sanit'].map(synthese_eval_mapping)

# Calculer la note moyenne par département
average_score_by_department = data.groupby('Departement')['Synthese_eval_sanit'].mean().reset_index()

# Créer un diagramme à barres pour afficher la note moyenne par département
plt.figure(figsize=(12, 6))
plt.bar(average_score_by_department['Departement'], average_score_by_department['Synthese_eval_sanit'])
plt.xlabel('Département')
plt.ylabel('Note Moyenne')
plt.title('Note Moyenne par Département')
plt.xticks(rotation=45)  # Rotation des étiquettes pour une meilleure lisibilité
plt.tight_layout()
plt.show()
