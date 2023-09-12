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

## TODO : Remplacer par son chemin
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

########## Modèle 2 : Support Vector Machine
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Création et entraînement du modèle SVM
model = SVC(kernel='linear', random_state=42)  # Utilisation d'un noyau linéaire
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle : {:.2f}%".format(accuracy * 100))



#################### Test : Prédiction pour le département 25
departement = 25
prediction_departement = model.predict([[departement]])

print("Prédiction pour le département ", departement, " :", prediction_departement[0])
