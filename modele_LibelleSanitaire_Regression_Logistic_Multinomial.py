import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Charger le dataset
export_data = pd.read_csv('export_alimconfiance@dgal.csv', sep=";")

# Encoder la variable "Synthese_eval_sanit" (catégorielle) en utilisant LabelEncoder
label_encoder = LabelEncoder()
export_data['Synthese_eval_sanit_encoded'] = label_encoder.fit_transform(export_data['Synthese_eval_sanit'])

# Encoder la variable "APP_Libelle_etablissement" (catégorielle) en utilisant LabelEncoder
export_data['APP_Libelle_etablissement_encoded'] = label_encoder.fit_transform(export_data['APP_Libelle_etablissement'])

# Encodage one-hot des noms d'établissements
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
X = one_hot_encoder.fit_transform(export_data[['APP_Libelle_etablissement_encoded']])
y = export_data['Synthese_eval_sanit_encoded']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression logistique multinomiale
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=0)


print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
