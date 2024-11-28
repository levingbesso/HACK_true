from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime

# Charger le modèle
model = joblib.load('modelML.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

# Définir une route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()

        # Récupérer les variables
        ville = data.get('Ville')
        taille_menage = data.get('Taille de ménage')
        type_habitation = data.get('Type_d\'habitation')

        # Mois actuel (automatique)
        mois = datetime.now().month

        # Vérification des entrées
        if ville is None or taille_menage is None or type_habitation is None:
            return jsonify({'error': 'Missing required fields: Ville, Taille de ménage, Type_d\'habitation'}), 400

        # Créer l'entrée pour le modèle
        features = np.array([ville, taille_menage, type_habitation, mois]).reshape(1, -1)

        # Faire une prédiction
        prediction = model.predict(features)

        # Retourner la partie entière de la prédiction
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True)