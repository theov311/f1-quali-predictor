# 🏁 Prédicteur de Qualifications F1

Application web interactive pour prédire les résultats des qualifications de Formule 1 en temps réel.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/theov311/f1-quali-predictor)

## 🎯 Fonctionnalités

- **Prédiction intelligente** : Analyse basée sur les données FastF1
- **3 types de sessions** : Q1, Q2, et Q3
- **Calcul automatique** : Prédiction aux 2/3 de chaque session (moment optimal)
- **Intervalle de confiance** : Fourchette de temps avec évolution de piste
- **Interface moderne** : Design épuré et responsive

## 📊 Comment ça marche ?

L'algorithme se place aux **2/3 de la session** (après la 1ère tentative, avant la 2ème) et :

1. Analyse les meilleurs tours de chaque pilote
2. Identifie la "bulle" des pilotes en danger (pour Q1/Q2)
3. Calcule un intervalle de prédiction statistique (t-Student)
4. Applique un facteur d'évolution de piste pour le temps restant

### Sessions détectées automatiquement

- **Q1 (18 min)** : Prédiction à 12 min → Cutoff P15
- **Q2 (15 min)** : Prédiction à 10 min → Cutoff P10
- **Q3 (12 min)** : Prédiction à 8 min → Pole Position

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/f1-quali-predictor.git
cd f1-quali-predictor
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer l'application**
```bash
python app.py
```

5. **Ouvrir dans le navigateur**
```
http://127.0.0.1:5000
```

## 📖 Utilisation

1. Sélectionnez une **année** (2018-2025)
2. Choisissez un **Grand Prix**
3. Sélectionnez le **type de session** (Q1, Q2 ou Q3)
4. Cliquez sur **"Prédire"**
5. Obtenez l'**intervalle de temps prédit** !

## 🛠️ Technologies utilisées

- **Backend** : Flask (Python)
- **Données** : FastF1 API
- **Analyse** : Pandas, NumPy, SciPy
- **Frontend** : HTML, CSS, JavaScript (Vanilla)

## 📁 Structure du projet

```
f1-quali-predictor/
├── app.py                      # Application Flask principale
├── f1_quali_predictor.py       # Script standalone (version CLI)
├── templates/
│   └── index.html              # Interface web
├── cache/                      # Cache FastF1 (généré automatiquement)
├── requirements.txt            # Dépendances Python
├── .gitignore
└── README.md
```

## 🔬 Algorithme de prédiction

### Q1/Q2 - Cutoff Prediction
- Utilise les pilotes dans la "bulle" (zone médiane/basse)
- Calcul d'intervalle avec distribution t-Student
- Facteur d'évolution : -0.03s par minute

### Q3 - Pole Position
- Compare le meilleur tour actuel
- Calcule l'Ultimate Lap théorique (meilleurs secteurs)
- Prédit entre ces deux valeurs avec évolution

## ⚠️ Limitations

- Nécessite des données FastF1 complètes
- Fonctionne pour les saisons 2018-2025
- Prédictions basées sur des moyennes statistiques
- Ne prend pas en compte la météo ou incidents en temps réel

## 🌐 Déploiement

### Option 1 : Render.com (Gratuit - Recommandé) ⭐

1. Créez un compte sur [Render.com](https://render.com)
2. Cliquez sur "New +" → "Web Service"
3. Connectez votre repository GitHub `theov311/f1-quali-predictor`
4. Render détectera automatiquement la configuration (`render.yaml`)
5. Cliquez sur "Create Web Service"
6. Votre app sera disponible sur `https://f1-quali-predictor.onrender.com`

⚠️ **Note** : Le service gratuit se met en veille après 15 minutes d'inactivité. Le premier chargement peut prendre 30-60 secondes.

### Option 2 : Railway.app

1. Créez un compte sur [Railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. Sélectionnez `theov311/f1-quali-predictor`
4. Railway détectera Flask automatiquement
5. Ajoutez la commande de démarrage : `gunicorn app:app`

### Option 3 : PythonAnywhere

1. Compte gratuit sur [PythonAnywhere](https://www.pythonanywhere.com)
2. Clonez votre repo : `git clone https://github.com/theov311/f1-quali-predictor.git`
3. Créez un environnement virtuel et installez les dépendances
4. Configurez une Web App Flask dans l'onglet "Web"

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir des issues pour signaler des bugs
- Proposer des améliorations
- Soumettre des pull requests

## 📝 License

MIT License - Libre d'utilisation

## 👨‍💻 Auteur

Créé avec ❤️ pour les passionnés de F1

## 🙏 Remerciements

- [FastF1](https://github.com/theOehrly/Fast-F1) pour l'API de données
- La communauté F1 pour l'inspiration

---

**Bon prédiction ! 🏎️💨**
