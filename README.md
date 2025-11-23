# Data Science & MLOps project:

## Project: Risk scoring of the clients of a bank

### Placeholder

**for libraries: uv pip install -r requirements.txt**

**for the docker part **
```go
# Build image
docker build -t mlops-risk-scoring:latest .

# Arrêter et supprimer tout conteneur portant le même nom pour éviter les conflits
docker stop risk-api
docker rm risk-api

# Lancer le nouveau conteneur en mode détaché (-d), et -p 8000:8000 mappe le port interne du conteneur au port de la machine
docker run -d --name risk-api -p 8000:8000 mlops-risk-scoring:latest

# Vérifier API et chargement model
docker logs risk-api

# Check statut conteneur
docker ps
```

Performance pour un modèle de random forest classifier -> ROC AUC: 0.6321
Performance pour un modèle de gradient boosting classifier -> ROC AUC: 0.6876

Difficulté: plateau de l'amélioration du modèle
    -> travail de feature engineering pour avoir des meilleurs entrées