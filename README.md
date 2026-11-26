# MLOps Capstone – AI/ML Internship

Starter workspace for an end-to-end machine learning pipeline.

## Dataset
`data/breast_cancer_wisconsin.csv` is the Breast Cancer Wisconsin Diagnostic dataset
provided by scikit-learn. It contains 569 rows, 30 numeric input features, a numeric
target (`target`), and a readable target label (`target_name`).

Target:
- `0` = malignant
- `1` = benign

This dataset is suitable for a binary-classification MLOps demonstration. It is an
educational dataset, not for medical diagnosis.

## Project structure
- `data/`: input dataset
- `src/`: preprocessing and model training code
- `app/`: FastAPI application (to be implemented)
- `models/`: saved trained model artifacts
- `notebooks/`: exploratory analysis
- `tests/`: automated tests
- `results/`: metrics and generated outputs
- `docs/`: screenshots and documentation
- `.github/workflows/`: CI workflow

## First setup
Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then inspect the dataset:

```powershell
python src/inspect_data.py
```

## Important
This is a starter workspace. The dataset and folders are prepared; the MLflow,
DVC, API, Docker, CI/CD, and Evidently components will be added incrementally
as the internship tasks require.


## Live Deployment

- **Platform:** Render
- **API URL:** https://mlops-breast-cancer-api.onrender.com
- **Swagger Docs:** https://mlops-breast-cancer-api.onrender.com/docs
- **Docker Hub:** https://hub.docker.com/r/ashwitha07/mlops-breast-cancer-api

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Checks API health |
| POST | /predict | Predicts breast cancer classification |

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- MLflow
- Docker
- GitHub Actions
- Render