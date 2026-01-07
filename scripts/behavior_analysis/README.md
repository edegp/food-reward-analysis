Behavioral rating prediction

This small utility trains regression models to predict normative ratings (e.g. Valence) from the image metadata CSV.

Setup

1. Create a virtualenv and install requirements:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Run

   python train.py --csv ../../Food_Behavior/food.pics.database.value.csv --target Valence_omnivore_Male --out models

Outputs

- models/model_ridge.joblib
- models/model_rf.joblib
- models/preprocess.joblib
