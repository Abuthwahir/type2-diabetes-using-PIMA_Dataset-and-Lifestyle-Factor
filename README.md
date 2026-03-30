# Type 2 Diabetes Risk Assessment

An end-to-end, interpretable machine learning project for estimating Type 2 diabetes risk using the Pima Indians Diabetes Dataset plus lifestyle-aware risk adjustment in a Streamlit application.

## Why This Project

Type 2 diabetes screening models are often judged only by accuracy. In healthcare settings, that is not enough. This project emphasizes:

- Clinical relevance of input features
- Interpretability through SHAP explanations
- Practical UX for non-technical users
- Transparent post-model lifestyle risk adjustment

The result is a risk assessment workflow that is easier to inspect, discuss, and improve.

## Highlights

- Interpretable ML pipeline for diabetes risk prediction
- SHAP-based local feature contribution explanation support
- Streamlit app for clinical-style interactive assessment
- Lifestyle factor module (smoking, activity, diet, meal pattern, family history, cardio history)
- Model training support for Random Forest and XGBoost
- Evaluation utilities for confusion matrix, ROC, precision-recall, and feature importance plots
- Unit tests for preprocessing, training, and evaluation modules

## Dataset: Pima Indians Diabetes Dataset (PIMA)

The project uses the Pima Indians Diabetes dataset (commonly referred to as PIMA) with core clinical variables such as:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target)

### Data Quality Handling

The preprocessing pipeline addresses medically implausible zeros in selected fields:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These values are converted to missing values and imputed using median strategy before model training.

## System Design

### 1. Preprocessing

Implemented in `src/preprocess.py`:

- Zero-to-missing conversion for biologically implausible values
- Median imputation with `SimpleImputer`
- Train/test split with stratification
- Feature scaling with `StandardScaler`
- Clinical feature engineering:
  - `Glucose_BMI`
  - `Age_Pregnancies`
  - `BP_Glucose_Ratio`
  - `Metabolic_Index`

### 2. Model Training

Implemented in `src/train.py`:

- Random Forest baseline
- XGBoost option for stronger gradient-boosted performance
- Class imbalance adjustment (`class_weight='balanced'` for Random Forest, `scale_pos_weight` for XGBoost)
- Trained model persisted as `.pkl`

### 3. Prediction and Explainability

Implemented in `src/predict.py`:

- Predict risk probability or class label
- Accepts list or dictionary input
- Applies the same engineered clinical features
- Supports SHAP explanation output for tree-based models

### 4. Streamlit Clinical App

Implemented in `streamlit/app.py`:

- Sidebar-driven patient profile entry
- BMI direct input or computed from height/weight
- Optional insulin and pedigree inputs with sensible defaults
- Lifestyle factor capture:
  - Smoking
  - Physical activity
  - Diet pattern
  - Meals per day
  - Family history
  - Cardio history
- Final risk is shown as:
  - Raw model probability
  - Lifestyle-adjusted probability
  - Human-readable reasons for adjustments

## SHAP Explainability

The project supports SHAP-based local explanations to show feature-level contribution to a single prediction.

Why SHAP matters here:

- Moves model output from black-box probability to feature-level rationale
- Helps compare intuition vs model behavior for each patient
- Supports transparent discussion in educational and clinical decision-support contexts

## Lifestyle Factor Module

Unlike many pure tabular ML demos, the app includes a rule-based lifestyle adjustment layer on top of the model probability.

This layer does not replace the model. It provides transparent contextual adjustment with explicit reasons displayed to the user.

Benefits:

- Captures behavioral risk components often underrepresented in the base dataset
- Improves communication of modifiable risk
- Keeps adjustment logic inspectable and easy to revise

## Performance Summary

Based on project-reported model comparison:

| Model         | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| ------------- | ------: | -------: | --------: | -----: | -------: |
| XGBoost       |    0.85 |     0.79 |      0.76 |   0.78 |     0.77 |
| Random Forest |    0.83 |     0.79 |      0.74 |   0.75 |     0.75 |

### Why It Performs Better Than Typical Baselines

This project improves on basic implementations through:

- Explicit handling of biologically invalid zero values
- Clinical feature engineering instead of only raw variables
- Imbalance-aware training setup
- Probabilistic output with threshold-sensitive evaluation
- Explainability layer (SHAP) for post-hoc inspection
- Lifestyle context layer for practical risk interpretation

## Repository Structure

```text
Type2-Diabetes-Risk-Assessessment-Lates/
  data/
    diabetes.csv
  notebooks/
    diabetes-prediction.ipynb
  src/
    preprocess.py
    train.py
    evaluate.py
    predict.py
    images/
      Demo1.png
      Demo2.png
  streamlit/
    app.py
    requirements.txt
    clinical_diabetes_pipeline.pkl
    feature_names.pkl
  tests/
    test_preprocess.py
    test_train.py
    test_evaluate.py
  README.md
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Abuthwahir/type2-diabetes-using-PIMA_Dataset-and-Lifestyle-Factor.git
cd type2-diabetes-using-PIMA_Dataset-and-Lifestyle-Factor
```

### 2. Create Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```powershell
python -m pip install -r streamlit\requirements.txt
```

For full training/evaluation workflow, also install:

```powershell
python -m pip install xgboost seaborn pytest jupyter
```

### 4. Run Streamlit App

```powershell
python -m streamlit run streamlit\app.py
```

## Training and Evaluation

### Train Random Forest

```powershell
python -m src.train --data data/diabetes.csv --model random_forest --output models
```

### Train XGBoost

```powershell
python -m src.train --data data/diabetes.csv --model xgboost --output models
```

### Run Evaluation Script

```powershell
python src/evaluate.py
```

### Run Tests

```powershell
pytest -q
```

## Visuals

- Demo screenshots: `src/images/Demo1.png`, `src/images/Demo2.png`
- Generated evaluation figures are saved by the evaluation module under the configured output path.

## Clinical and Ethical Note

This tool is an educational and decision-support prototype. It is not a medical diagnosis system and must not be used as a substitute for professional medical evaluation.

## Limitations

- Dataset is relatively small and population-specific
- External validation on other cohorts is not included here
- Lifestyle adjustment is rule-based, not learned jointly with the model
- Calibration and prospective clinical validation are future work

## Future Improvements

- Probability calibration and threshold tuning by use-case
- Expanded external validation datasets
- Fairness and subgroup performance analysis
- End-to-end pipeline object (preprocess + model + explainer) versioning
- CI workflow for automated model quality checks

## Original Author and Project Updates

### Original Author

- Harsh Patel (original project foundation, baseline pipeline, and initial implementation)

### Current Maintainer and Contributor

- Abuthwahir HM

### Key Updates Made by Abuthwahir HM

- Revamped the Streamlit application workflow, improving usability, structure, and user interaction flow
- Integrated lifestyle-aware risk adjustments to enhance real-world relevance of predictions
- Redesigned and improved the frontend UI/UX, making the application more intuitive and visually consistent
- Enhanced input handling and display logic within the Streamlit interface
- Improved project documentation for clearer setup, usage, and model interpretation
- Refactored repository structure for better organization and open-source readability
- Extended model workflow to support retraining and evaluation pipelines
- Added contributor-level metadata and improved overall repository hygiene

### About Model Retraining

- The codebase supports retraining with both Random Forest and XGBoost using the PIMA dataset
- Retraining commands are documented in the Training and Evaluation section
- Evaluation workflow remains available for comparing model behavior after retraining

## License

This repository is released under the MIT License.

- Original copyright: Harsh Patel
- Modifications and ongoing maintenance: Abuthwahir HM


### Current Maintainer and Contributor and Owner 

- Abuthwahir HM