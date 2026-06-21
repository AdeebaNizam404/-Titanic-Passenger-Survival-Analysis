# Titanic Passenger Survival Prediction

An end-to-end machine learning project predicting passenger survival on the Titanic. This repository showcases two approaches to building a machine learning workflow: a traditional manual data preprocessing approach and a modular, production-ready design utilizing Scikit-Learn Pipelines.

## Project Highlights
* **Exploratory Data Analysis (EDA):** Completed thorough univariate and bivariate statistical analysis to isolate the most powerful passenger survival indicators.
* **Feature Engineering:** Implemented data cleaning pipelines utilizing customized Function Transformers for handling missing demographic values.
* **Predictive Modeling:** Developed and tuned a Logistic Regression classifier achieving a peak classification **accuracy score of 90.8%**.
* **Model Productionization:** Serialized the final workflow into a production-ready Pickle file (`pipe.pkl`) for rapid downstream inference.

---

## Repository Structure

```text
├── model/
│   ├── clf.pkl                    # Serialized Logistic Regression model
│   ├── ohe_embarked.pkl           # One-Hot Encoder for Embarked location
│   ├── ohe_sex.pkl                # One-Hot Encoder for Passenger Sex
│   ├── pipe.pkl                   # Full end-to-end production ML pipeline 
│   ├── predict-using-pipeline.ipynb # Inference testing via the unified pipeline
│   ├── predict-without-pipeline.ipynb # Inference testing via manual transformations
│   ├── titanic-using-pipeline.ipynb # Training workflow built with Sklearn Pipelines
│   └── titanic-without-using-pipeline.ipynb # Baseline model training workflow
├── bivariate-analysis.ipynb       # Statistical bivariate visualization notebook
├── function-transformer.ipynb     # Prototyping custom preprocessing logic
├── train.csv                      # Passenger training dataset
├── understanding-data-eda.ipynb   # Initial data profiling and structuring
└── univariate-analysis.ipynb      # Statistical univariate distribution analysis
```

---

## Workflow Execution Order

To reproduce the results or run inference successfully, execute the project notebooks in the following order:

### Phase 1: Exploratory Analysis & Prototyping
1. `understanding-data-eda.ipynb`
2. `univariate-analysis.ipynb` & `bivariate-analysis.ipynb`
3. `function-transformer.ipynb`

### Phase 2: Model Training & Component Export
4. **`model/titanic-without-using-pipeline.ipynb`**  
   *Establishes the baseline model and exports individual transformation pickles (`clf.pkl`, `ohe_sex.pkl`, `ohe_embarked.pkl`).*
5. **`model/titanic-using-pipeline.ipynb`**  
   *Constructs the unified preprocessing and modeling workflow, exporting the global `pipe.pkl`.*

### Phase 3: Live Inference Testing
6. `model/predict-without-pipeline.ipynb`
7. `model/predict-using-pipeline.ipynb`

---

## Tech Stack & Tools
* **Language:** Python 3.10+
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Pickle
