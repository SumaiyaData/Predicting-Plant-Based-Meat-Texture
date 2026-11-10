# Group-Aware Machine Learning for Predicting Plant-Based Meat Texture

### A Comparative Regression Study of Hardness and Chewiness Using Food Composition Data

## Overview

Texture is one of the most important characteristics influencing the quality and consumer acceptance of plant-based meat alternatives. Developing formulations with desirable textural properties often requires repeated experimentation, making product development time-consuming and resource-intensive.

This project explores the application of machine learning to predict two important textural properties of plant-based meat analogues:

- **Hardness**
- **Chewiness**

Using a published experimental dataset, this study investigates the relationship between proximate nutritional composition, moisture conditions, and textural characteristics.

Five regression algorithms are evaluated using group-aware cross-validation and hyperparameter optimization. Particular attention is given to model generalization across different formulation groups rather than relying solely on conventional random train-test splitting.

This work is an independent computational re-analysis and extension of an existing published study.

## Research Objectives

1. Explore relationships between food composition variables and textural properties.
2. Develop regression models for predicting hardness and chewiness.
3. Compare the predictive performance of different machine learning algorithms.
4. Evaluate model generalization using group-aware validation.
5. Investigate prediction errors across different formulation groups.
6. Compare machine learning models against a simple statistical baseline.

## Dataset Description

The dataset contains 54 experimental observations representing plant-based meat analogue formulations.

It includes seven predictive features, two target variables, and additional grouping and study identifiers.

### Input Features

| Feature | Description |
|---|---|
| Protein | Protein content |
| Target moisture | Targeted moisture content |
| Moisture | Moisture content |
| Ash | Ash content |
| Carbs | Carbohydrate content |
| Fat | Fat content |
| Fiber | Fibre content |

### Target Variables

| Target | Description |
|---|---|
| Hardness | Measured hardness of the meat analogue |
| Chewiness | Measured chewiness of the meat analogue |

`Group_ID` and `Study_ID` are retained as metadata for group-aware validation and data inspection. They are not used as predictive features.

**Dataset dimensions:** 54 observations × 7 predictive features.

The dataset originates from the supplementary material associated with the original research paper cited below.

## Methodology

The project follows a structured machine learning workflow.

### 1. Data Loading and Exploration

- Load the experimental dataset using Pandas.
- Inspect data types, missing values, and dataset structure.
- Perform exploratory data profiling.
- Remove unnecessary indexing and non-predictive columns.
- Separate predictive features and target variables.

### 2. Group-Aware Train-Test Splitting

A `GroupShuffleSplit` strategy is used to separate training and testing observations based on formulation groups.

**Configuration:**
- Test size: 20%
- Random state: 42
- Training observations: 33
- Testing observations: 21
- Training groups: 9
- Held-out testing groups: 3

This prevents observations belonging to the same formulation group from appearing in both training and testing sets.

### 3. Machine Learning Models

Five regression algorithms are investigated:

| Algorithm | Approach |
|---|---|
| Ridge Regression | Regularized linear regression |
| K-Nearest Neighbors (KNN) | Distance-based regression |
| Gaussian Process Regression (GPR) | Kernel-based probabilistic regression |
| Random Forest | Tree-based ensemble learning |
| Extra Trees | Extremely randomized tree ensemble |

StandardScaler is incorporated into pipelines for Ridge, KNN, and GPR.

### 4. Group-Aware Cross-Validation

`LeaveOneGroupOut` (LOGO) cross-validation is applied to the training dataset.

The model is trained on all but one formulation group and validated on the excluded group. This process is repeated across the nine training groups.

This approach helps evaluate whether a model can generalize to formulation groups not represented in its training fold.

### 5. Baseline Comparison

A `DummyRegressor` using the mean prediction strategy is implemented as a baseline.

Machine learning performance is compared against this baseline to determine whether the models provide meaningful predictive improvements.

### 6. Hyperparameter Optimization

`GridSearchCV` is used with LOGO cross-validation.

The optimization objective is to minimize Root Mean Squared Error (RMSE).

Models are tuned separately for Hardness and Chewiness, allowing different algorithms and hyperparameters to be selected for each target.

### 7. Final Model Evaluation

The selected models are evaluated using the previously untouched test groups.

The evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### 8. Additional Diagnostics

The analysis also includes:

- Actual versus predicted comparisons.
- Group-wise prediction error analysis.
- Feature distribution and range diagnostics.
- Prediction error inspection.
- Comparison against the mean baseline.

## Results

In the recorded experimental run, K-Nearest Neighbors (KNN) achieved the lowest LOGO cross-validation RMSE after hyperparameter optimization for both target variables.

### Selected Models

| Target | Selected Model | Best LOGO RMSE |
|---|---|---:|
| Hardness | KNN | 13.9337 |
| Chewiness | KNN | 10.0083 |

### Held-Out Test Performance

The following results are from the recorded group-based test split.

| Target | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Hardness | KNN | 7.6264 | 10.3138 | 0.7815 |
| Hardness | Dummy Baseline | 27.1886 | 28.1208 | -0.6242 |
| Chewiness | KNN | 6.3889 | 8.7152 | 0.7727 |
| Chewiness | Dummy Baseline | 20.9840 | 21.3351 | -0.3621 |

These results indicate that KNN outperformed the mean baseline on the held-out groups in this particular experimental split.

**Important:** The test set contains only 21 observations from three formulation groups. Therefore, these results should be interpreted as exploratory evidence rather than definitive proof of generalization to all plant-based meat formulations.

## Relationship to Previous Research

This project builds upon the research:

**Kircali Ata, S., et al. (2023).** Predicting the Textural Properties of Plant-Based Meat Analogs with Machine Learning. *Foods, 12*, 344.

https://doi.org/10.3390/foods12020344

The original study investigated Ridge, Random Forest, XGBoost, KNN, and Multilayer Perceptron models, incorporating feature selection and group-aware validation.

The present project takes an alternative benchmarking approach by:

- Evaluating Gaussian Process Regression and Extra Trees.
- Retaining Ridge, Random Forest, and KNN for comparison.
- Incorporating a mean-based baseline.
- Applying an explicit group-based holdout test split.
- Examining prediction errors across formulation groups.
- Evaluating performance using MAE, RMSE, and R².

This project does not claim to reproduce every experiment in the original paper. It investigates an alternative computational workflow using the published experimental dataset.

Original research code: https://github.com/sezinata/FoodML

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Jupyter Notebook
- Data Profiling

## Repository Structure

```text
plant-based-meat-texture-prediction/
│
├── data/
│   └── additional_30_12_12.csv
│
├── notebooks/
│   └── plant_based_meat_texture_ml.ipynb
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── requirements.txt
│
└── README.md
```

*The structure above is a suggested organization for the repository.*

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
pip install "fg-data-profiling[notebook]" xgboost
```

The latter packages are included to support the imports in the current notebook.

### 4. Run the Notebook

```bash
jupyter notebook
```

Open the notebook and execute the cells sequentially.

**Dataset path:** The current notebook uses a Google Colab path (`/content/additional_30_12_12.csv`). For local execution, update the CSV path to match your repository structure.

## Limitations

Several limitations should be considered when interpreting the findings:

1. The dataset contains only 54 observations, limiting the complexity of models that can be reliably trained.
2. The held-out evaluation involves only three formulation groups.
3. Performance may vary with a different group-based train-test split.
4. The project relies on previously published experimental data rather than newly conducted laboratory experiments.
5. The current workflow does not implement the original paper's feature-selection methodology.
6. Predictions outside the observed composition and moisture ranges may be unreliable.

Further validation using larger datasets and additional experimental formulations is necessary before applying the models to real-world product development.

## Future Directions

Potential extensions include:

- Reproducing the original study's feature-selection pipeline.
- Comparing additional regression algorithms.
- Repeated group-based validation.
- Investigating feature importance and interactions.
- Evaluating uncertainty in Gaussian Process predictions.
- Integrating new experimental observations.
- Exploring machine learning-assisted formulation optimization.

## Acknowledgements

This project is based on the published experimental dataset and research framework of Kircali Ata et al. (2023).

All original research and data sources are acknowledged. This repository represents an independent computational analysis and is not an official implementation of the original authors' work.

---

**Project Focus:** Food Science | Food Texture | Machine Learning | Regression | Predictive Modeling | Computational Food Research
