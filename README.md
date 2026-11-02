# Network Fault Detection
### A Supervised Machine Learning Application for Network Condition Classification

<p align="center">
  <strong>Data Mining Course Project</strong><br>
  Network telemetry analysis · Classification · Model comparison · Interactive prediction
</p>

---

## Table of Contents

1. [About the Project](#1-about-the-project)
2. [Project Overview](#2-project-overview)
3. [Why This Project Was Built](#3-why-this-project-was-built)
4. [Why Network Fault Detection Is Needed](#4-why-network-fault-detection-is-needed)
5. [Project Objectives](#5-project-objectives)
6. [Data Mining Technique](#6-data-mining-technique)
7. [Dataset Description](#7-dataset-description)
8. [Data Preparation and Preprocessing](#8-data-preparation-and-preprocessing)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Model Training Workflow](#10-model-training-workflow)
11. [Model Evaluation](#11-model-evaluation)
12. [Application Workspace](#12-application-workspace)
13. [End-to-End System Workflow](#13-end-to-end-system-workflow)
14. [How the Application Was Built](#14-how-the-application-was-built)
15. [Project Architecture and File Structure](#15-project-architecture-and-file-structure)
16. [Installation and Execution](#16-installation-and-execution)
17. [Using the Application](#17-using-the-application)
18. [Using a Custom Dataset](#18-using-a-custom-dataset)
19. [Example Prediction](#19-example-prediction)
20. [Technology Stack](#20-technology-stack)
21. [Important Design Decisions](#21-important-design-decisions)
22. [Limitations and Responsible Interpretation](#22-limitations-and-responsible-interpretation)
23. [Possible Future Improvements](#23-possible-future-improvements)
24. [Academic Summary](#24-academic-summary)
25. [Keywords](#25-keywords)

---

## 1. About the Project

**Network Fault Detection** is a Python-based Data Mining project that applies supervised machine learning classification to network performance data.

The application accepts network telemetry measurements such as latency, packet loss, available bandwidth, CPU usage, and active connections. It analyzes these measurements and predicts the most likely network condition using a selected trained classification model.

The project implements and compares three machine learning algorithms:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Gradient Boosting Classifier

A Streamlit-based graphical interface brings the complete workflow together. Users can inspect the dataset, train all three models, compare their evaluation metrics, investigate confusion matrices, and submit custom network measurements to receive a prediction.

The application includes a generated demonstration dataset so that the project can be executed without first finding an external dataset. It also supports uploading a labeled CSV dataset that follows the required schema.

> **Dataset transparency:** The built-in demonstration dataset is synthetically generated. It is intended for demonstrating the application and machine learning workflow. It must not be described as measurements collected from a real production network.

---

## 2. Project Overview

| Item | Description |
|---|---|
| Project title | Network Fault Detection |
| Project category | Data Mining / Machine Learning |
| Problem type | Multi-class classification |
| Data mining technique | Supervised learning |
| Learning task | Classification |
| Input data | Network performance metrics |
| Target variable | `fault_type` |
| Algorithms | Random Forest, SVM, Gradient Boosting |
| Interface | Streamlit web application |
| Programming language | Python |
| Evaluation | Accuracy, weighted Precision, weighted Recall, weighted F1-score, Confusion Matrix |
| Dataset options | Built-in synthetic dataset or custom labeled CSV |

### Network conditions represented

The demonstration dataset contains five target classes:

| Class | Meaning |
|---|---|
| `Normal` | Network measurements represent a normal operating condition. |
| `High Latency` | Network delay is unusually high. |
| `Packet Loss` | A significant proportion of transmitted packets are lost. |
| `Congestion` | Network resources are under heavy load or congestion. |
| `Link Failure` | Measurements represent a possible network connection or link failure. |

These labels are the classes used by the demonstration dataset. A real dataset may use different labels and definitions.

---

## 3. Why This Project Was Built

Networks support communication between computers, servers, applications, and users. Their performance can be affected by high latency, packet loss, limited bandwidth, overloaded devices, and connection failures.

The purpose of this project is to demonstrate how data mining can help analyze network measurements and classify potentially abnormal network conditions.

Instead of examining every measurement manually, a trained classification model can learn patterns from previously labeled examples and use those patterns to classify new input records.

The project was built to:

- Apply the concepts of data mining to a practical technical problem.
- Demonstrate how network telemetry can be transformed into machine learning features.
- Implement more than one classification algorithm.
- Compare model performance using common evaluation metrics.
- Provide an interactive interface for testing individual network measurements.
- Make the complete process understandable and demonstrable in an academic environment.

The system is an academic prototype. It does not independently monitor routers, switches, or live network traffic.

---

## 4. Why Network Fault Detection Is Needed

Network faults can affect availability, response time, application performance, and the reliability of communication.

Examples include:

- **High latency:** Delayed communication can affect video calls, online services, remote access, and interactive applications.
- **Packet loss:** Missing packets can cause retransmissions, unstable connections, and degraded communication quality.
- **Congestion:** Heavy traffic can reduce available bandwidth and increase response times.
- **Link failure:** A broken or unavailable connection can interrupt communication between network devices.
- **Normal operation:** Identifying normal conditions is equally important because a detection system must distinguish ordinary behavior from potential faults.

A data-driven classification approach can help organize network measurements into meaningful categories. In a real deployment, such predictions could support network administrators during troubleshooting and monitoring.

The application demonstrates this idea using a labeled dataset and trained machine learning models.

---

## 5. Project Objectives

The primary objectives are:

1. Build a network fault classification application using Python.
2. Prepare network telemetry data for supervised learning.
3. Define suitable input features and a target class.
4. Train three different classification algorithms.
5. Evaluate all models using the same train/test split.
6. Compare their performance using standard classification metrics.
7. Allow users to select a model independently for prediction.
8. Accept manually entered network measurements.
9. Display the predicted network condition and class probabilities.
10. Support custom labeled CSV datasets for further experiments.

---

## 6. Data Mining Technique

### 6.1 Selected technique: Supervised Learning — Classification

This project uses **Supervised Machine Learning**, specifically **multi-class classification**.

Supervised learning is appropriate because the training dataset contains input measurements along with a known target label. The model learns the relationship between the input features and their corresponding labels.

For this project:

- The input features describe network behavior.
- The target label identifies the network condition.
- The model learns from labeled examples.
- The trained model predicts a class for previously unseen measurements.

### 6.2 Why classification was selected

The goal is to identify which category a network measurement belongs to, rather than predict a continuous numerical value.

For example, the system should classify a record as `Normal`, `High Latency`, `Packet Loss`, `Congestion`, or `Link Failure`.

Therefore, classification directly matches the problem.

### 6.3 How classification is applied

The dataset is divided into:

- **Independent variables (X):** The five network performance features.
- **Dependent variable (y):** `fault_type`.

The algorithms are trained using the training portion of the dataset. Their predictions are then compared with the known labels in the test portion.

In simplified form:

```text
Labeled network dataset
          |
          v
Separate features (X) and target (y)
          |
          v
Train/test split
          |
          v
Train classification algorithm
          |
          v
Predict test-set classes
          |
          v
Compare predictions with actual labels
          |
          v
Evaluate model performance
          |
          v
Predict condition for new network input
```

### 6.4 Why not use clustering or regression?

- **Clustering** is an unsupervised technique. It could group similar network records without known labels, but this project has labeled classes and needs named fault predictions.
- **Regression** predicts continuous numerical values. This project predicts a category, not a continuous output.
- **Classification** uses known labels to learn how network measurements correspond to network conditions, making it the most direct fit for this project.

---

## 7. Dataset Description

### 7.1 Dataset source

The application generates a reproducible synthetic dataset by default using NumPy. The default dataset contains 2,500 records and five network-condition classes.

The generated records are illustrative. Their feature distributions are designed to create distinguishable example patterns for the classification algorithms.

The dataset generator is implemented in `data_utils.py`.

### 7.2 Input features

The five independent variables are:

| Feature | Data type | Unit | Description |
|---|---|---|---|
| `latency_ms` | Numeric | Milliseconds | Measured network response delay. |
| `packet_loss_pct` | Numeric | Percentage | Percentage of packets lost. |
| `bandwidth_mbps` | Numeric | Mbps | Available or measured bandwidth. |
| `cpu_usage_pct` | Numeric | Percentage | CPU utilization of a relevant network device or server. |
| `connections` | Numeric | Count | Number of active connections. |

### 7.3 Target variable

| Column | Type | Description |
|---|---|---|
| `fault_type` | Categorical | The known network condition used as the classification target. |

Example class values:

```text
Normal
High Latency
Packet Loss
Congestion
Link Failure
```

### 7.4 Illustrative data patterns

The synthetic generator creates different statistical patterns for the five classes:

- Normal records generally have lower latency and packet loss, with relatively healthy bandwidth and moderate resource usage.
- High Latency records generally have higher latency.
- Packet Loss records generally have higher packet-loss percentages.
- Congestion records generally combine high CPU usage and connection counts with reduced bandwidth.
- Link Failure records generally have very high latency, low bandwidth, and high packet loss.

These are simulated patterns, not universal rules for diagnosing real networks. Real network faults can overlap, and the same symptom can have several possible causes.

### 7.5 Dataset generation

The default generator:

1. Creates class labels using predefined class proportions.
2. Generates feature values from class-specific statistical distributions.
3. Applies reasonable numeric clipping boundaries.
4. Combines the generated values into a Pandas DataFrame.
5. Shuffles the records to avoid keeping the classes grouped by generation order.

The default class proportions are approximately:

| Class | Generation proportion |
|---|---:|
| Normal | 42% |
| High Latency | 16% |
| Packet Loss | 16% |
| Congestion | 16% |
| Link Failure | 10% |

Because records are randomly generated, the exact count for each class can vary.

---

## 8. Data Preparation and Preprocessing

Before model training, the application separates the dataset into input features and target labels.

### 8.1 Feature and target separation

```python
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN].astype(str)
```

`X` contains the five numerical network features. `y` contains the network-condition labels.

### 8.2 Train/test splitting

The application uses `train_test_split` from scikit-learn.

The default test size is 20%, and the user can change it in the Model Laboratory.

The split uses:

- `random_state=42` for reproducibility.
- `stratify=y` to preserve class proportions across training and testing data.

### 8.3 Missing-value handling

Each model pipeline uses `SimpleImputer(strategy="median")` to replace missing numerical feature values with the median of the corresponding training feature.

This preprocessing is fitted as part of the model pipeline, which helps prevent test-set information from leaking into the training process.

### 8.4 Feature scaling

SVM uses `StandardScaler` after imputation.

Scaling is important for SVM because its RBF kernel is sensitive to the scale of the input features. Without scaling, features with larger numerical ranges could dominate distance calculations.

Random Forest and Gradient Boosting are tree-based methods and do not require standard feature scaling in this implementation.

### 8.5 Why pipelines are used

Scikit-learn pipelines combine preprocessing and model training into a single object.

Benefits include:

- Consistent preprocessing during training and prediction.
- Reduced risk of preprocessing inconsistencies.
- Easier model management.
- Cleaner and more maintainable code.

---

## 9. Machine Learning Models

Three classification algorithms are implemented. They are trained on the same training data and evaluated on the same test data to make their results more comparable.

### Model 01 — Random Forest Classifier

**Category:** Ensemble learning  
**Base learner:** Decision trees

#### What it does

Random Forest builds multiple decision trees using randomized samples of the training records and feature subsets. The trees independently produce predictions, and the classifier combines their outputs through voting.

#### Why it was selected

- Can model non-linear relationships.
- Handles interactions between network features.
- Does not require feature scaling.
- Provides a strong tree-based baseline.
- Supports feature-importance analysis in further development.

#### Configuration in this project

```python
RandomForestClassifier(
    n_estimators=180,
    max_depth=14,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The model is wrapped in a pipeline with median imputation.

#### Model workflow

```text
Training records
      |
      v
Create randomized tree samples
      |
      v
Build multiple decision trees
      |
      v
Each tree predicts a class
      |
      v
Combine tree predictions
      |
      v
Final network-condition class
```

`class_weight="balanced"` adjusts class weights inversely according to their frequencies in the training data. This can help the model account for class imbalance.

---

### Model 02 — Support Vector Machine (SVM)

**Category:** Supervised learning  
**Kernel:** Radial Basis Function (RBF)

#### What it does

Support Vector Machine attempts to find a decision boundary that separates classes. The RBF kernel allows it to model non-linear decision boundaries by representing relationships in a transformed feature space.

#### Why it was selected

- Suitable for non-linear classification problems.
- Can classify patterns with overlapping feature relationships.
- Offers a different learning approach from tree-based models.
- Provides a useful comparison against ensemble classifiers.

#### Configuration in this project

```python
SVC(
    C=3.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

Its pipeline includes:

1. Median imputation.
2. Standard feature scaling.
3. SVM classification.

#### Model workflow

```text
Training records
      |
      v
Handle missing feature values
      |
      v
Standardize numerical features
      |
      v
Apply RBF kernel
      |
      v
Learn class-separating boundaries
      |
      v
Classify new network measurements
```

`probability=True` enables probability estimates for the prediction interface. These estimates are model-based and should not be interpreted as guaranteed probabilities of a real-world fault.

---

### Model 03 — Gradient Boosting Classifier

**Category:** Ensemble learning  
**Approach:** Sequential boosting with decision trees

#### What it does

Gradient Boosting builds a sequence of weak learners, generally shallow decision trees. Each new learner attempts to reduce the errors of the current ensemble by optimizing a loss function through gradient-based updates.

#### Why it was selected

- Learns complex non-linear relationships.
- Combines multiple weak learners into a stronger model.
- Provides a boosting-based comparison against Random Forest.
- Can perform well on structured tabular datasets.

#### Configuration in this project

```python
GradientBoostingClassifier(
    n_estimators=120,
    learning_rate=0.08,
    max_depth=3,
    random_state=42
)
```

Its pipeline includes median imputation.

#### Model workflow

```text
Training records
      |
      v
Train an initial weak learner
      |
      v
Calculate prediction errors
      |
      v
Train the next learner to reduce loss
      |
      v
Repeat sequential boosting
      |
      v
Combine learners
      |
      v
Final network-condition class
```

Unlike Random Forest, Gradient Boosting builds learners sequentially rather than independently.

---

### 9.1 Why three different models were used

Using three algorithms makes it possible to compare different approaches to the same classification problem.

| Comparison | Random Forest | SVM | Gradient Boosting |
|---|---|---|---|
| Main approach | Bagging-based tree ensemble | Kernel-based classification | Sequential boosting |
| Learner structure | Multiple trees | Support vectors and decision boundaries | Sequential trees |
| Feature scaling | Not required | Used | Not required |
| Non-linear patterns | Yes | Yes, with RBF kernel | Yes |
| Main comparison value | Tree ensemble baseline | Margin and kernel-based method | Boosting-based ensemble |

The purpose is not to assume that one algorithm is always best. Their actual performance must be measured on the selected dataset.

---

## 10. Model Training Workflow

The Model Laboratory trains all three algorithms using a common training and testing split.

### Step-by-step process

**Step 1 — Load the dataset**

The application uses the default generated dataset or the CSV currently loaded by the user.

**Step 2 — Separate X and y**

The five numerical features are assigned to `X`, and `fault_type` is assigned to `y`.

**Step 3 — Split the dataset**

The records are divided into training and testing subsets. The split is stratified so that class proportions are approximately maintained.

**Step 4 — Initialize the algorithms**

The application creates the Random Forest, SVM, and Gradient Boosting pipelines with their configured preprocessing steps.

**Step 5 — Train each model**

Each pipeline is fitted using:

```python
model.fit(X_train, y_train)
```

**Step 6 — Generate test predictions**

Each trained model predicts labels for the same `X_test` records:

```python
predictions = model.predict(X_test)
```

**Step 7 — Calculate evaluation metrics**

The predicted labels are compared with `y_test`.

**Step 8 — Store trained models in the application session**

The trained pipelines and evaluation results are stored in Streamlit session state so that users can move between the Model Laboratory, Fault Detection, and Evaluation workspaces without retraining immediately.

### Training process diagram

```text
                  Dataset
                     |
                     v
             Feature / Target
                 Separation
                     |
                     v
             Stratified Split
                /       \
               v         v
          Training      Testing
             Data         Data
               |
       +-------+-------+
       |       |       |
       v       v       v
      RF      SVM      GB
       |       |       |
       v       v       v
    Train    Train   Train
       |       |       |
       +-------+-------+
               |
               v
       Predict Test Data
               |
               v
       Evaluate Each Model
               |
               v
       Compare Performance
               |
               v
       Select Model for UI
```

RF = Random Forest, SVM = Support Vector Machine, GB = Gradient Boosting.

---

## 11. Model Evaluation

The application evaluates each model using four common classification metrics and a confusion matrix.

### 11.1 Accuracy

Accuracy measures the proportion of all test records that were classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### 11.2 Precision

Precision measures how many predictions assigned to a class were correct.

```text
Precision = TP / (TP + FP)
```

Where:

- TP = True Positives
- FP = False Positives

### 11.3 Recall

Recall measures how many actual records belonging to a class were correctly identified.

```text
Recall = TP / (TP + FN)
```

Where FN means False Negatives.

### 11.4 F1-score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 11.5 Weighted averages

The application calculates precision, recall, and F1-score using `average="weighted"`.

This means each class contributes to the overall metric according to its number of actual test records.

Weighted metrics are useful when classes are not equally represented. However, they can hide weak performance on smaller classes, so the per-class classification report should also be inspected.

### 11.6 Confusion matrix

A confusion matrix compares actual classes against predicted classes.

- Rows represent actual labels.
- Columns represent predicted labels.
- Diagonal values represent correct predictions.
- Off-diagonal values represent classification errors.

It helps identify which network conditions are being confused with one another.

### 11.7 Model comparison

All three models are evaluated on the same test split.

The Model Laboratory displays:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- A grouped comparison chart

The application identifies the model with the highest weighted F1-score as the best model on the current test split.

> A model's score on synthetic data does not establish its ability to detect faults in a real network. For meaningful real-world conclusions, evaluation must use representative, independently collected and labeled network data.

---

## 12. Application Workspace

The interface is divided into five workspaces. Each workspace has a separate responsibility in the complete data mining workflow.

### 12.1 Overview

**Purpose:** Provide a quick summary of the current project and dataset.

#### What it displays

- Total dataset records.
- Number of input features.
- Number of fault classes.
- Number of trained models available in the current session.
- Fault-class distribution chart.
- A short explanation of the detection workflow.

#### How it is implemented

The page reads the active dataset from Streamlit session state, calculates basic dataset statistics, and displays them using custom HTML/CSS metric cards and Plotly charts.

#### Why it is needed

The Overview gives the user a starting point and a quick understanding of the dataset and application status.

### 12.2 Dataset Explorer

**Purpose:** Inspect the data used for machine learning.

#### What it provides

- CSV dataset upload.
- Dataset preview.
- Feature statistical summary.
- Fault-class distribution.
- Required dataset schema information.

#### How it is implemented

Pandas reads uploaded CSV files. The validation function checks whether all required columns exist, whether there are enough records, whether the target has multiple classes, and whether feature columns contain usable numeric values.

If a valid dataset is uploaded, it replaces the active dataset in session state and clears the previous model results.

#### Why it is needed

Model performance depends on the quality and structure of the data. The Dataset Explorer makes the input data visible and helps users identify whether their dataset follows the required format.

### 12.3 Model Laboratory

**Purpose:** Train and compare all three classification models.

#### What it provides

- Test-size selection.
- Train-all-models action.
- Performance comparison table.
- Grouped metric chart.
- Best model identification based on weighted F1-score.

#### How it is implemented

The page calls the shared training function. That function creates a common stratified train/test split, initializes each model pipeline, trains all three models, and calculates the evaluation metrics.

The trained pipelines and results are stored in `st.session_state`.

#### Why it is needed

This workspace demonstrates the practical model-building stage of data mining. It also allows a direct comparison of three different algorithms under the same test conditions.

### 12.4 Fault Detection

**Purpose:** Predict the network condition from user-provided measurements.

#### What it provides

- Independent model selection.
- Manual network metric inputs.
- Prediction action.
- Predicted network condition.
- Model-reported prediction confidence.
- Probability distribution across the available classes.

#### Input fields

1. Latency (ms)
2. Packet loss (%)
3. Bandwidth (Mbps)
4. CPU usage (%)
5. Active connections

#### How it is implemented

The page creates a one-row Pandas DataFrame with the same feature names and order used during training.

It retrieves the selected trained pipeline from session state and calls:

```python
prediction = model.predict(input_data)
probabilities = model.predict_proba(input_data)
```

The model pipeline automatically applies the same imputation and scaling steps used during training.

#### Why it is needed

A model evaluation table alone does not demonstrate how the system can be used. This workspace makes the trained models interactive and allows users to test new network measurements.

### 12.5 Evaluation

**Purpose:** Inspect the detailed performance of an individual model.

#### What it provides

- Model selection.
- Confusion matrix.
- Accuracy, precision, recall, and F1-score.
- Per-class classification report.
- Classification report CSV download.

#### How it is implemented

The page retrieves the selected model's stored test labels and predictions, generates a confusion matrix using scikit-learn, and displays it with Plotly.

The classification report is converted to a Pandas DataFrame and can be downloaded as a CSV file.

#### Why it is needed

A single overall accuracy value is not sufficient to understand classification behavior. This workspace helps users inspect class-level performance and identify misclassification patterns.

### Workspace workflow

```text
Overview
   |
   v
Dataset Explorer
   |
   v
Model Laboratory
   |
   +--------------------+
   |                    |
   v                    v
Fault Detection     Evaluation
   |                    |
   v                    v
New Input          Performance Analysis
   |
   v
Predicted Condition
```

---

## 13. End-to-End System Workflow

The complete application follows this sequence:

1. The application starts and loads the default synthetic dataset.
2. The user inspects the dataset in Dataset Explorer.
3. The user may upload a custom labeled CSV dataset.
4. The application validates the required columns and target labels.
5. The user opens Model Laboratory.
6. The user selects the test-set percentage.
7. The application performs a stratified train/test split.
8. The three classification pipelines are trained.
9. The application calculates model evaluation metrics.
10. The user compares the models.
11. The user selects a trained model in Fault Detection.
12. The user enters new network measurements.
13. The selected model predicts a network-condition class.
14. The application displays the prediction and class probabilities.
15. The user can inspect detailed evaluation results in Evaluation.

---

## 14. How the Application Was Built

The application was developed using Python and a modular structure.

### 14.1 Python

Python is used for dataset generation, data manipulation, preprocessing, machine learning, evaluation, and application logic.

### 14.2 Pandas and NumPy

- NumPy generates numerical network measurements and supports numerical operations.
- Pandas stores, validates, transforms, and displays tabular data.

### 14.3 Scikit-learn

Scikit-learn provides:

- Train/test splitting.
- Imputation.
- Feature scaling.
- Classification algorithms.
- Pipelines.
- Evaluation metrics.
- Confusion matrices and classification reports.

### 14.4 Streamlit

Streamlit provides the web application interface.

It was selected because it allows Python-based interactive applications without requiring a separate frontend framework or manually implemented API server.

The application uses Streamlit forms, selectors, sliders, file upload controls, session state, metric cards, and navigation controls.

### 14.5 Plotly

Plotly is used to create interactive visualizations, including:

- Fault distribution charts.
- Model performance comparisons.
- Prediction probability charts.
- Confusion matrices.

### 14.6 Custom CSS

Custom CSS is injected into the Streamlit application to create a more polished interface with:

- A dark navigation sidebar.
- A deep navy project header.
- Consistent typography.
- Clean metric cards.
- Subtle borders and spacing.
- A light workspace background.

---

## 15. Project Architecture and File Structure

```text
Network_Fault_Detection/
│
├── app.py
│   ├── Streamlit configuration
│   ├── Custom interface styling
│   ├── Navigation and workspace pages
│   ├── Model pipeline definitions
│   ├── Model training and evaluation
│   ├── Prediction interface
│   └── Session-state management
│
├── data_utils.py
│   ├── Feature and target definitions
│   ├── Synthetic dataset generation
│   └── CSV dataset validation
│
├── requirements.txt
│   └── Python package dependencies
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Files and folders excluded from Git
```

### Separation of responsibilities

`data_utils.py` keeps dataset-related operations separate from the user interface.

`app.py` manages the application, model pipelines, training, evaluation, and prediction.

`requirements.txt` makes it easier to install the same required libraries in another Python environment.

This structure keeps the project straightforward and makes future changes easier.

---

## 16. Installation and Execution

### Requirements

- Windows, macOS, or Linux.
- Python 3.10 or later recommended.
- Internet access for initial package installation.
- A modern web browser.

### Step 1 — Extract the project

Extract the project ZIP file and open the `Network_Fault_Detection` folder in VS Code or another code editor.

### Step 2 — Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
```

macOS / Linux:

```bash
python3 -m venv .venv
```

### Step 3 — Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 — Start the application

```bash
streamlit run app.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Open that address in your browser.

---

## 17. Using the Application

Recommended first-time workflow:

1. Start the application.
2. Open Overview and inspect the dataset summary.
3. Open Dataset Explorer and inspect the generated records.
4. Open Model Laboratory.
5. Keep the default test size at 20% or choose another supported value.
6. Click **Train all three models**.
7. Review the performance comparison table and chart.
8. Open Evaluation and inspect each model's confusion matrix and classification report.
9. Open Fault Detection.
10. Select one of the three trained models.
11. Enter the five network measurements.
12. Click **Analyze network condition**.
13. Review the predicted condition and class probability chart.

If a new dataset is uploaded, train the models again because previous results are cleared when the active dataset changes.

---

## 18. Using a Custom Dataset

The Dataset Explorer accepts a CSV file.

The CSV must contain these exact column names:

```csv
latency_ms,packet_loss_pct,bandwidth_mbps,cpu_usage_pct,connections,fault_type
32,0.2,600,30,120,Normal
250,1.5,350,45,220,High Latency
100,14.0,400,55,180,Packet Loss
170,5.0,90,90,900,Congestion
900,55.0,5,10,5,Link Failure
```

The values above are illustrative examples only.

### Dataset validation rules

The current validation function checks:

- All required columns are present.
- At least 30 records exist.
- At least two target classes exist.
- Each target class has at least two records.
- Feature columns can be interpreted as numerical data.
- Target labels are not empty.

Missing numeric values can be handled by the model pipelines using median imputation.

### Important custom-dataset considerations

For reliable experiments:

- Use measurements collected with a consistent methodology.
- Ensure that target labels are assigned correctly.
- Avoid duplicate or contradictory records.
- Keep training and test records independent.
- Use representative examples for every class.
- Document the source and meaning of each feature.
- Do not mix incompatible measurement units.

The current UI expects the same five input features for manual prediction. If a different feature schema is used, the dataset definitions and prediction form must also be updated.

---

## 19. Example Prediction

Suppose the user enters:

| Feature | Input |
|---|---:|
| Latency | 35 ms |
| Packet loss | 0.5% |
| Bandwidth | 500 Mbps |
| CPU usage | 35% |
| Active connections | 120 |

The user selects a trained model, for example Random Forest, and submits the form.

The application:

1. Creates a DataFrame containing the five measurements.
2. Passes the DataFrame through the selected model pipeline.
3. Predicts the most likely target class.
4. Retrieves the model's class probability estimates.
5. Displays the predicted class and probability chart.

The exact result depends on the trained model and dataset. The example values do not guarantee a particular prediction.

---

## 20. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and tabular analysis |
| NumPy | Numerical operations and synthetic data generation |
| Scikit-learn | Preprocessing, classification, training, and evaluation |
| Plotly | Interactive charts and confusion matrix |
| HTML/CSS | Interface styling |

---

## 21. Important Design Decisions

### One common test split

All three models use the same train/test split. This makes their evaluation more comparable because they are tested against the same records.

### Stratified sampling

Stratification helps preserve the distribution of target classes in both subsets.

### Model pipelines

Pipelines ensure that the same preprocessing operations are applied consistently during training and prediction.

### Weighted metrics

Weighted precision, recall, and F1-score account for differences in class frequencies. Per-class results are also available for deeper analysis.

### Independent model selection

The Fault Detection workspace allows the user to select any of the three trained models rather than automatically forcing one model for every prediction.

### Session-based model storage

The trained models are held in Streamlit session state. This avoids immediate retraining while navigating between pages in the same active session.

The models are not permanently saved to disk in the current version. Restarting the application or losing the session may require training them again.

### Synthetic data for demonstration

A generated dataset makes the application immediately usable and supports reproducible demonstrations. Real-world claims require real, representative network measurements.

---

## 22. Limitations and Responsible Interpretation

This project is an academic prototype and has several limitations.

1. **Synthetic dataset:** The built-in data is generated by statistical rules and is not captured from a live network.
2. **Limited feature set:** Real network diagnosis may require throughput, jitter, interface errors, retransmissions, signal quality, routing information, device logs, and time-series behavior.
3. **Simplified fault labels:** Real faults can occur simultaneously and may not fit into one mutually exclusive class.
4. **No live monitoring:** The application does not connect to routers, switches, packet capture tools, or network monitoring systems.
5. **No automatic remediation:** It predicts a class but does not repair network problems.
6. **Session-based training:** Models are not persisted between application sessions.
7. **Probability interpretation:** SVM probability estimates and other model probability outputs are not guaranteed to be calibrated.
8. **Evaluation dependence:** Performance metrics depend on the dataset, labels, split, and model configuration.

A prediction should be treated as a model-generated indication, not a definitive diagnosis. Network administrators should verify suspected faults using appropriate monitoring tools and operational evidence.

---

## 23. Possible Future Improvements

The project can be extended in several directions:

- Integrate real network monitoring data through SNMP or monitoring APIs.
- Support public network intrusion and traffic datasets where appropriate.
- Add jitter, throughput, retransmission, and interface error features.
- Add time-series analysis for detecting changes over time.
- Add feature importance and model explainability.
- Add hyperparameter tuning and cross-validation.
- Add model persistence using Joblib.
- Add model versioning and experiment tracking.
- Add real-time network telemetry dashboards.
- Add fault severity estimation.
- Add alerting and historical prediction logs.
- Add a dedicated API for integration with other applications.
- Add automated retraining with newly labeled data.

These improvements would move the application closer to a practical network analytics platform.

---

## 24. Academic Summary

**Network Fault Detection** demonstrates the application of supervised machine learning classification to network performance analysis.

The project uses five numerical network features to predict one of five network-condition labels. Three classification algorithms—Random Forest, Support Vector Machine, and Gradient Boosting—are trained and evaluated using a common stratified train/test split.

The application provides an interactive environment for dataset inspection, model training, performance comparison, detailed evaluation, and manual network-condition prediction.

The main contribution of the project is the integration of the complete data mining workflow into one application:

```text
Data Collection / Generation
           |
           v
     Data Preparation
           |
           v
  Feature and Target Selection
           |
           v
    Train/Test Splitting
           |
           v
    Classification Models
           |
           v
     Model Evaluation
           |
           v
      Model Selection
           |
           v
   Interactive Prediction
           |
           v
  Network Condition Output
```

The project illustrates how data mining techniques can be used to transform structured network measurements into useful classification results.

---

## 25. Keywords

`Network Fault Detection` · `Data Mining` · `Supervised Learning` · `Classification` · `Machine Learning` · `Random Forest` · `Support Vector Machine` · `SVM` · `Gradient Boosting` · `Network Monitoring` · `Network Telemetry` · `Fault Classification` · `Python` · `Streamlit` · `Scikit-learn` · `Pandas` · `NumPy` · `Plotly`

---

<p align="center">
  <strong>NETWORK FAULT DETECTION</strong><br>
  Data Mining Course Project<br>
  Built with Python and Machine Learning
</p>
