
---

```markdown
# NLP CLI Tool – Text Classification Pipeline

This project is a **Command-Line Interface (CLI) tool** that implements a complete
Natural Language Processing (NLP) pipeline for text classification.

The tool is **generic**, **modular**, and **language-agnostic**, meaning it can be
applied to **any CSV dataset** containing text and labels (Arabic, English, or mixed).

---

 Pipeline Overview

The NLP workflow consists of the following stages:

EDA → Preprocessing → Embedding → Training

Each stage can be executed **individually** or **automatically** using a
**one-line pipeline command**.

---

 Project Structure

```

nlp-cli-tool/
├── main.py
├── commands/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── embedding.py
│   └── training.py
├── outputs/
│   ├── embeddings/
│   ├── models/
│   ├── reports/
│   └── visualizations/
├── requirements.txt
└── README.md

````

---

 Setup Instructions (Windows / PowerShell)

  1) Install Dependencies
```powershell
pip install -r requirements.txt
````

### 2) Verify CLI Commands

```powershell
python main.py --help
```

---

## Dataset Requirements

The dataset must be a **CSV file** containing:

* A **text column** (e.g., `text`, `review`, `content`)
* A **label column** (e.g., `label`, `sentiment`, `category`)

Column names are configurable through CLI options.

---

## Step-by-Step Execution

1- Exploratory Data Analysis (EDA)

 Class Distribution

```powershell
python main.py eda distribution --csv_path data.csv --label_col label
```

 Text Length Histogram

```powershell
python main.py eda histogram --csv_path data.csv --text_col text --unit words
```

 Remove Outliers (Bonus Feature)

```powershell
python main.py eda remove-outliers --csv_path data.csv --text_col text --unit words --output data_no_outliers.csv
```

Outputs:

* Visualizations saved in `outputs/visualizations/`
* Statistics printed in the console

---

2- Preprocessing

### Remove Unwanted Characters

```powershell
python main.py preprocess remove --csv_path data.csv --text_col text --output cleaned.csv
```

 Remove Arabic Stopwords

```powershell
python main.py preprocess stopwords --csv_path cleaned.csv --text_col text --output no_stops.csv
```

### Normalize Arabic Text

```powershell
python main.py preprocess replace --csv_path no_stops.csv --text_col text --output normalized.csv
```

Outputs:

* `cleaned.csv`
* `no_stops.csv`
* `normalized.csv`

---

 3️- Embedding (TF-IDF)

```powershell
python main.py embed tfidf --csv_path normalized.csv --text_col text --max_features 5000 --output outputs/embeddings/tfidf_vectors.pkl
```

Outputs:

* TF-IDF embeddings saved as a pickle file
* Embedding matrix shape printed in the console

---

4- Training & Evaluation

Supported models:

* K-Nearest Neighbors (KNN)
* Logistic Regression (LR)
* Random Forest (RF)

```powershell
python main.py train --csv_path normalized.csv --input_col outputs/embeddings/tfidf_vectors.pkl --output_col label --models knn --models lr --models rf
```

Outputs:

* Training report (`.md`) saved in `outputs/reports/`
* Confusion matrix images saved in `outputs/visualizations/`
* Best-performing model printed in the console

---

 One-Line Pipeline Command (Bonus Feature)

The entire NLP workflow can be executed using **a single PowerShell command**:

```powershell
python main.py pipeline --csv_path data_no_outliers.csv --text_col text --label_col label
```

 This command automatically performs:

1. Text preprocessing (remove → stopwords → normalization)
2. TF-IDF embedding
3. Model training and evaluation

This improves reproducibility and reduces manual errors.

---

 Language Support

* Supports Arabic, English, and **mixed-language datasets
* TF-IDF embedding and machine learning models are language-independent
* Arabic-specific preprocessing is applied safely
* English text passes through unchanged

---

 Outputs Summary

| Output              | Location                  |
| ------------------- | ------------------------- |
| Processed CSV files | `outputs/pipeline/`       |
| Embeddings          | `outputs/embeddings/`     |
| Confusion matrices  | `outputs/visualizations/` |
| Training reports    | `outputs/reports/`        |
| Best model          | Printed in console        |

---


