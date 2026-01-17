

```markdown

\# NLP Classification CLI Tool



This project is a \*\*Command-Line Interface (CLI) tool\*\* that implements a complete

Natural Language Processing (NLP) pipeline for text classification.



The tool is \*\*generic\*\*, \*\*modular\*\*, and \*\*language-agnostic\*\*, meaning it can be

applied to \*\*any CSV dataset\*\* containing text and labels (Arabic, English, or mixed).



---



\## Pipeline Overview



The NLP workflow consists of the following stages:



EDA → Preprocessing → Embedding → Training



Each stage can be executed \*\*individually\*\* or \*\*all at once\*\* using a one-line pipeline command.



---



\## Project Structure



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



\## Setup Instructions



\### 1) Install Dependencies

```bash

pip install -r requirements.txt

````



\### 2) Verify CLI Commands



```bash

python main.py --help

```



---



\## Dataset Requirements



The dataset must be a \*\*CSV file\*\* containing:



\* A \*\*text column\*\* (e.g., `text`, `review`, `content`)

\* A \*\*label column\*\* (e.g., `label`, `sentiment`, `category`)



Column names are configurable through CLI options.



---



\## Step-by-Step Execution



\## 1- Exploratory Data Analysis (EDA)



\### Class Distribution



```powershell

python main.py eda distribution --csv\_path data.csv --label\_col label

```



\### Text Length Histogram



```powershell

python main.py eda histogram --csv\_path data.csv --text\_col text --unit words

```



\### Remove Outliers (Bonus Feature)



```powershell

python main.py eda remove-outliers --csv\_path data.csv --text\_col text --unit words --output data\_no\_outliers.csv

```



Outputs:



\* Visualizations saved in `outputs/visualizations/`

\* Statistics printed in the console



---



\## 2- Preprocessing



\### Remove Unwanted Characters



```powershell

python main.py preprocess remove --csv\_path data.csv --text\_col text --output cleaned.csv

```



\### Remove Arabic Stopwords



```powershell

python main.py preprocess stopwords --csv\_path cleaned.csv --text\_col text --output no\_stops.csv

```



\### Normalize Arabic Text



```powershell

python main.py preprocess replace --csv\_path no\_stops.csv --text\_col text --output normalized.csv

```



Outputs:



\* `cleaned.csv`

\* `no\_stops.csv`

\* `normalized.csv`



---



\## 3- Embedding (TF-IDF)



```powershell

python main.py embed tfidf --csv\_path normalized.csv --text\_col text --max\_features 5000 --output outputs/embeddings/tfidf\_vectors.pkl

```



Outputs:



\* TF-IDF embeddings saved as a pickle file

\* Embedding matrix shape printed in the console



---



\## 4- Training \& Evaluation



Supported models:



\* K-Nearest Neighbors (KNN)

\* Logistic Regression (LR)

\* Random Forest (RF)



```powershell

python main.py train --csv\_path normalized.csv --input\_col outputs/embeddings/tfidf\_vectors.pkl --output\_col label --models knn --models lr --models rf

```



Outputs:



\* Training report (`.md`) saved in `outputs/reports/`

\* Confusion matrix images saved in `outputs/visualizations/`

\* Best-performing model printed in the console



---



\## One-Line Pipeline Command (Bonus Feature)



The entire NLP workflow can be executed using \*\*a single command\*\*:



```powershell

python main.py pipeline --csv\_path data\_no\_outliers.csv --text\_col text --label\_col label

```



\### This command performs:



1\. Text preprocessing (remove → stopwords → normalization)

2\. TF-IDF embedding

3\. Model training and evaluation



This improves reproducibility and reduces manual errors.



---



\## Language Support



\* Supports \*\*Arabic\*\*, \*\*English\*\*, and \*\*mixed-language datasets\*\*

\* TF-IDF embedding and machine learning models are language-independent

\* Arabic-specific preprocessing is applied safely

\* English text passes through unchanged



---



\## Output Summary



| Output              | Location                  |

| ------------------- | ------------------------- |

| Processed CSV files | `outputs/pipeline/`       |

| Embeddings          | `outputs/embeddings/`     |

| Confusion matrices  | `outputs/visualizations/` |

| Training reports    | `outputs/reports/`        |

| Best model          | Printed in console        |



---



\## Conclusion



This project demonstrates a complete, reusable NLP classification pipeline

implemented as a professional CLI tool.



The system is configurable, extensible, and suitable for academic coursework

as well as real-world NLP experiments.



```



---





