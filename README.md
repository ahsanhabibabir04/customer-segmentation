# Customer Segmentation Using K-Means

A machine learning project that groups mall customers based on annual income and spending score using K-Means clustering.

## Overview
This project uses unsupervised machine learning to identify customer groups based on spending patterns.

## Dataset
- 200 customer records
- Annual Income
- Spending Score
- Age and Gender for additional analysis

## Model Performance
- Algorithm: K-Means Clustering
- Number of clusters: 5
- Silhouette Score: 0.5547

## Customer Segments
1. Average Income - Average Spending
2. High Income - High Spending
3. Low Income - High Spending
4. High Income - Low Spending
5. Low Income - Low Spending

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Unittest

## Features
- Customer segmentation using K-Means
- Silhouette score evaluation
- Customer segment visualization
- Interactive Streamlit application
- Automated tests

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Create segment labels:

```bash
python label_clusters.py
```

Run the application:

```bash
python -m streamlit run app.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Limitations
This project is for educational and analytical purposes. The customer groups are based on patterns in a small dataset and do not guarantee real-world customer behavior.

## Author
M.A. Ahsan Habib Abir

GitHub: https://github.com/ahsanhabibabir04