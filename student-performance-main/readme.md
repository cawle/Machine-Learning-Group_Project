# Student Performance Clustering (Unsupervised Learning)

This project applies unsupervised machine learning techniques to cluster students based on their performance and demographic data using the [StudentsPerformance.csv](StudentsPerformance.csv) dataset.

## Project Overview

The goal is to discover natural groupings among students using clustering methods, and to analyze the characteristics of each cluster.

## Workflow

1. **Data Loading & Exploration**
   - Loaded the dataset into a pandas DataFrame.
   - Explored the data structure, types, and checked for missing values.

2. **Data Preprocessing**
   - Dropped less relevant columns: `race/ethnicity` and `parental level of education`.
   - Applied one-hot encoding to categorical variables (`gender`, `lunch`, `test preparation course`).

3. **Feature Scaling**
   - Standardized all features using z-score normalization to ensure comparability.

4. **Outlier Detection & Removal**
   - Used PCA to reduce features to 2D for visualization.
   - Identified and removed outliers based on distance in PCA space (top 1% farthest points).

5. **Clustering**
   - Used KMeans clustering on the filtered, scaled data.
   - Determined the optimal number of clusters using the Elbow Method and Silhouette Score.
   - Selected **k=7** as the final number of clusters for interpretability and quality.

6. **Visualization**
   - Visualized clusters in PCA-reduced space, colored by cluster assignment.

7. **Cluster Analysis**
   - Calculated the percentage of students in each cluster.
   - Computed the mean scores and feature proportions for each cluster.
   - Analyzed the distribution of categorical features (e.g., gender, lunch type, test preparation) within clusters.

## Key Findings

- Students were grouped into 7 clusters with distinct performance and demographic profiles.
- Cluster analysis revealed patterns in gender, lunch type, and test preparation course completion related to academic performance.

## How to Run

1. Open `unsupervised.ipynb` in Jupyter or VS Code.
2. Ensure `StudentsPerformance.csv` is in the same directory.
3. Run all cells to reproduce the analysis and visualizations.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:
```sh
pip install pandas numpy scikit-learn matplotlib
```

## Files

- [`unsupervised.ipynb`](unsupervised.ipynb): Main notebook with all code and analysis.
- [`StudentsPerformance.csv`](StudentsPerformance.csv): Dataset used for clustering.

---

*This project demonstrates the use of unsupervised learning for educational data mining and cluster interpretation.*