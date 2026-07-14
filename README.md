# Wholesale Customer Segmentation

## Project Overview

This capstone project uses unsupervised machine learning to segment wholesale customers based on their annual spending behaviour across six product categories.

The aim is to identify meaningful customer groups that can support targeted sales, cross-selling and account-management strategies.

## Dataset

The project uses the publicly available Wholesale Customers dataset from the UCI Machine Learning Repository.

The dataset contains:

- 440 wholesale customers
- 6 annual spending variables:
  - Fresh
  - Milk
  - Grocery
  - Frozen
  - Detergents and Paper
  - Delicatessen
- Channel and Region variables for cluster interpretation

## Methodology

The project includes:

1. Data cleaning and validation
2. Exploratory data analysis
3. Feature engineering
4. Log transformation and standardisation
5. Principal Component Analysis
6. K-Means clustering
7. Hierarchical clustering
8. DBSCAN
9. Model comparison
10. SHAP explainability
11. Bias and fairness analysis

## Key Results

K-Means with two clusters was selected as the final model.

- **Segment 0:** Higher spending on Fresh and Frozen products
- **Segment 1:** Higher spending on Grocery, Milk and Detergents and Paper

K-Means achieved:

- Silhouette Score: 0.290
- Davies–Bouldin Index: 1.352
- Calinski–Harabasz Score: 189.050

## Business Value

The customer segments can support:

- Targeted product promotions
- Cross-selling opportunities
- Customer-specific sales strategies
- Better account prioritisation
- More effective allocation of sales resources

## Repository Structure

```text
data/           Dataset
notebooks/      Full analysis and technical presentation notebooks
presentations/  Technical and business presentations
models/         Saved model and preprocessing files
reports/        Final project report
src/            Reusable Python scripts
