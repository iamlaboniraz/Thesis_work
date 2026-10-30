# Machine Learning for Inventory Demand Forecasting Across Retail Demand Patterns

## 1. Project Overview

This repository contains the ongoing research and Python implementation for my Master's thesis in **Data Science and Advanced Analytics** at **NOVA Information Management School (NOVA IMS), Universidade NOVA de Lisboa**.

Retail demand forecasting is challenging because products exhibit different sales patterns. Some products sell regularly, while others have intermittent, erratic, seasonal, or highly variable demand.

A single forecasting model may not perform equally well across all products.

This research investigates whether selecting forecasting methods according to demand characteristics can improve forecasting accuracy and support better inventory management decisions.

**Research status:** In progress.

## 2. Research Objectives

The main objectives of this research are:

1. Analyze retail sales data and identify different demand characteristics.
2. Classify product-level demand patterns using Average Demand Interval (ADI) and Squared Coefficient of Variation (CV²).
3. Develop and evaluate statistical and machine learning forecasting models.
4. Investigate whether demand-pattern-aware model selection improves forecasting accuracy.
5. Compare forecasting methods using appropriate evaluation metrics.
6. Evaluate the impact of forecasting performance on inventory management, including stockouts, service levels, and inventory-related costs.

## 3. Research Questions

This study aims to address three main research questions:

**RQ1:** How do statistical and machine learning forecasting models perform across different retail demand patterns?

**RQ2:** Can demand-pattern-aware model selection improve forecasting accuracy compared with using a single forecasting approach?

**RQ3:** How do differences in forecasting accuracy affect inventory performance in terms of stockouts, service levels, and inventory costs?

These questions provide the initial research direction and may be refined during the thesis.

## 4. Dataset

### M5 Forecasting Dataset

This project uses the **M5 Forecasting Dataset**, based on Walmart retail sales data.

The dataset includes:

- Historical daily product sales
- Product and store identifiers
- Product categories and departments
- Calendar information
- Special events
- Selling prices

The dataset will be used to study different product–store demand patterns and evaluate forecasting methods.

**Dataset source:** [M5 Forecasting — Accuracy Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

The raw dataset is not included in this GitHub repository.

## 5. Demand Pattern Classification

An important part of this research is classifying retail products according to their demand characteristics.

Two statistical measures are used:

### 5.1 Average Demand Interval (ADI)

ADI measures how frequently non-zero demand occurs.

**Formula:**

ADI = Total number of time periods / Number of periods with non-zero demand

For example, consider seven days of sales:

`[3, 0, 0, 2, 0, 0, 4]`

Total periods = 7

Periods with non-zero demand = 3

ADI = 7 / 3 = **2.33**

A higher ADI indicates that demand occurs less frequently.

### 5.2 Squared Coefficient of Variation (CV²)

CV² measures the variability of non-zero demand quantities relative to their mean.

**Formula:**

CV² = Variance of non-zero demand / (Mean of non-zero demand)²

Where:

- Variance measures the spread of non-zero demand quantities.
- Mean represents their average quantity.

For the previous example:

Non-zero demand = `[3, 2, 4]`

Mean = 3

Population variance = 0.667

CV² = 0.667 / 9 = **0.074**

A lower CV² indicates more stable non-zero demand quantities, while a higher CV² indicates greater variability.

For the initial implementation, population variance (`ddof=0`) is used. This convention will be documented consistently in the research methodology.

### 5.3 Demand Classification Table

The following conventional ADI/CV² thresholds are used for the initial classification.

| Demand Pattern | ADI | CV² | Characteristics |
|---|---|---|---|
| **Smooth** | ≤ 1.32 | ≤ 0.49 | Frequent demand with relatively stable quantities |
| **Intermittent** | > 1.32 | ≤ 0.49 | Infrequent demand with relatively stable quantities |
| **Erratic** | ≤ 1.32 | > 0.49 | Frequent demand with highly variable quantities |
| **Lumpy** | > 1.32 | > 0.49 | Infrequent demand with highly variable quantities |

The exact boundary convention is explicitly defined above.

**Example classification:**

For sales `[3, 0, 0, 2, 0, 0, 4]`:

- ADI = 2.33
- CV² = 0.074
- Demand Pattern = **Intermittent**

This is because the product has infrequent demand but relatively stable positive demand quantities.

**Reference:** Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society, 56*, 495–503.

## 6. Python Implementation

The following examples demonstrate the initial ADI and CV² calculations.

### 6.1 Calculate ADI

```python
import pandas as pd
import numpy as np

def calculate_adi(series):
    series = pd.Series(series)
    nonzero_count = (series > 0).sum()

    if nonzero_count == 0:
        return np.nan

    return len(series) / nonzero_count
```

### 6.2 Calculate CV²

```python
def calculate_cv2(series):
    series = pd.Series(series)
    nonzero = series[series > 0]

    if len(nonzero) < 2:
        return np.nan

    mean = nonzero.mean()
    variance = nonzero.var(ddof=0)

    return variance / (mean ** 2)
```

### 6.3 Classify Demand Pattern

```python
def classify_demand(adi, cv2):
    if pd.isna(adi) or pd.isna(cv2):
        return "Unclassified"

    if adi <= 1.32 and cv2 <= 0.49:
        return "Smooth"

    elif adi > 1.32 and cv2 <= 0.49:
        return "Intermittent"

    elif adi <= 1.32 and cv2 > 0.49:
        return "Erratic"

    else:
        return "Lumpy"
```

### 6.4 Example

```python
product = pd.Series([3, 0, 0, 2, 0, 0, 4])

adi = calculate_adi(product)
cv2 = calculate_cv2(product)
pattern = classify_demand(adi, cv2)

print("ADI:", round(adi, 3))
print("CV²:", round(cv2, 3))
print("Demand Pattern:", pattern)
```

Expected output:

```text
ADI: 2.333
CV²: 0.074
Demand Pattern: Intermittent
```

These examples demonstrate the initial classification approach. They are not the final results of the M5 dataset.

All-zero series and series with fewer than two positive observations are flagged for separate handling. The initial examples assume complete, non-negative demand observations.

## 7. Planned Forecasting Methodology

The research is organized into the following stages.

### Stage 1: Data Preparation

- Load M5 sales, calendar, and price data.
- Examine missing values and data quality.
- Transform data into a suitable time-series format.
- Select the required product–store series.
- Investigate sales variability and zero-demand frequency.

### Stage 2: Demand Classification

- Calculate ADI.
- Calculate CV².
- Classify demand into Smooth, Intermittent, Erratic, and Lumpy categories.
- Analyze the distribution of demand patterns.

### Stage 3: Forecasting Models

Planned candidate methods include:

**Statistical and baseline methods:**
- Naive Forecasting
- Seasonal Naive Forecasting
- Exponential Smoothing
- Croston-based methods for intermittent demand

**Machine learning methods:**
- Random Forest
- Gradient Boosting
- LightGBM

The final model selection will depend on literature findings, data characteristics, and computational feasibility.

### Stage 4: Model Evaluation

Planned forecast evaluation metrics include:

- RMSE — Root Mean Squared Error
- MASE — Mean Absolute Scaled Error

Models will be assessed using time-based evaluation to avoid future data leakage.

Performance will be compared across different demand patterns.

### Stage 5: Demand-Pattern-Aware Model Selection

The research will investigate whether different demand categories benefit from different forecasting methods.

Forecasting models will be selected based on validation performance and evaluated on separate unseen data.

### Stage 6: Inventory Simulation

Forecasting results will be used in a simplified inventory simulation to investigate:

- Stockouts
- Service levels
- Inventory holding costs
- Total inventory-related costs

The objective is to understand whether improved forecasting accuracy also leads to better inventory decisions.

## 8. Research Workflow

The overall planned workflow is:

```text
M5 Retail Dataset
        |
        v
Data Cleaning and Preparation
        |
        v
Exploratory Data Analysis
        |
        v
ADI and CV² Calculation
        |
        v
Demand Pattern Classification
        |
        v
Statistical and ML Forecasting
        |
        v
Forecast Accuracy Evaluation
        |
        v
Pattern-Aware Model Selection
        |
        v
Inventory Simulation
        |
        v
Results and Conclusions
```

## 9. Technologies and Tools

Current and planned tools:

| Tool | Purpose |
|---|---|
| Python | Data analysis and modeling |
| Pandas | Data manipulation |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Jupyter Notebook | Code development and experiments |
| Scikit-learn | Machine learning and evaluation |
| LightGBM | Gradient boosting forecasting experiments |
| Git and GitHub | Version control |
| Zotero | Research literature and citation management |

Not all listed libraries have been implemented yet.

## 10. Project Progress

### Research and Preparation

- [x] Master's thesis topic approved
- [x] Initial research objectives defined
- [x] Literature review started
- [x] Initial study of retail demand forecasting
- [x] ADI and CV² concepts studied

### Implementation

- [x] Initial Python practice with ADI and CV²
- [ ] Full M5 data preparation
- [ ] Exploratory Data Analysis completed
- [ ] Demand pattern classification across selected M5 series
- [ ] Statistical baseline models implemented
- [ ] Machine learning forecasting models implemented
- [ ] Model comparison completed
- [ ] Pattern-aware model selection evaluated
- [ ] Inventory simulation completed
- [ ] Final thesis results and discussion completed

## 11. Literature Review

The literature review focuses on:

1. Retail demand forecasting challenges
2. Intermittent and irregular demand
3. Demand pattern classification using ADI and CV²
4. Statistical versus machine learning forecasting methods
5. Forecast accuracy evaluation
6. Forecasting and inventory management

### Initial Reading

Fildes, R., Ma, S., & Kolassa, S. (2022). Retail forecasting: Research and practice. *International Journal of Forecasting, 38*(4), 1283–1318.

https://doi.org/10.1016/j.ijforecast.2019.06.004

The authors review retail forecasting methods and discuss important challenges involving product-level demand, benchmarking, intermittent demand, machine learning validation, and inventory control.

Further literature will be reviewed to establish the current research gap.

## 12. Repository Organization

The repository is being organized as the project develops.

The following is a proposed structure, not necessarily the current repository structure:

```text
Thesis_work/
|
|-- README.md
|-- .gitignore
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_demand_classification.ipynb
|   |-- 03_baseline_forecasting.ipynb
|   |-- 04_ml_forecasting.ipynb
|   |-- 05_inventory_simulation.ipynb
|
|-- src/
|   |-- demand_classification.py
|   |-- forecasting.py
|   |-- evaluation.py
|
|-- research_notes/
|
|-- dataset/              # Ignored by Git
```

Additional files and folders will be created as required.

## 13. Academic Information

**Thesis Title:** Machine Learning for Inventory Demand Forecasting Across Retail Demand Patterns

**Degree:** Master's in Data Science and Advanced Analytics

**Institution:** NOVA Information Management School (NOVA IMS), Universidade NOVA de Lisboa

**Supervisor:** Professor Mauro Castelli

**Research Type:** Master's Thesis — Project-Based Research

## 14. Project Status and Disclaimer

This repository is maintained for academic research and learning.

The project is currently under development. Forecasting experiments, model comparisons, and inventory simulations are planned and should not be interpreted as completed research findings.

Research methods, experimental settings, and results may be updated as the thesis progresses.
