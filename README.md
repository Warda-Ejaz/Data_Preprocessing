
# Day 2 - Data Preprocessing & Machine Learning Models

AI/ML Internship Task - Project Day 2 Implementation

### 1. Task Overview
This notebook implements complete Data Preprocessing pipeline and trains 2 ML regression models to predict student final grades. Models are compared and best performing model is selected based on R² score.

### 2. Tasks Completed
| Section | Steps Implemented |
| --- | --- |
| Data Cleaning | Dataset info, data types check, missing values = 0 |
| Missing Value Handling | Mean imputation with `fillna(df.mean())` |
| Duplicate Removal | `df.duplicated().sum()` check + `drop_duplicates()` |
| Feature Selection | Correlation heatmap + dropped target from X |
| Data Visualization | Heatmap for feature correlation analysis |
| Train-Test Split | 80/20 split with `random_state=42` |
| ML Models | Linear Regression + Decision Tree Regressor |
| Model Comparison | MAE & R² metrics comparison table |

### 3. Dataset
File: `student_data.csv`  
Records: 500 students  
Features: 5 input features  
1. study_hours_per_day 
2. attendance_percentage
3. assignments_completed 
4. previous_semester_marks
5. class_participation  
 Target: final_performance_grade - 0 to 100

### 4. Model Results
| Model | MAE | R² Score |
| --- | --- | --- |
| Linear Regression | ~3.45 | ~0.89 |
| Decision Tree Regressor | ~3.12 | ~0.91 |

Best Model Selected: Decision Tree Regressor  
Reason: Higher R² = 0.91 vs 0.89. Better at capturing non-linear relationships between study habits and grades.

### 5. Tech Stack
Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn  
Environment: Jupyter Notebook

### 6. How to Run
1. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Open `Day2_DataPreprocessing_Modeling.ipynb` 
3. Run all cells: `Kernel → Run All Cells`
4. Dataset auto-generates if `student_data.csv` not found

### 7. Project Structure
Day2_DataPreprocessing_WardaEjaz/
├── Day2_DataPreprocessing_Modeling.ipynb
├── student_data.csv
├── README.md
└── Day2_Demo.mp4

### 8. Key Insights
1. All features show positive correlation with final grade
2. `study_hours_per_day` has highest feature importance in Decision Tree
3. No missing values or duplicates found in synthetic dataset
4. StandardScaler applied before training to avoid scaling bias

---
Submitted by: Warda Ejaz  
Date:  16-June2026  
Task: AI/ML Internship - Day 2
