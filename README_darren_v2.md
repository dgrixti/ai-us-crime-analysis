

---

# **Machine Learning Approaches to Ethical Analysis of Statistics**

## **1. Introduction**

This project explores machine learning techniques to analyze demographic and categorical data for predicting the **"Weapon Category"** (Firearm vs. Non-Firearm). The objectives include:

- Implementing and comparing various machine learning algorithms.
- Evaluating model performance across metrics like precision, recall, and F1-score.
- Developing an interactive tool to visualize and utilize predictions.
- Conducting an ethical analysis of the models, addressing bias and fairness.

The dataset comprises structured tabular data with demographic and contextual information. The workflow encompasses:

- **Data Cleaning and Preprocessing**
- **Feature Engineering and Selection**
- **Model Training and Evaluation**
- **Ethical Review**
- **Interactive Web Tool Development**

---

## **2. Data Preparation**

### **2.1 Data Cleaning**

The dataset was cleaned and prepared using the following steps:
1. **Handling Missing Values**:
   - Rows with missing critical features were removed.
   - Missing categorical values were imputed with the placeholder **"Unknown"**.
2. **Duplicate Removal**:
   - Duplicate records were identified and removed to avoid redundancy.
3. **Data Type Conversion**:
   - Categorical features were explicitly converted to their appropriate data types for efficient preprocessing.

---

### **2.2 Feature Engineering**

Key transformations were introduced to simplify and enhance the dataset:

1. **Age Grouping**:
   - The continuous "Victim Age" feature was categorized into the following groups:
     - Under 10, 10-20, 21-30, 31-40, 41-50, 51-60, 61+.

2. **Season Grouping**:
   - The "Month" feature was grouped into seasons for better interpretability:
     - **Winter**: December, January, February  
     - **Spring**: March, April, May  
     - **Summer**: June, July, August  
     - **Autumn**: September, October, November.

3. **Categorical Encoding**:
   - All categorical features (e.g., State, Victim Sex, Relationship Category) were OneHotEncoded to prepare them for machine learning models.

---

### **2.3 Feature Selection**

- **Included Features**:
  - Demographic data: `Victim Sex`, `Victim Race`, `Relationship Category`.
  - Contextual features: `State`, `Season`, and the newly created `Age Group`.
- **Excluded Features**:
  - Features irrelevant to the target variable or redundant were excluded to reduce noise and improve efficiency.

---

### **2.4 Data Balancing**

The original dataset showed a significant class imbalance between "Firearm" and "Non-Firearm" categories. To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied, ensuring balanced representation of both classes during training.

---

## **3. Experiments**

### **3.1 Machine Learning Models**

Four machine learning models were implemented and compared:

1. **Random Forest Classifier**:
   - Parameters:
     - `n_estimators=100`
     - `class_weight='balanced'`
   - **Result**: High accuracy with balanced performance across both classes.

2. **XGBoost Classifier**:
   - Parameters:
     - `n_estimators=100`
     - `eval_metric='logloss'`
   - **Result**: Faster training and high recall for "Firearm," but struggled with "Non-Firearm."

3. **LightGBM Classifier**:
   - Parameters:
     - `n_estimators=100`
     - `class_weight='balanced'`
   - **Result**: Moderate performance, with improved recall for "Non-Firearm" but lower overall accuracy.

4. **CatBoost Classifier**:
   - Parameters:
     - `iterations=100`
     - `class_weights=[1, 1]`
   - **Result**: High precision and recall for "Firearm" but limited performance for "Non-Firearm."

---

### **3.2 Model Comparisons**

| **Model**       | **Precision (Firearm)** | **Recall (Firearm)** | **F1-Score (Firearm)** | **Precision (Non-Firearm)** | **Recall (Non-Firearm)** | **F1-Score (Non-Firearm)** | **Accuracy** |
|------------------|------------------------|---------------------|------------------------|----------------------------|--------------------------|----------------------------|--------------|
| Random Forest    | 0.85                  | 0.86               | 0.86                  | 0.74                      | 0.72                    | 0.73                      | 0.81         |
| XGBoost          | 0.73                  | 0.92               | 0.82                  | 0.71                      | 0.36                    | 0.48                      | 0.72         |
| LightGBM         | 0.78                  | 0.70               | 0.74                  | 0.52                      | 0.63                    | 0.57                      | 0.68         |
| CatBoost         | 0.73                  | 0.92               | 0.81                  | 0.69                      | 0.36                    | 0.47                      | 0.72         |

---

### **Conclusion**

- **Random Forest** demonstrated the best overall performance, achieving high accuracy and balanced metrics across both classes.
- **XGBoost** and **CatBoost** excelled in "Firearm" predictions but struggled significantly with "Non-Firearm."
- **LightGBM** showed moderate performance, improving recall for "Non-Firearm" but with lower precision.

---

## **4. Ethical Review**

### **4.1 Data Bias**
- Addressed class imbalance using **SMOTE** to ensure fair representation of both classes during training.
- Categorical features were carefully preprocessed to ensure sensitive demographic features (e.g., race, gender) were handled appropriately.

### **4.2 Fairness Considerations**
- **Performance Metrics**: Evaluated models for equitable performance across different demographic groups.
- **Bias Mitigation**: Ensured models did not disproportionately disadvantage any group through careful feature selection and preprocessing.

### **4.3 Privacy Concerns**
- **Data Anonymization**: Ensured all personal identifiers were removed from the dataset.
- **Ethical Data Use**: Followed ethical guidelines to prevent harm to individuals represented in the dataset.

---

## **5. Web Tool Guide**

### **5.1 Overview**

An interactive web tool was developed using **Gradio**, enabling users to input features and receive predictions on the **Weapon Category**.

### **5.2 Features**

- **Model Selection**: Allows users to choose between Random Forest, XGBoost, LightGBM, and CatBoost.
- **Input Fields**: Features such as `State`, `Season`, `Victim Sex`, and `Relationship Category` are available for user input.
- **Output**: Displays the predicted **Weapon Category** along with probabilities.

---

### **5.3 How to Run**

1. **Install Dependencies**:
   ```bash
   pip install gradio pandas joblib xgboost lightgbm catboost
   ```

2. **Run the Tool**:
   ```bash
   python3 gradio_weapon_predictor_with_models.py
   ```

3. **Access the Interface**:
   - The tool can be accessed via the localhost URL provided after running the script.

---

## **6. Results**

### **6.1 Key Findings**
- **Random Forest** emerged as the most balanced and accurate model.
- **XGBoost** and **CatBoost** excelled at "Firearm" predictions but underperformed for "Non-Firearm."
- **LightGBM** achieved moderate performance, with improved recall for "Non-Firearm" but lower overall precision.

### **6.2 Limitations**
- Grouping features (e.g., age and season) reduced granularity, potentially impacting accuracy.
- Bias in model predictions for the "Non-Firearm" class remains a challenge.

---

## **7. Conclusion**

The project successfully implemented and compared machine learning models to predict weapon categories. Random Forest emerged as the best-performing model, while XGBoost and CatBoost showed potential with further tuning. The developed Gradio web tool provides an intuitive interface for real-time predictions.

**Future Work**:
- Hyperparameter tuning for models to improve accuracy.
- Exploring additional features for better class balance.
- Incorporating fairness-focused metrics to assess bias.

---

