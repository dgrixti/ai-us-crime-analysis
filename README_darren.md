# **Machine Learning Approaches to Ethical Analysis of Statistics**

## **1. Introduction**

This project explores **Machine Learning Approaches** to analyze demographic and categorical data to predict the **"Weapon Category"** (Firearm vs. Non-Firearm) based on input features such as victim and perpetrator characteristics. The goal is to experiment with different ML techniques, evaluate their accuracy, and develop an interactive tool for model usage.

The dataset used is **tabular data** comprising demographic features, extracted from structured data files. The project workflow includes:

- Data Cleaning and Feature Engineering
- Application of Machine Learning Algorithms
- Comparison of Model Performance Metrics
- Ethical Analysis of Predictions
- Building an Interactive Web Tool

---

## **2. Data Preparation**

### **2.1 Data Cleaning**
The raw data was cleaned and preprocessed to handle inconsistencies:
- **Missing Values**:
   - Removed rows with missing or incomplete information for essential features.
   - Filled missing categorical values with "Unknown."
- **Duplicated Entries**:
   - Checked and removed duplicates.

### **2.2 Feature Engineering**
We introduced new features and transformations to improve model performance:
1. **Age Grouping**:
   - Victim ages were grouped into categories:
     - Under 10, 10-20, 21-30, 31-40, 41-50, 51-60, 61+.
2. **Season Grouping**:
   - Months were grouped into seasons:
     - Winter: December, January, February  
     - Spring: March, April, May  
     - Summer: June, July, August  
     - Autumn: September, October, November.
3. **Categorical Encoding**:
   - OneHotEncoder was applied to all categorical features, including:
     - State, Victim Sex, Victim Race, Relationship Category, and others.

### **2.3 Balancing the Data**
To address class imbalance, we applied **SMOTE (Synthetic Minority Oversampling Technique)**. This ensured both classes, "Firearm" and "Non-Firearm," had equal representation for better training performance.

---

## **3. Experiments**

### **3.1 ML Techniques**
We implemented and compared three machine learning techniques to predict the **Weapon Category**:

1. **Random Forest Classifier**:
   - Used as a baseline ensemble model.  
   - Parameters:  
     - `n_estimators=100`  
     - `class_weight='balanced'`  
   - **Result**: High accuracy and balanced performance across both classes.

2. **XGBoost Classifier**:
   - Gradient boosting algorithm optimized for speed and performance.  
   - Parameters:  
     - `n_estimators=100`  
     - `eval_metric='logloss'`  
   - **Result**: Faster training, but accuracy was lower for minority classes ("Non-Firearm").

3. **Other Algorithms**:
   - Suggestions include **LightGBM** and **CatBoost** for future experimentation.

---

### **3.2 Model Comparisons**

| **Model**       | **Precision (Firearm)** | **Recall (Firearm)** | **Precision (Non-Firearm)** | **Recall (Non-Firearm)** |
|------------------|------------------------|---------------------|----------------------------|--------------------------|
| Random Forest    | 0.85                  | 0.86               | 0.74                      | 0.72                    |
| XGBoost          | 0.73                  | 0.92               | 0.71                      | 0.36                    |

**Conclusion**: Random Forest achieved better accuracy and balance between classes, while XGBoost excelled in precision for "Firearm" but struggled with "Non-Firearm."

---

## **4. Ethical Review**

### **4.1 Data Bias**
The model predictions showed disparities in class performance:
- **XGBoost** exhibited bias against the "Non-Firearm" class.
- **Feature Groupings**:
   - Grouping ages and seasons simplified features but may have reduced model granularity.

### **4.2 Fairness Considerations**
- The dataset includes sensitive demographic features (e.g., race, gender). Ethical considerations were applied to ensure no specific group was disproportionately disadvantaged.

### **4.3 Privacy Concerns**
- The data used was anonymized, ensuring no personal identifiers were included.

---

## **5. Web Tool Guide**

### **5.1 Overview**
An interactive web tool was developed using **Gradio**. The tool allows users to:
1. Input key features (e.g., State, Victim Sex, Relationship Category).
2. Select the model to use (Random Forest or XGBoost).
3. Obtain predictions for **Weapon Category** (Firearm vs. Non-Firearm) with probabilities.

### **5.2 Features**
- **Model Selection**:  
   Dropdown to choose between Random Forest and XGBoost models.
- **Random Input**:  
   Default values are pre-filled randomly to speed up testing.
- **Output**:  
   Displays the likelihood of each category.

### **5.3 How to Run**
1. Install dependencies:
   ```bash
   pip install gradio pandas joblib xgboost
   ```
2. Run the Gradio script:
   ```bash
   python3 gradio_weapon_predictor_with_models.py
   ```
3. Access the tool via the provided **localhost URL**.

---

## **6. Results**

### **6.1 Key Findings**
- Random Forest performed consistently across both classes.
- XGBoost, while faster, exhibited lower recall for "Non-Firearm."

### **6.2 Limitations**
- Grouping of numerical features (e.g., age, seasons) simplified the data but might have reduced accuracy.
- Further experimentation with hyperparameter tuning is needed for XGBoost.

---

## **7. Conclusion**
The project successfully implemented and compared machine learning models to predict weapon categories. An interactive Gradio web tool was built for real-time predictions. Random Forest emerged as the best-performing model, though XGBoost showed potential with further tuning.

**Future Work**:
- Testing additional algorithms like **LightGBM** and **CatBoost**.
- Addressing model bias and fairness.
- Refining feature engineering to improve accuracy.

---

## **8. References**
- Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
- XGBoost Documentation: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Gradio Library: [https://gradio.app](https://gradio.app)

---
