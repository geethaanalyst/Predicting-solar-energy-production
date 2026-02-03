# Predicting-solar-energy-production
Develop a machine learning model to accurately predict the annual solar energy production for future installations, considering variables such as developer, region, and equipment. By leveraging this model, the company aims to provide customers with informed choices regarding developers, equipment, and locations, thereby optimizing project planning and maximizing returns on investment.

## **Introduction**
Develop a machine learning model to predict annual solar energy production for future installations, using factors such as developer, location, and system characteristics. 
The goal is to enable data-driven decisions that optimize project planning, equipment selection, and maximize return on investment for customers.

## **Dataset Overview**
<img width="823" height="302" alt="image" src="https://github.com/user-attachments/assets/da4edf95-b2fa-44fb-b13a-a49c7ffa1704" />

## **Data Preprocessing**
* **Checked for missing values: Missing data was found in several columns.**
* **Cleaned and imputed missing categorical and numerical features.**
* **Numeric columns filled with median values.**
* **Categorical columns filled with 'Unknown’.**
* **Dropped non-informative or mostly missing columns: 'Data Through Date', 'Project ID', 'Division', 'Substation', 'Circuit ID’, 'Energy Storage System Size (kWac)’.**
* **Handled skewness in numeric features using log transformation for highly skewed features (Zip, Interconnect_Year, County_Avg_Size).**
* **Converted categorical features to numeric using Ordinal Encoding.**
* **Using standardscaler for numeric features.**
* **Split data into training (80%) and testing (20%) sets for model building.**

## **Machine Learning Models**
All models were evaluated using metrics MAE, MSE, RMSE and R² Score to ensure robust performance comparison.

### **Models Used**
* **Random Forest Regressor**
* **XGBoost Regressor**
* **LightGBM Regressor**
### **Hyperparameter tuning**
* **Best parameter in Random forest  max depth: 15, min_sample_split: 2, min_sample_leaf: 1.**
* **Best parameter in XGBoost max depth: 8, learning rate: 0.01.**
* **Best parameter in LightGBM max depth: 30, number of leaves: 50, learning rate: 0.05.**

## **Model Performance**
<img width="856" height="453" alt="image" src="https://github.com/user-attachments/assets/9cfd69b8-5f7e-421a-9ac4-37b7b3cf06f3" />

## **Exploratory Data Analysis**
<img width="982" height="593" alt="image" src="https://github.com/user-attachments/assets/c837f7f9-377b-44c7-8a1b-06fe7c442fb9" />

