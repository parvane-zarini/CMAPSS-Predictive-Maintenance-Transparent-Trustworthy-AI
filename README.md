# CMAPSS Predictive Maintenance – Transparent & Trustworthy AI
This project implements a predictive maintenance system using the NASA **CMAPSS FD001** turbofan engine dataset, with a strong focus on **algorithmic transparency**.  
The goal is not only to predict engine failures but also to explain *why* the model makes a certain prediction, making the system more trustworthy for operators and managers.

 **Project Purpose**
Traditional AI models often behave like **black boxes**.  
In high-risk domains such as aircraft engine maintenance, this lack of transparency reduces human trust.
This project integrates transparency mechanisms into the predictive model so that:
- **Operators** can validate AI decisions using domain knowledge  
- **Managers** can justify maintenance decisions with confidence  
- **Researchers** can study how transparency influences human trust  
The pipeline includes global explanations, local explanations, detailed confusion matrices, and a structured trust-study dataset.


**Project Structure**
CMAPSS-Transparency-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── train_FD001.txt
│ ├── test_FD001.txt
│ └── RUL_FD001.txt
│
├── src/
│ ├── main.py
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── explainability.py
│ ├── trust_export.py
│ └── utils.py
│
└── outputs/
├── confusion_matrix_Train.png
├── confusion_matrix_Validation.png
├── confusion_matrix_Test.png
├── rf_feature_importance.png
├── local_explanation_example.png
└── trust_study_samples_no_shap.csv


 **What the Project Does**

**Data Preparation**
- Computes Remaining Useful Life (RUL) for each engine  
- Converts RUL into a binary risk label:  
  - `1 = engine at risk (RUL ≤ 30 cycles)`  
  - `0 = normal operating condition`  

**Model Training**
A **Random Forest classifier** is used because it offers a good balance of:
- Accuracy  
- Stability  
- Built-in interpretability

  

**Model Evaluation**
The pipeline generates confusion matrices for:
- Train  
- Validation  
- Test  

(Stored in the `outputs/` directory)

** Transparency & Explainability**

** Global Explainability**  
A feature-importance plot shows which sensors are most influential fleet-wide.  
Example: sensors **s11**, **s4**, **s12** typically dominate.

<img width="2400" height="1800" alt="rf_feature_importance (2)" src="https://github.com/user-attachments/assets/f2be9452-b2a5-4b6a-a17d-b24e7e8b6824" />


** Local Explainability  **
Explains why a single engine was labeled as high-risk.  
Shows the top contributing sensors for that specific instance.


<img width="2400" height="1800" alt="local_explanation_example (1)" src="https://github.com/user-attachments/assets/485766e1-a54e-4d54-880b-2264bd13315e" />

 
** Trust Study Export**  
A CSV file is generated containing:
- Engine ID  
- Cycle  
- True label  
- Model output probability  
- Top 3 contributing sensors  
- Their local importance scores

  <img width="1920" height="1440" alt="confusion_<img width="1920" height="1440" alt="confusion_matrix_Test (2)" src="https://github.com/user-attachments/assets/620fb4b9-ecd3-4a42-86c6-03514f1d9a9b" />
matrix_Train (1)" src="https://github.com/user-attachments/assets/a5dba8d0-2e0e-4f15-91e0-8d2f0b03e938" /><img width="1920" height="1440" alt="confusion_matrix_Validation (1)" src="https://github.com/user-attachments/assets/4ff83dec-db5c-4aef-b6de-62deb42b7c58" />



This dataset is ideal for **human trust experiments** or survey-based research.

The outputs (feature-importance plots, local explanations, confusion matrices, and trust-study CSV) are directly used in the case study section of the research.
Key transparency mechanisms included:
Global feature reasoning
Instance-level explanations
Inspectability of model logic
User-interpretable justification of predictions

Technologies Used:
-Python 3.9+
-NumPy
-Pandas
-Scikit-learn
-Matplotlib
-NASA CMAPSS dataset

