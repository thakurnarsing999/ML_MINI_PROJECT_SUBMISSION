"""
========================================================================================
MNR UNIVERSITY — SCHOOL OF ENGINEERING, TECHNOLOGY AND AI
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING)
MACHINE LEARNING MINI PROJECT (COURSE CODE: 25CSM201, SEMESTER: 2-1)
----------------------------------------------------------------------------------------
Project Title : Employee Attrition Prediction Using Machine Learning
Project Guide : Dr. M. Praveen
Team Members  : 1. B. Sam Soloman        (Roll No: 25RBAIM0021)
                2. Thakur Narsingh Singh  (Roll No: 25RBAIM0027)
========================================================================================
Description:
An end-to-end machine learning classification solution engineered to predict employee
turnover risk using 30 enterprise organizational attributes from the IBM HR Analytics
dataset. Benchmarks six supervised classifiers (Logistic Regression, Decision Tree,
Random Forest, K-Nearest Neighbors, Support Vector Machine, and Gaussian Naive Bayes)
with leakage-free Scikit-learn Pipelines and 5-fold stratified cross-validation.
========================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
)

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def find_dataset(filename="attrition.csv"):
    """Search for the dataset in local or parent directories."""
    candidate_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "data", filename)
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate dataset '{filename}'. Please place it in the project root or data/ directory.")


def load_and_preprocess_data(dataset_path):
    """Load dataset, drop non-informative constants, and encode categoricals."""
    print("=" * 75)
    print("STEP 1: DATA INGESTION & AUDITING")
    print("=" * 75)
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset successfully from: {dataset_path}")
    print(f"Raw dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Verify missing values
    missing = df.isnull().sum().sum()
    print(f"Missing / Null values detected: {missing} (Dataset completeness verified)")
    
    # Drop zero-variance and identification constants
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    dropped = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=dropped)
    print(f"Dropped non-predictive constant columns: {dropped}")
    print(f"Working dimensions: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns (30 features + 1 target)")

    # Exploratory Class distribution check
    attr_counts = df_clean['Attrition'].value_counts()
    stayed = attr_counts.get('No', 0)
    left = attr_counts.get('Yes', 0)
    total = len(df_clean)
    print("\nClass Distribution:")
    print(f"  • Retained (Class 0 / 'No') : {stayed} ({stayed/total*100:.2f}%)")
    print(f"  • Churned  (Class 1 / 'Yes'): {left} ({left/total*100:.2f}%)")
    print(f"  • Class Imbalance Ratio     : {stayed/left:.2f} : 1")

    # Encode categorical text features
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
    
    # Map target variable
    if df_clean['Attrition'].dtype == object:
        df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0}).astype(int)

    return df, df_clean, label_encoders


def generate_exploratory_plots(df_raw, df_clean, viz_dir):
    """Generate and save publication-grade exploratory visualizations."""
    os.makedirs(viz_dir, exist_ok=True)
    print("\n" + "=" * 75)
    print("STEP 2: EXPLORATORY DATA VISUALIZATIONS")
    print("=" * 75)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(7, 5))
    class_counts = df_raw['Attrition'].value_counts()
    bars = plt.bar(['Stayed (0)', 'Left (1)'], 
                   [class_counts.get('No', 0), class_counts.get('Yes', 0)], 
                   color=['#1f77b4', '#d95f02'], width=0.45)
    plt.title('Class Distribution (Employee Attrition)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Attrition Status', fontsize=11)
    plt.ylabel('Number of Employees', fontsize=11)
    plt.ylim(0, 1450)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df_raw)) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f'{yval} ({pct:.1f}%)', 
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(viz_dir, 'figure_1_class_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved -> {p1}")

    # 2. Monthly Income Distribution
    plt.figure(figsize=(8, 5))
    plt.hist([df_raw[df_raw['Attrition']=='No']['MonthlyIncome'], 
              df_raw[df_raw['Attrition']=='Yes']['MonthlyIncome']],
             bins=20, stacked=True, color=['#7ea6c9', '#d95f02'], 
             label=['Stayed (Attrition=0)', 'Left (Attrition=1)'], alpha=0.9)
    plt.title('Monthly Income Distribution by Attrition Status', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Monthly Income ($)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    p2 = os.path.join(viz_dir, 'figure_2_income_distribution.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved -> {p2}")

    # 3. Correlation Heatmap
    corr_features = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 
                     'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                     'YearsWithCurrManager', 'Attrition']
    corr_matrix = df_clean[corr_features].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-0.4, vmax=1.0, 
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Correlation Heatmap of Key Workforce Metrics with Attrition', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    p3 = os.path.join(viz_dir, 'figure_3_correlation_heatmap.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  Saved -> {p3}")


def benchmark_models(df_clean, viz_dir):
    """Partition data, train 6 classification models, evaluate and plot metrics."""
    print("\n" + "=" * 75)
    print("STEP 3: STRATIFIED PARTITIONING & MODEL BENCHMARKING")
    print("=" * 75)

    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition']

    # 80:20 Stratified Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} records (80.0%)")
    print(f"Testing set : {X_test.shape[0]} records (20.0%)")
    print(f"Test Class Balance: Stayed={( y_test==0 ).sum()} | Churned={( y_test==1 ).sum()} (Exact 16.12% churn preserved)")

    # Define Six Classifiers with Pipelines for scale-dependent models
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN (K=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'Gaussian Naive Bayes': GaussianNB()
    }

    # 5-fold Stratified Cross-Validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    probs = {}
    preds = {}

    print("\nTraining and evaluating 6 classification models...")
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]
        
        preds[name] = p
        probs[name] = pr

        acc = accuracy_score(y_test, p)
        prec = precision_score(y_test, p, zero_division=0)
        rec = recall_score(y_test, p, zero_division=0)
        f1 = f1_score(y_test, p, zero_division=0)
        auc = roc_auc_score(y_test, pr)

        results.append({
            'Model': name,
            'CV_Acc': cv_acc,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        })

    comparison_df = pd.DataFrame(results)
    print("\n" + "-" * 75)
    print("MODEL PERFORMANCE COMPARISON (HELD-OUT 20% TEST SET):")
    print("-" * 75)
    print(comparison_df.round(4).to_string(index=False))
    print("-" * 75)

    # Champion Model Selection (Ranked by F1-Score with AUC as tie-breaker)
    best_row = comparison_df.sort_values(['F1', 'AUC'], ascending=False).iloc[0]
    best_model_name = best_row['Model']
    print(f"\nCHAMPION MODEL SELECTED: {best_model_name}")
    print(f"  • Test Accuracy : {best_row['Accuracy']*100:.2f}%")
    print(f"  • Precision     : {best_row['Precision']:.4f}")
    print(f"  • Recall        : {best_row['Recall']:.4f}")
    print(f"  • F1-Score      : {best_row['F1']:.4f}")
    print(f"  • ROC-AUC       : {best_row['AUC']:.4f}")

    # Generate Evaluation Diagnostic Charts
    # 4. Model Accuracy Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    acc_vals = comparison_df['Accuracy'].values
    model_names = comparison_df['Model'].values
    bars = plt.bar(model_names, acc_vals, color='#2c5282', width=0.52)
    plt.title('Comparison of Models Using Accuracy', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Test Accuracy', fontsize=11)
    plt.ylim(0.70, 0.95)
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval*100:.2f}%', 
                 ha='center', va='bottom', fontweight='bold', fontsize=9.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p4 = os.path.join(viz_dir, 'figure_4_model_accuracy_comparison.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"\n  Saved -> {p4}")

    # 5. ROC Curves for all 6 models
    plt.figure(figsize=(7, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for (name, pr), col in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, pr)
        auc_val = roc_auc_score(y_test, pr)
        plt.plot(fpr, tpr, label=f'{name} ({auc_val:.3f})', color=col, linewidth=1.75)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance (0.500)')
    plt.title('ROC Curves for All Six Models', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    p5 = os.path.join(viz_dir, 'figure_5_roc_curves.png')
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"  Saved -> {p5}")

    # 6. Confusion Matrix for Champion Model (Logistic Regression)
    cm = confusion_matrix(y_test, preds['Logistic Regression'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Stayed (0)', 'Left (1)'], yticklabels=['Stayed (0)', 'Left (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix: Logistic Regression', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted label', fontsize=11)
    plt.ylabel('True label', fontsize=11)
    plt.tight_layout()
    p6 = os.path.join(viz_dir, 'figure_6_confusion_matrix.png')
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"  Saved -> {p6}")
    print("\nConfusion Matrix Breakdown:")
    print(f"  • True Negatives  (TN) : {cm[0, 0]} (Correctly predicted Retained)")
    print(f"  • False Positives (FP) : {cm[0, 1]} (False Alarms)")
    print(f"  • False Negatives (FN) : {cm[1, 0]} (Missed Departures)")
    print(f"  • True Positives  (TP) : {cm[1, 1]} (Correctly predicted Departures)")

    # 7. KNN Sensitivity Analysis (K=1 to 20)
    k_vals = list(range(1, 21))
    k_accs = []
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr_sc, y_train)
        k_accs.append(accuracy_score(y_test, knn.predict(X_te_sc)))

    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, k_accs, marker='o', linewidth=2, color='#1f77b4')
    plt.title('Effect of K in K-Nearest Neighbors (KNN)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('K (Number of Neighbors)', fontsize=11)
    plt.ylabel('Test Accuracy', fontsize=11)
    plt.xticks(range(1, 21, 2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p7 = os.path.join(viz_dir, 'figure_7_knn_k_sensitivity.png')
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"  Saved -> {p7}")

    # 8. Feature Importance Ranking (Random Forest)
    rf = models['Random Forest']
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    top10_feats = feat_df.sort_values('Importance', ascending=True).tail(10)
    
    plt.figure(figsize=(8.5, 5))
    plt.barh(top10_feats['Feature'], top10_feats['Importance'], color='#d95f02', edgecolor='#7f2704', height=0.6)
    plt.title('Top 10 Feature Importances (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Gini Importance Score', fontsize=11)
    plt.tight_layout()
    p8 = os.path.join(viz_dir, 'figure_8_feature_importances.png')
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"  Saved -> {p8}")

    print("\nTop 10 Random Forest Features Driving Attrition:")
    for _, row in top10_feats.sort_values('Importance', ascending=False).iterrows():
        print(f"  • {row['Feature']:22s} : {row['Importance']:.4f}")

    # Single-sample predictive demonstration (Test sample matching Chapter 21.8)
    print("\n" + "=" * 75)
    print("STEP 4: SINGLE-EMPLOYEE REAL-TIME PREDICTIVE DEMONSTRATION")
    print("=" * 75)
    sample_emp = X_test.iloc[[0]]
    true_label = y_test.iloc[0]
    best_clf = models[best_model_name]
    pred_label = best_clf.predict(sample_emp)[0]
    prob_churn = best_clf.predict_proba(sample_emp)[0][1]

    print("Sample Employee Profile (From Test Set):")
    print(f"  • Age                     : {sample_emp['Age'].values[0]}")
    print(f"  • Monthly Income          : ${sample_emp['MonthlyIncome'].values[0]:,}")
    print(f"  • Total Working Years     : {sample_emp['TotalWorkingYears'].values[0]}")
    print(f"  • Years at Company        : {sample_emp['YearsAtCompany'].values[0]}")
    print(f"  • OverTime (1=Yes, 0=No)  : {sample_emp['OverTime'].values[0]}")
    print(f"  • Distance From Home (mi) : {sample_emp['DistanceFromHome'].values[0]}")
    print("\nInference Output:")
    print(f"  • Ground Truth Label      : {'Left (1)' if true_label == 1 else 'Stayed (0)'}")
    print(f"  • Predicted Class         : {'Left (1)' if pred_label == 1 else 'Stayed (0)'}")
    print(f"  • Churn Probability       : {prob_churn*100:.2f}%")
    print(f"  • Decision Match          : {'CORRECT PREDICTION' if pred_label == true_label else 'MISMATCH'}")

    print("\n" + "=" * 75)
    print("PROJECT EXECUTION COMPLETE — ALL DELIVERABLES GENERATED")
    print("=" * 75)


def main():
    print("""
    =============================================================
     MNR UNIVERSITY — SCHOOL OF ENGINEERING, TECHNOLOGY & AI
     EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING
     Lab Experiment 30: End-to-End Real-World Machine Learning
    =============================================================
    """)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = find_dataset("attrition.csv")
    viz_dir = os.path.join(base_dir, "visualizations")

    df_raw, df_clean, encoders = load_and_preprocess_data(dataset_path)
    generate_exploratory_plots(df_raw, df_clean, viz_dir)
    benchmark_models(df_clean, viz_dir)


if __name__ == "__main__":
    main()
