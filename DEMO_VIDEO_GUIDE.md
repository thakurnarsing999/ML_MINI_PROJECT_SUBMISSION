# Demo Video Presentation & Recording Guide

This guide provides instructions and a ready-to-read script for recording the 3–5 minute project demo video required for the **MNR University Machine Learning Mini Project Submission**.

---

## 📹 How to Record Your Screen on Windows

1. **Option A: Built-in Windows Game Bar (Easiest)**
   - Press `Win + Alt + R` to instantly start recording your screen.
   - Speak through your microphone clearly.
   - Press `Win + Alt + R` again to stop recording.
   - The video is saved to `Videos\Captures`.

2. **Option B: OBS Studio or Clipchamp**
   - Use OBS Studio or Microsoft Clipchamp (built-in Windows 11 app) to record screen and audio.

---

## 🎙️ 3–5 Minute Presentation Script

### ⏱️ Minute 0:00 – 0:45 | Introduction & Problem Statement
> *"Hello respected faculty and evaluator. My name is Thakur Narsingh Singh (Roll No: 25RBAIM0027), along with my team member B. Sam Soloman (Roll No: 25RBAIM0021), from MNR University, School of Engineering, Technology & AI, Section A, 2nd year 1st semester. Today we present our Machine Learning Mini Project entitled **'Employee Attrition Prediction Using Machine Learning'** under the guidance of Dr. M. Praveen.*
>
> *Employee turnover is a major challenge for enterprises, costing between 50% to 200% of an employee's annual salary to replace them. Our goal is to build an end-to-end predictive system using 30 enterprise attributes to predict whether an employee will leave or stay, shifting HR from reactive exit interviews to proactive retention."*

### ⏱️ Minute 0:45 – 1:30 | Dataset & Preprocessing Architecture
> *"We benchmarked our models using the real-world IBM HR Analytics dataset with 1,470 employee records. The dataset has zero missing values. We dropped four constant non-predictive columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`), leaving 30 predictive features.*
>
> *We split the data into 80% training and 20% testing sets using stratified sampling to preserve the 16.12% turnover class proportion. To strictly prevent data leakage, we encapsulated `StandardScaler` inside Scikit-learn `Pipeline` architectures."*

### ⏱️ Minute 1:30 – 3:00 | Code Demonstration & Model Results
*(Switch screen to VS Code / Terminal or Jupyter Notebook and run `python employee_attrition_prediction.py`)*
> *"Now let us demonstrate the live execution. Running our Python pipeline trains and benchmarks six distinct classification algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine with an RBF kernel, and Gaussian Naive Bayes using 5-fold stratified cross-validation.*
>
> *Looking at our results table: Logistic Regression achieved the highest overall test accuracy of **87.41%**, an F1-score of **0.4932**, cross-validation accuracy of **86.90%**, and an Area Under the ROC Curve of **0.8057**. Random Forest feature importance analysis shows that Monthly Income, Age, Total Working Years, OverTime, and Distance From Home are the predominant determinants of attrition.*
>
> *On our single-employee live test record, the model accurately flagged a high-risk churner with a 74.72% attrition probability, matching the ground truth."*

### ⏱️ Minute 3:00 – 3:45 | Conclusion & Ethical AI
> *"In conclusion, regularized linear models like Logistic Regression excel on enterprise tabular workforce data because economic and workload factors have strong monotonic trends. Finally, from an ethical standpoint, our model is designed strictly as an advisory decision-support tool to facilitate empathetic mentorship, and never for automated dismissals.*
>
> *Thank you very much."*

---

## 🌐 How to Upload and Share Your Video Link

1. **YouTube (Recommended):**
   - Go to [YouTube Studio](https://studio.youtube.com).
   - Click **Create → Upload videos**.
   - Set visibility to **Unlisted** (anyone with the link can view; not listed publicly).
   - Copy the video URL (e.g., `https://youtu.be/your-video-id`).

2. **Google Drive:**
   - Upload the MP4 file to Google Drive.
   - Right-click → **Share** → Change to **"Anyone with the link can view"**.
   - Copy the shareable link.

3. **Update `README.md`:**
   - Open `README.md` and replace the demo video placeholder with your actual link:
   ```markdown
   [![Watch Project Demo Video](https://img.shields.io/badge/YouTube-Project%20Demo%20Video-red?logo=youtube)](https://youtu.be/YOUR_ACTUAL_VIDEO_LINK)
   ```
