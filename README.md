# 🎓 Student Performance Prediction System

A machine learning web app that predicts whether a student is likely to **pass or fail**, and estimates their **expected final grade**, based on academic, family, and lifestyle factors — built with scikit-learn and deployed as an interactive Streamlit app.

🔗 **Live App:** [https://studentperformanceprediction-5u8wusvejx8dhaddvp9rfm.streamlit.app/]
📂 **Dataset:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

## 📌 Project Overview

This project uses the UCI Student Performance dataset (395 students from two Portuguese secondary schools) to predict academic outcomes based on 30 features — including study time, past failures, family background, and lifestyle habits.

**Two prediction modes:**
- **Classification** — Pass / At-risk (with confidence %)
- **Regression** — Predicted final grade (0–20 scale)

A key design choice: the model **does not use G1/G2** (the student's earlier-period grades) as inputs. This makes the prediction genuinely useful — it estimates outcomes from background and behavioral factors *before* those grades exist, rather than trivially predicting a grade from an almost-identical prior grade.

## 🛠️ Tech Stack

Python, pandas, numpy, scikit-learn, Streamlit, matplotlib/seaborn, Google Colab

## 📊 Model Performance

| Model | Task | Metric | Score |
|---|---|---|---|
| Random Forest (tuned) | Classification | Accuracy | [0.6582278481012658] |
| Random Forest (tuned) | Regression | RMSE / R² | 3.69 / 0.337 |

## 🔍 Features Used

Demographics, family background (parents' education/jobs), study time, past failures, school/family support, alcohol consumption, health, absences, and more — 30 features total.

## 🚀 How to Run Locally

\`\`\`bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 💡 Future Improvements

- Merge the Portuguese-language dataset for more training data
- Add SHAP explanations for individual predictions
