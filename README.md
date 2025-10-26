# 🧠 Loan Default Risk Prediction & Offline Reinforcement Learning (CQL + FQE)

This repository covers **Exploratory Data Analysis (EDA)**, **Deep Learning (DL)**, and **Offline Reinforcement Learning (RL)** for the **LendingClub Loan Dataset**.  
It combines traditional predictive modeling with reinforcement learning to explore **data-driven credit risk optimization**.

---

## 📘 Overview

The project is divided into three main parts:

1. **Exploratory Data Analysis (EDA)**
   - Statistical summaries, feature correlations, and outlier analysis.
   - Visualization of borrower characteristics and default patterns.
   - Feature engineering and preprocessing pipeline for numerical and categorical variables.

2. **Deep Learning Model (TensorFlow / Keras)**
   - A **Multi-Layer Perceptron (MLP)** model trained to predict the probability of loan default.
   - Evaluation using accuracy, precision-recall, ROC-AUC, and loss curves.
   - Demonstrates supervised credit-risk prediction.

3. **Offline Reinforcement Learning (d3rlpy)**
   - Implements **Discrete Conservative Q-Learning (CQL)** for offline policy learning.
   - Uses **Fitted Q Evaluation (FQE)** for offline performance estimation.
   - Compares model-based decision policies with traditional predictive models for loan approval optimization.

---


---

## ⚙️ Environment Setup

You can recreate the exact environment using either `pip` or a virtual environment.  
The code is tested with **Python 3.10+** and **d3rlpy 2.8.1**.

### 🧰 Option 1 — Using pip
```bash
git clone https://github.com/DhairyaJain30/ReinforcementLearning_LoanPredictor


# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


