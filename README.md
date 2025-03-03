# MSc AI LAB - Thesis Project 🎓🤖

Welcome to the **MSc AI LAB Thesis Project** repository! This project is part of our Master's research in **Artificial Intelligence** at UWE Bristol.

## 🚀 Project Overview
This project focuses on **Explainable AI (XAI) for Dermatological Screening**, aiming to classify **malignant vs. benign tumors** while ensuring model interpretability.

## 📂 Repository Structure
```text
thesis-project/
│– datasets/         # Dataset & preprocessing scripts
│– notebooks/        # Jupyter Notebooks for model development
│– reports/          # Research findings & documentation
│– src/              # Source code for AI model & pipeline
│– README.md         # Project documentation (this file)
│– requirements.txt  # Dependencies
```

### 1. Clone the Repository
```bash
git clone https://github.com/msc-ai-lab/thesis-project.git
cd thesis-project
```

### 2. Set Up Virtual Environment (Python 3.9+)
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebook
```bash
jupyter notebook
```

## 📌 Contribution Workflow
1. Create a feature branch:
```bash
git checkout -b feature/your-feature
```

2.	Commit changes (following Jira issue format):
```bash
git commit -m "feat(#TP-12): Implement dataset preprocessing"
```

3. Push & Open a Pull Request:
```bash
git push origin feature/your-feature
```

4. Request a review and merge into dev.

## 📌 Branching Strategy
- main → Stable production-ready branch (protected)
- dev → Active development branch
- feature/* → Feature branches for new additions
- fix/* → Bug fix branches

## 🤝 Contributors
LAB:
- 👤 Lukasz
- 👤 Avishek
- 👤 Berkay

## 📜 License

This project is for academic research purposes.