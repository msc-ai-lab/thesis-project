# MSc AI LAB - Thesis Project 🎓🤖

Welcome to the **MSc AI LAB Thesis Project** repository! This project is part of our Master's research in **Artificial Intelligence** at UWE Bristol.

## 🚀 Project Overview
This project focuses on **Explainable AI (XAI) for Dermatological Screening**, aiming to classify **malignant vs. benign tumors** while ensuring model interpretability using XAI methods and Large Language Models (LLMs).

## 📂 Repository Structure
```text
thesis-project/
│
├── data/             # Contains raw and processed datasets.
├── models/           # Stores final trained model artifacts (.pth files).
├── notebooks/        # Jupyter Notebooks for the project workflow.
├── results/          # Destination for outputs like XAI images and reports.
├── user_inputs/      # For placing sample images for XAI analysis.
│
├── .gitignore        # Specifies files for Git to ignore.
├── README.md         # Project documentation (this file).
└── requirements.txt  # Project dependencies.
```

## Local Environment Setup
This project uses a dedicated Python virtual environment to ensure all team members have an identical and reproducible setup.

1. Clone the Repository
    ```bash
    git clone [https://github.com/msc-ai-lab/thesis-project.git](https://github.com/msc-ai-lab/thesis-project.git)
    cd thesis-project
    ```

2. Set Up Virtual Environment (Python 3.9+)
This command creates a local, self-contained environment folder named `thesis-env`.
    ```bash
    python3 -m venv thesis-env
    ```

3. Activate the Environment
To start using the environment, you need to activate it.
    - On macOS/Linux:
    ```bash
    source thesis-env/bin/activate
    ```

    - On Windows:
    ```bash
    .\thesis-env\Scripts\activate
    ```

Your terminal prompt will change to show `(thesis-env)` to indicate it's active.

4. Install Dependencies
This command reads the `requirements.txt` file and installs the exact versions of all necessary packages into your new environment.

```bash
pip install -r requirements.txt
```

## 🔬 Project Workflow
The project is structured as a sequence of Jupyter notebooks that should be run in order to generate the necessary data and models.

First, start Jupyter Lab from your activated environment:

```bash
jupyter lab
```

### Execution Order:
1. `01_data_exploration.ipynb`: Cleans the raw metadata and saves `metadata_updated.csv`.
2. `02_data_preparation.ipynb`: Splits the data and saves processed training/validation/test sets as `.pt` files.
3. `03_model.ipynb`: Trains the CNN model and saves the final `cnn_trained_model.pth`.
4. `04_xai_methods.ipynb`: Loads the trained model to generate the final XAI outputs (Grad-CAM, SHAP, etc.).

## 📌 Contribution Workflow
1. Create a feature branch:
```bash
git checkout -b feature/your-feature
```
2. Commit changes (following Jira issue format):
```bash
git commit -m "feat(#TP-12): Implement dataset preprocessing"
```
3. Push & Open a Pull Request:
```bash
git push origin feature/your-feature
```
4. Request a review and merge into `dev`.

## 📌 Branching Strategy
- main → Stable production-ready branch (protected)
- dev → Active development branch
- feature/* → Feature branches for new additions
- fix/* → Bug fix branches

## 🤝 Contributors
### LAB:
- 👤 Lukasz
- 👤 Avishek
- 👤 Berkay