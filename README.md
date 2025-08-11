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

First, ensure Jupyter Lab itself is installed in your new environment.

```bash
pip install jupyterlab ipykernel
```

Next, install all project-specific libraries from the requirements file.

```bash
pip install -r requirements.txt
```

This command reads the `requirements.txt` file and installs the exact versions of all necessary packages into your new environment.

5. Create .env file that includes content from .env.example, with your own OpenAI API key.  

## 🚀 Running the Project
After setting up the environment, follow these steps to run the project notebooks.

1. Link Your Environment to Jupyter
This important one-time command makes the `thesis-env` selectable as a "kernel" inside Jupyter Lab.

```bash
python -m ipykernel install --user --name=thesis-env --display-name="Python (thesis-env)"
```

2. Start Jupyter Lab
Now, launch the Jupyter Lab interface from your terminal.

```bash
jupyter lab
```

3. Select the Correct Kernel in Your Notebook

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
5. `05_llm_api.ipynb` : Loads XAI outputs from the previous notebook and, using a set of instructions, prompts LLM to generate a user-friendly interpretation

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

# 🐳 Docker Deployment Guide

Comprehensive containerized deployment for the Skin Cancer Detection system using Docker Compose with separate frontend and backend services.

## 📋 Prerequisites

- **Docker Desktop** 4.0+ with Docker Compose
- **8GB+ RAM** and **10GB+ disk space**

## 🏗️ Architecture Overview

| Service | Technology | Size | Port | Purpose |
|---------|------------|------|------|---------|
| **Backend** | FastAPI + PyTorch | ~2.47 GB | 8000 | AI inference, XAI, LLM reports |
| **Frontend** | React + Nginx | ~50 MB | 80 | Professional medical UI |

## 🚀 Quick Start

```bash
# Clone and navigate
cd thesis-project

# Build and start all services
docker compose up --build

# Access applications
# Frontend: http://localhost
# API Docs: http://localhost:8000/docs
# Health Check: http://localhost:8000
```

## 📊 Expected Build Process

**Build Time:** ~2-3 minutes total
- Backend: ~84s (includes PyTorch + ML dependencies)
- Frontend: ~32s (React build + Nginx setup)

**Startup Logs to Expect:**
```bash
✔ Service backend   Built                84.1s 
✔ Service frontend  Built                32.5s 
```

**Runtime Initialization:**
```bash
# Backend model download
Downloading: "resnet34-b627a593.pth"
100%|██████████| 83.3M/83.3M [00:02<00:00, 29.9MB/s]
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.

# Frontend ready
nginx/1.28.0 ready for start up
```

## 🔍 Verification & Testing

### Docker Desktop Status
- **Containers**: Both services showing 🟢 green status
- **CPU Usage**: ~51% during model loading, then stabilizes
- **Memory**: ~454MB baseline usage

### Health Checks
```bash
# Backend API
curl http://localhost:8000
# Response: {"status": "API is running", "model_loaded": true}

# Frontend
curl http://localhost
# Response: HTML content of React app
```

### Complete Pipeline Test
1. **Access**: http://localhost
2. **Upload**: Dermatological image (JPEG/PNG)
3. **Analyze**: Click "Run Analysis" (30-60s processing)
4. **Results**: Prediction + Confidence + LLM Report + XAI Visualizations

## 🛠️ Development Features

### Hot Reload Enabled
- **Backend**: Auto-reload on code changes in `backend/app/`
- **Frontend**: Rebuild required for changes in `frontend/src/`

### Useful Commands
```bash
# View logs
docker compose logs -f

# Restart specific service
docker compose restart backend

# Clean rebuild
docker compose down && docker compose up --build
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port conflicts** | `lsof -i :80 -i :8000` - stop conflicting services |
| **Memory issues** | Allocate 4GB+ to Docker Desktop |
| **Build failures** | `docker builder prune -a` then rebuild |
| **Model download fails** | Check internet, restart backend container |

## 📁 Key Configuration Files

```
thesis-project/
├── docker-compose.yml          # Service orchestration
├── requirements-docker.txt     # Clean Python deps (no git)
├── backend/Dockerfile          # Python + ML stack
├── frontend/Dockerfile         # Node.js build + Nginx
└── frontend/nginx.conf         # SPA routing config
```

## 🎯 Access Points

| URL | Description |
|-----|-------------|
| http://localhost | Main application interface |
| http://localhost:8000/docs | **Interactive API documentation** |
| http://localhost:8000 | Backend health check |

## 🔒 Production Notes

- Remove volume mounts and `--reload` for production
- Add SSL/HTTPS configuration in nginx
- Set resource limits and health checks
- Consider GPU containers for faster inference

---

**Next Steps:** Use the [API Documentation](http://localhost:8000/docs) to explore endpoints and refer to the User Guide for application workflow.

## 🤝 Contributors
### LAB:
- 👤 Lukasz
- 👤 Avishek
- 👤 Berkay