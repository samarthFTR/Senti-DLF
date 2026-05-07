# Senti-DLF 🧠✨

**Senti-DLF** is a modern, end-to-end Machine Learning application designed for ultra-fast, high-accuracy Twitter sentiment analysis. It leverages a fine-tuned **BERT** transformer (`bert-base-uncased`) wrapped in a dynamic Glassmorphism React frontend and a lightning-fast FastAPI backend.

---

## 🏗️ Architecture

The project is split into three decoupled microservices:

1. **Model (`/model`)**
   - **TensorFlow 2.10 & Python 3.10**: Specifically pinned to support Native Windows GPU training.
   - **Transformers**: HuggingFace `TFBertModel` using a Partial Fine-Tuning approach. By freezing the embedding layers and the first 10 transformer blocks, it trains only ~14M parameters (instead of 110M), allowing it to train comfortably on a 4GB RTX 3050.
   - **Data Pipeline**: Robust `tf.data.Dataset` pipelines mapping massive `.parquet` datasets through `BertTokenizerFast`.

2. **Backend (`/backend`)**
   - **FastAPI**: Asynchronous HTTP API.
   - **Singleton Engine**: Loads the massive BERT `.h5` model weights into system RAM *exactly once* at startup via a `lifespan` hook, ensuring inference takes milliseconds.
   - **Pydantic**: Strict schema validation for single and batch prediction routes.

3. **Frontend (`/frontend`)**
   - **React + Vite**: Lightning-fast modern frontend.
   - **Design**: Premium Glassmorphism UI, smooth `framer-motion` micro-animations, and dynamic probability bars.
   - **Integration**: Communicates seamlessly with the FastAPI backend.

---

## 🚀 Quick Start

### 1. Environment Setup
The project requires a Conda environment running **Python 3.10** with native Windows CUDA support.
```cmd
conda create -n myenv python=3.10
conda activate myenv
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install -r requirements.txt
```

### 2. Training the Model
*(Note: Requires the raw Kaggle Twitter datasets placed in `model/data/raw/`)*
```cmd
# Clean and stratify the datasets
python -m model.src.preprocessing

# Train the BERT model
python -m model.src.trainer
```

### 3. Running the Application
You will need two terminal windows to run the full stack:

**Terminal 1 (Start the Backend API):**
```cmd
conda activate myenv
python -m backend.main
```
*API Docs available at: http://localhost:8000/docs*

**Terminal 2 (Start the Frontend UI):**
```cmd
cd frontend
npm run dev
```
*UI available at: http://localhost:5173*

---

## 📈 Performance Details
- **Base Model**: `bert-base-uncased`
- **Classes**: 3 (`negative`, `neutral`, `positive`)
- **Validation Accuracy**: ~90%
- **Training Time**: ~20 minutes on an NVIDIA RTX 3050 (4GB VRAM)
