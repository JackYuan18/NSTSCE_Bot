# Testing Environment Setup

This directory contains scripts for testing RAG systems against various datasets.

## Environment Setup

### Option 1: Using the provided environment.yml (Recommended)

Create and activate the conda environment:

```bash
cd /home/zyuan/RAGbot/Testing
conda env create -f environment.yml
conda activate rag-testing
```

### Option 2: Manual Installation

If you prefer to install dependencies manually:

```bash
conda create -n rag-testing python=3.9
conda activate rag-testing
pip install flask datasets PyPDF2 torch transformers sentence-transformers faiss-cpu huggingface-hub nltk requests tqdm
```

## Running the Tests

### Command Line Interface

```bash
# Activate the environment
conda activate rag-testing

# Run tests with naive-rag
python test_qasper_rag.py --rag-system naive-rag --dataset qasper --articles 1

# Run tests with self-rag (requires selfrag environment)
conda activate selfrag
python test_qasper_rag.py --rag-system self-rag --dataset qasper --articles 1
```

### Web Interface

```bash
# Activate the environment
conda activate rag-testing

# Start the web interface
python qasper_interface.py

# The interface will be available at http://localhost:5051
```

## Using Self-RAG

Self-RAG requires a separate conda environment with vLLM installed. Use the environment from `RAGSystem/self-rag/`:

```bash
# Create the selfrag environment (if not already created)
cd /home/zyuan/RAGbot/RAGSystem/self-rag
conda env create -f environment.yml
conda activate selfrag

# Then run tests with self-rag
cd /home/zyuan/RAGbot/Testing
python test_qasper_rag.py --rag-system self-rag --dataset qasper
```

## Notes

- The `rag-testing` environment is optimized for naive-rag and general testing
- For Self-RAG, use the `selfrag` environment which includes vLLM and CUDA dependencies
- Make sure to activate the correct environment before running scripts
- The web interface (`qasper_interface.py`) will use the Python from the currently active conda environment

