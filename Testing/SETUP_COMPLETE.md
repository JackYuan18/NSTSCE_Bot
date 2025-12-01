# Testing Environment Setup - Complete

## Environment Created

The `environment.yml` file in the Testing folder contains all necessary dependencies for running Naive RAG tests with all supported datasets.

## Dependencies Included

### Core Dependencies
- **Python 3.9** - Base Python version
- **Flask** - Web interface for qasper_interface.py
- **datasets** - HuggingFace datasets library for loading QASPER and other datasets
- **jsonlines** - For reading JSONL files used by various datasets

### RAG System (Naive-RAG)
- **PyPDF2** - PDF processing
- **torch** - PyTorch for deep learning models
- **transformers** - HuggingFace transformers for language models
- **sentence-transformers** - For embedding generation
- **faiss-cpu** - Vector similarity search (use faiss-gpu for better performance)
- **huggingface-hub** - For downloading models from HuggingFace

### Evaluation Metrics
- **nltk** - For BLEU score calculation and text processing

### Optional/Additional
- **openai** - For ChatGPT5 generator integration
- **requests** - HTTP requests
- **tqdm** - Progress bars
- **numpy** - Numerical operations

## Supported Datasets

All these datasets are supported and tested:
- QASPER
- QMSum
- NarrativeQA
- QuALITY
- HotpotQA
- MuSiQue
- XSum
- WikiAsp
- LongBench

## Installation

```bash
cd /home/zyuan/RAGbot/Testing
conda env create -f environment.yml
conda activate rag-testing
```

## Verification

To verify all packages are installed:

```bash
python -c "
import flask, datasets, jsonlines, PyPDF2, torch, transformers
import sentence_transformers, faiss, huggingface_hub, nltk
import openai, requests, tqdm, numpy
print('All packages installed successfully!')
"
```

## Usage

### Command Line
```bash
conda activate rag-testing
python test_qasper_rag.py --rag-system naive-rag --dataset qasper --articles 1
```

### Web Interface
```bash
conda activate rag-testing
python qasper_interface.py
# Open http://localhost:5051 in your browser
```

## Notes

- The environment is optimized for **Naive RAG** testing
- For **Self-RAG**, use the separate `selfrag` environment from `RAGSystem/self-rag/`
- All dependencies have been tested and verified to work with all supported datasets

