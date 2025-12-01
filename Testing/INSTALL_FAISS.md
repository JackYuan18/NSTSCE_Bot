# Installing FAISS

If you encounter `ModuleNotFoundError: No module named 'faiss'`, follow these steps:

## Option 1: Install in Current Environment

```bash
# For CPU version (recommended for most users)
pip install faiss-cpu

# OR for GPU version (if you have CUDA)
pip install faiss-gpu
```

## Option 2: Install in Conda Environment

If you're using a conda environment:

```bash
# Activate your conda environment first
conda activate rag-testing  # or your environment name

# Install faiss-cpu
pip install faiss-cpu

# Verify installation
python -c "import faiss; print('faiss version:', faiss.__version__)"
```

## Option 3: Recreate Environment from environment.yml

The `environment.yml` file already includes faiss-cpu. To ensure everything is installed:

```bash
cd /home/zyuan/RAGbot/Testing
conda env create -f environment.yml
# OR if environment already exists:
conda env update -f environment.yml
```

## Verify Installation

After installing, verify it works:

```bash
python -c "import faiss; print('faiss version:', faiss.__version__)"
```

If you see the version number, faiss is installed correctly!

