# Merged Environment Configuration

This document explains the merged `environment.yml` file that combines dependencies from both:
- **Testing/environment.yml** (for Naive-RAG testing)
- **RAGSystem/self-rag/environment.yml** (for Self-RAG)

## Key Decisions

### Python Version
- **Chosen: Python 3.10** (required for vLLM 0.2.6+)
- vLLM 0.2.6+ requires Python 3.10+ because it uses the `|` union type syntax (e.g., `str | None`) which was introduced in Python 3.10.
- Python 3.10 is compatible with both Naive-RAG and Self-RAG systems.

### Channels
Merged channels from both environments:
- `nvidia/label/cuda-12.1.0` - For CUDA support (required for vLLM)
- `conda-forge` - General packages
- `pytorch` - PyTorch packages
- `defaults` - Default conda packages

### Package Version Strategy
- Used `>=` constraints for flexibility
- Minimum versions set to meet both systems' requirements
- Self-RAG specific packages added with appropriate minimum versions

## Included Packages

### Core Testing Framework
- `flask` - Web interface
- `datasets` - Dataset loading
- `jsonlines` - JSONL file processing

### Naive-RAG Dependencies
- `PyPDF2` - PDF processing
- `torch` - PyTorch (>=1.13.0, compatible with both)
- `transformers` - HuggingFace transformers (>=4.21.0, compatible with both)
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search
- `huggingface-hub` - Model downloading

### Self-RAG Dependencies
- `vllm` - High-performance LLM inference (REQUIRES CUDA 12.1+)
- `accelerate` - Model acceleration
- `bitsandbytes` - Quantization
- `xformers` - Efficient attention
- `triton` - GPU kernels
- `spacy` - NLP processing
- `fastapi` & `uvicorn` - API services
- `ray` - Distributed computing
- `peft` - Parameter-efficient fine-tuning
- And many more utilities

### Evaluation Metrics
- `nltk` - Text processing
- `rouge-score` - ROUGE metrics
- `sacrebleu` - BLEU metrics
- `evaluate` - Evaluation library

## Usage

### Create the Environment

```bash
cd /home/zyuan/RAGbot/Testing

# If you already have the environment, remove it first
conda env remove -n rag-testing

# Create the new environment with Python 3.10
conda env create -f environment.yml
conda activate rag-testing
```

**Important:** The environment requires Python 3.10+ because vLLM 0.2.6+ uses Python 3.10+ type union syntax. If you have an existing environment with Python 3.9 or earlier, you need to recreate it.

### Test Naive-RAG

```bash
python test_qasper_rag.py --rag-system naive-rag --dataset qasper --articles 1
```

### Test Self-RAG

```bash
# Ensure CUDA 12.1+ is available
python test_qasper_rag.py --rag-system self-rag --dataset qasper --articles 1
```

## Important Notes

1. **CUDA Requirements**: Self-RAG requires CUDA 12.1+ and sufficient GPU memory (24GB+ recommended)

2. **Package Conflicts**: If you encounter version conflicts:
   - The environment uses flexible `>=` constraints
   - You may need to pin specific versions for compatibility
   - Check the original `RAGSystem/self-rag/environment.yml` for exact versions if needed

3. **Optional Packages**: Some large packages like `deepspeed` are included but optional. You can comment them out if not needed.

4. **CUDA Packages**: NVIDIA CUDA packages are typically installed automatically with CUDA-enabled PyTorch. If vLLM has issues, you may need to install specific CUDA 12.1 packages (see comments in environment.yml).

## Troubleshooting

### vLLM Installation Issues
If vLLM fails to install or import:
1. Verify CUDA 12.1+ is installed: `nvcc --version`
2. Check GPU availability: `nvidia-smi`
3. Install specific CUDA packages (uncomment in environment.yml)
4. Consider using the original self-rag environment for Self-RAG only

### Version Conflicts
If you encounter package version conflicts:
1. Check which package is causing the issue
2. Pin the version in environment.yml (use `==` instead of `>=`)
3. Or use separate environments for Naive-RAG and Self-RAG

## Original Files

- **Testing/environment.yml** - Original Naive-RAG testing environment
- **RAGSystem/self-rag/environment.yml** - Original Self-RAG environment (unchanged)

The merged file is in **Testing/environment.yml** and supports both systems.

