#!/usr/bin/env python3
"""
Utility script to exercise the RAG system against various datasets.

This script loads papers/articles/meetings from various datasets, indexes their
content with the existing RAG pipeline, and evaluates questions. It prints both
the generated answers and snippets of the retrieved context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for error handler utilities
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import error handler utilities
try:
    from error_handler_utils import handle_dependency_error, is_dependency_error
except ImportError:
    # Fallback if utility module is not available
    def handle_dependency_error(error, conda_env='rag-testing'):
        return error
    def is_dependency_error(error):
        return False

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:
    if is_dependency_error(exc):
        enhanced_error = handle_dependency_error(exc)
        raise type(exc)(str(enhanced_error)) from exc
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install it with `pip install datasets`."
    ) from exc

# RAG system imports will be loaded dynamically based on --rag-system argument
RAGConfig = None
RAGSystem = None
setup_logging = None

# Import our new modules
from dataset_loaders import (
    load_dataset_with_fallback,
    load_hotpot_from_local,
    load_longbench_from_local,
    load_musique_from_local,
    load_narrativeqa_from_local,
    load_qmsum_from_local,
    load_quality_from_local,
    load_wikiasp_from_local,
    load_xsum_from_local,
)
from dataset_iterators import iter_article_questions
from document_builders import build_article_documents
from evaluation_metrics import (
    calculate_bleu_score,
    calculate_exact_match,
    calculate_f1_score,
    calculate_recall_score,
    calculate_rouge_l_score,
    extract_reference_answers,
)


def load_rag_system(rag_system_name: str) -> None:
    """Dynamically load the RAG system based on the selection."""
    global RAGConfig, RAGSystem, setup_logging
    import importlib.util
    
    # Clear any existing imports
    if 'RAGSystem' in sys.modules:
        del sys.modules['RAGSystem']
    
    # Add necessary paths
    for extra_path in (PROJECT_ROOT / "NSTSCE", CURRENT_DIR):
        if extra_path.exists():
            path_str = str(extra_path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    
    if rag_system_name == "naive-rag":
        # Load naive-rag from RAGSystem/naive-rag/RAGSystem.py
        naive_rag_path = PROJECT_ROOT / "RAGSystem" / "naive-rag" / "RAGSystem.py"
        if not naive_rag_path.exists():
            raise FileNotFoundError(f"Naive RAG system not found at {naive_rag_path}")
        
        module_dir = str(naive_rag_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        from RAGSystem import RAGConfig, RAGSystem, setup_logging  # type: ignore
        
    elif rag_system_name == "self-rag":
        # Load self-rag from RAGSystem/self-rag/RAGSystem.py
        self_rag_path = PROJECT_ROOT / "RAGSystem" / "self-rag" / "RAGSystem.py"
        if not self_rag_path.exists():
            raise FileNotFoundError(
                f"Self-RAG system not found at {self_rag_path}. "
                "Please ensure self-rag has a RAGSystem.py file with RAGConfig, RAGSystem, and setup_logging."
            )
        
        # Use importlib to load the module with a unique name
        spec = importlib.util.spec_from_file_location("self_rag_system", self_rag_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {self_rag_path}")
        
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            # Check if it's a dependency error and enhance the message
            if is_dependency_error(e):
                enhanced_error = handle_dependency_error(e)
                error_str = str(e).lower()
                if "vllm" in error_str or "vllm is required" in error_str:
                    raise ImportError(
                        f"Self-RAG requires vLLM which is not installed.\n"
                        f"{str(enhanced_error)}\n"
                        f"Or use naive-rag instead by setting --rag-system naive-rag"
                    ) from e
                raise type(e)(str(enhanced_error)) from e
            raise
        
        RAGConfig = module.RAGConfig
        RAGSystem = module.RAGSystem
        setup_logging = module.setup_logging
        
    elif rag_system_name == "flare":
        print(f"[TEST] Loading FLARE system...", flush=True)
        logging.info("[TEST] Loading FLARE system...")
        # Load FLARE from RAGSystem/FLARE/RAGSystem.py
        flare_path = PROJECT_ROOT / "RAGSystem" / "FLARE" / "RAGSystem.py"
        print(f"[TEST] FLARE path: {flare_path}", flush=True)
        logging.info(f"[TEST] FLARE path: {flare_path}")
        if not flare_path.exists():
            error_msg = (
                f"FLARE system not found at {flare_path}. "
                "Please ensure FLARE has a RAGSystem.py file with RAGConfig, RAGSystem, and setup_logging."
            )
            print(f"[TEST ERROR] {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)
        
        # Use importlib to load the module with a unique name
        spec = importlib.util.spec_from_file_location("flare_system", flare_path)
        if spec is None or spec.loader is None:
            error_msg = f"Failed to create spec for {flare_path}"
            print(f"[TEST ERROR] {error_msg}", flush=True)
            raise ImportError(error_msg)
        
        module = importlib.util.module_from_spec(spec)
        try:
            print(f"[TEST] Executing FLARE module...", flush=True)
            logging.info("[TEST] Executing FLARE module...")
            spec.loader.exec_module(module)
            print(f"[TEST] FLARE module loaded successfully", flush=True)
            logging.info("[TEST] FLARE module loaded successfully")
        except ImportError as e:
            # Check if it's a dependency error and enhance the message
            print(f"[TEST ERROR] ImportError loading FLARE: {e}", flush=True)
            logging.error(f"[TEST] ImportError loading FLARE: {e}")
            if is_dependency_error(e):
                enhanced_error = handle_dependency_error(e)
                raise type(e)(str(enhanced_error)) from e
            raise
        except Exception as e:
            error_msg = str(e)
            # Mask API keys in error message
            import re
            masked_error = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], error_msg)
            masked_error = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_error)
            print(f"[TEST ERROR] Exception loading FLARE: {masked_error}", flush=True)
            logging.error(f"[TEST] Exception loading FLARE: {masked_error}")
            import traceback
            tb_str = traceback.format_exc()
            # Mask API keys in traceback
            masked_tb = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], tb_str)
            masked_tb = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_tb)
            logging.error(f"[TEST] Traceback: {masked_tb}")
            raise
        
        RAGConfig = module.RAGConfig
        RAGSystem = module.RAGSystem
        setup_logging = module.setup_logging
        print(f"[TEST] FLARE RAGConfig, RAGSystem, setup_logging loaded", flush=True)
        logging.info("[TEST] FLARE RAGConfig, RAGSystem, setup_logging loaded")
    else:
        raise ValueError(f"Unknown RAG system: {rag_system_name}. Choose from: naive-rag, self-rag, flare")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the RAG system on questions from various datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"],
        default="qasper",
        help="Dataset to use",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to use (train, validation, test, or a slice).",
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=1,
        help="Number of distinct papers to evaluate. Use 0 to process all papers in the split.",
    )
    parser.add_argument(
        "--questions-per-article",
        type=int,
        default=3,
        help="Maximum number of questions to test for each paper (0 means all questions).",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum character length for each chunk fed to the retriever.",
    )
    parser.add_argument(
        "--generator-model",
        default="t5-small",
        help="Hugging Face model identifier used by the answer generator (or 'chatgpt5').",
    )
    parser.add_argument(
        "--chatgpt5-api-key",
        default=None,
        help="API key required when --generator-model chatgpt5 is selected.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key required when --rag-system flare is selected.",
    )
    parser.add_argument(
        "--retrieval-instruction-method",
        default=None,
        choices=['cot', 'strategyqa', 'summary'],
        help="RetrievalInstruction method for FLARE: 'cot' (Chain of Thought), 'strategyqa', 'summary'. Omit to disable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--save-json",
        default="qasper_results.json",
        help="Optional path (relative to this script) to save results as JSON. Set to empty string to skip saving.",
    )
    parser.add_argument(
        "--show-context",
        dest="show_context",
        action="store_true",
        help="Display retrieved context in outputs.",
    )
    parser.add_argument(
        "--hide-context",
        dest="show_context",
        action="store_false",
        help="Hide retrieved context in outputs.",
    )
    parser.set_defaults(show_context=True)
    parser.add_argument(
        "--rag-system",
        choices=["naive-rag", "self-rag", "flare"],
        default="naive-rag",
        help="RAG system to use: 'naive-rag' (default), 'self-rag', or 'flare'. For multiple systems, call the script multiple times.",
    )
    args = parser.parse_args()
    if args.generator_model.lower() == "chatgpt5" and not args.chatgpt5_api_key:
        parser.error("generator_model 'chatgpt5' requires --chatgpt5-api-key.")
    # Note: FLARE can be loaded without API key - it will handle the error during query
    return args


def load_dataset_data(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Load dataset based on name with appropriate loader."""
    dataset_name_lower = dataset_name.lower()
    
    dataset_configs = {
        "qasper": {
            "local_dir": None,
            "hf_name": "allenai/qasper",
            "loader": None,  # Use HuggingFace directly
        },
        "narrativeqa": {
            "local_dir": CURRENT_DIR / "Datasets" / "narrativeqa",
            "hf_name": "google-deepmind/narrativeqa",
            "loader": load_narrativeqa_from_local,
        },
        "qmsum": {
            "local_dir": CURRENT_DIR / "Datasets" / "QMSum" / "data",
            "hf_name": "Yale-LILY/qmsum",
            "loader": load_qmsum_from_local,
        },
        "quality": {
            "local_dir": CURRENT_DIR / "Datasets" / "quality" / "data",
            "hf_name": None,
            "loader": load_quality_from_local,
        },
        "hotpot": {
            "local_dir": CURRENT_DIR / "Datasets" / "hotpot",
            "hf_name": "hotpot_qa",
            "loader": load_hotpot_from_local,
        },
        "musique": {
            "local_dir": CURRENT_DIR / "Datasets" / "musique" / "data",
            "hf_name": None,
            "loader": load_musique_from_local,
        },
        "xsum": {
            "local_dir": CURRENT_DIR / "Datasets" / "xsum" ,
            "hf_name": "xsum",
            "loader": load_xsum_from_local,
        },
        "wikiasp": {
            "local_dir": CURRENT_DIR / "Datasets" / "wikiasp",
            "hf_name": None,
            "loader": load_wikiasp_from_local,
        },
        "longbench": {
            "local_dir": CURRENT_DIR / "Datasets" / "LongBench" / "data",
            "hf_name": "THUDM/LongBench",
            "loader": load_longbench_from_local,
        },
    }
    
    config = dataset_configs.get(dataset_name_lower)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # QASPER uses HuggingFace directly
    if dataset_name_lower == "qasper":
        logging.info("Loading QASPER dataset split '%s'...", split)
        dataset_split = load_dataset("allenai/qasper", split=split)
        return list(dataset_split)
    
    # Others use load_dataset_with_fallback
    return load_dataset_with_fallback(
        dataset_name=dataset_name,
        split=split,
        local_dir=config["local_dir"],
        hf_dataset_name=config["hf_name"],
        load_local_func=config["loader"],
    )


def print_results(
    article_id: str,
    title: str,
    question: str,
    answer: str,
    reference_answers: List[str],
    retrieved_metadata: List[Dict[str, Any]],
    exact_match: Optional[float],
    f1_score: Optional[float],
    recall_score: Optional[float],
    rouge_l_score: Optional[float],
    bleu_score: Optional[float],
    show_context: bool,
) -> None:
    """Print formatted results."""
    print("=" * 120)
    print(f"Article ID: {article_id}")
    print(f"Title: {title}")
    print(f"Question: {question}")
    print("\nGenerated Answer:\n")
    print(answer)
    
    if reference_answers:
        print("\nReference Answers:")
        for ref_idx, ref_answer in enumerate(reference_answers, start=1):
            print(f"  [{ref_idx}] {ref_answer}")
    
    print("\nScores:")
    if exact_match is not None:
        print(f"  Exact Match: {exact_match:.4f}")
    if f1_score is not None:
        print(f"  F1 Score: {f1_score:.4f}")
    if recall_score is not None:
        print(f"  Recall Score: {recall_score:.4f}")
    if rouge_l_score is not None:
        print(f"  ROUGE-L Score: {rouge_l_score:.4f}")
    if bleu_score is not None:
        print(f"  BLEU Score: {bleu_score:.4f}")
    
    if show_context and retrieved_metadata:
        print("\nRetrieved Context:")
        for meta_idx, meta in enumerate(retrieved_metadata, start=1):
            preview = meta.get("text_preview") or meta.get("chunk", "")[:200]
            section = meta.get("section_title") or "Unknown Section"
            paragraph_index = meta.get("paragraph_index")
            print(f"  ({meta_idx}) {section} ¶{paragraph_index}: {preview}")
    
    print("=" * 120)


def main() -> None:
    """Main execution function."""
    # Force unbuffered output for print statements
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
    
    print("[TEST] ===== test_qasper_rag.py main() called =====", flush=True)
    args = parse_args()
    print(f"[TEST] Parsed args. RAG system: {args.rag_system}", flush=True)
    
    # Load the selected RAG system
    print(f"[TEST] About to load RAG system: {args.rag_system}", flush=True)
    load_rag_system(args.rag_system)
    print(f"[TEST] RAG system loaded: {args.rag_system}", flush=True)
    
    print(f"[TEST] About to setup logging with level: {args.log_level}", flush=True)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    print(f"[TEST] Logging configured. Level: {args.log_level}", flush=True)
    
    # Ensure logging goes to stdout so it's captured by the interface
    # Get the root logger and ensure it has a StreamHandler for stdout
    root_logger = logging.getLogger()
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout 
        for h in root_logger.handlers
    )
    if not has_stdout_handler:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(getattr(logging, args.log_level.upper()))
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(stdout_handler)
        print(f"[TEST] Added stdout handler to root logger", flush=True)
    
    logger.info("Using RAG system: %s", args.rag_system)
    
    use_chatgpt5 = args.generator_model.lower() == "chatgpt5"
    # Keep the generator_model as-is (chatgpt5 or the actual model name)
    # When chatgpt5 is selected, pass it through so RAG systems know to use ChatGPT5
    generator_model_name = args.generator_model
    
    logger.info("Using %s generator", "ChatGPT5 via API" if use_chatgpt5 else f"model: {generator_model_name}")
    
    # Load dataset
    dataset_name = args.dataset.lower()
    logger.info("Processing dataset: %s (split: %s)", dataset_name, args.split)
    dataset_split = load_dataset_data(dataset_name, args.split)
    logger.info("Loaded %d records from %s dataset", len(dataset_split), dataset_name)
    
    # Initialize RAG system
    # For FLARE, use openai_api_key argument; for others, use chatgpt5_api_key if using ChatGPT5
    openai_api_key = None
    if args.rag_system == "flare":
        openai_api_key = args.openai_api_key
    elif use_chatgpt5:
        openai_api_key = args.chatgpt5_api_key
    
    # Build config arguments - only include retrieval_instruction_method for FLARE
    config_kwargs = {
        'chunk_size': args.chunk_size,
        'retrieval_k': args.retrieval_k,
        'generator_model': generator_model_name,
        'use_chatgpt5': use_chatgpt5,
        'openai_api_key': openai_api_key,
    }
    
    # Only add retrieval_instruction_method for FLARE
    if args.rag_system == 'flare' and args.retrieval_instruction_method:
        config_kwargs['retrieval_instruction_method'] = args.retrieval_instruction_method
    
    print(f"[TEST] Creating RAGConfig with kwargs: { {k: v if k != 'openai_api_key' else ('***' if v else None) for k, v in config_kwargs.items()} }", flush=True)
    logging.info(f"[TEST] Creating RAGConfig for {args.rag_system}")
    config = RAGConfig(**config_kwargs)
    print(f"[TEST] RAGConfig created. About to create RAGSystem...", flush=True)
    logging.info(f"[TEST] About to create RAGSystem instance...")
    try:
        rag_system = RAGSystem(config)
        print(f"[TEST] RAGSystem instance created successfully", flush=True)
        logging.info(f"[TEST] RAGSystem instance created successfully")
    except Exception as e:
        error_msg = str(e)
        # Mask API keys in error message
        masked_error = error_msg
        if 'api' in error_msg.lower() or 'key' in error_msg.lower():
            import re
            # Mask API key patterns
            masked_error = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], masked_error)
            masked_error = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_error)
        print(f"[TEST ERROR] Exception creating RAGSystem: {masked_error}", flush=True)
        logging.error(f"[TEST] Exception creating RAGSystem: {masked_error}")
        import traceback
        tb_str = traceback.format_exc()
        # Mask API keys in traceback
        masked_tb = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], tb_str)
        masked_tb = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_tb)
        logging.error(f"[TEST] Traceback: {masked_tb}")
        raise
    
    # Initialize results tracking
    processed_articles = 0
    total_questions = 0
    total_questions_in_dataset = 0  # Total questions in entire dataset (unfiltered)
    total_questions_available = 0  # Questions available for testing based on config limits
    collected_results: List[Dict[str, Any]] = []
    output_path: Optional[Path] = None
    
    def write_results() -> None:
        """Write results to JSON file."""
        if not args.save_json or not collected_results:
            return
        
        nonlocal output_path
        if output_path is None:
            candidate = Path(args.save_json)
            if not candidate.is_absolute():
                candidate = CURRENT_DIR / candidate
            candidate.parent.mkdir(parents=True, exist_ok=True)
            output_path = candidate.resolve()
        
        try:
            logger.debug("Writing %d results to %s (dataset: %s)", len(collected_results), output_path, dataset_name)
            # Ensure all results have the correct dataset name
            for result in collected_results:
                if "dataset" in result:
                    result["dataset"] = dataset_name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(collected_results, f, ensure_ascii=False, indent=2)
            logger.debug("Successfully wrote results to %s", output_path)
        except Exception as exc:
            logger.error("Failed to write results to %s: %s", output_path, exc)
            raise
    
    # First pass: Count total questions in entire dataset (unfiltered) - no limits applied
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        total_questions_in_dataset += len(questions)
    
    # Second pass: Count questions available for testing based on article and question limits
    article_count = 0
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        # Count questions that would be tested based on article and question limits
        if args.articles == 0:  # If 0, process all articles
            # Process all articles
            if args.questions_per_article == 0:
                questions_limit = len(questions)  # All questions
            else:
                questions_limit = min(args.questions_per_article, len(questions))
        else:
            # Limited articles
            if article_count < args.articles:
                if args.questions_per_article == 0:
                    questions_limit = len(questions)  # All questions
                else:
                    questions_limit = min(args.questions_per_article, len(questions))
            else:
                questions_limit = 0
                break
        
        total_questions_available += questions_limit
        article_count += 1
        if args.articles and article_count >= args.articles:
            break
    
    # Process articles and questions
    processed_articles = 0
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        if args.articles and processed_articles >= args.articles:
            break
        
        # Build documents
        documents, doc_metadata = build_article_documents(
            rag_system=rag_system,
            article_id=article_id,
            record=record,
            chunk_size=args.chunk_size,
            dataset_name=dataset_name,
        )
        
        if not documents:
            logger.warning("Skipping article %s because no documents were extracted", article_id)
            continue
        
        # Build index
        # For self-rag and flare, build_index is on the RAGSystem itself, not on retriever
        # For naive-rag, build_index is on the retriever
        if args.rag_system in ("self-rag", "flare"):
            rag_system.build_index(documents, doc_metadata)
        else:
            rag_system.retriever.build_index(documents, doc_metadata)
        logger.info(
            "Indexed %d chunks for article %s (%s)",
            len(documents),
            article_id,
            record.get("title", "unknown title"),
        )
        
        # Process questions
        questions_to_run = questions[:args.questions_per_article] if args.questions_per_article else questions
        questions_processed = 0
        
        for example in questions_to_run:
            question_text = (example.get("question") or "").strip()
            if not question_text:
                continue
            
            total_questions += 1
            questions_processed += 1
            
            logger.info("Q%d: %s", total_questions, question_text)
            
            # Query RAG system and measure generation time
            start_time = time.time()
            
            # Check if FLARE is selected and API key is missing
            if args.rag_system == "flare" and not args.openai_api_key:
                logger.warning("FLARE selected but no OpenAI API key provided. Skipping query.")
                print(f"[TEST] FLARE selected without API key - skipping query", flush=True)
                answer = "No OpenAI API key provided. FLARE requires OpenAI API for generation."
                retrieved_metadata = []
                generation_time = 0.0
            # Check if naive-rag with chatgpt5 is selected and API key is missing
            elif args.rag_system == "naive-rag" and args.generator_model.lower() == "chatgpt5" and not args.chatgpt5_api_key:
                logger.warning("Naive RAG with ChatGPT5 selected but no OpenAI API key provided. Skipping query.")
                print(f"[TEST] Naive RAG with ChatGPT5 selected without API key - skipping query", flush=True)
                answer = "No OpenAI API key provided. Running Naive RAG with ChatGPT5 requires OpenAI API for generation."
                retrieved_metadata = []
                generation_time = 0.0
            else:
                logger.info(f"About to call rag_system.query for question: {question_text[:100]}...")
                print(f"[TEST] About to call rag_system.query for FLARE", flush=True)
                try:
                    _, answer, retrieved_metadata = rag_system.query(question_text, k=args.retrieval_k)
                    logger.info(f"rag_system.query returned answer: {answer[:100] if answer else 'None'}...")
                    print(f"[TEST] rag_system.query returned answer length: {len(answer) if answer else 0}", flush=True)
                except Exception as e:
                    error_msg = str(e)
                    # Mask API keys in error message
                    import re
                    masked_error = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], error_msg)
                    masked_error = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_error)
                    logger.error(f"Error calling rag_system.query: {masked_error}")
                    import traceback
                    tb_str = traceback.format_exc()
                    # Mask API keys in traceback
                    masked_tb = re.sub(r'sk-proj-[A-Za-z0-9]{20,}', lambda m: m.group()[:10] + '...' + m.group()[-4:], tb_str)
                    masked_tb = re.sub(r'sk-[A-Za-z0-9]{20,}', lambda m: m.group()[:7] + '...' + m.group()[-4:], masked_tb)
                    logger.error(f"Traceback: {masked_tb}")
                    print(f"[TEST ERROR] Exception in rag_system.query: {masked_error}", flush=True)
                    answer = f"Error: {masked_error}"
                    retrieved_metadata = []
                generation_time = time.time() - start_time
            
            # Get device information from RAG system
            device_used = getattr(rag_system.config, 'device', 'cpu')
            if device_used == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        device_display = f"GPU ({gpu_name})"
                    else:
                        device_display = "CPU (CUDA requested but not available)"
                except (ImportError, AttributeError):
                    device_display = "CPU"
            else:
                device_display = "CPU"
            
            # Calculate metrics
            reference_answers = extract_reference_answers(example.get("answers"))
            exact_match = calculate_exact_match(answer, reference_answers)
            f1_score = calculate_f1_score(answer, reference_answers)
            rouge_l_score = calculate_rouge_l_score(answer, reference_answers)
            bleu_score = calculate_bleu_score(answer, reference_answers)
            
            # Calculate recall score
            recall_score = calculate_recall_score(answer, reference_answers)
            
            # Print results
            print_results(
                article_id=article_id,
                title=record.get("title") or "Unknown title",
                question=question_text,
                answer=answer,
                reference_answers=reference_answers,
                retrieved_metadata=retrieved_metadata,
                exact_match=exact_match,
                f1_score=f1_score,
                recall_score=recall_score,
                rouge_l_score=rouge_l_score,
                bleu_score=bleu_score,
                show_context=args.show_context,
            )
            
            # Store results - ensure dataset_name is used (not record.get("dataset") which might be sub-dataset name)
            result_entry = {
                "article_id": article_id,
                "title": record.get("title") or record.get("article_title") or "Unknown title",
                "question": question_text,
                "generated_answer": answer,
                "reference_answers": reference_answers,
                "retrieved_context": retrieved_metadata if args.show_context else [],
                "generator": "chatgpt5" if use_chatgpt5 else args.generator_model,
                "rag_system": args.rag_system,  # Track which RAG system was used
                "dataset": dataset_name,  # Use top-level dataset name, not sub-dataset from record
                "total_questions_in_dataset": total_questions_in_dataset,  # Total questions in entire dataset
                "total_questions_available": total_questions_available,  # Questions available based on config
                "bleu_score": bleu_score,
                "exact_match": exact_match,
                "f1_score": f1_score,
                "recall_score": recall_score,
                "rouge_l_score": rouge_l_score,
                "generation_time": generation_time,  # Time taken to generate answer in seconds
                "device": device_display,  # Device used (GPU or CPU)
            }
            # For LongBench, also store the sub-dataset name for reference
            if dataset_name == "longbench" and record.get("dataset"):
                result_entry["sub_dataset"] = record.get("dataset")
            collected_results.append(result_entry)
            write_results()
        
        if questions_processed > 0:
            processed_articles += 1
    
    logger.info("Completed %d articles and processed %d questions.", processed_articles, total_questions)
    
    # Final write
    if args.save_json and collected_results:
        write_results()
        logger.info("Final write: Saved %d results to %s", len(collected_results), output_path)


# Print at module level to verify script is being executed
print("[TEST] test_qasper_rag.py module loaded", flush=True)

if __name__ == "__main__":
    print("[TEST] test_qasper_rag.py __main__ block executing", flush=True)
    main()
