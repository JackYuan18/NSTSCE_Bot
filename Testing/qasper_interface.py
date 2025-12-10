#!/usr/bin/env python3
"""
Interactive viewer for QASPER RAG evaluation results.

Launch this script to serve a lightweight web interface where you can trigger
`test_qasper_rag.py`, inspect generated answers, and compare them with reference
answers from the dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request  # type: ignore

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_RESULTS_PATH = CURRENT_DIR / "qasper_results.json"
DEFAULT_TEST_SCRIPT = CURRENT_DIR / "test_qasper_rag.py"

# Set up logging directory
LOG_DIR = CURRENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create log file with timestamp
LOG_FILE = LOG_DIR / f"qasper_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

app = Flask(__name__)

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)

# Set Flask app logger level
app.logger.setLevel(logging.INFO)

# Log startup information
app.logger.info("=" * 80)
app.logger.info("QASPER Interface Starting")
app.logger.info("=" * 80)
app.logger.info("Log file: %s", LOG_FILE)
app.logger.info("Working directory: %s", CURRENT_DIR)
app.logger.info("Python executable: %s", sys.executable)
app.logger.info("Python version: %s", sys.version)

# Shared state for UI polling
RUN_LOCK = threading.Lock()
RUN_STATE: dict[str, Any] = {
    "status": "idle",
    "message": "Idle. Click 'Run Tests' to generate fresh results.",
    "last_result": None,
}

RESULTS_PATH: Path = DEFAULT_RESULTS_PATH
TEST_OPTIONS: argparse.Namespace


def run_test_script(options: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    """Run test script for a single dataset."""
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    dataset = getattr(options, "dataset", "qasper")
    # Ensure RESULTS_PATH is absolute
    results_path_abs = RESULTS_PATH.resolve()
    results_path_abs.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the Python from the conda environment if available
    # Check if we're in a conda environment
    python_executable = sys.executable
    if 'CONDA_DEFAULT_ENV' in os.environ:
        # Try to use conda's python explicitly
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_python = os.path.join(conda_prefix, 'bin', 'python')
            if os.path.exists(conda_python):
                python_executable = conda_python
                app.logger.info(f"Using conda environment Python: {python_executable}")
    
    cmd = [
        python_executable,
        str(DEFAULT_TEST_SCRIPT),
        "--dataset",
        dataset,
        "--split",
        options.split,
        "--articles",
        str(options.articles),
        "--questions-per-article",
        str(options.questions_per_article),
        "--retrieval-k",
        str(options.retrieval_k),
        "--chunk-size",
        str(options.chunk_size),
        "--generator-model",
        options.generator_model,
        "--rag-system",
        getattr(options, "rag_system", "naive-rag"),
        "--log-level",
        options.log_level,
        "--save-json",
        str(results_path_abs),
    ]

    if getattr(options, "chatgpt5_api_key", None):
        cmd.extend(["--chatgpt5-api-key", options.chatgpt5_api_key])

    if getattr(options, "openai_api_key", None):
        cmd.extend(["--openai-api-key", options.openai_api_key])

    if getattr(options, "retrieval_instruction_method", None):
        cmd.extend(["--retrieval-instruction-method", options.retrieval_instruction_method])

    if getattr(options, "show_context", False):
        cmd.append("--show-context")
    else:
        cmd.append("--hide-context")

    display_cmd = list(cmd)
    if "--chatgpt5-api-key" in display_cmd:
        try:
            idx = display_cmd.index("--chatgpt5-api-key")
            if idx + 1 < len(display_cmd):
                display_cmd[idx + 1] = "****"
        except ValueError:
            pass
    if "--openai-api-key" in display_cmd:
        try:
            idx = display_cmd.index("--openai-api-key")
            if idx + 1 < len(display_cmd):
                display_cmd[idx + 1] = "****"
        except ValueError:
            pass

    app.logger.info("Running test script: %s", " ".join(display_cmd))
    app.logger.info("Results will be saved to: %s (absolute: %s)", RESULTS_PATH, results_path_abs)
    app.logger.info("Working directory: %s", CURRENT_DIR)
    app.logger.info("Python executable: %s", sys.executable)
    app.logger.info("Test script path: %s (exists: %s)", DEFAULT_TEST_SCRIPT, DEFAULT_TEST_SCRIPT.exists())
    
    try:
        # Use Popen to capture output in real-time for progress messages
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            cwd=str(CURRENT_DIR),
            universal_newlines=True
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Read output line by line to capture progress messages
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.rstrip()
            stdout_lines.append(line)
            
            # Check if this is a [Self-RAG] progress message
            if "[Self-RAG]" in line:
                # Extract the message after [Self-RAG]
                # Format: "2025-12-02 15:31:01,940 - self_rag_system - INFO - [Self-RAG] Embedding 100 passages..."
                # Or simpler: "[Self-RAG] Embedding 100 passages..."
                message_start = line.find("[Self-RAG]")
                if message_start != -1:
                    progress_msg = line[message_start + len("[Self-RAG]"):].strip()
                    # Update RUN_STATE with progress message
                    RUN_STATE["message"] = f"Test in progress... {progress_msg}"
                    app.logger.info("Progress: %s", progress_msg)
            
            # Check if this is a [FLARE] log message
            elif "[FLARE]" in line:
                # Extract the message after [FLARE]
                message_start = line.find("[FLARE]")
                if message_start != -1:
                    flare_msg = line[message_start + len("[FLARE]"):].strip()
                    # Update RUN_STATE with progress message
                    RUN_STATE["message"] = f"Test in progress... {flare_msg}"
                    # Log FLARE messages at INFO level
                    app.logger.info("FLARE: %s", flare_msg)
                else:
                    # Log the full line if it contains [FLARE]
                    app.logger.info("FLARE output: %s", line)
            
            # Check for ERROR or WARNING in the line (important messages)
            elif any(level in line.upper() for level in ["ERROR", "WARNING", "CRITICAL", "EXCEPTION"]):
                # Log errors and warnings at appropriate levels
                if "ERROR" in line.upper() or "CRITICAL" in line.upper() or "EXCEPTION" in line.upper():
                    app.logger.error("Test script: %s", line)
                elif "WARNING" in line.upper():
                    app.logger.warning("Test script: %s", line)
                else:
                    app.logger.info("Test output: %s", line)
            
            # Log other important messages at INFO level (not just DEBUG)
            # This includes INFO level messages from the test script
            elif "INFO" in line.upper() or any(keyword in line for keyword in ["Processing", "Loading", "Initializing", "Completed", "Finished"]):
                app.logger.info("Test output: %s", line)
            
            # Log everything else at DEBUG level
            else:
                app.logger.debug("Test output: %s", line)
        
        # Wait for process to complete
        returncode = process.wait(timeout=3600)
        
        # Combine stdout and stderr
        stdout = "\n".join(stdout_lines)
        stderr = ""  # Already captured in stdout
        
        result = subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
        
    except subprocess.TimeoutExpired:
        if 'process' in locals():
            process.kill()
        app.logger.error("Test script timed out after 1 hour")
        return subprocess.CompletedProcess(cmd, 1, "", "Test script timed out")
    except Exception as exc:
        app.logger.error("Failed to run test script: %s", exc)
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))
    
    if result.returncode != 0:
        app.logger.error("Test script failed with return code %d", result.returncode)
        app.logger.error("STDOUT: %s", result.stdout[-1000:] if result.stdout else "(empty)")
        app.logger.error("STDERR: %s", result.stderr[-1000:] if result.stderr else "(empty)")
    else:
        app.logger.info("Test script completed successfully")
        if results_path_abs.exists():
            app.logger.info("Results file exists at: %s", results_path_abs)
            try:
                with results_path_abs.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    app.logger.info("Results file contains %d entries", len(data) if isinstance(data, list) else 0)
            except Exception as e:
                app.logger.error("Failed to read results file: %s", e)
        else:
            app.logger.warning("Results file not found at: %s", results_path_abs)
    return result


def run_multiple_datasets(options: argparse.Namespace, datasets: List[str]) -> subprocess.CompletedProcess[str]:
    """Run test script for multiple datasets sequentially and combine results."""
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    all_results: List[Dict[str, Any]] = []
    combined_output = []
    combined_error = []
    last_returncode = 0

    # Ensure RESULTS_PATH is absolute
    results_path_abs = RESULTS_PATH.resolve()
    results_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Clear results file at the start
    clear_results_file()

    for idx, dataset in enumerate(datasets):
        app.logger.info("Running tests for dataset %d/%d: %s", idx + 1, len(datasets), dataset)
        RUN_STATE["message"] = f"Running tests for dataset {idx + 1}/{len(datasets)}: {dataset}..."
        
        # Create options for this dataset
        dataset_options = argparse.Namespace(**vars(options))
        dataset_options.dataset = dataset
        
        # Run test for this dataset (this will write to RESULTS_PATH)
        result = run_test_script(dataset_options)
        
        # Collect output
        if result.stdout:
            combined_output.append(f"\n=== Dataset: {dataset} ===\n")
            combined_output.append(result.stdout)
        if result.stderr:
            combined_error.append(f"\n=== Dataset: {dataset} ===\n")
            combined_error.append(result.stderr)
        
        # Update last returncode (keep error if any dataset fails)
        if result.returncode != 0:
            last_returncode = result.returncode
        
        # Load results from this dataset run and add to combined results
        try:
            if results_path_abs.exists():
                with results_path_abs.open("r", encoding="utf-8") as f:
                    dataset_results = json.load(f)
                    if isinstance(dataset_results, list):
                        all_results.extend(dataset_results)
                        app.logger.info("Loaded %d results from dataset %s (total: %d)", len(dataset_results), dataset, len(all_results))
                    else:
                        app.logger.warning("Results file for dataset %s does not contain a list", dataset)
            else:
                app.logger.warning("Results file not found after dataset %s run: %s", dataset, results_path_abs)
        except Exception as exc:
            app.logger.error("Failed to load results for dataset %s: %s", dataset, exc)
    
    # Save combined results
    try:
        results_path_abs.parent.mkdir(parents=True, exist_ok=True)
        with results_path_abs.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        app.logger.info("Combined %d results from %d datasets into %s", len(all_results), len(datasets), results_path_abs)
        if not all_results:
            app.logger.warning("No results collected from any dataset!")
    except Exception as exc:
        app.logger.error("Failed to save combined results to %s: %s", results_path_abs, exc)
        raise
    
    # Return combined result
    return subprocess.CompletedProcess(
        args=[],
        returncode=last_returncode,
        stdout="".join(combined_output),
        stderr="".join(combined_error),
    )


def run_multiple_rag_systems(options: argparse.Namespace, rag_systems: List[str]) -> subprocess.CompletedProcess[str]:
    """Run test script for multiple RAG systems sequentially and combine results."""
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    all_results: List[Dict[str, Any]] = []
    combined_output = []
    combined_error = []
    last_returncode = 0

    # Ensure RESULTS_PATH is absolute
    results_path_abs = RESULTS_PATH.resolve()
    results_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Clear results file at the start
    clear_results_file()

    for idx, rag_system in enumerate(rag_systems):
        app.logger.info("Running tests for RAG system %d/%d: %s", idx + 1, len(rag_systems), rag_system)
        RUN_STATE["message"] = f"Test in progress... Running RAG system {idx + 1}/{len(rag_systems)}: {rag_system}..."
        
        # Create options for this RAG system
        rag_options = argparse.Namespace(**vars(options))
        rag_options.rag_system = rag_system
        
        # Run test for this RAG system (this will write to RESULTS_PATH)
        result = run_test_script(rag_options)
        
        # Collect output
        if result.stdout:
            combined_output.append(f"\n=== RAG System: {rag_system} ===\n")
            combined_output.append(result.stdout)
        if result.stderr:
            combined_error.append(f"\n=== RAG System: {rag_system} ===\n")
            combined_error.append(result.stderr)
        
        # Update last returncode (keep error if any RAG system fails)
        if result.returncode != 0:
            last_returncode = result.returncode
        
        # Load results from this RAG system run and add to combined results
        try:
            if results_path_abs.exists():
                with results_path_abs.open("r", encoding="utf-8") as f:
                    rag_results = json.load(f)
                    if isinstance(rag_results, list):
                        all_results.extend(rag_results)
                        app.logger.info("Loaded %d results from RAG system %s (total: %d)", len(rag_results), rag_system, len(all_results))
                    else:
                        app.logger.warning("Results file for RAG system %s does not contain a list", rag_system)
            else:
                app.logger.warning("Results file not found after RAG system %s run: %s", rag_system, results_path_abs)
        except Exception as exc:
            app.logger.error("Failed to load results for RAG system %s: %s", rag_system, exc)
    
    # Save combined results
    try:
        results_path_abs.parent.mkdir(parents=True, exist_ok=True)
        with results_path_abs.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        app.logger.info("Combined %d results from %d RAG systems into %s", len(all_results), len(rag_systems), results_path_abs)
        if not all_results:
            app.logger.warning("No results collected from any RAG system!")
    except Exception as exc:
        app.logger.error("Failed to save combined results to %s: %s", results_path_abs, exc)
        raise
    
    # Return combined result
    return subprocess.CompletedProcess(
        args=[],
        returncode=last_returncode,
        stdout="".join(combined_output),
        stderr="".join(combined_error),
    )


def start_test_run(options: argparse.Namespace, *, async_run: bool = True) -> bool:
    if not RUN_LOCK.acquire(blocking=False):
        app.logger.warning("Cannot start test run: another run is already in progress")
        return False

    RUN_STATE["status"] = "running"
    RUN_STATE["message"] = "Test in progress..."
    RUN_STATE["last_result"] = None
    app.logger.info("Starting test run with options: dataset=%s, split=%s, articles=%s, questions_per_article=%s",
                   getattr(options, "dataset", "unknown"), getattr(options, "split", "unknown"),
                   getattr(options, "articles", "unknown"), getattr(options, "questions_per_article", "unknown"))

    def runner() -> None:
        try:
            app.logger.info("Test runner thread started")
            # Check if multiple RAG systems are requested
            rag_systems = getattr(options, "rag_systems", None)
            datasets = getattr(options, "datasets", None)
            
            if rag_systems and isinstance(rag_systems, list) and len(rag_systems) > 1:
                app.logger.info("Running multiple RAG systems: %s", rag_systems)
                result = run_multiple_rag_systems(options, rag_systems)
            elif datasets and isinstance(datasets, list) and len(datasets) > 1:
                app.logger.info("Running multiple datasets: %s", datasets)
                result = run_multiple_datasets(options, datasets)
            else:
                dataset = getattr(options, "dataset", "qasper")
                app.logger.info("Running single dataset: %s", dataset)
                result = run_test_script(options)
            
            app.logger.info("Test script finished with return code: %d", result.returncode)
            if result.returncode == 0:
                dataset_count = len(datasets) if datasets and isinstance(datasets, list) else 1
                RUN_STATE["message"] = f"Last run completed successfully ({dataset_count} dataset(s))."
                RUN_STATE["last_result"] = "success"
                app.logger.info("test_qasper_rag.py completed successfully.")
                # Verify results file was created
                results_path_abs = RESULTS_PATH.resolve()
                if results_path_abs.exists():
                    try:
                        with results_path_abs.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 0
                            app.logger.info("Results file verified: %d entries", count)
                    except Exception as e:
                        app.logger.error("Failed to verify results file: %s", e)
                else:
                    app.logger.warning("Results file missing after successful run: %s", results_path_abs)
            else:
                error_msg = f"Run failed with code {result.returncode}."
                if result.stderr:
                    error_msg += f" Error: {result.stderr[:200]}"
                RUN_STATE["message"] = error_msg
                RUN_STATE["last_result"] = "error"
                app.logger.error("test_qasper_rag.py failed (%s):\nSTDOUT:\n%s\nSTDERR:\n%s", 
                               result.returncode, result.stdout[-500:] if result.stdout else "(empty)", 
                               result.stderr[-500:] if result.stderr else "(empty)")
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Error: {str(exc)}"
            RUN_STATE["message"] = error_msg
            RUN_STATE["last_result"] = "error"
            app.logger.exception("Unexpected error during test run: %s", exc)
        finally:
            RUN_STATE["status"] = "idle"
            RUN_LOCK.release()
            app.logger.info("Test runner thread finished")

    if async_run:
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        app.logger.info("Test runner thread started (daemon=%s)", thread.daemon)
    else:
        app.logger.info("Running test synchronously")
        runner()

    return True


def load_results() -> List[dict[str, Any]]:
    results_path_abs = RESULTS_PATH.resolve()
    if results_path_abs.exists():
        try:
            with results_path_abs.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    app.logger.debug("Loaded %d results from %s", len(data), results_path_abs)
                    return data
                else:
                    app.logger.warning("Results file does not contain a list: %s", results_path_abs)
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.error("Failed to read %s: %s", results_path_abs, exc)
    else:
        app.logger.debug("Results file does not exist: %s", results_path_abs)
    return []


def calculate_dataset_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics (average, min, max) for each dataset.
    
    Returns a dictionary mapping dataset name to statistics dict.
    """
    from collections import defaultdict
    
    # Group results by dataset
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        dataset = result.get("dataset", "unknown")
        by_dataset[dataset].append(result)
    
    statistics: Dict[str, Dict[str, Any]] = {}
    
    for dataset, dataset_results in by_dataset.items():
        metrics = {
            "exact_match": [],
            "f1_score": [],
            "recall_score": [],
            "rouge_l_score": [],
            "bleu_score": [],
        }
        
        # Collect all scores
        for result in dataset_results:
            for metric_name in metrics.keys():
                score = result.get(metric_name)
                if score is not None:
                    try:
                        score_float = float(score)
                        if not (score_float is None or (isinstance(score_float, float) and score_float != score_float)):  # Check for NaN
                            metrics[metric_name].append(score_float)
                    except (TypeError, ValueError):
                        pass
        
        # Calculate total questions in dataset and available (from first result's metadata, all should have same value)
        total_questions_in_dataset = None
        total_questions_available = None
        if dataset_results:
            first_result = dataset_results[0]
            total_questions_in_dataset = first_result.get("total_questions_in_dataset")
            total_questions_available = first_result.get("total_questions_available")
        
        # Calculate statistics for each metric
        stats: Dict[str, Any] = {
            "count": len(dataset_results),
            "total_questions_in_dataset": total_questions_in_dataset,
            "total_questions_available": total_questions_available,
            "metrics": {}
        }
        
        for metric_name, scores in metrics.items():
            if scores:
                stats["metrics"][metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }
            else:
                stats["metrics"][metric_name] = {
                    "average": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                }
        
        statistics[dataset] = stats
    
    return statistics


def clear_results_file() -> None:
    try:
        RESULTS_PATH.write_text("[]", encoding="utf-8")
    except Exception as exc:
        app.logger.error("Failed to clear results file %s: %s", RESULTS_PATH, exc)


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Testing RAG pipelines</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }
    h1 { margin-bottom: 0.5rem; }
    .meta { margin-bottom: 1.5rem; color: #495057; }
    .actions { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; }
    button { padding: 0.6rem 1.2rem; border: none; border-radius: 0.3rem; background: #0d6efd; color: #fff; font-size: 1rem; cursor: pointer; }
    button:disabled { background: #6c757d; cursor: not-allowed; }
    #run-status { font-style: italic; color: #495057; }
    table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 0 12px rgba(0,0,0,0.08); }
    th, td { padding: 0.85rem; border-bottom: 1px solid #dee2e6; vertical-align: top; }
    th { background: #e9ecef; text-align: left; }
    tr:nth-child(even) { background: #fefefe; }
    .question { font-weight: 600; }
    .answer { white-space: pre-wrap; }
    ul { margin: 0; padding-left: 1.25rem; }
    .badge { display: inline-block; padding: 0.25rem 0.55rem; margin-bottom: 0.5rem; background: #20c997; color: #fff; border-radius: 0.3rem; font-size: 0.8rem; }
    .context { font-size: 0.9rem; color: #495057; margin-top: 0.6rem; }
    .context-item { margin-bottom: 0.4rem; }
    .config-form { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; background: #ffffff; border: 1px solid #dee2e6; border-radius: 0.5rem; padding: 0.75rem 1rem; box-shadow: 0 0 6px rgba(0,0,0,0.05); }
    .config-form label { display: flex; flex-direction: column; align-items: flex-start; gap: 0.35rem; font-size: 0.95rem; color: #495057; }
    .param-input { width: 6rem; padding: 0.35rem 0.5rem; border: 1px solid #ced4da; border-radius: 0.35rem; background: #f8f9fa; color: #6c757d; transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease; }
    .config-form select.param-input { width: 11rem; }
    .param-input.modified { background: #fff; color: #212529; border-color: #495057; }
    .config-value { color: #6c757d; transition: color 0.2s ease; }
    .config-value.modified { color: #212529; }
    .api-key-input { width: 16rem; padding: 0.35rem 0.5rem; border: 1px solid #ced4da; border-radius: 0.35rem; }
    .api-key-actions { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    .api-key-actions button { padding: 0.35rem 0.75rem; border-radius: 0.35rem; background: #198754; border: none; color: #fff; cursor: pointer; font-size: 0.9rem; }
    .api-key-actions button:disabled { background: #6c757d; }
    .api-key-status { font-size: 0.9rem; color: #495057; }
    .api-key-status.valid { color: #198754; }
    .api-key-status.invalid { color: #dc3545; }
    .dataset-tab { margin-bottom: 1.5rem; background: #ffffff; border: 1px solid #dee2e6; border-radius: 0.5rem; box-shadow: 0 0 6px rgba(0,0,0,0.05); }
    .tab-header { padding: 1rem 1.25rem; background: #e9ecef; border-bottom: 1px solid #dee2e6; cursor: pointer; user-select: none; border-radius: 0.5rem 0.5rem 0 0; }
    .tab-header:hover { background: #dee2e6; }
    .tab-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .tab-title { font-weight: 600; font-size: 1.1rem; color: #212529; }
    .tab-toggle { font-size: 1.2rem; color: #495057; transition: transform 0.2s ease; }
    .tab-header.collapsed .tab-toggle { transform: rotate(-90deg); }
    .tab-stats { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem; }
    .tab-stat-item { display: flex; align-items: center; gap: 0.4rem; }
    .tab-stat-label { color: #495057; font-weight: 500; }
    .tab-stat-value { color: #212529; font-weight: 600; }
    .tab-content { padding: 0; display: none; }
    .tab-content.expanded { display: block; }
    .tab-results-table { width: 100%; border-collapse: collapse; background: #fff; table-layout: fixed; }
    .tab-results-table th, .tab-results-table td { padding: 0.85rem; border-bottom: 1px solid #dee2e6; vertical-align: top; position: relative; box-sizing: border-box; overflow: hidden; }
    .tab-results-table th { background: #f8f9fa; text-align: left; font-weight: 600; }
    .tab-results-table tbody tr:nth-child(even) { background: #fefefe; }
    .resize-handle { position: absolute; top: 0; right: -2px; width: 5px; height: 100%; cursor: col-resize; background: #000000; z-index: 10; }
    .resize-handle:hover { background: #333333; }
    .resize-handle.active { background: #000000; }
    .tab-empty-state { padding: 2rem; text-align: center; color: #6c757d; }
    .summary-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
    .summary-table th, .summary-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
    .summary-table th { background: #f8f9fa; font-weight: 600; color: #495057; }
    .summary-table td { color: #212529; }
    .summary-table tr:last-child td { border-bottom: none; }
    .metric-value { font-weight: 500; }
    .metric-value.avg { color: #0d6efd; }
    .metric-value.min { color: #dc3545; }
    .metric-value.max { color: #198754; }
    .metric-na { color: #6c757d; font-style: italic; }
  </style>
</head>
<body>
  {% set generator_label = 'ChatGPT 5' if config.generator_model == 'chatgpt5' else config.generator_model %}
  <h1>Testing RAG pipelines</h1>
  <p style="font-size: 0.9rem; color: #6c757d; margin-top: -0.5rem; margin-bottom: 1rem;">Testing within single document RAG*</p>
  <div class="meta">
    <p>Results file: <code>{{ results_path }}</code></p>
    <p>
      <strong>Generator:</strong> <code id="config-generator" class="config-value">{{ generator_label }}</code>
      &nbsp;|&nbsp; <strong>Chunk size:</strong> <code id="config-chunk_size" class="config-value">{{ config.chunk_size }}</code>
      &nbsp;|&nbsp; <strong>Retrieval k:</strong> <code id="config-retrieval_k" class="config-value">{{ config.retrieval_k }}</code>
      &nbsp;|&nbsp; <strong>Articles:</strong> <code id="config-articles" class="config-value">{{ config.articles }}</code>
      &nbsp;|&nbsp; <strong>Questions/article:</strong> <code id="config-questions_per_article" class="config-value">{{ config.questions_per_article }}</code>
      &nbsp;|&nbsp; <strong>Split:</strong> <code id="config-split" class="config-value">{{ config.split }}</code>
      &nbsp;|&nbsp; <strong>Datasets:</strong> <code id="config-dataset" class="config-value">{{ config.datasets or config.dataset }}</code>
      &nbsp;|&nbsp; <strong>RAG Systems:</strong> <code id="config-rag_system" class="config-value">{{ config.rag_systems or config.rag_system }}</code>
      &nbsp;|&nbsp; <strong>Show context:</strong> <code id="config-show_context" class="config-value">{{ 'Yes' if config.show_context else 'No' }}</code>
    </p>
    <p>Total questions processed: <strong id="total-questions">0</strong></p>
    <p id="device-info" style="font-size: 0.9rem; color: #6c757d; margin-top: 0.25rem;">Device: <span id="device-display">-</span></p>
  </div>
  <div class="config-form" id="config-form">
    <label style="display: flex; flex-direction: column; gap: 0.5rem;">
      <span style="font-size: 0.95rem; color: #495057; font-weight: 500;">Datasets</span>
      <div style="display: flex; flex-direction: column; gap: 0.4rem;">
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-qasper" name="datasets" value="qasper" class="dataset-checkbox" {% if config.dataset == 'qasper' or (config.datasets and 'qasper' in config.datasets) %}checked{% endif %}>
          <span>QASPER</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-qmsum" name="datasets" value="qmsum" class="dataset-checkbox" {% if config.dataset == 'qmsum' or (config.datasets and 'qmsum' in config.datasets) %}checked{% endif %}>
          <span>QMSum</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-narrativeqa" name="datasets" value="narrativeqa" class="dataset-checkbox" {% if config.dataset == 'narrativeqa' or (config.datasets and 'narrativeqa' in config.datasets) %}checked{% endif %}>
          <span>NarrativeQA</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-quality" name="datasets" value="quality" class="dataset-checkbox" {% if config.dataset == 'quality' or (config.datasets and 'quality' in config.datasets) %}checked{% endif %}>
          <span>QuALITY</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-hotpot" name="datasets" value="hotpot" class="dataset-checkbox" {% if config.dataset == 'hotpot' or (config.datasets and 'hotpot' in config.datasets) %}checked{% endif %}>
          <span>HotpotQA</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-musique" name="datasets" value="musique" class="dataset-checkbox" {% if config.dataset == 'musique' or (config.datasets and 'musique' in config.datasets) %}checked{% endif %}>
          <span>MuSiQue</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-xsum" name="datasets" value="xsum" class="dataset-checkbox" {% if config.dataset == 'xsum' or (config.datasets and 'xsum' in config.datasets) %}checked{% endif %}>
          <span>XSum</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-wikiasp" name="datasets" value="wikiasp" class="dataset-checkbox" {% if config.dataset == 'wikiasp' or (config.datasets and 'wikiasp' in config.datasets) %}checked{% endif %}>
          <span>WikiAsp</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-longbench" name="datasets" value="longbench" class="dataset-checkbox" {% if config.dataset == 'longbench' or (config.datasets and 'longbench' in config.datasets) %}checked{% endif %}>
          <span>LongBench</span>
        </label>
      </div>
    </label>
    <label style="display: flex; flex-direction: column; gap: 0.5rem;">
      <span style="font-size: 0.95rem; color: #495057; font-weight: 500;">RAG Systems</span>
      <div style="display: flex; flex-direction: column; gap: 0.4rem;">
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-rag_system-naive" name="rag_systems" value="naive-rag" class="rag-system-checkbox" {% if config.rag_system == 'naive-rag' or (config.rag_systems and 'naive-rag' in config.rag_systems) %}checked{% endif %}>
          <span>Naive RAG</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-rag_system-self" name="rag_systems" value="self-rag" class="rag-system-checkbox" {% if config.rag_system == 'self-rag' or (config.rag_systems and 'self-rag' in config.rag_systems) %}checked{% endif %}>
          <span>Self-RAG</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-rag_system-flare" name="rag_systems" value="flare" class="rag-system-checkbox" {% if config.rag_system == 'flare' or (config.rag_systems and 'flare' in config.rag_systems) %}checked{% endif %}>
          <span>FLARE</span>
        </label>
      </div>
    </label>
    <label for="input-generator_model">
      Generator
      <select id="input-generator_model" name="generator_model" class="param-input" data-default="{{ config.generator_model }}" data-type="string" data-target="config-generator">
        <option value="t5-small" {% if config.generator_model == 't5-small' %}selected{% endif %}>Local T5 Small</option>
        <option value="chatgpt5" id="option-chatgpt5" {% if config.generator_model == 'chatgpt5' %}selected{% endif %}>ChatGPT 5</option>
        <option value="gpt-3.5-turbo-instruct" id="option-gpt-3.5-turbo-instruct" {% if config.generator_model == 'gpt-3.5-turbo-instruct' %}selected{% endif %}>OpenAI gpt-3.5-turbo-instruct</option>
        <option value="text-davinci-003" {% if config.generator_model == 'text-davinci-003' %}selected{% endif %}>OpenAI text-davinci-003</option>
      </select>
    </label>
    <label for="input-retrieval_k">
      Retrieval k
      <input id="input-retrieval_k" type="number" min="1" step="1" name="retrieval_k" class="param-input" value="{{ config.retrieval_k }}" data-default="{{ config.retrieval_k }}" data-type="number" data-target="config-retrieval_k">
    </label>
    <label for="input-chunk_size">
      Chunk size
      <input id="input-chunk_size" type="number" min="1" step="1" name="chunk_size" class="param-input" value="{{ config.chunk_size }}" data-default="{{ config.chunk_size }}" data-type="number" data-target="config-chunk_size">
    </label>
    <label for="input-articles">
      Articles
      <input id="input-articles" type="number" min="1" step="1" name="articles" class="param-input" value="{{ config.articles }}" data-default="{{ config.articles }}" data-type="number" data-target="config-articles">
    </label>
    <label for="input-questions_per_article">
      Questions/article
      <input id="input-questions_per_article" type="number" min="0" step="1" name="questions_per_article" class="param-input" value="{{ config.questions_per_article }}" data-default="{{ config.questions_per_article }}" data-type="number" data-target="config-questions_per_article">
    </label>
    <label id="chatgpt5-key-field" for="input-chatgpt5_api_key" style="display: {{ 'flex' if config.generator_model == 'chatgpt5' else 'none' }};">
      OpenAI API key
      <div class="api-key-actions">
        <input id="input-chatgpt5_api_key" type="password" class="api-key-input" placeholder="Enter OpenAI API key" autocomplete="off">
        <button type="button" id="validate-api-key">Validate</button>
        <span id="api-key-status" class="api-key-status"></span>
      </div>
    </label>
    <label id="flare-retrieval-instruction-field" for="input-retrieval_instruction_method" style="display: none; flex-direction: column; gap: 0.5rem;">
      Retrieval Instruction Method
      <select id="input-retrieval_instruction_method" name="retrieval_instruction_method" class="param-input">
        <option value="">None (disabled)</option>
        <option value="cot" selected>CoT (Chain of Thought)</option>
        <option value="strategyqa">StrategyQA</option>
        <option value="summary">Summary</option>
      </select>
    </label>
    <label for="input-show_context" style="flex-direction: row; align-items: center; gap: 0.5rem;">
      <input id="input-show_context" type="checkbox" name="show_context" class="param-input" data-default="{{ 'true' if config.show_context else 'false' }}" data-type="boolean" data-target="config-show_context" {% if config.show_context %}checked{% endif %}>
      Show retrieved context
    </label>
  </div>
  <div class="actions">
    <button id="run-tests" type="button" {% if run_state.status == 'running' %}disabled{% endif %}>Run Tests</button>
    <span id="run-status" data-status="{{ run_state.status }}" data-last-result="{{ run_state.last_result or '' }}">{{ run_state.message }}</span>
  </div>
  <div id="dataset-tabs-container"></div>
  <div id="empty-state-container" style="text-align: center; padding: 2rem; color: #6c757d;">
    <p>No results yet. Click <strong>Run Tests</strong> to begin.</p>
  </div>
  <script>
    console.log('JavaScript script loading...');
    const runButton = document.getElementById('run-tests');
    const runStatus = document.getElementById('run-status');
    const datasetTabsContainer = document.getElementById('dataset-tabs-container');
    const emptyStateContainer = document.getElementById('empty-state-container');
    const totalQuestions = document.getElementById('total-questions');
    const deviceInfo = document.getElementById('device-display');
    const generatorSelect = document.getElementById('input-generator_model');
    const apiKeyField = document.getElementById('chatgpt5-key-field');
    const apiKeyInput = document.getElementById('input-chatgpt5_api_key');
    const validateButton = document.getElementById('validate-api-key');
    const apiKeyStatus = document.getElementById('api-key-status');
    const paramInputs = Array.from(document.querySelectorAll('.param-input'));
    const datasetCheckboxes = Array.from(document.querySelectorAll('.dataset-checkbox'));
    const ragSystemCheckboxes = Array.from(document.querySelectorAll('.rag-system-checkbox'));
    let renderedCount = 0;
    
    console.log('Button element:', runButton);
    console.log('Button exists:', !!runButton);
    if (runButton) {
      console.log('Button type:', runButton.type);
      console.log('Button disabled:', runButton.disabled);
    }

    function updateDatasetDisplay() {
      const checked = Array.from(datasetCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
      const target = document.getElementById('config-dataset');
      if (target) {
        const displayText = checked.map(d => {
          if (d === 'qasper') return 'QASPER';
          if (d === 'qmsum') return 'QMSum';
          if (d === 'narrativeqa') return 'NarrativeQA';
          if (d === 'quality') return 'QuALITY';
          if (d === 'hotpot') return 'HotpotQA';
          if (d === 'musique') return 'MuSiQue';
          if (d === 'xsum') return 'XSum';
          if (d === 'wikiasp') return 'WikiAsp';
          if (d === 'longbench') return 'LongBench';
          return d;
        }).join(', ') || 'None';
        target.textContent = displayText;
        const defaultDatasets = Array.from(datasetCheckboxes)
          .filter(cb => cb.hasAttribute('checked'))
          .map(cb => cb.value);
        const isModified = JSON.stringify(checked.sort()) !== JSON.stringify(defaultDatasets.sort());
        target.classList.toggle('modified', isModified);
      }
    }

    datasetCheckboxes.forEach((checkbox) => {
      checkbox.addEventListener('change', updateDatasetDisplay);
    });
    updateDatasetDisplay();

    function updateRAGSystemDisplay() {
      const checked = Array.from(ragSystemCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
      const target = document.getElementById('config-rag_system');
      if (target) {
        const displayText = checked.map(r => {
          if (r === 'naive-rag') return 'Naive RAG';
          if (r === 'self-rag') return 'Self-RAG';
          if (r === 'flare') return 'FLARE';
          return r;
        }).join(', ') || 'None';
        target.textContent = displayText;
        const defaultRAGSystems = Array.from(ragSystemCheckboxes)
          .filter(cb => cb.hasAttribute('checked'))
          .map(cb => cb.value);
        const isModified = JSON.stringify(checked.sort()) !== JSON.stringify(defaultRAGSystems.sort());
        target.classList.toggle('modified', isModified);
      }
      // Show/hide FLARE retrieval instruction method field
      const isFlareSelected = checked.includes('flare');
      const flareRetrievalInstructionField = document.getElementById('flare-retrieval-instruction-field');
      if (flareRetrievalInstructionField) {
        flareRetrievalInstructionField.style.display = isFlareSelected ? 'flex' : 'none';
      }
      
      // Update generator model dropdown when FLARE is selected/deselected
      if (generatorSelect) {
        // Get all options in the generator dropdown
        const generatorOptions = Array.from(generatorSelect.options);
        const chatgpt5Option = generatorOptions.find(opt => opt.value === 'chatgpt5');
        const gpt35Option = generatorOptions.find(opt => opt.value === 'gpt-3.5-turbo-instruct');
        
        if (isFlareSelected) {
          // When FLARE is selected, hide/disable the chatgpt5 option
          if (chatgpt5Option) {
            chatgpt5Option.style.display = 'none';
            chatgpt5Option.disabled = true;
          }
          // Ensure gpt-3.5-turbo-instruct option exists and select it
          if (!gpt35Option) {
            const newOption = document.createElement('option');
            newOption.value = 'gpt-3.5-turbo-instruct';
            newOption.id = 'option-gpt-3.5-turbo-instruct';
            newOption.textContent = 'OpenAI gpt-3.5-turbo-instruct';
            generatorSelect.appendChild(newOption);
          }
          // Set generator to gpt-3.5-turbo-instruct
          generatorSelect.value = 'gpt-3.5-turbo-instruct';
          generatorSelect.disabled = false;  // Keep it enabled but show the selected model
          generatorSelect.title = 'FLARE uses gpt-3.5-turbo-instruct as generator model';
          applyInputAppearance(generatorSelect);
          // Update API key field visibility
          updateGeneratorState();
        } else {
          // When FLARE is deselected, show/enable the chatgpt5 option again
          if (chatgpt5Option) {
            chatgpt5Option.style.display = '';
            chatgpt5Option.disabled = false;
          }
          generatorSelect.disabled = false;
          generatorSelect.title = '';
          // Update API key field visibility
          updateGeneratorState();
        }
      }
    }

    ragSystemCheckboxes.forEach((checkbox) => {
      checkbox.addEventListener('change', updateRAGSystemDisplay);
    });
    updateRAGSystemDisplay();

    function applyInputAppearance(input) {
      const defaultValue = input.dataset.default ?? '';
      const targetId = input.dataset.target;
      const target = targetId ? document.getElementById(targetId) : null;
      const type = input.dataset.type || input.type || 'text';
      let value = input.value.trim();
      if (type === 'boolean' && input instanceof HTMLInputElement) {
        value = input.checked ? 'true' : 'false';
      }
      const isModified = value !== '' && value !== defaultValue;
      input.classList.toggle('modified', isModified);
      if (target) {
        let displayValue = value === '' ? defaultValue : value;
        if (targetId === 'config-generator') {
          if (displayValue === '') {
            displayValue = defaultValue;
          }
          if (displayValue === 'chatgpt5') {
            displayValue = 'ChatGPT 5';
          } else if (displayValue === 'gpt-3.5-turbo-instruct') {
            displayValue = 'OpenAI gpt-3.5-turbo-instruct';
          } else if (displayValue === 'text-davinci-003') {
            displayValue = 'OpenAI text-davinci-003';
          }
        } else if (targetId === 'config-show_context') {
          displayValue = displayValue === 'true' ? 'Yes' : 'No';
        } else if (targetId === 'config-rag_system') {
          // Display RAG system names (handled by updateRAGSystemDisplay)
          // This is a fallback for non-checkbox updates
          if (Array.isArray(displayValue)) {
            displayValue = displayValue.map(r => {
              if (r === 'naive-rag') return 'Naive RAG';
              if (r === 'self-rag') return 'Self-RAG';
              if (r === 'flare') return 'FLARE';
              return r;
            }).join(', ');
          } else if (displayValue === 'naive-rag') {
            displayValue = 'Naive RAG';
          } else if (displayValue === 'self-rag') {
            displayValue = 'Self-RAG';
          } else if (displayValue === 'flare') {
            displayValue = 'FLARE';
          }
        } else if (targetId === 'config-dataset') {
          // Display selected datasets
            if (Array.isArray(displayValue)) {
            displayValue = displayValue.map(d => {
              if (d === 'qasper') return 'QASPER';
              if (d === 'qmsum') return 'QMSum';
              if (d === 'narrativeqa') return 'NarrativeQA';
              if (d === 'quality') return 'QuALITY';
              if (d === 'hotpot') return 'HotpotQA';
              if (d === 'musique') return 'MuSiQue';
              return d;
            }).join(', ');
          } else {
            // Fallback for single dataset
            if (displayValue === 'qasper') {
              displayValue = 'QASPER';
            } else if (displayValue === 'qmsum') {
              displayValue = 'QMSum';
            } else if (displayValue === 'narrativeqa') {
              displayValue = 'NarrativeQA';
            } else if (displayValue === 'quality') {
              displayValue = 'QuALITY';
            } else if (displayValue === 'hotpot') {
              displayValue = 'HotpotQA';
            } else if (displayValue === 'musique') {
              displayValue = 'MuSiQue';
            } else if (displayValue === 'xsum') {
              displayValue = 'XSum';
            } else if (displayValue === 'wikiasp') {
              displayValue = 'WikiAsp';
            } else if (displayValue === 'longbench') {
              displayValue = 'LongBench';
            }
          }
        }
        target.textContent = displayValue;
        target.classList.toggle('modified', isModified);
      }
    }

    paramInputs.forEach((input) => {
      applyInputAppearance(input);
      const handler = () => applyInputAppearance(input);
      input.addEventListener('input', handler);
      if (input.tagName === 'SELECT' || input.dataset.type === 'boolean') {
        input.addEventListener('change', handler);
      }
    });

    function setApiKeyStatus(text, className) {
      if (!apiKeyStatus) {
        return;
      }
      apiKeyStatus.textContent = text;
      apiKeyStatus.className = className || 'api-key-status';
    }

    function validateApiKey() {
      if (!apiKeyInput) {
        return;
      }
      const key = apiKeyInput.value.trim();
      if (!key) {
        setApiKeyStatus('Enter a key to validate.', 'api-key-status invalid');
        return;
      }
      setApiKeyStatus('Validating...', 'api-key-status');
      if (validateButton) {
        validateButton.disabled = true;
      }
      fetch('/validate-chatgpt5', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key }),
      })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok || !data.success) {
            throw new Error(data.message || 'Invalid API key');
          }
          setApiKeyStatus('API key is valid.', 'api-key-status valid');
        })
        .catch((err) => {
          setApiKeyStatus(err.message || 'Validation failed.', 'api-key-status invalid');
        })
        .finally(() => {
          if (validateButton) {
            validateButton.disabled = generatorSelect && generatorSelect.value !== 'chatgpt5' && generatorSelect.value !== 'gpt-3.5-turbo-instruct';
          }
        });
    }

    if (validateButton) {
      validateButton.addEventListener('click', validateApiKey);
    }

    if (apiKeyInput) {
      apiKeyInput.addEventListener('input', () => setApiKeyStatus('', 'api-key-status'));
    }


    function updateGeneratorState() {
      if (!generatorSelect || !apiKeyField) {
        return;
      }
      // Check if FLARE is selected
      const checkedRAGSystems = Array.from(ragSystemCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
      const isFlareSelected = checkedRAGSystems.includes('flare');
      
      // Show API key field if chatgpt5 is selected OR if FLARE is selected
      const useChatGPT = generatorSelect.value === 'chatgpt5' || generatorSelect.value === 'gpt-3.5-turbo-instruct';
      const shouldShowApiKey = useChatGPT || isFlareSelected;
      
      apiKeyField.style.display = shouldShowApiKey ? 'flex' : 'none';
      if (!shouldShowApiKey) {
        if (apiKeyInput) {
          apiKeyInput.value = '';
        }
        if (apiKeyStatus) {
          apiKeyStatus.textContent = '';
          apiKeyStatus.className = 'api-key-status';
        }
      }
      if (validateButton) {
        validateButton.disabled = !shouldShowApiKey;
      }
      applyInputAppearance(generatorSelect);
    }

    if (generatorSelect) {
      updateGeneratorState();
      generatorSelect.addEventListener('change', updateGeneratorState);
    }

    function updateStatus(data) {
      runButton.disabled = data.status === 'running';
      runStatus.textContent = data.message || '';
      runStatus.dataset.status = data.status || '';
      runStatus.dataset.lastResult = data.last_result || '';
    }

    function pollStatus() {
      fetch('/api/status')
        .then((res) => res.json())
        .then(updateStatus)
        .catch((err) => console.error('Status poll failed', err));
    }


    function formatMetricValue(value) {
      if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
      }
      return value.toFixed(4);
    }

    function formatMetricShort(value) {
      if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
      }
      return value.toFixed(3);
    }

    let renderedResultsByDataset = {};
    const tabExpandedState = {}; // Track expanded/collapsed state for each dataset
    const columnWidthsState = {}; // Track column widths for each dataset table: { dataset: [width1, width2, ...] }

    function setupColumnResizing(table) {
      const resizeHandles = table.querySelectorAll('.resize-handle');
      let isResizing = false;
      let currentColumn = null;
      let nextColumn = null;
      let startX = 0;
      let startLeftWidth = 0;
      let startRightWidth = 0;
      let columnIndex = -1;
      let allHeaders = [];
      let allRows = [];

      function setColumnWidth(column, width) {
        const widthPx = width + 'px';
        // Use !important via setProperty to override any CSS
        column.style.setProperty('width', widthPx, 'important');
        column.style.setProperty('min-width', widthPx, 'important');
        column.style.setProperty('max-width', widthPx, 'important');
        column.style.setProperty('flex-shrink', '0', 'important');
        column.style.setProperty('flex-grow', '0', 'important');
        // Store width in data attribute for persistence
        column.dataset.resizedWidth = width;
      }

      function updateColumnWidths(leftWidth, rightWidth) {
        // Update header cells
        if (currentColumn) {
          setColumnWidth(currentColumn, leftWidth);
        }
        if (nextColumn) {
          setColumnWidth(nextColumn, rightWidth);
        }
        
        // Update all body cells in these columns
        allRows.forEach((row) => {
          const cells = Array.from(row.querySelectorAll('td'));
          if (cells[columnIndex]) {
            setColumnWidth(cells[columnIndex], leftWidth);
          }
          if (nextColumn && cells[columnIndex + 1]) {
            setColumnWidth(cells[columnIndex + 1], rightWidth);
          }
        });
      }

      resizeHandles.forEach((handle) => {
        handle.addEventListener('mousedown', (e) => {
          e.preventDefault();
          e.stopPropagation();
          
          isResizing = true;
          currentColumn = handle.parentElement;
          columnIndex = parseInt(handle.dataset.columnIndex);
          allHeaders = Array.from(table.querySelectorAll('thead th'));
          allRows = Array.from(table.querySelectorAll('tbody tr'));
          
          // Find next column
          if (columnIndex < allHeaders.length - 1) {
            nextColumn = allHeaders[columnIndex + 1];
          } else {
            nextColumn = null;
          }
          
          // Convert all columns to pixel widths to prevent recalculation
          allHeaders.forEach((th, idx) => {
            const currentWidth = th.getBoundingClientRect().width;
            const widthPx = currentWidth + 'px';
            th.style.width = widthPx;
            th.style.minWidth = widthPx;
            th.style.maxWidth = widthPx;
            // Update all cells in this column
            allRows.forEach((row) => {
              const cells = Array.from(row.querySelectorAll('td'));
              if (cells[idx]) {
                cells[idx].style.width = widthPx;
                cells[idx].style.minWidth = widthPx;
                cells[idx].style.maxWidth = widthPx;
              }
            });
          });
          
          // Get initial widths in pixels
          startX = e.clientX;
          startLeftWidth = currentColumn.getBoundingClientRect().width;
          if (nextColumn) {
            startRightWidth = nextColumn.getBoundingClientRect().width;
          }
          
          handle.classList.add('active');
          document.body.style.cursor = 'col-resize';
          document.body.style.userSelect = 'none';
          
          // Prevent text selection during resize
          document.addEventListener('selectstart', preventSelection);
        });
      });

      function preventSelection(e) {
        if (isResizing) {
          e.preventDefault();
        }
      }

      function handleMouseMove(e) {
        if (!isResizing || !currentColumn) return;
        
        e.preventDefault();
        e.stopPropagation();
        
        const diff = e.clientX - startX;
        const newLeftWidth = Math.max(1, startLeftWidth + diff);
        const newRightWidth = nextColumn ? Math.max(1, startRightWidth - diff) : startRightWidth;
        
        // Apply new widths immediately
        updateColumnWidths(newLeftWidth, newRightWidth);
        
        // Prevent default browser behavior that might interfere
        return false;
      }

      function handleMouseUp(e) {
        if (isResizing) {
          // Calculate final widths from mouse position (more accurate than getBoundingClientRect)
          const finalX = e ? e.clientX : (window.event ? window.event.clientX : startX);
          const diff = finalX - startX;
          const finalLeftWidth = Math.max(1, startLeftWidth + diff);
          const finalRightWidth = nextColumn ? Math.max(1, startRightWidth - diff) : 0;
          
          // Lock in the final widths
          updateColumnWidths(finalLeftWidth, finalRightWidth);
          
          // Save column widths to state for persistence
          const tableContainer = table.closest('.dataset-tab');
          if (tableContainer) {
            const datasetId = tableContainer.id.replace('tab-', '');
            const allHeaders = Array.from(table.querySelectorAll('thead th'));
            const savedWidths = allHeaders.map(th => {
              const width = th.style.width || th.getBoundingClientRect().width + 'px';
              return width;
            });
            columnWidthsState[datasetId] = savedWidths;
          }
          
          // Force a reflow to ensure styles are applied
          table.offsetHeight;
          
          // Clean up
          if (currentColumn) {
            const handle = currentColumn.querySelector('.resize-handle');
            if (handle) {
              handle.classList.remove('active');
            }
          }
          
          isResizing = false;
          document.body.style.cursor = '';
          document.body.style.userSelect = '';
          document.removeEventListener('selectstart', preventSelection);
          currentColumn = null;
          nextColumn = null;
          columnIndex = -1;
          allHeaders = [];
          allRows = [];
        }
      }

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      
      // Also handle mouseleave to ensure cleanup
      document.addEventListener('mouseleave', handleMouseUp);
    }

    function appendResultRowToTable(tbody, questionKey, questionResults) {
      // questionResults is an array of results for the same question
      const firstResult = questionResults[0];
      const row = document.createElement('tr');
      
      // Article cell (same for all results)
      const articleCell = document.createElement('td');
      articleCell.innerHTML = '<div class="badge">ID: ' + (firstResult.article_id || 'N/A') + '</div><div>' + (firstResult.title || 'Unknown title') + '</div>';
      
      // Question cell (same for all results)
      const questionCell = document.createElement('td');
      questionCell.className = 'question';
      questionCell.textContent = firstResult.question || '';
      
      // Answer cell - show multiple answers in separate rows
      const answerCell = document.createElement('td');
      answerCell.className = 'answer';
      questionResults.forEach((item, idx) => {
        const answerDiv = document.createElement('div');
        answerDiv.style.marginBottom = idx < questionResults.length - 1 ? '1rem' : '0';
        answerDiv.style.paddingBottom = idx < questionResults.length - 1 ? '1rem' : '0';
        answerDiv.style.borderBottom = idx < questionResults.length - 1 ? '1px solid #dee2e6' : 'none';
        
        // RAG system label
        const ragSystemLabel = document.createElement('div');
        ragSystemLabel.style.fontWeight = '600';
        ragSystemLabel.style.color = '#495057';
        ragSystemLabel.style.marginBottom = '0.5rem';
        ragSystemLabel.style.fontSize = '0.9rem';
        const ragSystemName = item.rag_system === 'naive-rag' ? 'Naive RAG' : (item.rag_system === 'self-rag' ? 'Self-RAG' : (item.rag_system === 'flare' ? 'FLARE' : item.rag_system || 'Unknown'));
        
        // Add generation time if available
        let ragSystemText = ragSystemName + ':';
        if (item.generation_time !== null && item.generation_time !== undefined && !isNaN(item.generation_time)) {
          const timeSeconds = item.generation_time;
          if (timeSeconds < 1) {
            ragSystemText += ` <span style="font-weight: 400; color: #6c757d; font-size: 0.85rem;">(${(timeSeconds * 1000).toFixed(0)}ms)</span>`;
          } else {
            ragSystemText += ` <span style="font-weight: 400; color: #6c757d; font-size: 0.85rem;">(${timeSeconds.toFixed(2)}s)</span>`;
          }
        }
        ragSystemLabel.innerHTML = ragSystemText;
        answerDiv.appendChild(ragSystemLabel);
        
        // Answer text
        const answerText = document.createElement('div');
        answerText.textContent = item.generated_answer || '';
        answerDiv.appendChild(answerText);
        
        // Context (if enabled)
        const showContext = document.getElementById('input-show_context')?.checked !== false;
        if (Array.isArray(item.retrieved_context) && item.retrieved_context.length > 0 && showContext) {
          const contextDiv = document.createElement('div');
          contextDiv.className = 'context';
          contextDiv.style.marginTop = '0.5rem';
          const contextTitle = document.createElement('strong');
          contextTitle.textContent = 'Retrieved Context:';
          contextDiv.appendChild(contextTitle);
          const maxPreview = Math.min(item.retrieved_context.length, 3);
          for (let i = 0; i < maxPreview; i++) {
            const ctx = item.retrieved_context[i] || {};
            const ctxDiv = document.createElement('div');
            ctxDiv.className = 'context-item';
            const section = ctx.section_title || 'Unknown Section';
            const para = ctx.paragraph_index !== undefined && ctx.paragraph_index !== null ? ctx.paragraph_index : '?';
            ctxDiv.innerHTML = '<em>' + section + ' ¶' + para + '</em><br>' + (ctx.text_preview || (ctx.chunk ? ctx.chunk.slice(0, 200) + (ctx.chunk.length > 200 ? '...' : '') : ''));
            contextDiv.appendChild(ctxDiv);
          }
          if (item.retrieved_context.length > 3) {
            const moreDiv = document.createElement('div');
            moreDiv.className = 'context-item';
            moreDiv.innerHTML = '<em>+ ' + (item.retrieved_context.length - 3) + ' more chunks</em>';
            contextDiv.appendChild(moreDiv);
          }
          answerDiv.appendChild(contextDiv);
        }
        
        answerCell.appendChild(answerDiv);
      });
      
      // Reference answers cell (same for all results)
      const refCell = document.createElement('td');
      if (Array.isArray(firstResult.reference_answers) && firstResult.reference_answers.length > 0) {
        const list = document.createElement('ul');
        for (const ref of firstResult.reference_answers) {
          const li = document.createElement('li');
          li.textContent = ref;
          list.appendChild(li);
        }
        refCell.appendChild(list);
      } else {
        const placeholder = document.createElement('em');
        placeholder.textContent = 'No reference answers provided.';
        refCell.appendChild(placeholder);
      }
      
      // Score cell - show multiple scores in separate rows
      const scoreCell = document.createElement('td');
      questionResults.forEach((item, idx) => {
        const scoreDiv = document.createElement('div');
        scoreDiv.style.marginBottom = idx < questionResults.length - 1 ? '1rem' : '0';
        scoreDiv.style.paddingBottom = idx < questionResults.length - 1 ? '1rem' : '0';
        scoreDiv.style.borderBottom = idx < questionResults.length - 1 ? '1px solid #dee2e6' : 'none';
        
        // RAG system label
        const ragSystemLabel = document.createElement('div');
        ragSystemLabel.style.fontWeight = '600';
        ragSystemLabel.style.color = '#495057';
        ragSystemLabel.style.marginBottom = '0.5rem';
        ragSystemLabel.style.fontSize = '0.9rem';
        const ragSystemName = item.rag_system === 'naive-rag' ? 'Naive RAG' : (item.rag_system === 'self-rag' ? 'Self-RAG' : (item.rag_system === 'flare' ? 'FLARE' : item.rag_system || 'Unknown'));
        ragSystemLabel.textContent = ragSystemName + ':';
        scoreDiv.appendChild(ragSystemLabel);
        
        // Scores
        const scoreParts = [];
        if (item.exact_match !== null && item.exact_match !== undefined) {
          scoreParts.push('<strong>EM:</strong> ' + formatMetricValue(item.exact_match));
        }
        if (item.f1_score !== null && item.f1_score !== undefined) {
          scoreParts.push('<strong>F1:</strong> ' + formatMetricValue(item.f1_score));
        }
        if (item.recall_score !== null && item.recall_score !== undefined) {
          scoreParts.push('<strong>Recall:</strong> ' + formatMetricValue(item.recall_score));
        }
        if (item.rouge_l_score !== null && item.rouge_l_score !== undefined) {
          scoreParts.push('<strong>R-L:</strong> ' + formatMetricValue(item.rouge_l_score));
        }
        if (item.bleu_score !== null && item.bleu_score !== undefined) {
          scoreParts.push('<strong>BLEU:</strong> ' + formatMetricValue(item.bleu_score));
        }
        const scoreContent = document.createElement('div');
        scoreContent.innerHTML = scoreParts.length > 0 ? scoreParts.join('<br>') : '<em>No scores</em>';
        scoreDiv.appendChild(scoreContent);
        
        scoreCell.appendChild(scoreDiv);
      });
      
      row.appendChild(articleCell);
      row.appendChild(questionCell);
      row.appendChild(answerCell);
      row.appendChild(refCell);
      row.appendChild(scoreCell);
      tbody.appendChild(row);
    }

    function renderDatasetTabs(allResults, statsData) {
      if (!datasetTabsContainer) {
        return;
      }

      // Group results by dataset
      const resultsByDataset = {};
      allResults.forEach((result) => {
        const dataset = result.dataset || 'unknown';
        if (!resultsByDataset[dataset]) {
          resultsByDataset[dataset] = [];
        }
        resultsByDataset[dataset].push(result);
      });

      const datasets = Object.keys(resultsByDataset);
      if (datasets.length === 0) {
        datasetTabsContainer.innerHTML = '';
        if (emptyStateContainer) {
          emptyStateContainer.style.display = 'block';
        }
        return;
      }

      if (emptyStateContainer) {
        emptyStateContainer.style.display = 'none';
      }

      // Preserve expanded state before clearing
      const existingTabs = datasetTabsContainer.querySelectorAll('.dataset-tab');
      existingTabs.forEach((tab) => {
        const datasetId = tab.id.replace('tab-', '');
        const header = tab.querySelector('.tab-header');
        const content = tab.querySelector('.tab-content');
        if (header && content) {
          const isExpanded = content.classList.contains('expanded');
          if (isExpanded) {
            tabExpandedState[datasetId] = true;
          } else if (tabExpandedState[datasetId] === undefined) {
            // Only set to false if not previously tracked
            tabExpandedState[datasetId] = false;
          }
        }
      });

      datasetTabsContainer.innerHTML = '';

      datasets.forEach((dataset) => {
        // Restore expanded state if previously expanded
        const shouldBeExpanded = tabExpandedState[dataset] === true;
        const datasetResults = resultsByDataset[dataset] || [];
        const stats = statsData[dataset] || { count: 0, metrics: {} };
        const datasetLabel = {
          'qasper': 'QASPER',
          'qmsum': 'QMSum',
          'narrativeqa': 'NarrativeQA',
          'quality': 'QuALITY',
          'hotpot': 'HotpotQA',
          'musique': 'MuSiQue',
          'xsum': 'XSum',
          'wikiasp': 'WikiAsp',
          'longbench': 'LongBench'
        }[dataset] || dataset;

        const tab = document.createElement('div');
        tab.className = 'dataset-tab';
        tab.id = 'tab-' + dataset;

        const header = document.createElement('div');
        header.className = shouldBeExpanded ? 'tab-header' : 'tab-header collapsed';

        const headerRow = document.createElement('div');
        headerRow.className = 'tab-header-row';

        const title = document.createElement('div');
        title.className = 'tab-title';
        let titleText = datasetLabel;
        const questionsTested = stats.count || 0;
        const questionsAvailable = stats.total_questions_available;
        const questionsInDataset = stats.total_questions_in_dataset;
        
        // Build title with question counts
        let countText = '';
        if (questionsInDataset !== null && questionsInDataset !== undefined) {
          // Show: "X of Y tested (Z total in dataset)"
          if (questionsAvailable !== null && questionsAvailable !== undefined) {
            countText = questionsTested + ' of ' + questionsAvailable + ' tested (' + questionsInDataset + ' total in dataset)';
          } else {
            countText = questionsTested + ' tested (' + questionsInDataset + ' total in dataset)';
          }
        } else if (questionsAvailable !== null && questionsAvailable !== undefined) {
          // Fallback: show available if we don't have total in dataset
          countText = questionsTested + ' of ' + questionsAvailable + ' tested';
        } else {
          // Last fallback: just show tested
          countText = questionsTested + ' questions tested';
        }
        titleText += ' (' + countText + ')';
        title.textContent = titleText;

        const toggle = document.createElement('span');
        toggle.className = 'tab-toggle';
        toggle.textContent = '▼';

        headerRow.appendChild(title);
        headerRow.appendChild(toggle);

        const statsRow = document.createElement('div');
        statsRow.className = 'tab-stats';
        
        const metrics = [
          { key: 'exact_match', label: 'EM' },
          { key: 'f1_score', label: 'F1' },
          { key: 'recall_score', label: 'Recall' },
          { key: 'rouge_l_score', label: 'ROUGE-L' },
          { key: 'bleu_score', label: 'BLEU' }
        ];

        metrics.forEach((metric) => {
          const metricStat = stats.metrics[metric.key] || {};
          const avg = metricStat.average;
          if (avg !== null && avg !== undefined && !isNaN(avg)) {
            const statItem = document.createElement('div');
            statItem.className = 'tab-stat-item';
            const label = document.createElement('span');
            label.className = 'tab-stat-label';
            label.textContent = metric.label + ':';
            const value = document.createElement('span');
            value.className = 'tab-stat-value';
            value.textContent = formatMetricShort(avg);
            statItem.appendChild(label);
            statItem.appendChild(value);
            statsRow.appendChild(statItem);
          }
        });

        header.appendChild(headerRow);
        header.appendChild(statsRow);

        const content = document.createElement('div');
        // Restore expanded state if previously expanded
        content.className = shouldBeExpanded ? 'tab-content expanded' : 'tab-content';

        if (datasetResults.length === 0) {
          const emptyState = document.createElement('div');
          emptyState.className = 'tab-empty-state';
          emptyState.textContent = 'No results for this dataset.';
          content.appendChild(emptyState);
        } else {
          const table = document.createElement('table');
          table.className = 'tab-results-table';
          
          const thead = document.createElement('thead');
          const headerRow = document.createElement('tr');
          const headers = ['Article', 'Question', 'Generated Answer', 'Reference Answers', 'Score'];
          const defaultWidths = ['15%', '20%', '35%', '15%', '15%']; // Default column widths
          
          // Restore saved column widths if available
          const savedWidths = columnWidthsState[dataset];
          const widthsToUse = savedWidths || defaultWidths;
          
          headers.forEach((h, idx) => {
            const th = document.createElement('th');
            th.textContent = h;
            th.style.width = widthsToUse[idx];
            // If using saved pixel widths, also set min/max
            if (savedWidths && savedWidths[idx] && savedWidths[idx].includes('px')) {
              th.style.minWidth = savedWidths[idx];
              th.style.maxWidth = savedWidths[idx];
            }
            // Add resize handle (except for last column)
            if (idx < headers.length - 1) {
              const resizeHandle = document.createElement('div');
              resizeHandle.className = 'resize-handle';
              resizeHandle.dataset.columnIndex = idx;
              th.appendChild(resizeHandle);
            }
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);
          table.appendChild(thead);
          
          // Apply saved widths to body cells if available
          if (savedWidths) {
            const tbody = table.querySelector('tbody') || document.createElement('tbody');
            const allRows = Array.from(tbody.querySelectorAll('tr'));
            allRows.forEach((row) => {
              const cells = Array.from(row.querySelectorAll('td'));
              cells.forEach((cell, idx) => {
                if (savedWidths[idx]) {
                  cell.style.width = savedWidths[idx];
                  if (savedWidths[idx].includes('px')) {
                    cell.style.minWidth = savedWidths[idx];
                    cell.style.maxWidth = savedWidths[idx];
                  }
                }
              });
            });
          }
          
          // Add column resizing functionality
          setupColumnResizing(table);

          const tbody = document.createElement('tbody');
          // Group results by question (article_id + question text)
          const resultsByQuestion = {};
          datasetResults.forEach((result) => {
            const questionKey = (result.article_id || '') + '|||' + (result.question || '');
            if (!resultsByQuestion[questionKey]) {
              resultsByQuestion[questionKey] = [];
            }
            resultsByQuestion[questionKey].push(result);
          });
          
          // Render grouped results
          Object.keys(resultsByQuestion).forEach((questionKey) => {
            appendResultRowToTable(tbody, questionKey, resultsByQuestion[questionKey]);
          });

          table.appendChild(tbody);
          content.appendChild(table);
        }

        header.onclick = function(e) {
          e.preventDefault();
          e.stopPropagation();
          const isCollapsed = header.classList.contains('collapsed');
          if (isCollapsed) {
            // Expand: remove collapsed, add expanded
            header.classList.remove('collapsed');
            content.classList.add('expanded');
            tabExpandedState[dataset] = true;
          } else {
            // Collapse: add collapsed, remove expanded
            header.classList.add('collapsed');
            content.classList.remove('expanded');
            tabExpandedState[dataset] = false;
          }
        };

        tab.appendChild(header);
        tab.appendChild(content);
        datasetTabsContainer.appendChild(tab);
      });
    }

    function fetchResults() {
      fetch('/api/results')
        .then((res) => res.json())
        .then((data) => {
          if (!Array.isArray(data)) {
            return;
          }
          if (totalQuestions) {
            totalQuestions.textContent = String(data.length);
          }
          
          // Extract and display device information from results
          if (data.length > 0 && deviceInfo) {
            // Get device from first result (all results should use same device)
            const firstDevice = data[0].device;
            if (firstDevice) {
              deviceInfo.textContent = firstDevice;
            } else {
              deviceInfo.textContent = 'Unknown';
            }
          } else if (deviceInfo && data.length === 0) {
            deviceInfo.textContent = '-';
          }
          
          // Fetch statistics and render tabs
          fetch('/api/statistics')
            .then((res) => res.json())
            .then((stats) => {
              renderDatasetTabs(data, stats);
            })
            .catch((err) => {
              console.error('Failed to fetch statistics', err);
              renderDatasetTabs(data, {});
            });
        })
        .catch((err) => console.error('Failed to fetch results', err));
    }

    function resetResultsDisplay() {
      renderedResultsByDataset = {};
      if (datasetTabsContainer) {
        datasetTabsContainer.innerHTML = '';
      }
      if (emptyStateContainer) {
        emptyStateContainer.style.display = 'block';
        emptyStateContainer.textContent = 'Running tests... results will appear here.';
      }
      if (totalQuestions) {
        totalQuestions.textContent = '0';
      }
      if (deviceInfo) {
        deviceInfo.textContent = '-';
      }
    }

    console.log('Setting up event listeners...');
    console.log('runButton found:', !!runButton);
    
    try {
      if (runButton) {
        console.log('Attaching click event listener to run button');
        runButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Run Tests button clicked');
        runButton.disabled = true;
        runStatus.textContent = 'Starting tests...';
        resetResultsDisplay();
        const overrides = {};
        paramInputs.forEach((input) => {
          const name = input.name;
          if (!name) {
            return;
          }
          if (name === 'generator_model') {
            const defaultValue = input.dataset.default ?? '';
            const value = input.value.trim();
            if (value !== '' && value !== defaultValue) {
              overrides.generator_model = value;
            }
            return;
          }
          const defaultValue = input.dataset.default ?? '';
          const type = input.dataset.type || input.type || 'text';
          let value = input.value.trim();
          if (type === 'boolean' && input instanceof HTMLInputElement) {
            value = input.checked ? 'true' : 'false';
          }
          if (value === '' || value === defaultValue) {
            return;
          }
          if (type === 'number') {
            const parsedValue = Number(value);
            if (!Number.isNaN(parsedValue)) {
              overrides[name] = parsedValue;
            }
          } else if (type === 'boolean') {
            overrides[name] = value === 'true';
          } else {
            overrides[name] = value;
          }
        });

        const generatorDefault = generatorSelect ? (generatorSelect.dataset.default ?? '') : '';
        const generatorValue = generatorSelect ? generatorSelect.value.trim() : generatorDefault;
        if (generatorSelect && generatorValue !== generatorDefault) {
          overrides.generator_model = generatorValue;
        }
        
        // Handle multiple RAG system selection from checkboxes
        const checkedRAGSystems = Array.from(ragSystemCheckboxes)
          .filter(cb => cb.checked)
          .map(cb => cb.value);
        
        if (checkedRAGSystems.length === 0) {
          runStatus.textContent = 'Please select at least one RAG system.';
          runButton.disabled = false;
          return;
        }
        
        if (checkedRAGSystems.length === 1) {
          // Single RAG system - use legacy 'rag_system' field for backward compatibility
          overrides.rag_system = checkedRAGSystems[0];
        } else {
          // Multiple RAG systems - use 'rag_systems' array
          overrides.rag_systems = checkedRAGSystems;
        }
        
        // Handle multiple dataset selection from checkboxes
        const checkedDatasets = Array.from(datasetCheckboxes)
          .filter(cb => cb.checked)
          .map(cb => cb.value);
        
        if (checkedDatasets.length === 0) {
          runStatus.textContent = 'Please select at least one dataset.';
          runButton.disabled = false;
          return;
        }
        
        if (checkedDatasets.length === 1) {
          // Single dataset - use legacy 'dataset' field for backward compatibility
          overrides.dataset = checkedDatasets[0];
        } else {
          // Multiple datasets - use 'datasets' array
          overrides.datasets = checkedDatasets;
        }
        // Check if FLARE is selected
        if (checkedRAGSystems.includes('flare')) {
          // FLARE uses gpt-3.5-turbo-instruct internally, doesn't need chatgpt5 selection
          // Get OpenAI API key (required for FLARE)
          const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
          if (!apiKey) {
            runStatus.textContent = 'Please enter an OpenAI API key (required for FLARE).';
            runButton.disabled = false;
            return;
          }
          if (!(apiKeyStatus && apiKeyStatus.classList.contains('valid'))) {
            runStatus.textContent = 'Please validate the OpenAI API key before running tests.';
            runButton.disabled = false;
            return;
          }
          // FLARE uses openai_api_key (not chatgpt5_api_key)
          overrides.openai_api_key = apiKey;
          // Add retrieval instruction method if specified
          const retrievalInstructionMethod = document.getElementById('input-retrieval_instruction_method')?.value.trim() || '';
          if (retrievalInstructionMethod) {
            overrides.retrieval_instruction_method = retrievalInstructionMethod;
          }
          // FLARE uses gpt-3.5-turbo-instruct, don't override generator_model
          // The generator dropdown is disabled when FLARE is selected
        } else if (generatorValue === 'chatgpt5') {
          // Only check for chatgpt5 API key when FLARE is NOT selected
          const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
          if (!apiKey) {
            runStatus.textContent = 'Please enter an OpenAI API key.';
            runButton.disabled = false;
            return;
          }
          if (!(apiKeyStatus && apiKeyStatus.classList.contains('valid'))) {
            runStatus.textContent = 'Please validate the OpenAI API key before running tests.';
            runButton.disabled = false;
            return;
          }
          overrides.chatgpt5_api_key = apiKey;
        }
        
        // Add split parameter from the config display (it should be available)
        const splitDisplay = document.getElementById('config-split');
        if (splitDisplay && splitDisplay.textContent) {
          overrides.split = splitDisplay.textContent.trim();
        }
        
        console.log('Sending test run request with overrides:', { ...overrides, chatgpt5_api_key: overrides.chatgpt5_api_key ? '***' : undefined, openai_api_key: overrides.openai_api_key ? '***' : undefined });
        
        fetch('/run-tests', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overrides),
        })
          .then(async (res) => {
            console.log('Response status:', res.status, res.statusText);
            const data = await res.json();
            console.log('Response data:', data);
            if (!res.ok) {
              throw new Error(data.message || 'Failed to start tests');
            }
            updateStatus(data);
            fetchResults();
          })
          .catch((err) => {
            console.error('Fetch error:', err);
            runStatus.textContent = 'Error: ' + (err.message || 'Failed to start tests. Check console for details.');
            runButton.disabled = false;
          });
        });
        
        console.log('Event listener attached successfully');
      } else {
        console.error('runButton element not found! Cannot attach event listener.');
      }
    } catch (err) {
      console.error('Error setting up event listeners:', err);
      if (runStatus) {
        runStatus.textContent = 'Error: Failed to set up button. Check console for details.';
      }
    }

    try {
      setInterval(() => {
        pollStatus();
        fetchResults();
      }, 3000);
      fetchResults();
      console.log('Polling interval set up successfully');
    } catch (err) {
      console.error('Error setting up polling:', err);
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    default_dataset = getattr(TEST_OPTIONS, "dataset", "qasper")
    default_datasets = getattr(TEST_OPTIONS, "datasets", None)
    return render_template_string(
        TEMPLATE,
        results_path=RESULTS_PATH,
        run_state=RUN_STATE,
        config={
            "generator_model": TEST_OPTIONS.generator_model,
            "chunk_size": TEST_OPTIONS.chunk_size,
            "retrieval_k": TEST_OPTIONS.retrieval_k,
            "articles": TEST_OPTIONS.articles,
            "questions_per_article": TEST_OPTIONS.questions_per_article,
            "split": TEST_OPTIONS.split,
            "dataset": default_dataset,  # For backward compatibility
            "datasets": default_datasets if default_datasets else [default_dataset],
            "rag_system": getattr(TEST_OPTIONS, "rag_system", "naive-rag"),
            "rag_systems": getattr(TEST_OPTIONS, "rag_systems", None),
            "show_context": TEST_OPTIONS.show_context,
        },
    )


@app.route("/api/status")
def api_status():
    return jsonify(RUN_STATE)


@app.route("/validate-chatgpt5", methods=["POST"])
def validate_chatgpt5():
    """Validate ChatGPT5 API key using ChatGPT5Automation."""
    payload = request.get_json(silent=True) or {}
    api_key = payload.get("api_key")
    if not api_key:
        return jsonify({"success": False, "message": "API key is required."}), 400

    try:
        # Use ChatGPT5Automation to validate the API key directly
        # This is the same approach used for FLARE OpenAI validation
        sys.path.insert(0, str(PROJECT_ROOT / "NSTSCE"))
        from ChatGPT5Automation import ChatGPT5Automation
        
        chatgpt5 = ChatGPT5Automation(api_key=api_key)
        is_valid, message = chatgpt5.validate_api_key()
        
        if is_valid:
            return jsonify({"success": True, "message": "API key is valid."})
        else:
            return jsonify({"success": False, "message": message}), 400
            
    except Exception as exc:
        app.logger.error("API key validation failed: %s", exc)
        import traceback
        app.logger.error("Traceback: %s", traceback.format_exc())
        return jsonify({"success": False, "message": f"Validation error: {str(exc)}"}), 500


@app.route("/validate-openai", methods=["POST"])
def validate_openai():
    """Validate OpenAI API key for FLARE using ChatGPT5Automation."""
    payload = request.get_json(silent=True) or {}
    api_key = payload.get("api_key")
    if not api_key:
        return jsonify({"success": False, "message": "API key is required."}), 400

    try:
        # Use ChatGPT5Automation to validate the API key directly
        # This is the same approach used for ChatGPT5 validation
        sys.path.insert(0, str(PROJECT_ROOT / "NSTSCE"))
        from ChatGPT5Automation import ChatGPT5Automation
        
        chatgpt5 = ChatGPT5Automation(api_key=api_key)
        is_valid, message = chatgpt5.validate_api_key()
        
        if is_valid:
            return jsonify({"success": True, "message": "API key is valid."})
        else:
            return jsonify({"success": False, "message": message}), 400
            
    except Exception as exc:
        app.logger.error("API key validation failed: %s", exc)
        import traceback
        app.logger.error("Traceback: %s", traceback.format_exc())
        return jsonify({"success": False, "message": f"Validation error: {str(exc)}"}), 500

@app.route("/api/results")
def api_results():
    return jsonify(load_results())


@app.route("/api/statistics")
def api_statistics():
    """Return statistics for each dataset."""
    results = load_results()
    stats = calculate_dataset_statistics(results)
    return jsonify(stats)


@app.route("/run-tests", methods=["POST"])
def trigger_run():
    app.logger.info("Received POST request to /run-tests")
    payload = request.get_json(silent=True) or {}
    app.logger.info("Request payload (without API key): %s", {k: v for k, v in payload.items() if k != "chatgpt5_api_key"})
    options_dict = vars(TEST_OPTIONS).copy()

    for field in ("retrieval_k", "chunk_size", "questions_per_article", "articles"):
        if field in payload:
            try:
                options_dict[field] = int(payload[field])
            except (TypeError, ValueError):
                app.logger.warning("Ignoring invalid override for %s: %s", field, payload[field])

    if "show_context" in payload:
        options_dict["show_context"] = bool(payload["show_context"])

    if "split" in payload and payload["split"]:
        options_dict["split"] = str(payload["split"]).strip()

    if "generator_model" in payload and payload["generator_model"]:
        options_dict["generator_model"] = str(payload["generator_model"]).strip()

    # Handle RAG system selection (single or multiple)
    if "rag_systems" in payload and isinstance(payload["rag_systems"], list) and payload["rag_systems"]:
        # Multiple RAG systems
        rag_systems = [str(r).strip().lower() for r in payload["rag_systems"]]
        valid_rag_systems = [r for r in rag_systems if r in ("naive-rag", "self-rag", "flare")]
        if valid_rag_systems:
            if len(valid_rag_systems) == 1:
                # Single RAG system - use legacy field for compatibility
                options_dict["rag_system"] = valid_rag_systems[0]
            else:
                # Multiple RAG systems
                options_dict["rag_systems"] = valid_rag_systems
                options_dict["rag_system"] = valid_rag_systems[0]  # Keep for backward compatibility
        else:
            options_dict["rag_system"] = "naive-rag"
    elif "rag_system" in payload and payload["rag_system"]:
        # Single RAG system (legacy support)
        rag_system_value = str(payload["rag_system"]).strip().lower()
        if rag_system_value in ("naive-rag", "self-rag", "flare"):
            options_dict["rag_system"] = rag_system_value
        else:
            options_dict["rag_system"] = "naive-rag"

    # Handle dataset selection (single or multiple)
    if "datasets" in payload and isinstance(payload["datasets"], list) and payload["datasets"]:
        # Multiple datasets
        datasets = [str(d).strip().lower() for d in payload["datasets"]]
        valid_datasets = [d for d in datasets if d in ("qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench")]
        if valid_datasets:
            if len(valid_datasets) == 1:
                # Single dataset - use legacy field for compatibility
                options_dict["dataset"] = valid_datasets[0]
            else:
                # Multiple datasets
                options_dict["datasets"] = valid_datasets
                options_dict["dataset"] = valid_datasets[0]  # Keep for backward compatibility
        else:
            options_dict["dataset"] = "qasper"
    elif "dataset" in payload and payload["dataset"]:
        # Single dataset (legacy support)
        options_dict["dataset"] = str(payload["dataset"]).strip().lower()
        if options_dict["dataset"] not in ("qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"):
            options_dict["dataset"] = "qasper"

    # Default generator model based on RAG system
    # FLARE uses 'gpt-3.5-turbo-instruct' internally (not selectable), others use 't5-small'
    rag_systems = options_dict.get("rag_systems", [])
    rag_system = options_dict.get("rag_system", TEST_OPTIONS.rag_system if hasattr(TEST_OPTIONS, 'rag_system') else "naive-rag")
    
    # If FLARE is selected, it uses gpt-3.5-turbo-instruct internally (not chatgpt5)
    is_flare_selected = (rag_systems and 'flare' in rag_systems) or rag_system == 'flare'
    
    # FLARE doesn't use the generator_model from the dropdown - it uses gpt-3.5-turbo-instruct
    # For other RAG systems, default to t5-small
    default_generator = "t5-small"
    generator_value = str(options_dict.get("generator_model", TEST_OPTIONS.generator_model or default_generator)).lower()
    
    # Handle FLARE first
    if is_flare_selected:
        # FLARE uses gpt-3.5-turbo-instruct internally
        generator_value = "gpt-3.5-turbo-instruct"
        options_dict["generator_model"] = "gpt-3.5-turbo-instruct"
        options_dict["use_chatgpt5"] = True  # Set use_chatgpt5 flag for FLARE
        
        # FLARE requires OpenAI API key - check both openai_api_key and chatgpt5_api_key
        openai_api_key = payload.get("openai_api_key") or payload.get("chatgpt5_api_key")
        if openai_api_key:
            options_dict["openai_api_key"] = openai_api_key
        else:
            return jsonify({"status": "error", "message": "FLARE requires an OpenAI API key."}), 400
        options_dict["chatgpt5_api_key"] = None  # FLARE doesn't use chatgpt5_api_key
        
        # Handle retrieval instruction method for FLARE
        if "retrieval_instruction_method" in payload:
            retrieval_method = str(payload["retrieval_instruction_method"]).strip()
            if retrieval_method and retrieval_method in ("cot", "strategyqa", "summary"):
                options_dict["retrieval_instruction_method"] = retrieval_method
            else:
                options_dict["retrieval_instruction_method"] = None
        else:
            options_dict["retrieval_instruction_method"] = None
    elif generator_value == "chatgpt5":
        # Only handle chatgpt5 when FLARE is NOT selected
        api_key_override = payload.get("chatgpt5_api_key")
        if not api_key_override:
            return jsonify({"status": "error", "message": "ChatGPT5 generator requires an API key."}), 400
        options_dict["chatgpt5_api_key"] = api_key_override
        options_dict["openai_api_key"] = None
        options_dict["retrieval_instruction_method"] = None
    else:
        options_dict["chatgpt5_api_key"] = None
        options_dict["openai_api_key"] = None
        options_dict["retrieval_instruction_method"] = None

    updated_options = argparse.Namespace(**options_dict)
    
    # Log the options being used (excluding API keys)
    app.logger.info("Triggering test run with options: %s", {k: v for k, v in options_dict.items() if k not in ("chatgpt5_api_key", "openai_api_key")})

    clear_results_file()
    RUN_STATE["message"] = "Test run initiated..."
    if not start_test_run(updated_options, async_run=True):
        app.logger.warning("Failed to start test run: lock already held")
        return jsonify({"status": "busy", "message": "A run is already in progress."}), 409
    app.logger.info("Test run started successfully")
    return jsonify({"status": "running", "message": "Test run started..."})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the QASPER RAG interface.")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument("--articles", type=int, default=1, help="Number of articles to evaluate (0 means all).")
    parser.add_argument(
        "--questions-per-article",
        type=int,
        default=3,
        help="Maximum questions per article (0 means all questions).",
    )
    parser.add_argument("--retrieval-k", type=int, default=5, help="Chunks to retrieve for each query.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters for indexing.")
    parser.add_argument(
        "--generator-model",
        default="t5-small",
        help="Hugging Face model identifier for answer generation.",
    )
    parser.add_argument("--chatgpt5-api-key", default=None, help="API key for ChatGPT5 generator.")
    parser.add_argument("--show-context", dest="show_context", action="store_true", help="Display retrieved context in outputs.")
    parser.add_argument("--hide-context", dest="show_context", action="store_false", help="Hide retrieved context in outputs.")
    parser.set_defaults(show_context=False)
    parser.add_argument("--rag-system", choices=["naive-rag", "self-rag", "flare"], default="naive-rag", help="RAG system to use: 'naive-rag' (default), 'self-rag', or 'flare'.")
    parser.add_argument("--dataset", choices=["qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"], default="qasper", help="Dataset to use: 'qasper', 'qmsum', 'narrativeqa', 'quality', 'hotpot', 'musique', 'xsum', 'wikiasp', or 'longbench'.")
    parser.add_argument("--log-level", default="INFO", help="Log level passed to the test script.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH), help="Where to write the results JSON.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface host.")
    parser.add_argument("--port", type=int, default=5051, help="Interface port.")
    parser.add_argument("--run-on-start", action="store_true", help="Run the test suite automatically at startup.")
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false", help="Do not open the interface in the default browser.")
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


if __name__ == "__main__":
    import atexit
    import signal
    
    def shutdown_handler(signum=None, frame=None):
        """Handle shutdown signals and log termination."""
        app.logger.info("=" * 80)
        app.logger.info("QASPER Interface Terminating")
        app.logger.info("=" * 80)
        app.logger.info("Log file: %s", LOG_FILE)
        app.logger.info("Shutdown signal received: %s", signum if signum else "atexit")
        # Flush all log handlers
        for handler in logging.root.handlers:
            handler.flush()
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(shutdown_handler)
    
    TEST_OPTIONS = parse_args()
    RESULTS_PATH = Path(TEST_OPTIONS.results_path).resolve()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    app.logger.info("Configuration:")
    app.logger.info("  Host: %s", TEST_OPTIONS.host)
    app.logger.info("  Port: %s", TEST_OPTIONS.port)
    app.logger.info("  Results path: %s", RESULTS_PATH)
    app.logger.info("  Dataset: %s", TEST_OPTIONS.dataset)
    app.logger.info("  RAG system: %s", getattr(TEST_OPTIONS, "rag_system", "naive-rag"))
    app.logger.info("  Run on start: %s", TEST_OPTIONS.run_on_start)

    if TEST_OPTIONS.run_on_start:
        clear_results_file()
        start_test_run(TEST_OPTIONS, async_run=False)
    else:
        app.logger.info("Startup configured without automatic test run.")

    if TEST_OPTIONS.open_browser:
        target_host = TEST_OPTIONS.host if TEST_OPTIONS.host not in ("0.0.0.0", "::") else "127.0.0.1"
        Timer = threading.Timer
        Timer(1.5, lambda: webbrowser.open(f"http://{target_host}:{TEST_OPTIONS.port}")).start()
        app.logger.info("Browser will open at: http://%s:%s", target_host, TEST_OPTIONS.port)

    app.logger.info("Starting Flask server...")
    try:
        app.run(host=TEST_OPTIONS.host, port=TEST_OPTIONS.port, debug=True)
    except KeyboardInterrupt:
        shutdown_handler()
    except Exception as e:
        app.logger.exception("Fatal error during server execution: %s", e)
        shutdown_handler()


