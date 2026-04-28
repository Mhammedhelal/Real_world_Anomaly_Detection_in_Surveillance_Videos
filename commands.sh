#!/usr/bin/env bash
# =============================================================================
# Real-world Anomaly Detection in Surveillance Videos
# COMMAND GUIDE — Feature Extraction → Training → Evaluation → Testing
# =============================================================================
# Usage:
#   chmod +x commands.sh
#   bash commands.sh          # prints this guide
#   source commands.sh        # loads helper aliases into your shell
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. PROJECT SETUP
# ---------------------------------------------------------------------------

# Activate your Python virtual environment (adjust path as needed)
#   source /home/$USER/ml_env/bin/activate

# Make sure you are in the project root
#   cd /path/to/Real_world_Anomaly_Detection_in_Surveillance_Videos

# Set PYTHONPATH so `src.*` imports resolve
export PYTHONPATH=.

# Install all dependencies
#   pip install -r requirements.txt
#   pip install pytorchvideo          # needed for I3D feature extractor
#   pip install pytest pytest-cov pytest-xdist   # needed for tests


# =============================================================================
# 1. FEATURE EXTRACTION
# =============================================================================
# Script : scripts/extract_features.py
# Purpose: Read raw video files, run I3D + YOLOv8, save .npz feature files.
#
# Required flags:
#   --video-folder NAME   subfolder under data/videos/ (e.g. normal, anomalous)
#   OR
#   --video-dir   PATH    explicit absolute/relative path to video directory
#
# Optional flags:
#   --split       train|test       which split label to embed  (default: train)
#   --batch-size  INT              videos processed per batch  (default: 50)
#   --max-videos  INT              stop after N videos         (default: all)
#   --resume                       skip already-extracted files (default: on)
#   --force                        re-extract even if file exists
#   --dry-run                      list what would be processed, no extraction
#   --yes                          skip the "Proceed?" prompt
#   --config      PATH             path to YAML config         (default: configs/default.yaml)
#   --log-dir     PATH             where to write .log files   (default: outputs/logs)
# -----------------------------------------------------------------------------

# --- 1a. Dry run (preview, no extraction) ------------------------------------
python scripts/extract_features.py \
    --video-folder normal \
    --split train \
    --dry-run

# --- 1b. Extract training features from the "normal" folder ------------------
python scripts/extract_features.py \
    --video-folder normal \
    --split train \
    --yes

# --- 1c. Extract test features from an explicit path -------------------------
python scripts/extract_features.py \
    --video-dir data/videos/anomalous \
    --split test \
    --yes

# --- 1d. Extract with a cap on the number of videos --------------------------
python scripts/extract_features.py \
    --video-folder normal \
    --split train \
    --max-videos 100 \
    --yes

# --- 1e. Force re-extraction (ignore cached .npz files) ----------------------
python scripts/extract_features.py \
    --video-folder normal \
    --split train \
    --force \
    --yes

# --- 1f. Extract with a custom config file -----------------------------------
python scripts/extract_features.py \
    --video-folder normal \
    --split train \
    --config configs/default.yaml \
    --log-dir outputs/logs \
    --yes

# Custom save directory
python scripts/extract_features.py --video-folder normal --split train \
    --save-dir my_project/features/normal_train --yes

# Custom save + custom metadata
python scripts/extract_features.py --video-folder normal --split train \
    --save-dir /data/features/normal \
    --metadata-dir /data/features/metadata \
    --yes

# Dry-run to preview where files would go
python scripts/extract_features.py --video-folder normal --split train \
    --save-dir /data/features/normal --dry-run

# =============================================================================
# 2. TRAINING
# =============================================================================
# Script : scripts/train.py
# Purpose: Train the Bi-GRU anomaly detector on extracted .npz features.
#
# Required flags:
#   --features-dir PATH    directory that contains train_*.npz files
#
# Optional flags:
#   --save-dir    PATH     checkpoint output directory       (default: ./checkpoints)
#   --config      PATH     YAML config file                  (default: configs/default.yaml)
#   --epochs      INT      number of training epochs         (default: from config → 100)
#   --batch-size  INT      samples per batch                 (default: from config → 32)
#   --lr          FLOAT    learning rate                     (default: from config → 1.0)
#   --input-size  INT      feature vector dimension          (default: 2131)
#   --hidden-size INT      GRU hidden units                  (default: 256)
#   --num-classes INT      number of output classes          (default: 14)
#   --device      cuda|cpu target device                     (default: auto)
#   --resume      PATH     checkpoint to resume training from
#   --run-name    STR      label for log filenames
#   --log-dir     PATH     log output directory              (default: outputs/logs)
# -----------------------------------------------------------------------------

# --- 2a. Minimal training run (uses all config defaults) ----------------------
python scripts/train.py \
    --features-dir data/features/extracted

# --- 2b. Full training with explicit arguments --------------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --save-dir     outputs/checkpoints \
    --epochs       100 \
    --batch-size   32 \
    --lr           1.0 \
    --device       cuda \
    --run-name     run_001 \
    --log-dir      outputs/logs

# --- 2c. Quick smoke-test training (5 epochs, CPU) ---------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --epochs       5 \
    --batch-size   4 \
    --device       cpu \
    --run-name     smoke_test

# --- 2d. Resume from a saved checkpoint --------------------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --resume       outputs/checkpoints/best_model.pt \
    --epochs       50 \
    --run-name     resumed_run

# --- 2e. Custom model architecture -------------------------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --input-size   2131 \
    --hidden-size  512 \
    --num-classes  14 \
    --epochs       100

# --- 2f. CPU-only training (no GPU) ------------------------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --device       cpu \
    --batch-size   8 \
    --epochs       10


# =============================================================================
# 3. EVALUATION
# =============================================================================
# Script : scripts/evaluate.py
# Purpose: Compute AUC-ROC, AUC-PR, accuracy, per-class accuracy, confusion
#          matrix, and optionally save plots.
#
# Required flags:
#   --features-dir  PATH    directory with test_*.npz (or train_*.npz) files
#   --checkpoint    PATH    path to a saved .pt checkpoint
#
# Optional flags:
#   --split         train|test   which split to evaluate on  (default: test)
#   --batch-size    INT          inference batch size         (default: from config)
#   --num-classes   INT          number of classes            (default: 14)
#   --config        PATH         YAML config file
#   --device        cuda|cpu     target device               (default: auto)
#   --save-dir      PATH         save plots + JSON results here
#   --no-plot                    skip matplotlib plots
#   --log-dir       PATH         log output directory        (default: outputs/logs)
# -----------------------------------------------------------------------------

# --- 3a. Standard evaluation on the test split --------------------------------
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt

# --- 3b. Evaluate with results saved to disk ----------------------------------
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt \
    --save-dir     outputs/evaluation \
    --split        test

# --- 3c. Evaluate on the training split (sanity check) -----------------------
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt \
    --split        train \
    --no-plot

# --- 3d. Evaluate on CPU (useful when no GPU is available) -------------------
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt \
    --device       cpu \
    --no-plot

# --- 3e. Evaluate a specific epoch checkpoint (not best) ---------------------
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/anomaly_detector_epoch0050.pt \
    --save-dir     outputs/evaluation/epoch50


# =============================================================================
# 4. TESTING (pytest)
# =============================================================================
# Framework : pytest
# Config    : tests/pytest.ini
# Coverage  : tests/conftest.py + test_*.py files
# -----------------------------------------------------------------------------

# --- 4a. Run the complete test suite -----------------------------------------
pytest tests/ -v

# --- 4b. Run only a specific test file ---------------------------------------
pytest tests/test_anomaly_detector_comprehensive.py -v
pytest tests/test_loss_comprehensive.py              -v
pytest tests/test_dataset_comprehensive.py           -v
pytest tests/test_feature_extractors_comprehensive.py -v
pytest tests/test_integration_comprehensive.py       -v
pytest tests/test_refactored_modules.py              -v

# --- 4c. Run a specific test class -------------------------------------------
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward -v
pytest tests/test_loss_comprehensive.py::TestMILRankingLossGradients             -v
pytest tests/test_dataset_comprehensive.py::TestCollateFunction                  -v

# --- 4d. Run a single named test ---------------------------------------------
pytest "tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward::test_forward_pass_basic" -v

# --- 4e. Filter by keyword (runs all matching tests across all files) ---------
pytest -v -k "determinism"               tests/   # reproducibility tests
pytest -v -k "gradient or backward"      tests/   # gradient-flow tests
pytest -v -k "edge or zero or single"    tests/   # edge-case tests
pytest -v -k "forward"                   tests/   # forward-pass tests
pytest -v -k "device or cuda or cpu"     tests/   # device-handling tests
pytest -v -k "integration"               tests/   # integration tests
pytest -v -k "not slow"                  tests/   # skip long-running tests

# --- 4f. Run tests in parallel (requires pip install pytest-xdist) -----------
pytest -n 4  tests/   # 4 workers
pytest -n auto tests/  # number of workers = CPU count

# --- 4g. Stop on the first failure -------------------------------------------
pytest -x tests/

# --- 4h. Generate a coverage report (HTML) -----------------------------------
pytest --cov=src --cov-report=html tests/
# Then open: htmlcov/index.html

# --- 4i. Generate a coverage report (terminal) --------------------------------
pytest --cov=src --cov-report=term-missing tests/

# --- 4j. Generate a JUnit XML report (for CI/CD) -----------------------------
pytest --junitxml=junit.xml tests/

# --- 4k. Debug a failing test (show print output + drop into pdb) ------------
pytest -vv -s --pdb \
    "tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward::test_forward_pass_basic"

# --- 4l. Force CPU-only testing (disable CUDA) --------------------------------
CUDA_VISIBLE_DEVICES="" pytest tests/ -v

# --- 4m. Show the 10 slowest tests -------------------------------------------
pytest --durations=10 tests/

# --- 4n. Collect (list) all tests without running them -----------------------
pytest --collect-only -q tests/


# =============================================================================
# 5. INFERENCE SERVICE (FastAPI)
# =============================================================================
# Entry point: inference_service/app/main.py
# The service receives base64 frames from a Node.js back-end and returns
# anomaly scores, class labels, and bounding boxes as JSON.
# -----------------------------------------------------------------------------

# --- 5a. Start the inference server locally ----------------------------------
uvicorn inference_service.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

# --- 5b. Start with auto-reload (development) --------------------------------
uvicorn inference_service.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload

# --- 5c. Run smoke tests against the live service ----------------------------
python inference_service/test_service.py

# --- 5d. Run smoke tests against a remote service ----------------------------
python inference_service/test_service.py \
    --url http://192.168.1.100:8000

# --- 5e. Docker: build and start the container --------------------------------
#   docker compose -f inference_service/docker-compose.yaml up --build

# --- 5f. Docker: run on CPU (edit docker-compose.yaml to set DEVICE=cpu) -----
#   DEVICE=cpu docker compose -f inference_service/docker-compose.yaml up


# =============================================================================
# 6. QUICK WORKFLOWS
# =============================================================================

# --- Full pipeline from raw videos to evaluation (GPU) -----------------------
# Step 1: extract features
python scripts/extract_features.py \
    --video-folder normal    --split train --yes
python scripts/extract_features.py \
    --video-dir data/videos/anomalous --split test --yes

# Step 2: train
python scripts/train.py \
    --features-dir data/features/extracted \
    --save-dir     outputs/checkpoints \
    --epochs       100 \
    --run-name     full_run

# Step 3: evaluate
python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt \
    --save-dir     outputs/evaluation

# Step 4: run tests
pytest tests/ -v --cov=src --cov-report=html


# --- Quick smoke-test cycle (CPU, no GPU needed) ------------------------------
python scripts/train.py \
    --features-dir data/features/extracted \
    --epochs 5 --batch-size 4 --device cpu --run-name smoke

python scripts/evaluate.py \
    --features-dir data/features/extracted \
    --checkpoint   outputs/checkpoints/best_model.pt \
    --device cpu --no-plot

pytest tests/ -q -k "not slow"


# =============================================================================
# 7. ENVIRONMENT & DEPENDENCY CHECKS
# =============================================================================

# Check Python and package versions
python  --version
pip show torch torchvision numpy opencv-python ultralytics pytorchvideo

# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
           print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Install / upgrade test dependencies
pip install pytest pytest-cov pytest-xdist --upgrade

# Install feature extraction dependencies
pip install pytorchvideo ultralytics --upgrade

# Install inference service dependencies
pip install fastapi uvicorn pydantic --upgrade


# =============================================================================
# 8. OUTPUT LOCATIONS (reference)
# =============================================================================
#
#   data/features/extracted/      ← .npz files from feature extraction
#   data/features/metadata/       ← extraction_progress.json
#   outputs/checkpoints/          ← .pt checkpoint files (best_model.pt + per-epoch)
#   outputs/logs/                 ← .log files + _metrics.jsonl files
#   outputs/evaluation/           ← evaluation_results.png + .json
#   htmlcov/                      ← pytest HTML coverage report
#   junit.xml                     ← pytest JUnit report (for CI/CD)
#
# =============================================================================
echo ""
echo "Command guide loaded. Copy any command above and run it in your shell."
echo "Remember to activate your virtual environment and set PYTHONPATH=. first."