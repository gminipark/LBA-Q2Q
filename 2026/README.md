# DramaQA Interactive Q2Q & Knowledge Graph Expansion (2026)

Interactive multimodal yes/no sub-questioning and dialogue-driven Knowledge Graph (KG) expansion over **DramaQA (Another Miss Oh)** images and episode-scene KGs using Qwen multimodal models.

## Directory Contents

```bash
2026/
├── run.py                  # Core CLI pipeline: DramaQA KB retrieval, VLM sub-questioning, KG expansion & HTML visualization
├── web_app.py              # FastAPI backend for multi-turn chat sessions, GPU/memory management, and KG expansion
├── web/
│   └── index.html          # Interactive browser UI for dialogue, representative scene frames, and KG viewer
└── requirements-web.txt    # FastAPI & Uvicorn dependencies for web_app.py
```

## Quick Start

### 1. CLI Mode (`run.py`)

Preview a real dataset sample without loading the model:
```bash
cd 2026
python run.py --show-sample
```

Dry-run KG and frame retrieval for a question:
```bash
python run.py --question "Why did Dokyung leave the room?" --dry-run --show-context
```

Run interactive sub-questioning and KG expansion:
```bash
python run.py --sample \
  --model Qwen/Qwen3.8-27B \
  --dtype bf16 \
  --top-scenes 2 \
  --max-images 4 \
  --min-subquestions 1 \
  --max-subquestions 5
```

Key flags:
- `--question`, `--vid`: Specify a custom question and/or comma-separated scene IDs.
- `--answers "yes,no"`: Run non-interactively with pre-supplied binary answers.
- `--no-kg-expansion`: Skip post-dialogue relation extraction and KG expansion.
- `--local-files-only`, `--cache-dir`: Control offline Hugging Face model loading.

### 2. Web Demo Mode (`web_app.py`)

Install web dependencies and start the server:
```bash
cd 2026
pip install -r requirements-web.txt
python web_app.py
```

Common environment variables:
- `LBA_MODEL`: Hugging Face model ID or local path (default: `Qwen/Qwen3.8-27B`)
- `LBA_DTYPE`: `auto`, `bf16`, `fp16`, or `fp32`
- `LBA_DATASET_ROOT`, `LBA_KG_PATH`, `LBA_QA_DIR`, `LBA_OUTPUT_DIR`: Override dataset/output paths
- `CUDA_VISIBLE_DEVICES`, `LBA_MAX_GPU_MEMORY_GIB`, `LBA_MAX_CPU_MEMORY_GIB`, `LBA_GPU_IDLE_TIMEOUT_MINUTES`: Runtime GPU allocation and idle auto-unload settings

## Acknowledgement

```text
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2022-II0951, Development of Uncertainty-Aware Agents Learning by Asking Questions)
```
