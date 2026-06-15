# Dixtral: Diarization-Conditioned Target-Speaker ASR, QA and Summarization
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/collections/BUT-FIT)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()

**Dixtral** extends the [DiCoW](https://github.com/BUTSpeechFIT/DiCoW) target-speaker ASR system with an LLM decoder, enabling **per-speaker QA and summarization** directly from meeting audio.

> **Live demo available** — see [`demo/README.md`](demo/README.md) for instructions on running the Gradio interface locally.

The architecture combines:
- **DiCoW encoder** — a Whisper encoder augmented with Frame-level Diarization-Dependent Transformations (FDDT) that inject Silence–Target–Non-Target–Overlap (STNO) diarization masks into every encoder layer.
- **Voxtral decoder** — `mistralai/Voxtral-Mini-3B-2507` as the LLM backbone, fine-tuned via LoRA to answer questions about what a specific speaker said.


---


## Checkpoints

| Model                | Description                               | Link                                            |
|----------------------|-------------------------------------------|-------------------------------------------------|
| **DiCoW v3.3 large** | Diarization-conditioned Whisper (encoder) | https://huggingface.co/BUT-FIT/DiCoW_v3_3_large |
| **Dixtral**          | Diarization-conditioned Voxtral           | https://huggingface.co/BUT-FIT/Dixtral          |
| **Dixtral QA**       | Q/A finetunned variant                    | https://huggingface.co/BUT-FIT/Dixtral_QA       |

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BUTSpeechFIT/Dixtral.git
cd Dixtral
```

### 2. Create a Python Environment

**Conda**
```bash
conda create -n dixtral python=3.11
conda activate dixtral
```

**venv**
```bash
python -m venv dixtral
source dixtral/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Paths

Edit [`configs/local_paths.sh`](configs/local_paths.sh) to set:

| Variable | Description |
|----------|-------------|
| `SRC_ROOT` | Root of this repository |
| `MANIFEST_DIR` | Directory containing Lhotse manifest files |
| `EXPERIMENT_PATH` | Output directory for checkpoints and logs |
| `MUSAN_ROOT` | Path to MUSAN noise corpus (optional) |
| `HF_HOME` | Hugging Face cache directory |

### 5. System Dependencies
```bash
conda install -c conda-forge ffmpeg sox
# or
sudo apt install ffmpeg sox
```

---

## Data Preparation

### ASR Data

For standard target-speaker ASR, prepare Lhotse manifests using the dedicated repository:
👉 [**mt-asr-data-prep**](https://github.com/BUTSpeechFIT/mt-asr-data-prep)

Follow its instructions, then set `MANIFEST_DIR` in `configs/local_paths.sh`.

---

## Usage

The codebase uses **[Hydra](https://hydra.cc/)** for configuration. All configs are in `./configs`.

### Run Modes

| Config group | Description |
|---|---|
| `+train=base` | Train Dixtral encoder + decoder for target-speaker ASR |
| `+train=qa_ft` | Fine-tune Dixtral with LoRA for QA / summarization |
| `+train=dec_gt` | Decode with ground-truth diarization (ASR) |
| `+train=dec_gt_qa` | Decode with a QA-fine-tuned checkpoint |

Scripts are written for SLURM. Drop `sbatch` to run locally.

### Train Dixtral (Target-Speaker ASR)

```bash
sbatch ./scripts/submit_slurm.sh +train=base
```

Key config knobs (`configs/train/base.yaml`):
- `model.dixtral_base_model` — Voxtral model ID (default: `mistralai/Voxtral-Mini-3B-2507`)
- `model.dixtral_load_fddt_from` — path to a pretrained DiCoW checkpoint (provides FDDT weights)
- `training.use_lora` — enable LoRA on the LLM decoder
- `data.train_cutsets` — list of Lhotse manifest paths

### QA Data Preparation (NSF-QA)

Dixtral QA/summarization fine-tuning requires speaker-level question-answer pairs and summaries
annotated on top of an existing Lhotse cutset.
We provide annotations for NOTSOFAR1 via the **NSF-QA** dataset on Hugging Face.

#### Step 1 — Download NSF-QA Annotations

```bash
python utils/download_nsf_qa.py --local-dir data/nsf_qa
```

This downloads only the QA (`*_flat.json`), and summary (`*_summaries.json`) files from `popcornell/NSF-QA`.

The downloaded directory will contain:
```
data/nsf_qa/
  qa_annotations/
    train_qa_flat.json          # flat list of {session_id, speaker, question, answer, category, type}
    dev_qa_flat.json
    eval_qa_flat.json
    summaries/
      train/<session_id>_summaries.json   # per-speaker GT summaries
      dev/<session_id>_summaries.json
      eval/<session_id>_summaries.json
```

Each `*_qa_flat.json` is a list of records:
```json
[
  {"session_id": "MTG_30860", "speaker": "Peter", "question": "...", "answer": "...", "category": "content", "type": "entity"},
  ...
]
```

Each `*_summaries.json` file has:
```json
{
  "speaker_summaries": {
    "<speaker_id>": ["summary text 1", "summary text 2"]
  }
}
```

#### Step 2 — Populate the Lhotse Cutset

`utils/populate_cutset.py` merges QA/summary annotations into an existing Lhotse cutset,
storing all prompts and ground-truth answers in `cut.custom["speakers"]`.

```bash
for SPLIT in train dev eval; do
    python utils/populate_cutset.py \
        --cutset_path  ${MANIFEST_DIR}/notsofar1/notsofar1_sdm_${SPLIT}_set_*_cutset.jsonl.gz \
        --split        ${SPLIT} \
        --qa_dir       data/nsf_qa/qa_annotations \
        --summary_dir  data/nsf_qa/qa_annotations/summaries \
        --output_cutset ${MANIFEST_DIR}/notsofar1/notsofar1_sdm_${SPLIT}_set_*_cutset_qa.jsonl.gz
done
```

The populated cutset contains one entry per original cut, each with a `custom.speakers` dict:
```python
cut.custom["speakers"] = {
    "SPK1": [
        {"prompt": "Summarize what this speaker said.", "gt_answer": "...", "qa_type": "summary", ...},
        {"prompt": "What did the speaker propose?",     "gt_answer": "...", "qa_type": "content",  ...},
    ],
    ...
}
```
This format is consumed directly by `TS_QA_Dataset` during training and evaluation.

### Fine-tune for QA / Summarization

```bash
sbatch ./scripts/submit_slurm.sh +train=qa_ft
```

Key differences from ASR training (`configs/train/qa_ft.yaml`):
- `training.train_for_qa: True` — switches dataset, collator, and checkpoint selection
- `training.predict_with_generate: False` — uses loss for checkpoint selection during training
- `training.metric_for_best_model: eval_<split>_loss`
- `data.train_cutsets` / `dev_cutsets` / `eval_cutsets` — point to `*_qa.jsonl.gz` cutsets

### Decode Only

```bash
# ASR decode with GT diarization
sbatch ./scripts/submit_slurm.sh +decode=w_lora

# QA decode with GT diarization
sbatch ./scripts/submit_slurm.sh +decode=qa_ft
```

#### Long-form Data

For optimal performance, recordings longer than ~5 minutes should be chunked before decoding. `utils/chunk_longform_cutset.py` splits a GT cutset (and optionally an aligned diarization-predicted cutset) at silence boundaries:

```bash
python utils/chunk_longform_cutset.py \
    ${MANIFEST_DIR}/<corpus>/<corpus>_cutset_test.jsonl.gz \
    [path/to/diar_predicted_cutset.jsonl.gz] \
    --target-duration 300 \
    --output-dir ${MANIFEST_DIR}/<corpus>/
```

Output filenames are derived from the input basenames with `_<duration>s` appended (e.g. `<corpus>_cutset_test_300s.jsonl.gz`).

#### Multi-channel Data

For optimal performance, multi-channel recordings should be reduced to a single channel before decoding. Two options:

**Select a specific channel** — `utils/select_channel.py` extracts one channel from a recordings + supervisions manifest pair into a cutset:

```bash
python utils/select_channel.py \
    --input-recset ${MANIFEST_DIR}/<corpus>/recordings.jsonl.gz \
    --input-supset ${MANIFEST_DIR}/<corpus>/supervisions.jsonl.gz \
    --channel 4 \
    --output ${MANIFEST_DIR}/<corpus>/<corpus>_cuts_ch4.jsonl.gz
```

**Sum all channels** — set `data.load_signal_sum: true` in your config to average all channels to mono at data-loading time (no manifest preprocessing needed).

#### Diarized Cutsets

To decode with predicted (rather than ground-truth) diarization, first obtain diarized cutsets using the diarization pipeline from TS-ASR-Whisper:
👉 [`scripts/diarize.sh`](https://github.com/BUTSpeechFIT/TS-ASR-Whisper/blob/main/scripts/diarize.sh)

Then wire the resulting cutsets into your decode config via `data.dev_diar_cutsets` / `data.eval_diar_cutsets`, following the pattern in:
👉 [`configs/decode/dicow_v3_beam_joint_diar.yaml`](https://github.com/BUTSpeechFIT/TS-ASR-Whisper/blob/main/configs/decode/dicow_v3_beam_joint_diar.yaml)

---

## Configuration Details

Hydra configs are modular. Each file starts with:
```yaml
# @package _global_
```
and overrides `configs/base.yaml`. A config can inherit from another using `defaults`:

```yaml
# @package _global_
defaults:
  - /train/base
```

All available training/data/model parameters are documented in `src/utils/training_args.py`.

### Environment Variables (set in `configs/local_paths.sh`)

| Variable | Used for |
|---|---|
| `SRC_ROOT` | Python path root |
| `MANIFEST_DIR` | Lhotse manifest directory |
| `EXPERIMENT_PATH` | Checkpoint and log output |
| `MUSAN_ROOT` | MUSAN noise augmentation |

---

## Model Export

Export a trained checkpoint to Hugging Face Hub:

1. Create a model card at `export_sources/readmes/<HUB_MODEL_NAME>.md`
2. Optionally update `export_sources/generation_config.json`
3. Run:

```bash
python ./export_dixtral.py \
  --model_path <MODEL_DIR> \
  --model_name <HUB_MODEL_NAME> \
  --org <HUB_ORG>
```

---

## License

Source code is licensed under the [Apache License 2.0](LICENSE).

---

## Citation

If you use this code or models, please cite:

```bibtex

```

---

## Contributing

Issues and pull requests are welcome.

## Contact

- [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
