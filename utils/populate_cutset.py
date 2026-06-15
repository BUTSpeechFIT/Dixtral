"""
Dixtral Cutset Population Script - Target Speaker Summarization + QA

Populates a Lhotse cutset with one cut per (cut, speaker, prompt) pair,
storing prompt/answer in cut.custom fields, then saves as .jsonl.

Each new cut gets custom fields:
    task        : "summary" | "qa"
    speaker     : str
    prompt      : str
    gt_answers  : List[str]   (all GT summaries, or [single QA answer])
    qa_type     : str
    qa_category : str

Cut id format: {original_id}__{speaker}__{task}__{index}

Usage:
    python src/populate_cutset.py \
        --cutset_path data/manifests/ami/test.jsonl \
        --qa_dir /path/to/qa_sessions \
        --summary_dir /path/to/summary_sessions \
        --output_cutset data/manifests/ami/test_populated.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import lhotse
from lhotse import CutSet, MonoCut, fastcopy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = (
    "You are summarizing what a single speaker said and proposed during a meeting. "
    "Write a single, concise summary in at most 50 words capturing what this speaker "
    "proposed, suggested, or said. Focus on key points and decisions. "
    "Output only the summary text, no preamble or label."
)


# ──────────────────────────────────────────────────────────────────────────────
# Session data loader
# ──────────────────────────────────────────────────────────────────────────────

class SessionDataLoader:
    """Loads QA pairs and GT summaries from NSF-QA flat files and per-session summary JSONs.

    Expected layout (as downloaded by utils/download_nsf_qa.py):
        qa_dir/
            train_qa_flat.json      # list of {session_id, speaker, question, answer, category, type}
            dev_qa_flat.json
            eval_qa_flat.json
        summary_dir/
            train/<session_id>_summaries.json
            dev/<session_id>_summaries.json
            eval/<session_id>_summaries.json
    """

    def __init__(self, qa_dir: str, summary_dir: str, split: str):
        self.qa_dir      = Path(qa_dir)
        self.summary_dir = Path(summary_dir)
        self.split       = split
        self._qa_index: Optional[Dict] = None

    def _session_id_to_stem(self, session_id: str) -> str:
        return session_id.replace('sdm_', '').split('_sc')[0]

    def _build_qa_index(self) -> None:
        from collections import defaultdict
        self._qa_index = defaultdict(list)
        flat_file = self.qa_dir / f"{self.split}_qa_flat.json"
        with open(flat_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            self._qa_index[(rec['session_id'], rec['speaker'])].append({
                'question': rec.get('question', ''),
                'answer':   rec.get('answer', ''),
                'category': rec.get('category', ''),
                'type':     rec.get('type', ''),
            })
        logger.info(f"Indexed {len(records)} QA records from {flat_file.name}")

    def load_gt_summaries(self, session_id: str, speaker_name: str) -> List[str]:
        stem = self._session_id_to_stem(session_id)
        path = self.summary_dir / self.split / f"{stem}_summaries.json"
        if not path.exists():
            logger.debug(f"Summary file not found: {path}")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('speaker_summaries', {}).get(speaker_name, [])
        except Exception as e:
            logger.warning(f"Error loading GT summaries for {speaker_name} in {session_id}: {e}")
            return []

    def load_session_qa(self, session_id: str, speaker_name: str,
                        categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self._qa_index is None:
            self._build_qa_index()
        if categories is None:
            categories = ['content', 'paralinguistic', 'gender']
        stem = self._session_id_to_stem(session_id)
        return [r for r in self._qa_index.get((stem, speaker_name), []) if r['category'] in categories]


# ──────────────────────────────────────────────────────────────────────────────
# Population
# ──────────────────────────────────────────────────────────────────────────────

def _get_cut_speakers(cut: MonoCut) -> List[str]:
    return sorted({sup.speaker for sup in cut.supervisions})

def populate_cutset(
        cutset: CutSet,
        session_loader: SessionDataLoader,
        output_cutset_path: str,
) -> CutSet:
    new_cuts: List[MonoCut] = []
    n_skipped_qa      = 0
    n_skipped_summary = 0

    for cut in cutset:
        session_id = cut.recording_id
        speakers   = _get_cut_speakers(cut)

        speakers_data: Dict[str, Any] = {}

        for speaker in speakers:
            # ── summaries ─────────────────────────────────────────────────────
            gt_summaries = session_loader.load_gt_summaries(session_id, speaker)
            if not gt_summaries:
                n_skipped_summary += 1

            # ── QA ────────────────────────────────────────────────────────────
            qa_pairs = session_loader.load_session_qa(session_id, speaker)
            if not qa_pairs:
                n_skipped_qa += 1

            speakers_data[speaker] = [
                *[
                    {
                        "prompt":      SUMMARIZE_PROMPT,
                        "gt_answer":   gt_summary,
                        "qa_type":     "summary",
                        "qa_category": "summary",
                        "speaker": speaker,
                    }
                    for gt_summary in gt_summaries
                ],
                *[
                    {
                        "prompt":      qa["question"],
                        "gt_answer":   qa["answer"],
                        "qa_type":     qa["type"],
                        "qa_category": qa["category"],
                        "speaker": speaker,
                    }
                    for qa in qa_pairs
                ],
            ]

        new_cut = fastcopy(cut, supervisions=[
            fastcopy(sup, text=None, alignment=None) for sup in cut.supervisions
        ])
        new_cut.custom = {"speakers": speakers_data}
        new_cuts.append(new_cut)

    populated = CutSet.from_cuts(new_cuts)

    output_path = Path(output_cutset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    populated.to_file(str(output_path))

    logger.info(f"✓ Saved {len(new_cuts)} cuts → {output_path}")
    if n_skipped_summary:
        logger.warning(f"  {n_skipped_summary} (cut, speaker) pairs had no GT summaries")
    if n_skipped_qa:
        logger.warning(f"  {n_skipped_qa} (cut, speaker) pairs had no QA data")

    return populated
# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Populate a Lhotse cutset with summarization + QA prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/populate_cutset.py \\
    --cutset_path data/manifests/ami/test.jsonl \\
    --qa_dir /path/to/qa_sessions \\
    --summary_dir /path/to/summary_sessions \\
    --output_cutset data/manifests/ami/test_populated.jsonl
        """
    )
    parser.add_argument('--cutset_path',   type=str, required=True,
                        help='Input Lhotse cutset manifest (.jsonl)')
    parser.add_argument('--split',         type=str, required=True, choices=['train', 'dev', 'eval'],
                        help='Dataset split (train/dev/eval)')
    parser.add_argument('--qa_dir',        type=str, required=True,
                        help='Directory containing {split}_qa_flat.json files')
    parser.add_argument('--summary_dir',   type=str, required=True,
                        help='Directory containing {split}/<session>_summaries.json files')
    parser.add_argument('--output_cutset', type=str, required=True,
                        help='Path to save the populated cutset (.jsonl)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Cutset Population: Summarization + QA")
    logger.info("=" * 60)
    logger.info(f"  cutset_path  : {args.cutset_path}")
    logger.info(f"  split        : {args.split}")
    logger.info(f"  qa_dir       : {args.qa_dir}")
    logger.info(f"  summary_dir  : {args.summary_dir}")
    logger.info(f"  output_cutset: {args.output_cutset}")
    logger.info("=" * 60)

    try:
        cutset = lhotse.load_manifest(args.cutset_path)
        logger.info(f"✓ Loaded {len(cutset)} cuts from {args.cutset_path}")

        session_loader = SessionDataLoader(
            qa_dir=args.qa_dir,
            summary_dir=args.summary_dir,
            split=args.split,
        )
        populate_cutset(cutset, session_loader, args.output_cutset)

        return 0

    except Exception as e:
        logger.error(f"Population failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())