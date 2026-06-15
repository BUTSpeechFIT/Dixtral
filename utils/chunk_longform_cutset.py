import argparse
import dataclasses
import os
from typing import List, Dict, Tuple

from lhotse import CutSet


def split_cuts_near_silence(cutset: CutSet, target_duration: float = 300.0) -> Tuple[
    CutSet, Dict[str, List[Tuple[float, float, int]]]]:
    """
    Splits cuts in a CutSet into chunks of approximately `target_duration` seconds.
    Returns the split CutSet and a dict mapping original cut.id -> list of (window_start, window_end, index).
    """
    out_cuts = []
    split_boundaries: Dict[str, List[Tuple[float, float, int]]] = {}

    def apply_new_id(cut, new_id):
        cut.id = new_id
        if getattr(cut, "recording", None) is not None:
            cut.recording = dataclasses.replace(cut.recording, id=new_id)
        for sup in cut.supervisions:
            sup.recording_id = new_id
        return cut

    for cut in cutset:
        boundaries = []

        if cut.duration <= target_duration:
            out_cuts.append(cut)
            boundaries.append((0.0, cut.duration, 0))
            split_boundaries[cut.id] = boundaries
            continue

        window_start = 0.0
        last_sup_end = 0.0
        window_index = 0
        sups = sorted(cut.supervisions, key=lambda s: s.start)

        if not sups:
            while window_start < cut.duration:
                dur = min(target_duration, cut.duration - window_start)
                new_cut = cut.truncate(offset=window_start, duration=dur)
                new_cut = apply_new_id(new_cut, f"{cut.id}-{window_index}")
                out_cuts.append(new_cut)
                boundaries.append((window_start, window_start + dur, window_index))
                window_start += dur
                window_index += 1
            split_boundaries[cut.id] = boundaries
            continue

        for sup in sups:
            gap_start = last_sup_end
            gap_end = sup.start
            if gap_end > gap_start:
                candidate_split_point = (gap_start + gap_end) / 2.0
                if candidate_split_point - window_start >= target_duration:
                    dur = candidate_split_point - window_start
                    new_cut = cut.truncate(offset=window_start, duration=dur)
                    new_cut = apply_new_id(new_cut, f"{cut.id}-{window_index}")
                    out_cuts.append(new_cut)
                    boundaries.append((window_start, window_start + dur, window_index))
                    window_start = candidate_split_point
                    window_index += 1
            last_sup_end = max(last_sup_end, sup.end)

        if window_start < cut.duration:
            dur = cut.duration - window_start
            new_cut = cut.truncate(offset=window_start, duration=dur)
            new_cut = apply_new_id(new_cut, f"{cut.id}-{window_index}")
            out_cuts.append(new_cut)
            boundaries.append((window_start, window_start + dur, window_index))

        split_boundaries[cut.id] = boundaries

    return CutSet.from_cuts(out_cuts), split_boundaries


def split_predicted_cutset_on_boundaries(
        pred_cutset: CutSet,
        split_boundaries: Dict[str, List[Tuple[float, float, int]]],
) -> CutSet:
    """
    Splits the predicted cutset using the same boundaries as the GT cutset.
    Supervisions that span a boundary are split at that boundary.
    """
    out_cuts = []

    def apply_new_id(cut, new_id):
        cut.id = new_id
        if getattr(cut, "recording", None) is not None:
            cut.recording = dataclasses.replace(cut.recording, id=new_id)
        for sup in cut.supervisions:
            sup.recording_id = new_id
        return cut

    for cut in pred_cutset:
        # Match by base cut id (pred cut may have same id as GT cut)
        base_id = cut.id
        if base_id not in split_boundaries:
            # No splitting needed or unknown cut
            out_cuts.append(cut)
            continue

        boundaries = split_boundaries[base_id]

        if len(boundaries) == 1:
            # No split was done
            out_cuts.append(cut)
            continue

        pred_sups = sorted(cut.supervisions, key=lambda s: s.start)

        for (win_start, win_end, win_idx) in boundaries:
            new_cut = cut.truncate(offset=win_start, duration=win_end - win_start)

            # Collect supervisions that overlap with [win_start, win_end)
            chunk_sups = []
            for sup in pred_sups:
                sup_start = sup.start
                sup_end = sup.end

                # Check overlap
                if sup_end <= win_start or sup_start >= win_end:
                    continue

                # Clip supervision to window
                clipped_start = max(sup_start, win_start)
                clipped_end = min(sup_end, win_end)

                # Adjust to be relative to the new cut's start
                rel_start = clipped_start - win_start
                rel_end = clipped_end - win_start

                new_sup = dataclasses.replace(
                    sup,
                    start=rel_start,
                    duration=rel_end - rel_start,
                    recording_id=f"{base_id}-{win_idx}",
                )
                chunk_sups.append(new_sup)

            # Replace supervisions on the truncated cut
            new_cut = dataclasses.replace(new_cut, supervisions=chunk_sups)
            new_cut = apply_new_id(new_cut, f"{base_id}-{win_idx}")
            out_cuts.append(new_cut)

    return CutSet.from_cuts(out_cuts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a long-form cutset into fixed-duration windows.")
    parser.add_argument("cutset", help="Path to the input GT cutset file.")
    parser.add_argument("pred_cutset", nargs="?", default=None,
                        help="Path to the predicted diarization cutset file (optional).")
    parser.add_argument("--target-duration", type=float, default=300.0,
                        help="Target chunk duration in seconds (default: 300).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the output cutset files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gt_cuts = CutSet.from_file(args.cutset)

    split_gt_cuts, split_boundaries = split_cuts_near_silence(gt_cuts, target_duration=args.target_duration)

    gt_basename = os.path.basename(args.cutset).replace(".jsonl.gz", f"_{int(args.target_duration)}s.jsonl.gz")
    split_gt_cuts.to_file(os.path.join(args.output_dir, gt_basename))

    if args.pred_cutset is not None:
        pred_cuts = CutSet.from_file(args.pred_cutset)
        pred_basename = os.path.basename(args.pred_cutset).replace(".jsonl.gz",
                                                                   f"_{int(args.target_duration)}s.jsonl.gz")
        split_pred_cuts = split_predicted_cutset_on_boundaries(pred_cuts, split_boundaries)
        split_pred_cuts.to_file(os.path.join(args.output_dir, pred_basename))
        print(f"GT cuts: {len(split_gt_cuts)}, Pred cuts: {len(split_pred_cuts)}")
    else:
        print(f"GT cuts: {len(split_gt_cuts)}")
