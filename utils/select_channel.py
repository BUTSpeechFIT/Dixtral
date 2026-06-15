import argparse

from lhotse import CutSet, RecordingSet, SupervisionSet, fix_manifests, load_manifest, fastcopy


def pick_channel(recording, channel_id: int):
    new_sources = [src for src in recording.sources if channel_id in src.channels]
    if not new_sources:
        return None
    return fastcopy(recording, sources=new_sources, channel_ids=[channel_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a single channel from a multi-channel Lhotse manifest.")
    parser.add_argument("--input-recset", required=True, help="Path to the input recordings manifest.")
    parser.add_argument("--input-supset", required=True, help="Path to the input supervisions manifest.")
    parser.add_argument("--channel", type=int, required=True, help="Channel index to select.")
    parser.add_argument("--output", required=True, help="Path to write the output cutset.")
    args = parser.parse_args()

    recordings = load_manifest(args.input_recset)
    recordings = RecordingSet.from_recordings(
        r for r in (pick_channel(rec, args.channel) for rec in recordings) if r is not None
    )

    supervisions = load_manifest(args.input_supset)
    supervisions = SupervisionSet.from_segments(
        fastcopy(s, channel=[args.channel]) for s in supervisions
    )

    cs = CutSet.from_manifests(*fix_manifests(recordings, supervisions))
    cs.to_file(args.output)
    print(f"Saved {len(cs)} cuts to {args.output}")
