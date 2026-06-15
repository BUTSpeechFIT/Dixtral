"""Download QA annotations and summaries from the NSF-QA dataset on Hugging Face."""

import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download NSF-QA annotations (QA + summaries only).")
    parser.add_argument("--local-dir", default="data/nsf_qa", help="Local directory to download into.")
    args = parser.parse_args()

    snapshot_download(
        repo_id="popcornell/NSF-QA",
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=["*_summaries.json", "*_flat.json"],
    )
    print(f"NSF-QA annotations downloaded to {args.local_dir}")

if __name__ == "__main__":
    main()
