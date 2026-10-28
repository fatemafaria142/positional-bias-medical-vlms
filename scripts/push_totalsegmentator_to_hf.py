"""
Push the staged 50-subject TotalSegmentatorV2 subset (data/totalsegmentator_50/,
produced by scripts/stage_totalsegmentator_50.py) to a Hugging Face dataset repo,
so Kaggle notebooks can snapshot_download it the same way they already do for the
Neurite-OASIS dataset (Fatema142/neurite-oasis-100-rq-analysis).

Requires an active `huggingface-cli login` (or HF_TOKEN env var) with write access.

Usage:
    python scripts/push_totalsegmentator_to_hf.py [--repo-id Fatema142/totalsegmentator-50-rq-analysis] [--private]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "totalsegmentator_50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Fatema142/totalsegmentator-50-rq-analysis")
    parser.add_argument("--private", action="store_true", help="create/keep the repo private (default: public)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"{DATA_DIR} does not exist -- run scripts/stage_totalsegmentator_50.py first")

    api = HfApi()
    print(f"[plan] creating (if needed) dataset repo {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"[plan] uploading {DATA_DIR} -> {args.repo_id}")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
    )
    print(f"[done] pushed {DATA_DIR} to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
