"""Push a local model/adapter directory to the Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py <local_dir> <repo_id> [--private]
"""

import argparse

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir")
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.local_dir, repo_id=args.repo_id)
    print(f"Pushed {args.local_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
