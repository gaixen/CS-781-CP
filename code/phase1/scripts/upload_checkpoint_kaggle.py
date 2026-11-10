"""CLI: publish a trained ``model-final`` folder as a (private) Kaggle dataset so Phase 2/3/4 can load it.

Example:
    python scripts/upload_checkpoint_kaggle.py --arch modern_decoder --username <you> --slug cs781-phase1-checkpoints-modern-decoder
Requires KAGGLE_KEY (env var or Kaggle secret) and the ``kaggle`` package.
"""
import argparse
import json
import os

import _bootstrap  # noqa: F401


def main():
    """Create the Kaggle dataset (or push a new version if it exists).

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True); ap.add_argument("--username", required=True); ap.add_argument("--slug", required=True)
    ap.add_argument("--title", default="CS781 Phase1 Checkpoints"); ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    from lm.config import build_config
    cfg = build_config(a.arch); os.environ["KAGGLE_USERNAME"] = a.username
    if "KAGGLE_KEY" not in os.environ:
        from kaggle_secrets import UserSecretsClient; os.environ["KAGGLE_KEY"] = UserSecretsClient().get_secret("KAGGLE_KEY")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate(); src = cfg["model_output_dir"]; did = f"{a.username}/{a.slug}"
    json.dump({"title": a.title, "id": did, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(src, "dataset-metadata.json"), "w"), indent=2)
    if did in [d.ref for d in api.dataset_list(search=a.slug, user=a.username)]:
        api.dataset_create_version(folder=src, version_notes=f"Checkpoints for {a.arch}", dir_mode="zip")
    else:
        api.dataset_create_new(folder=src, dir_mode="zip", public=a.public)
    print(f"Done: https://www.kaggle.com/datasets/{did}")


if __name__ == "__main__":
    main()
