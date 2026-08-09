#!/usr/bin/env python
"""Pre-download the pipeline models into HF_HOME. Run on the LOGIN node (has internet) BEFORE
submitting GPU jobs, so compute nodes can run with HF_HUB_OFFLINE=1 even if they're offline.

Only the models the REMAINING phases need: base (extract/LoRA/gen), text monitor (04/09),
faithfulness judge (05). The Phase-2 rewriter is skipped — Phase 2 is already done.

  COTMON_FS=~/scratch python scripts/prestage_models.py         # ~50 GB into $COTMON_FS/hf
  python scripts/prestage_models.py --all                        # also fetch the rewriter (~+28 GB)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also prestage the Phase-2 rewriter")
    args = ap.parse_args()

    fs = os.environ.get("COTMON_FS", os.path.expanduser("~/scratch"))
    os.environ.setdefault("HF_HOME", f"{fs}/hf")
    print(f"[prestage] HF_HOME={os.environ['HF_HOME']}")

    models = [config.BASE_MODEL, config.TEXT_MONITOR_MODEL, config.FAITHFULNESS_JUDGE_MODEL]
    if args.all:
        models.append(config.REWRITER_MODEL)

    from huggingface_hub import snapshot_download
    for m in dict.fromkeys(models):  # dedupe, keep order
        print(f"[prestage] downloading {m} ...", flush=True)
        path = snapshot_download(repo_id=m)
        print(f"[prestage]   -> {path}", flush=True)
    print("[prestage] done. On compute nodes set HF_HUB_OFFLINE=1 (gpu.sbatch does this by default).")


if __name__ == "__main__":
    main()
