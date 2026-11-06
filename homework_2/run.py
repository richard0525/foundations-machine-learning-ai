"""Reproduce HW2. Example: python homework_2/run.py --part all."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from threadpoolctl import threadpool_limits
from common import ROOT, dump, setup, verify_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "q1", "q2", "q3"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"config.json").read_text())
    manifest = verify_data(args.data_dir)
    setup(config["seed"], config["threads"])
    with threadpool_limits(limits=config["threads"]):
        if args.part in {"all", "q1"}:
            from q1_taxi import run
            run(args.data_dir, out, config)
        if args.part in {"all", "q2"}:
            from q2_retail import run
            run(args.data_dir, out, config)
        if args.part in {"all", "q3"}:
            from q3_features import run
            run(args.data_dir, out, config)
    source_names = ["run.py", "common.py", "models.py", "encoding.py", "dataloader.py",
                    "q1_taxi.py", "q2_retail.py", "q3_features.py", "config.json"]
    dump(out/f"metadata_{args.part}.json", {"part":args.part, "config":config,
         "python":platform.python_version(), "platform":platform.platform(), "device":"CPU",
         "versions":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","torch","matplotlib","pyarrow"]},
         "data":manifest, "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}})
    print("Completed", args.part, "in", out, flush=True)


if __name__ == "__main__":
    main()
