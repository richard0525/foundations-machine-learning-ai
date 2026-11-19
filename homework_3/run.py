"""Run every HW3 experiment using the included checksum-verified data."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

ROOT = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['all','q2','q3'],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--jobs',type=int,default=4,help='Concurrent Q2 configurations; each estimator uses one worker.')
    args=parser.parse_args()
    if args.jobs < 1: parser.error('--jobs must be positive')
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    for filename,expected in manifest['files'].items():
        actual=hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'Input checksum mismatch: {filename}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if args.part in ['all','q2']:
        from q2_wine import run
        run(ROOT/'data',args.output_dir,jobs=args.jobs)
    if args.part in ['all','q3']:
        from q3_boosting import run
        run(ROOT/'data',args.output_dir)
    metadata={'python':platform.python_version(),'platform':platform.platform(),
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','threadpoolctl']},
              'part':args.part,'jobs':args.jobs,'elapsed_seconds':time.perf_counter()-start,
              'input_sha256':manifest['files']}
    (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Completed {args.part} in {metadata["elapsed_seconds"]:.1f} seconds. Results: {args.output_dir}')

if __name__=='__main__': main()
