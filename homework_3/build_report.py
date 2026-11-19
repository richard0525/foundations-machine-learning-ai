"""Create LaTeX tables from saved results and compile the submitted PDF (TeX required)."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent

def generate_values(root=ROOT):
    q2=json.loads((root/'results/q2_summary.json').read_text())
    q3=json.loads((root/'results/q3_summary.json').read_text())
    lines=[]
    def macro(name,value): lines.append('\\newcommand{\\'+name+'}{'+value+'}')
    p=q2['selected']['random_forest']['parameters']
    depth='unrestricted depth' if p['max_depth'] is None else f"maximum depth {p['max_depth']}"
    macro('RFChoice',f"RF selects {p['n_estimators']} trees, {depth}, minimum leaf size {p['min_samples_leaf']}, feature fraction {p['max_features']}, and bootstrap fraction {p['max_samples']}.")
    p=q2['selected']['knn']['parameters']
    weights='inverse-distance weights' if p['weights']=='distance' else 'uniform weights'
    macro('KNNChoice',f"{p['n_neighbors']} neighbors, {weights}, and {'Manhattan' if p['p']==1 else 'Euclidean'} distance")
    table=[]
    for f,label in [('random_forest','Random forest'),('knn','K-nearest neighbors'),('training_mean','Training-mean baseline')]:
        m=q2['metrics'][f];diff=m['test']['mse']-m['validation']['mse']
        table.append(f"{label} & {m['train']['mse']:.4f} & {m['validation']['mse']:.4f} & {m['test']['mse']:.4f} & {diff:+.4f} \\\\")
    macro('WineRows','\n'.join(table))
    comparisons=[]
    for f,label in [('random_forest','RF'),('knn','KNN')]:
        m=q2['metrics'][f];d=m['test']['mse']-m['validation']['mse']
        comparisons.append(f"{label}'s test MSE is {abs(d):.4f} {'higher' if d>=0 else 'lower'} than its selected validation MSE")
    macro('WineInterpretation','; '.join(comparisons)+'. Validation and test estimate the risk of the same fitted predictor, but on different samples. Validation also influenced which predictor was selected. Their difference reflects both selection and sampling variability; its sign on one split is not a proof or disproof of the expected optimism in part (b). Both selected models improve on the constant baseline on the test set.')
    for name,key in [('WineDuplicates','duplicates_beyond_first'),('WineValRepeats','validation_rows_identical_to_any_training_row'),('WineTestRepeats','test_rows_identical_to_any_training_row')]:
        macro(name,str(q2['duplicates'][key]))
    rows=[]
    for r in q3['primary_metrics']:
        if r['dataset']=='dset1' and r['trees']==1:rows.append('\\midrule')
        rows.append(f"{r['dataset']} & {r['trees']:,} & {100*r['train_error']:.2f} & {100*r['test_error']:.2f} & {r['train_log_loss']:.5f} & {r['test_log_loss']:.4f} \\\\")
    macro('BoostRows','\n'.join(rows))
    rows=[]
    for ds in ['dset0','dset1']:
        cells=[]
        for n in [100,300,1000]:
            r=next(r for r in q3['repeated_split_summary'] if r['dataset']==ds and r['trees']==n)
            cells.append(f"${100*r['test_error_mean']:.2f}\\pm{100*r['test_error_sd']:.2f}$")
        rows.append(ds+' & '+' & '.join(cells)+' \\\\')
    macro('SensitivityRows','\n'.join(rows))
    (root/'results/report_values.tex').write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values-only',action='store_true')
    a=parser.parse_args()
    generate_values()
    if a.values_only:return
    executable=shutil.which('pdflatex')
    if not executable:raise SystemExit('Tables generated. Install a TeX distribution with pdflatex to rebuild the PDF; the submitted PDF is included in results/.')
    for _ in range(2):
        subprocess.run([executable,'-interaction=nonstopmode','-halt-on-error','-output-directory=results','-jobname=HW3_Solutions','solution.tex'],cwd=ROOT,check=True)
    print(ROOT/'results/HW3_Solutions.pdf')

if __name__=='__main__':main()
