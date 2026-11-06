"""Refresh report tables from saved experiments; optionally compile using LaTeX."""
import csv
import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

def read_json(name):
    return json.loads((OUT / name).read_text())

def rows(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))

def table(name, headers, data, align=None):
    align = align or 'l' + 'r'*(len(headers)-1)
    lines = [r'\begin{center}\small', r'\begin{tabular}{'+align+'}', r'\toprule',
             ' & '.join(headers)+r' \\', r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in data]
    lines += [r'\bottomrule',r'\end{tabular}\end{center}']
    (OUT/name).write_text('\n'.join(lines)+'\n')

def main():
    q1,q2,q3 = [read_json(f'q{i}_summary.json') for i in [1,2,3]]
    def short(s):
        return {'Boosted mean + constant SD':'Boosted mean, fixed SD',
                'Constant log-target mean':'Constant log-target'}.get(s,s)
    test1=[r for r in q1['metrics'] if r['split']=='test']
    table('table_q1_regression.tex',['Model','NLL','RMSE','MAE','90\\% cover','Width'],
          [[short(r['model']),f"{r['gaussian_nll']:.5f}",f"{r['rmse']:.4f}",f"{r['mae']:.4f}",
            f"{100*r['coverage_90']:.2f}\\%",f"{r['mean_interval_width_90']:.3f}"] for r in test1])
    table('table_q1_probability.tex',['Model','Brier score','Log-loss','Accuracy'],
          [[short(r['model']),f"{r['brier']:.6f}",f"{r['binary_log_loss']:.6f}",f"{100*r['accuracy']:.4f}\\%"]
           for r in [*test1,q1['binary_frequency_baseline']]])
    table('table_q2_metrics.tex',['Model','Val. log-MSE','Test log-MSE','RMSLE','MAE','RMSE'],
          [[short(r['model']),f"{next(v['log1p_mse'] for v in q2['metrics'] if v['model']==r['model'] and v['split']=='validation'):.6f}",
            f"{r['log1p_mse']:.6f}",f"{r['rmsle']:.5f}",f"{r['mae']:.4f}",f"{r['rmse']:.4f}"]
           for r in q2['metrics'] if r['split']=='test'])
    table('table_q2_epochs.tex',['Encoding','Input dimension','Parameters','Selected epoch','Epochs run'],
          [[name,{'Target encoding':2,'Embeddings':32,'Combined':34}[name],f"{spec['parameters']:,}",spec['selected_epoch'],spec['epochs_run']]
           for name,spec in q2['models'].items()])
    table('table_q2_subgroups.tex',['Test group','Rows','Target enc.','Embeddings','Combined'],
          [[group, f"{next(r['n'] for r in q2['test_subgroups'] if r['group']==group):,}"]+
           [f"{next(r['log1p_mse'] for r in q2['test_subgroups'] if r['group']==group and r['model']==mode):.5f}"
            for mode in ['target','embedding','combined']]
           for group in ['both categories seen','at least one unseen']])
    table('table_q3_rks.tex',['Total features $D$','Pairs $h$','Training MSE','Validation MSE'],
          [[int(r['D']),int(r['h']),f"{float(r['train_mse']):.8f}",f"{float(r['validation_mse']):.8f}"] for r in rows('q3_rks_sweep.csv')])
    table('table_q3_mlp.tex',['Hidden width','Selected epoch','Training MSE','Validation MSE'],
          [[int(r['hidden_width']),int(r['selected_epoch']),f"{float(r['train_mse']):.8f}",f"{float(r['validation_mse']):.8f}"] for r in rows('q3_mlp_sweep.csv')])
    table('table_q3_comparison.tex',['Refitted model','Training MSE','Validation MSE'],
          [[r['model'],f"{r['train_mse']:.8f}",f"{r['validation_mse']:.8f}"] for r in q3['comparison']])
    # Quantify, rather than speculate about, the retail model's extreme errors.
    with gzip.open(OUT/'q2_test_predictions.csv.gz','rt') as handle:
        pred=list(csv.DictReader(handle))
    extreme={}
    for mode in ['target','embedding','combined']:
        squared=[(float(r[mode+'_predicted_quantity'])-float(r['Quantity']))**2 for r in pred]
        k=max(range(len(squared)),key=squared.__getitem__)
        extreme[mode]={'row_id':int(pred[k]['row_id']),'actual':int(pred[k]['Quantity']),
                      'predicted':float(pred[k][mode+'_predicted_quantity']),
                      'fraction_total_squared_error':squared[k]/sum(squared)}
    table('table_q2_extremes.tex',['Model','Original row index','Actual','Prediction','Share of squared error'],
          [[name,extreme[mode]['row_id'],extreme[mode]['actual'],f"{extreme[mode]['predicted']:.2f}",
            f"{100*extreme[mode]['fraction_total_squared_error']:.2f}\\%"]
           for mode,name in [('target','Target encoding'),('embedding','Embeddings'),('combined','Combined')]])
    (OUT/'q2_extreme_error_audit.json').write_text(json.dumps(extreme,indent=2)+'\n')
    macros={'TaxiEpoch':q1['selected_epoch'],'TaxiEpochsRun':q1['epochs_run'],
            'TaxiValNLL':f"{next(r['gaussian_nll'] for r in q1['metrics'] if r['split']=='validation' and r['model']=='Gaussian MLP'):.6f}",
            'RKSD':q3['best_rks']['D'],'RKSPairs':q3['best_rks']['h'],
            'MLPWidth':q3['best_mlp']['hidden_width'],'MLPEpoch':q3['best_mlp']['selected_epoch'],
            'MLPImprovement':f"{100*(1-q3['best_mlp']['validation_mse']/q3['best_rks']['validation_mse']):.2f}",
            'RuntimeMinutes':f"{sum(q['runtime_seconds'] for q in [q1,q2,q3])/60:.1f}"}
    for label,q in [('Taxi',q1),('Retail',q2)]:
        for split,value in q['split_sizes'].items():
            macros[label+split.title()]=f'{value:,}'
    (OUT/'report_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
    executable=shutil.which('pdflatex')
    if not executable:
        print('Tables refreshed. PDF compilation skipped: pdflatex is not installed. The supplied PDF remains available.')
        return
    command=[executable,'-interaction=nonstopmode','-halt-on-error','-output-directory=results','solution.tex']
    for _ in range(2):
        result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:
            print(result.stdout)
            raise SystemExit(result.returncode)
    print('Updated',OUT/'solution.pdf')

if __name__=='__main__':
    main()
