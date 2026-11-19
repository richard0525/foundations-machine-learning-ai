"""HW3 Q2: prespecified RF/KNN grids, training-only fits, one test evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from threadpoolctl import threadpool_limits

RF_GRID = {'n_estimators':[50,100,200], 'max_depth':[None,3,5,10,20],
           'min_samples_leaf':[1,2,4,8], 'max_features':[.33,.66,1.0],
           'max_samples':[.6,.8,1.0]}
KNN_GRID = {'n_neighbors':list(range(1,126)), 'weights':['uniform','distance'], 'p':[1,2]}
SEED = 2026

def estimator(family, params):
    if family == 'random_forest':
        return RandomForestRegressor(**params, criterion='squared_error', bootstrap=True,
                                     random_state=SEED, n_jobs=1)
    return KNeighborsRegressor(**params, metric='minkowski', algorithm='brute', n_jobs=1)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def run(data_dir: Path, output_dir: Path, *, jobs=4):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(jobs))
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'figures').mkdir(exist_ok=True)
    source = data_dir/'winequality-white.csv'
    df = pd.read_csv(source, sep=';')
    assert df.shape == (4898,12) and df.columns[-1] == 'quality'
    x, y = df.iloc[:,:-1].to_numpy(), df.iloc[:,-1].to_numpy(dtype=float)
    assert np.isfinite(x).all() and np.isfinite(y).all() and ((1 <= y)&(y <= 10)).all()
    trainval, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED)
    train, val = train_test_split(trainval, test_size=.125, random_state=SEED+1)
    splits = {'train':train, 'validation':val, 'test':test}
    pd.DataFrame([{'row_id':int(i),'split':s} for s, ids in splits.items() for i in ids]).to_csv(output_dir/'q2_splits.csv',index=False)
    scaler = StandardScaler().fit(x[train])
    scaled = scaler.transform(x)
    write_json(output_dir/'q2_scaler.json', {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'n_samples_seen':int(scaler.n_samples_seen_)})
    grids = {'random_forest':list(ParameterGrid(RF_GRID)), 'knn':list(ParameterGrid(KNN_GRID))}
    assert all(len(grid) > len(val) for grid in grids.values())
    frozen = {'split_seeds':[SEED,SEED+1], 'model_seed':SEED, 'split_sizes':{s:len(ids) for s,ids in splits.items()},
              'grids':{'random_forest':RF_GRID,'knn':KNN_GRID}, 'selection':'minimum validation MSE; first candidate wins exact ties',
              'refit_on_train_plus_validation':False, 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    write_json(output_dir/'q2_protocol.json',frozen)
    allrows, winners, models = [], {}, {}
    with threadpool_limits(limits=1):
        for family, grid in grids.items():
            features = scaled if family == 'knn' else x
            def evaluate_candidate(index, params):
                model = estimator(family,params).fit(features[train], y[train])
                return {'family':family,'candidate_id':index,'parameters':json.dumps(params,sort_keys=True),
                        'validation_mse':float(mean_squared_error(y[val],model.predict(features[val])))}
            print(f'{family}: fitting {len(grid)} configurations...',flush=True)
            rows = Parallel(n_jobs=jobs,prefer='threads')(delayed(evaluate_candidate)(i,p) for i,p in enumerate(grid))
            winner = min(rows,key=lambda r:(r['validation_mse'],r['candidate_id']))
            params = grid[winner['candidate_id']]
            winners[family] = {'candidate_id':winner['candidate_id'],'parameters':params,'validation_mse':winner['validation_mse'],'configurations':len(grid)}
            models[family] = estimator(family,params).fit(features[train],y[train])
            allrows.extend(rows)
            print(f'{family}: selected {params}, validation MSE {winner["validation_mse"]:.6f}',flush=True)
        # Persist both selections before reading any test predictions or test losses.
        write_json(output_dir/'q2_selection.json',winners)
        pd.DataFrame(allrows).to_csv(output_dir/'q2_grid_scores.csv',index=False)
        metrics, predictions = {}, []
        for family in [*models,'training_mean']:
            metrics[family] = {}
            features = scaled if family == 'knn' else x
            for split, ids in splits.items():
                pred = np.full(len(ids),y[train].mean()) if family == 'training_mean' else models[family].predict(features[ids])
                mse = float(mean_squared_error(y[ids],pred))
                metrics[family][split] = {'n':len(ids),'mse':mse,'rmse':float(np.sqrt(mse)), 'mae':float(mean_absolute_error(y[ids],pred))}
                predictions.extend({'family':family,'split':split,'row_id':int(i),'y':float(a),'prediction':float(b)} for i,a,b in zip(ids,y[ids],pred))
    assert all(np.isclose(metrics[f]['validation']['mse'],winners[f]['validation_mse'],atol=1e-12) for f in winners)
    pd.DataFrame(predictions).to_csv(output_dir/'q2_predictions.csv',index=False)
    # Exact repeats are preserved, as requested by the random-row split protocol.
    row_tuples = [tuple(row) for row in df.to_numpy()]
    train_records = {row_tuples[i] for i in train}
    duplicate_info = {'duplicates_beyond_first':int(df.duplicated().sum()),
                      'validation_rows_identical_to_any_training_row':sum(row_tuples[i] in train_records for i in val),
                      'test_rows_identical_to_any_training_row':sum(row_tuples[i] in train_records for i in test)}
    summary = {'protocol':frozen,'selected':winners,'metrics':metrics,'duplicates':duplicate_info,
               'observed_quality_range':[int(y.min()),int(y.max())], 'elapsed_seconds':time.perf_counter()-start}
    write_json(output_dir/'q2_summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    names={'random_forest':'Random forest','knn':'K-nearest neighbors','training_mean':'Training-mean baseline'}
    for family in winners:
        vals=np.array([r['validation_mse'] for r in allrows if r['family']==family])
        axes[0].plot(np.arange(1,len(vals)+1),np.minimum.accumulate(vals),label=names[family])
    axes[0].set(xlabel='Configurations evaluated (fixed grid order)',ylabel='Best validation MSE so far')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    position=np.arange(3)
    for j,(split,label,color) in enumerate([('validation','Validation','#1f4e79'),('test','Test','#c55a11')]):
        axes[1].bar(position+(j-.5)*.35,[metrics[f][split]['mse'] for f in metrics],width=.35,label=label,color=color)
    axes[1].set_xticks(position,['Random\nforest','K-nearest\nneighbors','Mean\nbaseline'])
    axes[1].set_ylabel('MSE');axes[1].legend(frameon=False);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(output_dir/'figures'/'q2_model_comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--jobs',type=int,default=4)
    a=parser.parse_args()
    run(a.data_dir,a.output_dir,jobs=a.jobs)
