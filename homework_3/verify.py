"""Independently check the submitted data, experiment records, and Poisson calculus.

This reads saved predictions rather than trusting metrics reported by the fitting
scripts. It also checks split membership, source labels, grid coverage, selection,
training-only scaling, and repeat-split aggregation. No model is tuned here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln

ROOT = Path(__file__).resolve().parent
COUNTS = (1, 10, 100, 300, 1000)


class Checks:
    def __init__(self):
        self.passed = []

    def require(self, condition, description):
        if not bool(condition):
            raise AssertionError(description)
        self.passed.append(description)

    def close(self, actual, expected, description, *, atol=1e-11, rtol=1e-10):
        self.require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), description)


def read_json(path):
    return json.loads(Path(path).read_text())


def check_calculus(c):
    """Central differences of full NLL, including the constant log-factorial."""
    first_errors, second_errors = [], []
    for y, f in itertools.product((0, 1, 5, 20), (-4., -1., 0., .7, 3.)):
        def nll(z):
            return np.exp(z) - y*z + gammaln(y+1)
        step = 1e-5
        numeric_first = (nll(f+step)-nll(f-step))/(2*step)
        expected_first = np.exp(f)-y
        c.close(numeric_first, expected_first, f'Poisson gradient y={y}, f={f}', atol=2e-9, rtol=2e-8)
        step = 1e-3
        numeric_second = (nll(f+step)-2*nll(f)+nll(f-step))/(step*step)
        expected_second = np.exp(f)
        c.close(numeric_second, expected_second, f'Poisson curvature y={y}, f={f}', atol=1e-6, rtol=2e-6)
        c.require(expected_second > 0, f'Positive Poisson curvature y={y}, f={f}')
        first_errors.append(abs(numeric_first-expected_first))
        second_errors.append(abs(numeric_second-expected_second))
    return {'points':20, 'max_gradient_absolute_error':max(first_errors),
            'max_hessian_absolute_error':max(second_errors)}


def check_sources(c, data_dir, manifest_path):
    manifest = read_json(manifest_path)
    c.require(isinstance(manifest.get('files'), dict), 'Source manifest has a files mapping')
    required = {'winequality-white.csv', 'dset0-1.csv', 'dset1-1.csv'}
    c.require(required.issubset(manifest['files']), 'All three experimental datasets have checksums')
    for name, expected in manifest['files'].items():
        path = data_dir/name
        c.require(path.is_file(), f'Source exists: {name}')
        c.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'SHA256 matches: {name}')
    return manifest


def check_wine(c, data_dir, results_dir):
    source = pd.read_csv(data_dir/'winequality-white.csv', sep=';')
    x, y = source.iloc[:, :-1].to_numpy(), source.iloc[:, -1].to_numpy()
    c.require(source.shape == (4898, 12) and source.columns[-1] == 'quality', 'Wine shape and target column')
    c.require(np.isfinite(x).all() and np.isfinite(y).all(), 'Wine contains finite values')
    summary = read_json(results_dir/'q2_summary.json')
    protocol = read_json(results_dir/'q2_protocol.json')
    selected = read_json(results_dir/'q2_selection.json')
    c.require(summary['protocol'] == protocol, 'Q2 summary and protocol agree')
    c.require(summary['selected'] == selected, 'Q2 summary and selections agree')
    c.require(protocol['sha256'] == hashlib.sha256((data_dir/'winequality-white.csv').read_bytes()).hexdigest(), 'Q2 run used the supplied wine bytes')
    splits = pd.read_csv(results_dir/'q2_splits.csv')
    c.require(len(splits) == len(y) and splits.row_id.is_unique, 'Q2 each source row occurs in exactly one split')
    c.require(set(splits.row_id) == set(range(len(y))), 'Q2 split rows cover the whole source')
    expected_sizes = {'train':3428, 'validation':490, 'test':980}
    c.require(splits.groupby('split').size().to_dict() == expected_sizes, 'Q2 70/10/20 split sizes')
    c.require(protocol['split_sizes'] == expected_sizes, 'Q2 protocol records actual split sizes')
    ids = {split:splits.loc[splits.split == split, 'row_id'].to_numpy(dtype=int) for split in expected_sizes}
    scale = read_json(results_dir/'q2_scaler.json')
    c.require(scale['n_samples_seen'] == len(ids['train']), 'KNN scaler fitted on training count')
    c.close(scale['mean'], x[ids['train']].mean(axis=0), 'KNN scaler means equal training-only means')
    expected_scale = x[ids['train']].std(axis=0, ddof=0)
    expected_scale[expected_scale == 0] = 1
    c.close(scale['scale'], expected_scale, 'KNN scaler scales equal training-only standard deviations')

    grid = pd.read_csv(results_dir/'q2_grid_scores.csv', float_precision='round_trip')
    c.require(set(grid.family) == {'knn','random_forest'}, 'Q2 contains exactly the two selected model families')
    c.require(len(grid) == 1040, 'Q2 contains all 1040 configurations')
    for family, expected_count in (('random_forest', 540), ('knn', 500)):
        rows = grid.loc[grid.family == family].sort_values('candidate_id')
        c.require(len(rows) == expected_count > len(ids['validation']), f'{family}: grid larger than validation set')
        c.require(rows.candidate_id.tolist() == list(range(expected_count)), f'{family}: candidate IDs are complete and unique')
        options = protocol['grids'][family]
        keys = sorted(options)
        expected_params = [dict(zip(keys, choices)) for choices in itertools.product(*(options[key] for key in keys))]
        actual_params = [json.loads(value) for value in rows.parameters]
        c.require(actual_params == expected_params, f'{family}: every planned configuration was evaluated in order')
        canonical = [json.dumps(p, sort_keys=True) for p in actual_params]
        c.require(len(set(canonical)) == expected_count, f'{family}: configurations are distinct')
        c.require(np.isfinite(rows.validation_mse).all() and (rows.validation_mse >= 0).all(), f'{family}: all validation losses finite and nonnegative')
        best = rows.sort_values(['validation_mse','candidate_id'], kind='stable').iloc[0]
        c.require(selected[family]['candidate_id'] == int(best.candidate_id), f'{family}: selected the minimum validation MSE with stated tie rule')
        c.require(selected[family]['parameters'] == json.loads(best.parameters), f'{family}: selected parameters match score table')
        c.require(selected[family]['configurations'] == expected_count, f'{family}: selected record counts the full grid')
        c.close(selected[family]['validation_mse'], best.validation_mse, f'{family}: saved selected MSE matches grid')
    c.require(protocol['refit_on_train_plus_validation'] is False, 'Q2 retained training-only fits for validation/test comparison')

    predictions = pd.read_csv(results_dir/'q2_predictions.csv', float_precision='round_trip')
    c.require(len(predictions) == 3*len(y), 'Q2 predictions cover three models on every row')
    c.require(set(predictions.family) == {'random_forest','knn','training_mean'}, 'Q2 prediction family names')
    for family, recorded in summary['metrics'].items():
        for split, index in ids.items():
            rows = predictions.loc[(predictions.family == family)&(predictions.split == split)]
            c.require(rows.row_id.is_unique and set(rows.row_id) == set(index), f'{family}/{split}: predictions match split membership')
            row_ids = rows.row_id.to_numpy(dtype=int)
            c.close(rows.y, y[row_ids], f'{family}/{split}: targets equal original source labels', atol=0, rtol=0)
            prediction = rows.prediction.to_numpy()
            c.require(np.isfinite(prediction).all(), f'{family}/{split}: finite predictions')
            c.require(((prediction >= y[ids['train']].min()-1e-10)&(prediction <= y[ids['train']].max()+1e-10)).all(), f'{family}/{split}: predictions remain inside training-label range')
            errors = prediction-y[row_ids]
            mse = np.mean(errors**2)
            c.require(recorded[split]['n'] == len(index), f'{family}/{split}: recorded count')
            c.close(recorded[split]['mse'], mse, f'{family}/{split}: MSE independently recomputed')
            c.close(recorded[split]['rmse'], np.sqrt(mse), f'{family}/{split}: RMSE independently recomputed')
            c.close(recorded[split]['mae'], np.mean(np.abs(errors)), f'{family}/{split}: MAE independently recomputed')
            if family == 'training_mean':
                c.close(prediction, np.full(len(index), y[ids['train']].mean()), f'Baseline/{split}: uses training mean only')
        if family != 'training_mean':
            c.close(recorded['validation']['mse'], selected[family]['validation_mse'], f'{family}: refitted selected model reproduces selected validation MSE')
    tuples = [tuple(row) for row in source.to_numpy()]
    training_records = {tuples[i] for i in ids['train']}
    expected_duplicates = {'duplicates_beyond_first':int(source.duplicated().sum()),
        'validation_rows_identical_to_any_training_row':sum(tuples[i] in training_records for i in ids['validation']),
        'test_rows_identical_to_any_training_row':sum(tuples[i] in training_records for i in ids['test'])}
    c.require(summary['duplicates'] == expected_duplicates, 'Q2 duplicate-record caveat matches actual split')
    c.require(summary['observed_quality_range'] == [int(y.min()),int(y.max())], 'Q2 observed target range')
    return {'grid_counts':{'random_forest':540,'knn':500}, 'split_sizes':expected_sizes,
            'prediction_rows':len(predictions), 'duplicates':expected_duplicates}


def check_boosting(c, data_dir, results_dir):
    summary = read_json(results_dir/'q3_summary.json')
    primary = pd.read_csv(results_dir/'q3_primary_metrics.csv', float_precision='round_trip')
    curves = pd.read_csv(results_dir/'q3_learning_curves.csv', float_precision='round_trip')
    predictions = pd.read_csv(results_dir/'q3_primary_predictions.csv', float_precision='round_trip')
    assignments = pd.read_csv(results_dir/'q3_split_assignments.csv')
    repeats = pd.read_csv(results_dir/'q3_repeated_split_metrics.csv', float_precision='round_trip')
    aggregate = pd.read_csv(results_dir/'q3_repeated_split_summary.csv', float_precision='round_trip')
    c.require(summary['primary_counts'] == list(COUNTS), 'Q3 all five assigned tree budgets recorded')
    c.require(len(primary) == 10 and len(curves) == 2000 and len(predictions) == 5000, 'Q3 complete primary metric, curve, and prediction records')
    c.require(summary['model_parameters']['n_iter_no_change'] is None, 'Q3 early stopping disabled')
    c.require(summary['robustness_split_seeds'] == list(range(20)), 'Q3 all 20 prespecified sensitivity seeds')
    c.require(len(repeats) == 200 and len(aggregate) == 10, 'Q3 repeated-split records complete')
    json_primary = {(r['dataset'],r['trees']):r for r in summary['primary_metrics']}
    json_aggregate = {(r['dataset'],r['trees']):r for r in summary['repeated_split_summary']}
    c.require(len(json_primary) == 10 and len(json_aggregate) == 10, 'Q3 JSON summary includes every dataset/budget')
    for dataset in ('dset0','dset1'):
        original = np.loadtxt(data_dir/f'{dataset}-1.csv', delimiter=',')
        c.require(original.shape == (500,6), f'{dataset}: headerless source retains all 500 rows and five inputs')
        y = original[:,0].astype(int)
        c.require(set(y) == {0,1}, f'{dataset}: original target is binary')
        info = summary['datasets'][dataset]
        c.require(info['sha256'] == hashlib.sha256((data_dir/f'{dataset}-1.csv').read_bytes()).hexdigest(), f'{dataset}: run source hash')
        split = assignments.loc[assignments.dataset == dataset]
        c.require(split.csv_row_1based.is_unique and set(split.csv_row_1based) == set(range(1,501)), f'{dataset}: disjoint splits cover every source row')
        c.require(split.groupby('split').size().to_dict() == {'test':150,'train':350}, f'{dataset}: 350/150 split counts')
        c.close(split.y, y[split.csv_row_1based.to_numpy(dtype=int)-1], f'{dataset}: split labels match source', atol=0, rtol=0)
        curve = curves.loc[curves.dataset == dataset]
        c.require(curve.trees.tolist() == list(range(1,1001)), f'{dataset}: every tree stage is present')
        rows_primary = primary.loc[primary.dataset == dataset]
        c.require(rows_primary.trees.tolist() == list(COUNTS), f'{dataset}: all requested budgets in order')
        for budget in COUNTS:
            metric = rows_primary.loc[rows_primary.trees == budget].iloc[0]
            stage = curve.loc[curve.trees == budget].iloc[0]
            for field in metric.index:
                if field != 'dataset':
                    c.close(metric[field], stage[field], f'{dataset}/{budget}: {field} agrees with full curve')
                    c.close(metric[field], json_primary[(dataset,budget)][field], f'{dataset}/{budget}: {field} agrees with JSON summary')
            checks = info['independent_fit_max_abs_score_difference'][str(budget)]
            for split_name in ('train','test'):
                c.require(0 <= checks[split_name] <= 1e-12, f'{dataset}/{budget}/{split_name}: independent fit matches staged scores')
                rows = predictions.loc[(predictions.dataset == dataset)&(predictions.trees == budget)&(predictions.split == split_name)]
                ids = split.loc[split.split == split_name,'csv_row_1based'].to_numpy(dtype=int)
                c.require(rows.csv_row_1based.is_unique and set(rows.csv_row_1based) == set(ids), f'{dataset}/{budget}/{split_name}: prediction membership')
                labels = y[rows.csv_row_1based.to_numpy(dtype=int)-1]
                scores = rows.score.to_numpy()
                c.close(rows.y, labels, f'{dataset}/{budget}/{split_name}: prediction labels', atol=0, rtol=0)
                c.require(np.isfinite(scores).all(), f'{dataset}/{budget}/{split_name}: finite scores')
                c.close(rows.probability_1, expit(scores), f'{dataset}/{budget}/{split_name}: sigmoid probabilities')
                c.close(rows.prediction, (scores > 0).astype(int), f'{dataset}/{budget}/{split_name}: classification threshold and tie rule', atol=0, rtol=0)
                wrong = int(np.sum(rows.prediction.to_numpy() != labels))
                loss = np.mean(np.logaddexp(0,scores)-labels*scores)
                for field, expected in (('n',len(ids)),('wrong',wrong),('error',wrong/len(ids)),('log_loss',loss)):
                    c.close(metric[f'{split_name}_{field}'], expected, f'{dataset}/{budget}/{split_name}: {field} independently recomputed')
            repeat_rows = repeats.loc[(repeats.dataset == dataset)&(repeats.trees == budget)]
            c.require(sorted(repeat_rows.split_seed.tolist()) == list(range(20)), f'{dataset}/{budget}: each prespecified split represented exactly once')
            agg = aggregate.loc[(aggregate.dataset == dataset)&(aggregate.trees == budget)]
            c.require(len(agg) == 1 and int(agg.iloc[0]['splits']) == 20, f'{dataset}/{budget}: aggregate split count')
            agg = agg.iloc[0]
            for field in ('train_error','test_error','train_log_loss','test_log_loss'):
                values = repeat_rows[field].to_numpy()
                c.require(np.isfinite(values).all(), f'{dataset}/{budget}/{field}: finite repeated-split metrics')
                for suffix, expected in (('mean',values.mean()),('sd',values.std(ddof=1))):
                    name = f'{field}_{suffix}'
                    c.close(agg[name], expected, f'{dataset}/{budget}: sensitivity {name} independently recomputed')
                    c.close(json_aggregate[(dataset,budget)][name], expected, f'{dataset}/{budget}: JSON sensitivity {name}')
    return {'datasets':2,'budgets':list(COUNTS),'primary_prediction_rows':len(predictions),'sensitivity_splits_per_dataset':20}


def run(data_dir=ROOT/'data', results_dir=ROOT/'results', *, manifest_path=ROOT/'data_manifest.json'):
    data_dir, results_dir, manifest_path = Path(data_dir), Path(results_dir), Path(manifest_path)
    c = Checks()
    report = {'status':'running','python':platform.python_version()}
    try:
        report['calculus'] = check_calculus(c)
        check_sources(c, data_dir, manifest_path)
        report['q2'] = check_wine(c, data_dir, results_dir)
        report['q3'] = check_boosting(c, data_dir, results_dir)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['checks_passed'] = len(c.passed)
        report['checks'] = c.passed
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir/'checks_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Verification passed: {len(c.passed)} checks (source hashes, calculus, splits, grids, predictions, metrics, aggregation).', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--manifest',type=Path,default=ROOT/'data_manifest.json')
    args = parser.parse_args()
    run(args.data_dir,args.results_dir,manifest_path=args.manifest)
