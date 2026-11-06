"""Q1: heteroscedastic Gaussian regression and P(duration < 45 minutes)."""
import time
import numpy as np
import pandas as pd
import torch
from scipy import special, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from common import (setup, dump, savefig, split_indices, save_split, tensor,
                    fit_network, batched_predict, save_csv_gz, plt)
from models import GaussianMLP, gaussian_nll


def probability_metrics(y, probability, log_p=None, log_one_minus_p=None):
    event = np.asarray(y) < 45
    p = np.asarray(probability)
    if log_p is None:
        log_p, log_one_minus_p = np.log(p), np.log1p(-p)
    return {"brier":float(np.mean((p-event)**2)),
            "binary_log_loss":float(-np.mean(np.where(event, log_p, log_one_minus_p))),
            "accuracy":float(np.mean((p >= .5) == event))}


def gaussian_metrics(y, mu, sigma):
    residual = np.asarray(y, float)-np.asarray(mu, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), residual.shape)
    assert (sigma > 0).all() and np.isfinite(sigma).all()
    z = residual/sigma
    cutoff = (45-np.asarray(mu))/sigma
    p = special.ndtr(cutoff)
    return {"gaussian_nll":float(np.mean(.5*np.log(2*np.pi)+np.log(sigma)+.5*z*z)),
            "rmse":float(np.sqrt(np.mean(residual**2))), "mae":float(np.mean(abs(residual))),
            "coverage_90":float(np.mean(abs(z) <= stats.norm.ppf(.95))),
            "mean_interval_width_90":float(2*stats.norm.ppf(.95)*np.mean(sigma)),
            **probability_metrics(y, p, special.log_ndtr(cutoff), special.log_ndtr(-cutoff))}


def run(data, out, config):
    start = time.monotonic()
    seed, c = config["seed"], config["taxi"]
    setup(seed, config["threads"])
    frame = pd.read_parquet(data/"taxi_trips.parquet")
    features = [s for s in frame.columns if s != "trip_duration"]
    assert len(features)==7 and np.isfinite(frame.to_numpy()).all()
    x, y = frame[features].to_numpy(np.float32), frame.trip_duration.to_numpy(np.float64)
    train, val, test = split_indices(len(y), seed, c["train_fraction"], c["validation_fraction"])
    save_split(out, "q1", train, val, test)
    target_mean, target_scale = float(y[train].mean()), float(y[train].std())
    print(f"Q1 taxi: {len(train)} train, {len(val)} validation, {len(test)} test", flush=True)
    # The seven supplied features are already standardized. Only target scaling
    # is learned here, using training rows, and is inverted for every score.
    xt, yt = tensor(x[train]), tensor((y[train]-target_mean)/target_scale)
    xv, yv = tensor(x[val]), tensor((y[val]-target_mean)/target_scale)
    model = GaussianMLP(7, c["hidden"], c["sigma_floor_standardized"])
    history, epoch = fit_network(model, [xt], yt, [xv], yv, gaussian_nll,
        **{k:c[k] for k in ["epochs","patience","batch_size","learning_rate","weight_decay"]}, seed=seed)
    history = pd.DataFrame(history)
    history["train_nll_minutes"] = history.train_loss+np.log(target_scale)
    history["validation_nll_minutes"] = history.validation_loss+np.log(target_scale)
    history.to_csv(out/"q1_learning_curve.csv", index=False)
    torch.save(model.state_dict(), out/"q1_gaussian_mlp_weights.pt")

    linear = LinearRegression().fit(x[train], y[train])
    linear_sigma = float(np.sqrt(np.mean((y[train]-linear.predict(x[train]))**2)))
    boosted = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31, min_samples_leaf=50,
               learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=seed)
    boosted.fit(x[train], y[train])
    boosted_sigma = float(np.sqrt(np.mean((y[train]-boosted.predict(x[train]))**2)))
    metrics = []
    test_predictions = pd.DataFrame({"row_id":test,"trip_duration":y[test],"under_45":(y[test]<45).astype(int)})
    final_prediction = None
    for split, ids in [("validation", val), ("test", test)]:
        neural = batched_predict(model, [tensor(x[ids])])
        predictions = {
            "Constant Gaussian":(np.full(len(ids),target_mean),np.full(len(ids),target_scale)),
            "Linear Gaussian":(linear.predict(x[ids]),np.full(len(ids),linear_sigma)),
            "Boosted mean + constant SD":(boosted.predict(x[ids]),np.full(len(ids),boosted_sigma)),
            "Gaussian MLP":(target_mean+target_scale*neural[:,0].astype(float),target_scale*neural[:,1].astype(float))}
        for name,(mu,sigma) in predictions.items():
            metrics.append({"split":split,"model":name,"n":len(ids),**gaussian_metrics(y[ids],mu,sigma)})
            if split=="test":
                key={"Constant Gaussian":"constant","Linear Gaussian":"linear","Boosted mean + constant SD":"boosted","Gaussian MLP":"mlp"}[name]
                test_predictions[key+"_mu"] = mu
                test_predictions[key+"_sigma"] = sigma
                test_predictions[key+"_p_under_45"] = special.ndtr((45-mu)/sigma)
        if split=="test":
            final_prediction=predictions["Gaussian MLP"]
    metric_frame=pd.DataFrame(metrics)
    metric_frame.to_csv(out/"q1_metrics.csv",index=False)
    prevalence=float(np.mean(y[train]<45))
    binary_baseline={"model":"Training event frequency", "probability":prevalence,
                     **probability_metrics(y[test],np.full(len(test),prevalence))}
    save_csv_gz(test_predictions,out/"q1_test_predictions.csv.gz")

    mu,sigma=final_prediction
    event=y[test]<45
    probability=special.ndtr((45-mu)/sigma)
    late_probability=1-probability
    bins=pd.qcut(late_probability,10,duplicates="drop")
    reliability=(pd.DataFrame({"bin":bins,"predicted":late_probability,"observed":~event})
                 .groupby("bin",observed=True).agg(n=("observed","size"),predicted=("predicted","mean"),observed=("observed","mean")).reset_index())
    reliability["bin"]=reliability.bin.astype(str)
    reliability.to_csv(out/"q1_reliability.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),constrained_layout=True)
    axes[0].hist(y[train],bins=np.linspace(0,100,81),color="#176b89")
    axes[0].set(yscale="log",xlabel="Trip duration (minutes)",ylabel="Training count (log)",title="Training trip durations (display 0–100)")
    axes[1].plot(history.epoch,history.train_nll_minutes,label="Training")
    axes[1].plot(history.epoch,history.validation_nll_minutes,label="Validation")
    axes[1].axvline(epoch,color="#a75631",ls="--",label=f"Selected epoch {epoch}")
    axes[1].set(xlabel="Epoch",ylabel="Gaussian NLL (minutes scale)",title="Heteroscedastic MLP")
    axes[1].legend(fontsize=8)
    savefig(fig,out,"q1_data_and_training")
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),constrained_layout=True)
    z=(y[test]-mu)/sigma
    axes[0].hist(z,bins=np.linspace(-4,4,81),weights=np.ones(len(z))/len(z)/.1,alpha=.65,color="#176b89",label="Test standardized residuals")
    grid=np.linspace(-4,4,200)
    axes[0].plot(grid,stats.norm.pdf(grid),color="#a75631",label="Standard normal")
    axes[0].set(xlabel="(duration − predicted mean) / predicted SD",ylabel="Density",title="Distributional diagnostic")
    axes[0].legend(fontsize=8)
    axes[1].plot(reliability.predicted,reliability.observed,"o-",color="#176b89",label="MLP, quantile bins")
    high=max(reliability.predicted.max(),reliability.observed.max())*1.12
    axes[1].plot([0,high],[0,high],"--",color="#a75631",label="Calibrated")
    axes[1].set(xlabel="Mean predicted P(duration ≥ 45)",ylabel="Observed fraction ≥ 45",title="Tail-event reliability")
    axes[1].legend(fontsize=8)
    savefig(fig,out,"q1_distribution_and_calibration")
    summary={"n":len(y),"features":features,"split_sizes":{"train":len(train),"validation":len(val),"test":len(test)},
             "missing_values":int(frame.isna().sum().sum()),"exact_duplicate_rows":int(frame.duplicated().sum()),
             "target_mean_train":target_mean,"target_sd_train":target_scale,"selected_epoch":epoch,
             "epochs_run":int(history.epoch.iloc[-1]),"training_event_prevalence":prevalence,
             "test_under_45_count":int(event.sum()),"test_at_least_45_count":int((~event).sum()),
             "test_event_prevalence":float(event.mean()),"binary_frequency_baseline":binary_baseline,
             "metrics":metrics,"test_standardized_residual_mean":float(z.mean()),"test_standardized_residual_sd":float(z.std()),
             "sigma_test_quantiles":np.quantile(sigma,[0,.1,.5,.9,1]).tolist(),
             "runtime_seconds":time.monotonic()-start}
    dump(out/"q1_summary.json",summary)
    print(metric_frame.to_string(index=False),flush=True)
    print("Frequency baseline:",binary_baseline,flush=True)
    return summary
