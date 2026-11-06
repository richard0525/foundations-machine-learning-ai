"""Q2: raw-target means, embeddings and their combination for heavy-tailed retail counts."""
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from common import (setup, dump, savefig, split_indices, save_split, tensor,
                    fit_network, batched_predict, save_csv_gz, plt)
from encoding import CrossFitTargetEncoder, CategoryVocabulary
from models import RetailMLP


def metrics(y, predicted_log):
    y = np.asarray(y, float)
    pred = np.expm1(np.asarray(predicted_log,float))
    assert np.isfinite(pred).all() and (pred>=0).all()
    residual = pred-y
    return {"log1p_mse":float(np.mean((np.log1p(y)-predicted_log)**2)),
            "rmsle":float(np.sqrt(np.mean((np.log1p(y)-predicted_log)**2))),
            "mae":float(np.mean(abs(residual))),"rmse":float(np.sqrt(np.mean(residual**2))),
            "median_absolute_error":float(np.median(abs(residual)))}


def run(data, out, config):
    start=time.monotonic()
    seed,c=config["seed"],config["retail"]
    setup(seed,config["threads"])
    frame=pd.read_csv(data/"retail.csv",dtype={"StockCode":"string","CustomerID":"string"})
    names=["StockCode","CustomerID"]
    X=frame[names]
    y=frame.Quantity.to_numpy(float)
    assert (y>0).all() and np.equal(y,np.floor(y)).all() and not frame.isna().any().any()
    train,val,test=split_indices(len(y),seed,c["train_fraction"],c["validation_fraction"])
    save_split(out,"q2",train,val,test)
    print(f"Q2 retail: {len(train)} train, {len(val)} validation, {len(test)} test",flush=True)
    enc=CrossFitTargetEncoder(c["encoding_smoothing"],c["encoding_folds"],seed)
    raw_oof=enc.fit_transform(X.iloc[train],y[train])
    scaler=StandardScaler().fit(np.log1p(raw_oof))
    te={"train":scaler.transform(np.log1p(raw_oof)),
        "validation":scaler.transform(np.log1p(enc.transform(X.iloc[val]))),
        "test":scaler.transform(np.log1p(enc.transform(X.iloc[test])))}
    vocab=CategoryVocabulary().fit(X.iloc[train])
    ids={"train":train,"validation":val,"test":test}
    cats={key:vocab.transform(X.iloc[ix]) for key,ix in ids.items()}
    arrays={key:[tensor(cats[key],integer=True),tensor(te[key])] for key in ids}
    targets={key:tensor(np.log1p(y[ix])) for key,ix in ids.items()}
    np.savez_compressed(out/"q2_encoding_fold_ids.npz",fold_ids=enc.fold_ids_)
    dump(out/"q2_preprocessing.json",{"category_vocabulary":vocab.maps_,
         "target_encoding_full_training_maps":{name:{str(k):float(v) for k,v in mapping.items()} for name,mapping in enc.maps_.items()},
         "raw_target_global_mean_train":enc.global_mean_,"encoding_smoothing":enc.smoothing,
         "log_encoding_scaler_mean":scaler.mean_.tolist(),"log_encoding_scaler_scale":scaler.scale_.tolist()})
    rows=[]
    baseline=float(np.log1p(y[train]).mean())
    for split in ["validation","test"]:
        rows.append({"model":"Constant log-target mean","split":split,"selected_epoch":0,
                     **metrics(y[ids[split]],np.full(len(ids[split]),baseline))})
    predictions=pd.DataFrame({"row_id":test,"Quantity":y[test].astype(np.int64)})
    predictions["stock_seen_in_train"]=cats["test"][:,0]>0
    predictions["customer_seen_in_train"]=cats["test"][:,1]>0
    all_histories={}
    choices={}
    for mode,label in [("target","Target encoding"),("embedding","Embeddings"),("combined","Combined")]:
        setup(seed,config["threads"])
        model=RetailMLP(vocab.sizes_,mode,c["embedding_dim"],c["hidden"],c["dropout"])
        print("Retail model:",label,flush=True)
        history,epoch=fit_network(model,arrays["train"],targets["train"],arrays["validation"],targets["validation"],F.mse_loss,
            **{k:c[k] for k in ["epochs","patience","batch_size","learning_rate","weight_decay"]},seed=seed)
        all_histories[label]=pd.DataFrame(history)
        all_histories[label].to_csv(out/f"q2_{mode}_history.csv",index=False)
        choices[label]={"selected_epoch":epoch,"epochs_run":history[-1]["epoch"],
                        "parameters":sum(p.numel() for p in model.parameters())}
        torch.save(model.state_dict(),out/f"q2_{mode}_weights.pt")
        for split in ["validation","test"]:
            z=batched_predict(model,arrays[split]).astype(float)
            rows.append({"model":label,"split":split,"selected_epoch":epoch,**metrics(y[ids[split]],z)})
            if split=="test":
                predictions[mode+"_predicted_log1p"]=z
                predictions[mode+"_predicted_quantity"]=np.expm1(z)
    results=pd.DataFrame(rows)
    results.to_csv(out/"q2_metrics.csv",index=False)
    save_csv_gz(predictions,out/"q2_test_predictions.csv.gz")
    seen_both=predictions.stock_seen_in_train & predictions.customer_seen_in_train
    subgroups=[]
    for group,mask in [("both categories seen",seen_both),("at least one unseen",~seen_both)]:
        if not mask.any():continue
        for mode in ["target","embedding","combined"]:
            subgroups.append({"group":group,"model":mode,"n":int(mask.sum()),
                              **metrics(y[test][mask],predictions.loc[mask,mode+"_predicted_log1p"].to_numpy())})
    pd.DataFrame(subgroups).to_csv(out/"q2_test_subgroups.csv",index=False)
    cardinalities={name:int(X[name].nunique()) for name in names}
    train_cardinalities={name:int(X.iloc[train][name].nunique()) for name in names}
    unseen={split:{name:int((cats[split][:,j]==0).sum()) for j,name in enumerate(names)} for split in ["validation","test"]}
    summary={"n":len(y),"cardinalities_all":cardinalities,"cardinalities_train":train_cardinalities,
             "split_sizes":{key:len(ix) for key,ix in ids.items()},"unseen_category_rows":unseen,
             "quantity_quantiles_all":{str(q):float(np.quantile(y,q)) for q in [0,.5,.9,.99,.999,1]},
             "quantity_mean_all":float(y.mean()),"quantity_mean_train":float(y[train].mean()),
             "quantity_max_by_split":{key:float(y[ix].max()) for key,ix in ids.items()},
             "dense_one_hot_float32_bytes":len(y)*sum(cardinalities.values())*4,
             "sparse_one_hot_float32_csr_bytes_approx":2*len(y)*8+(len(y)+1)*4,
             "training_log_target_mean":baseline,"models":choices,"metrics":rows,
             "target_encoding":"5-fold OOF smoothed means of RAW Quantity; log1p followed by OOF-training feature scaling",
             "training_objective":"MSE of log1p(Quantity); nonnegative log-scale output via softplus",
             "test_subgroups":subgroups,"runtime_seconds":time.monotonic()-start}
    dump(out/"q2_summary.json",summary)
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),constrained_layout=True)
    axes[0].hist(y[train],bins=np.geomspace(1,max(y[train])+1,65),color="#176b89")
    axes[0].set(xscale="log",yscale="log",xlabel="Quantity (log scale)",ylabel="Training count (log)",title="Heavy-tailed retail quantities")
    for name,hist in all_histories.items():
        axes[1].plot(hist.epoch,hist.validation_loss,label=name)
    axes[1].set(xlabel="Epoch",ylabel="Validation log1p-MSE",title="Training-only encodings; validation stopping")
    axes[1].legend(fontsize=8)
    savefig(fig,out,"q2_tail_and_validation")
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,["log1p_mse","mae"],["Primary: log1p-MSE","Original units: MAE"]):
        subset=results.loc[results.split=="test"]
        ax.bar(range(len(subset)),subset[metric],color=["#b4bec4","#176b89","#4c956c","#a75631"])
        ax.set_xticks(range(len(subset)),["Constant","Target\nencoding","Embeddings","Combined"])
        ax.set(title=title,ylabel=metric)
    savefig(fig,out,"q2_test_comparison")
    print(results.to_string(index=False),flush=True)
    return summary
