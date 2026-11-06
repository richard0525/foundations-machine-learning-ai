"""Independent checks of the likelihood, ridge solver, and leakage boundaries."""
from pathlib import Path
import unittest
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import torch
from common import ROOT, setup, split_indices, verify_data, exact_row_groups
from encoding import CrossFitTargetEncoder, CategoryVocabulary
from models import RandomKitchenSinks, GaussianMLP, gaussian_nll
from q1_taxi import gaussian_metrics, gaussian_tail_probability
from dataloader import HW2Dataset


class HomeworkTests(unittest.TestCase):
    def setUp(self):
        setup(2026,1)

    def test_gaussian_likelihood_and_analytic_gradients(self):
        y=torch.tensor([1.,-2.,.7],dtype=torch.float64)
        pred=torch.tensor([[.2,.4],[-1.,2.],[3.,.9]],dtype=torch.float64,requires_grad=True)
        loss=gaussian_nll(pred,y)
        reference=-stats.norm.logpdf(y.numpy(),loc=pred[:,0].detach().numpy(),scale=pred[:,1].detach().numpy()).mean()
        self.assertAlmostEqual(float(loss.detach()),reference,places=13)
        loss.backward()
        mu,sigma=pred.detach().T
        expected=torch.stack([(mu-y)/sigma**2,1/sigma-(y-mu)**2/sigma**3],dim=1)/len(y)
        torch.testing.assert_close(pred.grad,expected)

    def test_target_scaling_preserves_gaussian_nll(self):
        y=torch.tensor([2.,5.,11.],dtype=torch.float64)
        original=torch.tensor([[3.,2.],[4.,3.],[8.,4.]],dtype=torch.float64)
        mean,scale=4.,2.5
        transformed=torch.stack([(original[:,0]-mean)/scale,original[:,1]/scale],dim=1)
        self.assertAlmostEqual(float(gaussian_nll(original,y)),float(gaussian_nll(transformed,(y-mean)/scale))+np.log(scale),places=13)

    def test_gaussian_mlp_positive_scale_and_finite_gradients(self):
        model=GaussianMLP(7)
        x=torch.randn(25,7)*20
        pred=model(x)
        self.assertTrue(torch.all(pred[:,1]>.001))
        gaussian_nll(pred,torch.randn(25)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_probability_scores_and_45_minute_boundary(self):
        y=np.array([44.,45.,46.]);mu=np.full(3,45.);sigma=np.ones(3)
        result=gaussian_metrics(y,mu,sigma)
        self.assertAlmostEqual(result['brier'],.25)
        self.assertAlmostEqual(result['binary_log_loss'],np.log(2))
        self.assertAlmostEqual(result['accuracy'],1/3)

    def test_gaussian_tail_probability_preserves_tiny_positive_values(self):
        z=np.array([10.,12.,20.])
        tail=gaussian_tail_probability(45.-z,np.ones_like(z))
        expected=np.array([.5*math.erfc(float(value)/math.sqrt(2)) for value in z])
        self.assertTrue((tail>0).all())
        np.testing.assert_allclose(tail,expected,rtol=1e-12,atol=0)
        self.assertEqual(float(gaussian_tail_probability(45.,1.)),.5)

    def test_rks_matches_independent_ridge_primal_and_dual(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(25,2));y=rng.normal(size=25);xe=rng.normal(size=(11,2))
        for D in [12,60]:
            model=RandomKitchenSinks(D=D,ridge_lambda=.07).fit(x,y)
            reference=Ridge(alpha=len(y)*.07,fit_intercept=True,solver='svd').fit(model._features(x),y)
            np.testing.assert_allclose(model.predict(xe),reference.predict(model._features(xe)),atol=1e-12)

    def test_rks_paired_features_and_nested_random_draws(self):
        x=np.random.default_rng(5).normal(size=(10,2));y=np.arange(10.)
        small=RandomKitchenSinks(8).fit(x,y);large=RandomKitchenSinks(20).fit(x,y)
        phi=small._features(x)
        np.testing.assert_allclose(phi[:,0::2]**2+phi[:,1::2]**2,1,atol=1e-14)
        np.testing.assert_array_equal(phi,large._features(x)[:,:8])

    def test_oof_encoding_excludes_own_outcome(self):
        x=pd.DataFrame({'a':['x','y','x','z']*10,'b':['u','v']*20})
        y=np.arange(40.)+1
        first=CrossFitTargetEncoder(folds=5).fit_transform(x,y)
        changed=y.copy();changed[7]+=100000
        second=CrossFitTargetEncoder(folds=5).fit_transform(x,changed)
        np.testing.assert_array_equal(first[7],second[7])

    def test_encoding_mean_formula_and_unseen_fallback(self):
        x=pd.DataFrame({'a':['x','x','y']});y=np.array([1.,5.,9.])
        encoder=CrossFitTargetEncoder(smoothing=2).fit(x,y)
        result=encoder.transform(pd.DataFrame({'a':['x','y','unseen']})).ravel()
        np.testing.assert_allclose(result,[(6+2*5)/4,(9+2*5)/3,5])

    def test_category_ids_reserve_zero_for_unknown(self):
        vocab=CategoryVocabulary().fit(pd.DataFrame({'a':['b','a','b'],'c':['x','z','y']}))
        ids=vocab.transform(pd.DataFrame({'a':['b','new'],'c':['x','new']}))
        self.assertTrue((ids[0]>0).all());np.testing.assert_array_equal(ids[1],[0,0])

    def test_splits_are_disjoint_complete_and_equal_holdouts(self):
        train,val,test=split_indices(397924,2026)
        self.assertEqual(len(val),len(test))
        np.testing.assert_array_equal(np.sort(np.r_[train,val,test]),np.arange(397924))

    def test_exact_duplicates_never_cross_splits_even_with_distinct_indices(self):
        base=pd.DataFrame({'x':np.arange(100.),'y':np.arange(100.)**2})
        frame=pd.concat([base,base],ignore_index=True).sample(frac=1,random_state=19)
        groups=exact_row_groups(frame)
        renamed=frame.set_axis(np.arange(len(frame))+10000)
        np.testing.assert_array_equal(groups,exact_row_groups(renamed))
        self.assertEqual(len(np.unique(groups)),100)
        splits=split_indices(len(frame),2026,groups=groups)
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)),np.arange(len(frame)))
        self.assertEqual([len(ix) for ix in splits],[140,30,30])
        for a,b in [(0,1),(0,2),(1,2)]:
            self.assertEqual(np.intersect1d(groups[splits[a]],groups[splits[b]]).size,0)
        repeated=split_indices(len(frame),2026,groups=groups)
        for first,second in zip(splits,repeated):np.testing.assert_array_equal(first,second)

    def test_provided_data_integrity_and_unlabeled_test_contract(self):
        verify_data(ROOT/'data')
        ds=HW2Dataset.from_csv(ROOT/'data/train.csv',ROOT/'data/val.csv',ROOT/'data/test.csv')
        self.assertEqual(ds.train.X.shape,(600,2));self.assertEqual(ds.val.X.shape,(200,2))
        self.assertEqual(ds.test.X.shape,(200,2));self.assertIsNone(ds.test.y)


if __name__=='__main__':
    unittest.main()
