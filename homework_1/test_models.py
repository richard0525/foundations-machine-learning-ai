"""Numerical and analytic checks independent of the experiment's reported metrics."""
import unittest
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
from data_utils import load_ale, FEATURES
from models import (Affine, logistic_loss, probit_loss, mse_loss,
                    student_t_loss, fit_sgd, predict)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.z = torch.tensor([-2.0, -0.3, 0.7, 2.5], dtype=torch.float64, requires_grad=True)
        self.y = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)

    def test_logistic_bce_and_gradient(self):
        p = torch.sigmoid(self.z)
        direct = -(self.y * p.log() + (1-self.y) * torch.log1p(-p)).mean()
        self.assertAlmostEqual(float(logistic_loss(self.z, self.y).detach()), float(direct.detach()), places=13)
        actual, = torch.autograd.grad(logistic_loss(self.z, self.y), self.z)
        torch.testing.assert_close(actual, (p.detach()-self.y)/len(self.y))

    def test_probit_against_scipy(self):
        z = self.z.detach().numpy()
        y = self.y.numpy()
        reference = -(y*stats.norm.logcdf(z)+(1-y)*stats.norm.logcdf(-z)).mean()
        self.assertAlmostEqual(float(probit_loss(self.z, self.y).detach()), reference, places=13)

    def test_extreme_classification_logits_have_finite_losses_and_gradients(self):
        for loss in (logistic_loss, probit_loss):
            z = torch.tensor([-100., 100.], dtype=torch.float64, requires_grad=True)
            value = loss(z, torch.tensor([1., 0.], dtype=torch.float64))
            grad, = torch.autograd.grad(value, z)
            self.assertTrue(torch.isfinite(value) and torch.isfinite(grad).all())

    def test_student_t_against_scipy_and_gradient(self):
        residual = self.y - self.z.detach()
        reference = -stats.t.logpdf(residual.numpy(), df=5).mean()
        value = student_t_loss(self.z, self.y)
        self.assertAlmostEqual(float(value.detach()), reference, places=12)
        actual, = torch.autograd.grad(value, self.z)
        torch.testing.assert_close(actual, -6*residual/(5+residual**2)/len(self.y))

    def test_autograd_finite_difference(self):
        for fn in (logistic_loss, probit_loss, mse_loss, student_t_loss):
            self.assertTrue(torch.autograd.gradcheck(lambda z: fn(z, self.y), (self.z,)))

    def test_mse_matrix_gradient(self):
        x = torch.tensor([[1., 3.], [-2., 1.], [0.5, -1.]], dtype=torch.float64)
        y = torch.tensor([2., -1., 3.], dtype=torch.float64)
        model = Affine(2)
        mse_loss(model(x), y).backward()
        torch.testing.assert_close(model.weight.grad, 2*x.T@(-y)/len(y))
        torch.testing.assert_close(model.bias.grad, -2*y.mean())

    def test_sgd_recovers_known_linear_solution(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 3))
        y = x @ np.array([1.2, -0.7, 2.1]) + .4
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        model, _, _, _ = fit_sgd(x, y, "mse", epochs=500, learning_rate=.05,
             decay_epochs=100, batch_size=12, seed=7, checkpoint_every=100)
        reference = LinearRegression().fit(x, y).predict(x)
        np.testing.assert_allclose(predict(model, x, "mse"), reference, atol=1e-8)

    def test_ale_schema_and_checksum(self):
        x, y, frame = load_ale(Path(__file__).parent / "data")
        self.assertEqual(x.shape, (107, 4))
        self.assertEqual(y.shape, (107,))
        self.assertNotIn("sd_ale", FEATURES)
        np.testing.assert_array_equal(x, frame.iloc[:, :4].to_numpy())


if __name__ == "__main__":
    unittest.main()
