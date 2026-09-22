import unittest

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import ModelNotSetException, RequiresFittingException
from src.pytorch_ood.detector import CARef
from src.pytorch_ood.utils import OODMetrics
from tests.helpers import ClassificationModel, sample_dataset


class IdentityHead(torch.nn.Module):
    """
    Head that turns a feature vector directly into logits, so that the predicted class of a
    feature vector is simply its argmax. Makes hand-computed expectations readable.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class TestCARef(unittest.TestCase):
    """
    Tests for the CARef detector
    """

    def _train_model(self, loader, num_outputs=3, n_hidden=128):
        model = ClassificationModel(num_outputs=num_outputs, n_hidden=n_hidden)
        sgd = SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9, nesterov=True)
        for _ in range(20):
            for x, y in loader:
                sgd.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                sgd.step()
        return model.eval()

    def test_requires_fit_flag(self):
        self.assertTrue(CARef.requires_fit)

    def test_nofitting(self):
        """Scoring before fitting must fail"""
        model = ClassificationModel().eval()
        detector = CARef(encoder=model.features, head=model.classifier)

        with self.assertRaises(RequiresFittingException):
            detector(torch.randn(size=(16, 10)))

        with self.assertRaises(RequiresFittingException):
            detector.predict_features(torch.randn(size=(16, 10)))

    def test_no_encoder(self):
        """predict() without an encoder must fail"""
        model = ClassificationModel().eval()
        detector = CARef(encoder=None, head=model.classifier)
        detector.fit_features(torch.randn(size=(64, 10)))

        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(size=(16, 10)))

        with self.assertRaises(ModelNotSetException):
            detector.fit(DataLoader(TensorDataset(torch.randn(16, 10), torch.zeros(16).long())))

    def test_no_head(self):
        with self.assertRaises(ValueError):
            CARef(encoder=None, head=None)

    def test_rejects_grid_input(self):
        """CARef operates on pooled feature vectors, not feature maps"""
        detector = CARef(encoder=None, head=IdentityHead())

        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(size=(16, 3, 4, 4)))

        detector.fit_features(torch.randn(size=(16, 3)))
        with self.assertRaises(ValueError):
            detector.predict_features(torch.randn(size=(16, 3, 4, 4)))

    def test_score_shape_and_finiteness(self):
        detector = CARef(encoder=None, head=torch.nn.Linear(16, 4))
        detector.fit_features(torch.randn(size=(256, 16)))

        scores = detector.predict_features(torch.randn(size=(32, 16)))

        self.assertEqual(scores.shape, (32,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_exact_score(self):
        """
        Hand-computed score. With an identity head the predicted class is the argmax of the
        feature vector, so the centroids and the relative error can be written down directly.
        """
        # class 0 wins for the first two rows, class 1 for the last two
        z_fit = torch.tensor(
            [
                [4.0, 0.0],
                [6.0, 2.0],
                [0.0, 10.0],
                [2.0, 20.0],
            ]
        )
        # centroid of class 0 = (5, 1), centroid of class 1 = (1, 15)
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(z_fit)

        self.assertTrue(
            torch.allclose(detector.train_means, torch.tensor([[5.0, 1.0], [1.0, 15.0]]))
        )

        z_test = torch.tensor([[7.0, 3.0], [-1.0, 5.0]])
        # row 0: predicted class 0, |7-5| + |3-1| = 4, ||z||_1 = 10 -> 0.4
        # row 1: predicted class 1, |-1-1| + |5-15| = 12, ||z||_1 = 6 -> 2.0
        expected = torch.tensor([0.4, 2.0])

        self.assertTrue(torch.allclose(detector.predict_features(z_test), expected, atol=1e-6))

    def test_score_sign(self):
        """
        A feature vector that equals its class centroid has zero error, anything else is
        strictly larger. This pins down the direction of the score.
        """
        z_fit = torch.tensor([[4.0, 0.0], [6.0, 2.0]])
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(z_fit)

        centroid = detector.train_means[0].unsqueeze(0)
        off_centroid = centroid + torch.tensor([[0.5, 0.25]])

        self.assertAlmostEqual(detector.predict_features(centroid).item(), 0.0, places=6)
        self.assertGreater(detector.predict_features(off_centroid).item(), 0.0)

    def test_normalization_is_scale_invariant(self):
        """
        Scaling a feature vector *and* its centroid by the same factor must leave the relative
        error unchanged; an implementation that forgets the normalization would not.
        """
        z_fit = torch.tensor([[4.0, 0.0], [6.0, 2.0]])
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(z_fit)

        plain = detector.predict_features(torch.tensor([[7.0, 3.0]]))

        scaled = CARef(encoder=None, head=IdentityHead())
        scaled.fit_features(z_fit * 3.0)
        scaled_score = scaled.predict_features(torch.tensor([[21.0, 9.0]]))

        self.assertTrue(torch.allclose(plain, scaled_score, atol=1e-6))

    def test_reduction_is_over_feature_dimension(self):
        """
        The error must be reduced over the feature dimension only, giving one score per sample.
        A reduction over the batch would make the scores identical.
        """
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(torch.tensor([[4.0, 0.0], [6.0, 2.0]]))

        scores = detector.predict_features(torch.tensor([[5.0, 1.0], [50.0, 1.0]]))

        self.assertEqual(scores.shape, (2,))
        self.assertNotAlmostEqual(scores[0].item(), scores[1].item())

    def test_centroids_use_predicted_not_true_labels(self):
        """
        The paper conditions the centroids on the predicted label. Passing deliberately wrong
        ground-truth labels must therefore not change the fitted state.
        """
        torch.manual_seed(5)
        head = torch.nn.Linear(8, 3)
        z = torch.randn(size=(128, 8))

        with_true = CARef(encoder=None, head=head)
        with_true.fit_features(z, torch.arange(128) % 3)

        with_shuffled = CARef(encoder=None, head=head)
        with_shuffled.fit_features(z, (torch.arange(128) + 1) % 3)

        self.assertTrue(torch.allclose(with_true.train_means, with_shuffled.train_means))

    def test_uses_features_not_logits(self):
        """
        The relative error lives in feature space; the head is only used to pick the class.
        With a head whose output dimension differs from the feature dimension, an
        implementation that scored logits instead of features cannot produce this result.
        """
        head = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            head.bias.copy_(torch.tensor([0.0, 100.0]))  # class 1 always wins

        z_fit = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        detector = CARef(encoder=None, head=head)
        detector.fit_features(z_fit)

        # centroids live in feature space (2 classes x 3 feature dimensions)
        self.assertEqual(detector.train_means.shape, (2, 3))
        # both fitting samples are assigned to class 1
        self.assertTrue(torch.allclose(detector.train_means[1], torch.tensor([2.0, 3.0, 4.0])))

        # |4-2| + |4-3| + |4-4| = 3, ||z||_1 = 12
        score = detector.predict_features(torch.tensor([[4.0, 4.0, 4.0]]))
        self.assertAlmostEqual(score.item(), 3.0 / 12.0, places=6)

    def test_no_batch_coupling(self):
        """Scoring a sample alone must give the same value as scoring it inside a batch"""
        torch.manual_seed(7)
        head = torch.nn.Linear(6, 3)
        detector = CARef(encoder=None, head=head)
        detector.fit_features(torch.randn(size=(256, 6)))

        batch = torch.randn(size=(16, 6))
        batch[3] *= 40  # an extreme sample that a batch-level normalization would leak from

        batched = detector.predict_features(batch)
        individually = torch.cat([detector.predict_features(row.unsqueeze(0)) for row in batch])

        self.assertTrue(torch.allclose(batched, individually, atol=1e-6))

    def test_fit_ignores_ood_samples(self):
        """OOD samples in the fitting set must not enter the centroid estimate"""
        torch.manual_seed(17)
        head = torch.nn.Linear(8, 3)
        z_id = torch.randn(size=(256, 8))
        z_ood = torch.randn(size=(256, 8)) * 20 + 50

        clean = CARef(encoder=None, head=head)
        clean.fit_features(z_id, torch.arange(256) % 3)

        mixed = CARef(encoder=None, head=head)
        mixed.fit_features(
            torch.cat([z_id, z_ood]),
            torch.cat([torch.arange(256) % 3, -torch.ones(256).long()]),
        )

        self.assertTrue(torch.allclose(clean.train_means, mixed.train_means, atol=1e-5))

    def test_fit_without_id_samples(self):
        detector = CARef(encoder=None, head=torch.nn.Linear(8, 3))

        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(size=(32, 8)), -torch.ones(32).long())

    def test_unpredicted_class_falls_back_to_global_mean(self):
        """Classes without assigned samples must not produce NaN centroids"""
        # identity head, third dimension never wins
        z_fit = torch.tensor([[4.0, 0.0, -9.0], [0.0, 3.0, -9.0]])
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(z_fit)

        self.assertTrue(torch.isfinite(detector.train_means).all())
        self.assertTrue(torch.allclose(detector.train_means[2], z_fit.mean(dim=0)))

    def test_fit_matches_fit_features(self):
        """The loader based and the tensor based interface must agree"""
        torch.manual_seed(11)
        model = ClassificationModel().eval()
        x = torch.randn(size=(256, 10))
        loader = DataLoader(TensorDataset(x, torch.arange(256) % 3), batch_size=32)

        from_loader = CARef(encoder=model.features, head=model.classifier)
        from_loader.fit(loader)

        with torch.no_grad():
            z = model.features(x)
        from_features = CARef(encoder=model.features, head=model.classifier)
        from_features.fit_features(z)

        self.assertTrue(
            torch.allclose(from_loader.train_means, from_features.train_means, atol=1e-5)
        )

    def test_predict_matches_predict_features(self):
        """predict() must only add the encoder forward pass"""
        torch.manual_seed(13)
        model = ClassificationModel().eval()
        x = torch.randn(size=(128, 10))

        detector = CARef(encoder=model.features, head=model.classifier)
        with torch.no_grad():
            detector.fit_features(model.features(x))

        x_test = torch.randn(size=(16, 10))
        with torch.no_grad():
            expected = detector.predict_features(model.features(x_test))

        self.assertTrue(torch.allclose(detector.predict(x_test), expected, atol=1e-6))

    def test_fit_chunking_is_exact(self):
        """The chunked head evaluation during fit must not change the result"""
        torch.manual_seed(19)
        head = torch.nn.Linear(6, 4)
        z = torch.randn(size=(200, 6))

        whole = CARef(encoder=None, head=head)
        whole.fit_features(z, batch_size=1000)

        chunked = CARef(encoder=None, head=head)
        chunked.fit_features(z, batch_size=7)

        self.assertTrue(torch.allclose(whole.train_means, chunked.train_means, atol=1e-6))

    def test_repr(self):
        self.assertEqual(repr(CARef(encoder=None, head=torch.nn.Linear(2, 2))), "CARef()")

    def test_detects_outliers(self):
        """End-to-end check on a trained model with separable ID/OOD data"""
        torch.manual_seed(42)
        dataset = sample_dataset(n_samples=1000, n_dim=10, centers=3, seed=42, loc=3)
        loader = DataLoader(dataset, batch_size=1000)
        model = self._train_model(loader)

        detector = CARef(encoder=model.features, head=model.classifier)
        detector.fit(loader)

        x, y = next(iter(loader))
        scores_in = detector.predict(x)

        ood = sample_dataset(n_samples=100, n_dim=10, centers=3, std=5, seed=42, loc=-3)
        x_ood, _ = next(iter(DataLoader(ood, batch_size=1000)))
        scores_out = detector.predict(x_ood)

        metrics = OODMetrics()
        metrics.update(scores_in, y)
        metrics.update(scores_out, -torch.ones_like(scores_out).long())

        auroc = metrics.compute()["AUROC"]
        print(f"AUROC: {auroc}")
        self.assertGreater(auroc, 0.8)

    def test_scores_are_graph_free(self):
        """Scores must never carry an autograd graph back to the caller"""
        head = torch.nn.Linear(6, 3)
        detector = CARef(encoder=None, head=head)
        detector.fit_features(torch.randn(size=(64, 6), requires_grad=True))

        scores = detector.predict_features(torch.randn(size=(8, 6), requires_grad=True))

        self.assertFalse(scores.requires_grad)
        self.assertIsNone(scores.grad_fn)
        self.assertFalse(detector.train_means.requires_grad)

    def test_fit_features_rejects_empty_input(self):
        """An empty fitting set must fail with a clear error, not inside torch"""
        detector = CARef(encoder=None, head=torch.nn.Linear(6, 3))

        with self.assertRaises(ValueError):
            detector.fit_features(torch.zeros(size=(0, 6)))

    def test_features_are_cast_to_float32(self):
        """
        float64 input is accepted and downcast, which is what lets a standard float32 head
        consume it without the caller having to convert.
        """
        detector = CARef(encoder=None, head=torch.nn.Linear(4, 2))
        detector.fit_features(torch.randn(size=(32, 4), dtype=torch.float64))

        self.assertEqual(detector.train_means.dtype, torch.float32)

        scores = detector.predict_features(torch.randn(size=(8, 4), dtype=torch.float64))
        self.assertEqual(scores.dtype, torch.float32)

    def test_small_norm_features_are_not_clamped_away(self):
        """
        The normalization guard must only protect against a zero norm. A feature vector with
        a small but nonzero norm has to be normalized by its actual norm.
        """
        detector = CARef(encoder=None, head=IdentityHead())
        # centroid of class 0 is (0.08, 0.0)
        detector.fit_features(torch.tensor([[0.06, 0.0], [0.10, 0.0]]))

        # |0.2 - 0.08| + |0.0 - 0.0| = 0.12, ||z||_1 = 0.2 -> 0.6
        score = detector.predict_features(torch.tensor([[0.2, 0.0]]))
        self.assertAlmostEqual(score.item(), 0.6, places=5)

    def test_zero_feature_vector_is_finite(self):
        """An all-zero feature vector must not produce NaN"""
        detector = CARef(encoder=None, head=IdentityHead())
        detector.fit_features(torch.tensor([[1.0, 0.0], [3.0, 0.0]]))

        score = detector.predict_features(torch.zeros(size=(1, 2)))
        self.assertTrue(torch.isfinite(score).all())

    def test_state_follows_to_device(self):
        """Fitted state must be moved by to(), also on CPU-only machines"""
        head = torch.nn.Linear(6, 3)
        detector = CARef(encoder=None, head=head)
        detector.fit_features(torch.randn(size=(64, 6)))

        detector.to("cpu")

        self.assertIsNotNone(detector.train_means)
        self.assertEqual(detector.train_means.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_fit_features_moves_cpu_features_to_detector_device(self):
        """
        fit_features() is not wrapped by the base class, so it must move CPU features onto the
        detector's device itself. Otherwise the fitted state ends up on the wrong device.
        """
        model = ClassificationModel().eval().cuda()
        detector = CARef(encoder=model.features, head=model.classifier).to("cuda")

        detector.fit_features(torch.randn(size=(256, 10)))  # deliberately on the CPU

        self.assertEqual(detector.train_means.device.type, "cuda")

        scores = detector.predict_features(torch.randn(size=(8, 10)))
        self.assertEqual(scores.device.type, "cuda")
        self.assertTrue(torch.isfinite(scores).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_cuda(self):
        """State must follow the detector onto the GPU"""
        model = ClassificationModel().eval().cuda()
        detector = CARef(encoder=model.features, head=model.classifier).to("cuda")
        detector.fit_features(torch.randn(size=(256, 10)).cuda())

        self.assertEqual(detector.train_means.device.type, "cuda")

        scores = detector.predict(torch.randn(size=(16, 10)).cuda())

        self.assertEqual(scores.device.type, "cuda")
        self.assertTrue(torch.isfinite(scores).all())
