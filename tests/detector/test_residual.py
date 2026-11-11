import pytest
import torch

from pytorch_ood.detector import Residual, ViM


def _reference_vim(features, train_features, w, b, d):
    """
    Reference ViM implementation (as before ViM was refactored to extend Residual),
    used to ensure the refactoring does not change ViM's scores.
    """
    u = -(torch.linalg.pinv(w) @ b)
    X = train_features - u
    cov = (X.T @ X) / X.shape[0]
    _, eig_vecs = torch.linalg.eigh(cov)
    ns = eig_vecs[:, : eig_vecs.shape[1] - d]
    alpha = (train_features @ w.T + b).max(dim=-1).values.mean() / (X @ ns).norm(dim=-1).mean()
    vlogit = ((features - u) @ ns).norm(dim=-1) * alpha
    energy = torch.logsumexp(features @ w.T + b, dim=-1)
    return vlogit - energy


class TestResidual:
    def setup_method(self) -> None:
        torch.manual_seed(123)
        self.n_dim = 10
        self.w = torch.randn(size=(3, self.n_dim))
        self.b = torch.randn(size=(3,))
        self.train = torch.randn(size=(200, self.n_dim))
        self.labels = torch.randint(0, 3, size=(200,))
        self.test = torch.randn(size=(64, self.n_dim))

    @pytest.mark.parametrize("cls", (Residual, ViM))
    @pytest.mark.parametrize("excess", (0, 1))
    def test_d_of_at_least_the_feature_dimension_is_halved_without_changing_the_setting(
        self, cls, excess
    ):
        d = self.n_dim + excess
        halved = cls(None, d=self.n_dim // 2, w=self.w, b=self.b)
        expected = halved.fit_features(self.train, self.labels).predict_features(self.test)

        detector = cls(None, d=d, w=self.w, b=self.b)
        detector.fit_features(self.train, self.labels)

        torch.testing.assert_close(detector.predict_features(self.test), expected)
        assert detector.d == d

    def test_principal_subspace_has_zero_residual(self):
        """
        Features that lie entirely in the principal subspace (after centering) have a residual norm of zero.
        """
        detector = Residual(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        ns = detector.residual_subspace
        x = self.test - detector.u
        # remove the component in the residual subspace, keeping only the principal component
        x = x - (x @ ns) @ ns.T + detector.u
        torch.testing.assert_close(
            detector.predict_features(x), torch.zeros(64), atol=1e-4, rtol=0
        )

    def test_vim_scores_unchanged(self):
        vim = ViM(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        expected = _reference_vim(self.test, self.train, self.w, self.b, d=4)
        torch.testing.assert_close(vim.predict_features(self.test), expected)

    def test_vim_extends_residual(self):
        residual = Residual(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        vim = ViM(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)

        assert isinstance(vim, Residual)
        energy = torch.logsumexp(self.test @ self.w.T + self.b, dim=-1)
        expected = vim.alpha * residual.predict_features(self.test) - energy
        torch.testing.assert_close(vim.predict_features(self.test), expected)

    def test_principal_subspace_alias_is_deprecated(self):
        detector = Residual(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        with pytest.warns(DeprecationWarning):
            assert detector.principal_subspace is detector.residual_subspace

    def test_repr(self):
        assert repr(Residual(None, d=4, w=self.w, b=self.b)) == "Residual(d=4)"
        assert repr(ViM(None, d=4, w=self.w, b=self.b)) == "ViM(d=4)"
