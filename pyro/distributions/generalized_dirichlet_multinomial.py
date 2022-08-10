import numbers

import torch
from torch.distributions.utils import broadcast_all

from pyro.ops.special import log_beta, log_binomial

from . import constraints
from .torch import Beta, Binomial, Dirichlet, Gamma, Multinomial, Poisson
from .torch_distribution import TorchDistribution
from .util import broadcast_shape

class GeneralizedDirichletMultinomial(TorchDistribution):
    r"""
    Compound distribution comprising of a dirichlet-multinomial pair. The probability of
    classes (``probs`` for the :class:`~pyro.distributions.Multinomial` distribution)
    is unknown and randomly drawn from a :class:`~pyro.distributions.Dirichlet`
    distribution prior to a certain number of Categorical trials given by
    ``total_count``.
    :param float or torch.Tensor concentration: concentration parameter (alpha) for the
        Dirichlet distribution.
    :param int or torch.Tensor total_count: number of Categorical trials.
    :param bool is_sparse: Whether to assume value is mostly zero when computing
        :meth:`log_prob`, which can speed up computation when data is sparse.
    """
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1),
        "total_count": constraints.nonnegative_integer,
    }
    support = Multinomial.support

    def __init__(
        self, alpha, beta, total_count=1, is_sparse=False, validate_args=None
    ):
        assert alpha.shape == beta.shape
        dummy_alpha = torch.cat((alpha, alpha[...,-1:]), axis=-1)
        batch_shape = alpha.shape[:-1]
        event_shape = alpha.shape[-1:]
        if isinstance(total_count, numbers.Number):
            total_count = dummy_alpha.new_tensor(total_count)
        else:
            batch_shape = broadcast_shape(batch_shape, total_count.shape)
            alpha = alpha.expand(batch_shape + (-1,))
            beta = beta.expand(batch_shape + (-1,))
            total_count = total_count.expand(batch_shape)
        self.alpha = alpha
        self.beta = beta
        self.total_count = total_count
        self.is_sparse = is_sparse
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @staticmethod
    def infer_shapes(alpha, total_count=()):
        batch_shape = broadcast_shape(alpha[:-1], total_count)
        event_shape = alpha[-1:]
        return batch_shape, event_shape

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedDirichletMultinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.beta = self.beta.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.is_sparse = self.is_sparse
        super(GeneralizedDirichletMultinomial, new).__init__(
            new.alpha.shape[:-1], new.alpha.shape[:-1], validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        # probs = self._dirichlet.sample(sample_shape)
        # total_count = int(self.total_count.max())
        # if not self.total_count.min() == total_count:
        #     raise NotImplementedError(
        #         "Inhomogeneous total count not supported by `sample`."
        #     )
        # return Multinomial(total_count, probs).sample()
        raise NotImplementedError

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_m = torch.lgamma(self.total_count + 1)
        log_factorial_ys = torch.lgamma(value).sum(axis=-1)
        return _component_prob(self.alpha, self.beta, value)[..., :-1].sum(axis=-1) + log_factorial_m - log_factorial_ys

    @property
    def mean(self):
        ratios = self.alpha / (self.alpha + self.beta)
        mean_d0 = ratios[0]
        log_ratio_sum_before_d_j = ratios.log().unsqueeze(-1) * torch.tril(torch.ones(ratios.shape[-1], ratios.shape[-1]), diagonal = -1).sum(axis=1)
        mean_d1_last = (ratios.log() + log_ratio_sum_before_d_j)[:-1].exp()
        mean_d_last = log_ratio_sum_before_d_j[-1]
        return torch.cat((mean_d0, mean_d1_last, mean_d_last), axis=-1)

    @property
    def variance(self):
        raise NotImplementedError


def _component_prob(alpha, beta, value):
    z = (value.unsqueeze(-1) * (1.0-torch.tril(torch.ones((value.dims[-1], value.dims[-1]))))).sum(axis=0)
    torch.lgamma(alpha + value) - torch.lgamma(alpha) + torch.lgamma(beta + z) - torch.lgamma(beta) + torch.lgamma(alpha + beta) - torch.lgamma(alpha + beta + z)