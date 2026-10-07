"""Three joint-embedding losses with one contract: two views z1, z2 of shape (B, D) -> scalar.

infonce: SimCLR NT-Xent, negatives from the batch, cosine / tau.
vicreg:  Bardes et al. 2022, invariance 25, variance 25, covariance 1.
sigreg:  LeJEPA (Balestriero, LeCun 2025): views pulled to their mean, plus the
         Epps-Pulley test of every random 1-D projection against N(0, 1).
"""

import torch
import torch.nn.functional as F

TAU = 0.2
SIGREG_LAMBDA = 0.05
SIGREG_SLICES = 256


def infonce(z1_BD, z2_BD):
    z_2BD = F.normalize(torch.cat([z1_BD, z2_BD]), dim=1)
    b = z1_BD.shape[0]
    logits_2B2B = z_2BD @ z_2BD.T / TAU
    logits_2B2B = logits_2B2B.masked_fill(torch.eye(2 * b, dtype=torch.bool, device=z_2BD.device), float("-inf"))
    target_2B = torch.cat([torch.arange(b, 2 * b), torch.arange(b)]).to(z_2BD.device)
    return F.cross_entropy(logits_2B2B, target_2B)


def vicreg(z1_BD, z2_BD):
    invariance = F.mse_loss(z1_BD, z2_BD)

    def variance(z_BD):
        return F.relu(1 - (z_BD.var(dim=0) + 1e-4).sqrt()).mean()

    def covariance(z_BD):
        b, d = z_BD.shape
        z_BD = z_BD - z_BD.mean(dim=0)
        cov_DD = z_BD.T @ z_BD / (b - 1)
        off_diagonal = cov_DD - torch.diag(torch.diag(cov_DD))
        return off_diagonal.square().sum() / d

    return 25 * invariance + 25 * (variance(z1_BD) + variance(z2_BD)) + covariance(z1_BD) + covariance(z2_BD)


def epps_pulley(z_BD, num_slices=SIGREG_SLICES):
    """Distance between the empirical characteristic function of each projection and exp(-t^2/2)."""
    direction_DM = torch.randn(z_BD.shape[1], num_slices, device=z_BD.device)
    direction_DM = direction_DM / direction_DM.norm(dim=0)
    t_T = torch.linspace(-5, 5, 17, device=z_BD.device)
    gauss_T = torch.exp(-0.5 * t_T.square())
    xt_BMT = (z_BD @ direction_DM).unsqueeze(2) * t_T
    real_MT = xt_BMT.cos().mean(0)
    imag_MT = xt_BMT.sin().mean(0)
    error_MT = ((real_MT - gauss_T).square() + imag_MT.square()) * gauss_T
    return torch.trapezoid(error_MT, t_T, dim=1).mean() * z_BD.shape[0]


def sigreg(z1_BD, z2_BD):
    z_VBD = torch.stack([z1_BD, z2_BD])
    invariance = (z_VBD - z_VBD.mean(0)).square().mean()
    regularizer = (epps_pulley(z1_BD) + epps_pulley(z2_BD)) / 2
    return (1 - SIGREG_LAMBDA) * invariance + SIGREG_LAMBDA * regularizer


def none(z1_BD, z2_BD):
    """No SSL signal: the labels-only control for the guided runs."""
    return 0 * z1_BD.sum()


LOSSES = {"infonce": infonce, "vicreg": vicreg, "sigreg": sigreg, "none": none}
