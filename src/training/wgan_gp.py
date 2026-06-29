"""WGAN-GP objective (Section 3.6, Algorithm 1: ``L_adv = wgan_gp(...)``).

Standard Wasserstein GAN with a two-sided gradient penalty
(Gulrajani et al., 2017), made conditional through the projection critic.
"""
from __future__ import annotations

import torch


def gradient_penalty(D, real: torch.Tensor, fake: torch.Tensor,
                     c_ext: torch.Tensor, gp_lambda: float) -> torch.Tensor:
    """Two-sided gradient penalty on interpolates between real and fake."""
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = D(interp, c_ext)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=interp,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.reshape(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp_lambda * gp


def critic_loss(D, real: torch.Tensor, fake: torch.Tensor, c_ext: torch.Tensor,
                gp_lambda: float) -> torch.Tensor:
    """Critic (discriminator) loss: E[D(fake)] - E[D(real)] + GP."""
    d_real = D(real, c_ext).mean()
    d_fake = D(fake.detach(), c_ext).mean()
    gp = gradient_penalty(D, real, fake.detach(), c_ext, gp_lambda)
    return d_fake - d_real + gp


def generator_adv_loss(D, fake: torch.Tensor, c_ext: torch.Tensor) -> torch.Tensor:
    """Generator adversarial term: -E[D(fake)]."""
    return -D(fake, c_ext).mean()
