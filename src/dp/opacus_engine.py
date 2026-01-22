from typing import Tuple

from opacus import PrivacyEngine


def make_private(
    model,
    optimizer,
    train_loader,
    noise_multiplier: float,
    max_grad_norm: float,
    poisson_sampling: bool = False,
) -> Tuple:
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )
    return model, optimizer, train_loader, privacy_engine


def make_private_with_epsilon(
    model,
    optimizer,
    train_loader,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    poisson_sampling: bool = False,
) -> Tuple:
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )
    return model, optimizer, train_loader, privacy_engine


def get_epsilon(privacy_engine: PrivacyEngine, delta: float) -> float:
    return float(privacy_engine.get_epsilon(delta))
