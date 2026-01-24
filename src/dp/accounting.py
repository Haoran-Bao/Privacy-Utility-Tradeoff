from opacus.accountants import RDPAccountant


def default_delta(num_samples: int) -> float:
    return 1.0 / float(num_samples)


def get_sample_rate(batch_size: int, num_samples: int) -> float:
    return float(batch_size) / float(num_samples)


def create_accountant() -> RDPAccountant:
    return RDPAccountant()


def get_accountant_epsilon(accountant: RDPAccountant, delta: float) -> float:
    return float(accountant.get_epsilon(delta))


def _compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta))


def find_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    max_noise: float = 20.0,
    tol: float = 1e-3,
    max_iters: int = 30,
) -> float:
    if steps <= 0:
        return 0.0
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")

    low = 1e-6
    high = max_noise
    eps_high = _compute_epsilon(high, sample_rate, steps, target_delta)
    while eps_high > target_epsilon and high < 100.0:
        high *= 2.0
        eps_high = _compute_epsilon(high, sample_rate, steps, target_delta)

    if eps_high > target_epsilon:
        return high

    for _ in range(max_iters):
        mid = 0.5 * (low + high)
        eps_mid = _compute_epsilon(mid, sample_rate, steps, target_delta)
        if abs(eps_mid - target_epsilon) <= tol:
            return mid
        if eps_mid > target_epsilon:
            low = mid
        else:
            high = mid
    return high
