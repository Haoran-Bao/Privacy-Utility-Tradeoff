from opacus.accountants import RDPAccountant


def default_delta(num_samples: int) -> float:
    return 1.0 / float(num_samples)


def get_sample_rate(batch_size: int, num_samples: int) -> float:
    return float(batch_size) / float(num_samples)


def create_accountant() -> RDPAccountant:
    return RDPAccountant()


def get_accountant_epsilon(accountant: RDPAccountant, delta: float) -> float:
    return float(accountant.get_epsilon(delta))
