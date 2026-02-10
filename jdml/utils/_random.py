import numbers
import numpy as np

def check_random_generator(seed):
    """Convert seed to np.random.Generator.

    Parameters
    ----------
    seed : None, int, RandomState, or Generator

    Returns
    -------
    np.random.Generator
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState):
        # Use RandomState to generate a seed for Generator
        return np.random.default_rng(seed.randint(2**31))
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"Invalid seed type: {type(seed)}")


def check_random_state(seed):
    # this function is directly copied from scikit-learn
    """Turn seed into an np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
