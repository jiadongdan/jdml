__all__ = [
    'load_h5_data',
]


def __getattr__(name):
    if name == 'load_h5_data':
        from jdml.io.dataio import load_h5_data

        return load_h5_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
