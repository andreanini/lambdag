from .lambdag import LambdaGMethod

__all__ = ["LambdaGMethod"]


def __getattr__(name):
    if name == "visualize":
        import importlib
        try:
            return importlib.import_module("lambdag.visualize")
        except ImportError as e:
            raise ImportError(
                "The 'visualize' submodule requires optional dependencies. "
                "Install them with: pip install lambdag[visualize]"
            ) from e
    raise AttributeError(f"module 'lambdag' has no attribute {name!r}")
