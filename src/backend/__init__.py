"""Accelerator backend seam.

`create_backend()` is the only thing callers need. It selects from the
environment and defers the import, so a GPU box never touches `torch_xla`:

    from src.backend import create_backend

    backend = create_backend()
    backend.setup()
    model = backend.wrap_model(model.to(backend.device))

Do NOT import `src.backend.tpu_backend` directly from shared code --
`scripts/ci/check_backend_seam.sh` exists to keep `torch_xla` out of the import
graph of everything except that one module.
"""

from src.backend.base import Backend, create_backend

__all__ = ["Backend", "create_backend"]
