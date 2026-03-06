import importlib
from typing import *

if TYPE_CHECKING:
    from .v1 import MoGeModel as MoGeModelV1
    from .v2 import MoGeModel as MoGeModelV2


def import_model_class_by_version(version: str) -> Type[Union['MoGeModelV1', 'MoGeModelV2']]:
    assert version in ['v1', 'v2'], f'Unsupported model version: {version}'
    
    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError as exc:
        # Do not hide missing dependency errors (e.g., utils3d) as missing model version.
        missing_name = getattr(exc, "name", "")
        expected_module = f"{__package__}.{version}" if __package__ else version
        if missing_name in {expected_module, version}:
            raise ValueError(f'Model version "{version}" not found.') from exc
        raise

    cls = getattr(module, 'MoGeModel')
    return cls
