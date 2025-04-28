
__version__ = "8.3.90"

import os

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1" 

from yolo12.utils import ASSETS, SETTINGS
from yolo12.utils.checks import check_yolo as checks
from yolo12.utils.downloads import download
from yolo12.models import YOLO, YOLOWorld

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "checks",
    "download",
    "settings",
)
