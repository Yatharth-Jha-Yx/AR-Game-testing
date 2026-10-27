"""MotionPlay: turn a PC + webcam into a living-room motion console."""

import os

# Media Foundation (Windows) can take many seconds to open a webcam while it
# sets up hardware colour conversion; OpenCV's documented switch turns that off.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

__version__ = "0.1.0"
