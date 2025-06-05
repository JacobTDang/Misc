# Package Delcaration by creating an __init__.py file is basically telling Python that the folder should be treated like a package
# Allows import accessibility which will make it accessible when importing utils package
# the "." before video_utils is a relative import which would basically says look for video_utils.py in this directory
from .video_utils import read_video, save_video

from .bbox_utils import get_center_of_bbox, get_bbox_width