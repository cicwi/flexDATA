from . import data
from . import geometry
from . import display
from . import correct

__version__ = "1.0.1"

# We prefer info log messages to be written to console output.
# This can always be disabled by the user.
import logging
logging.basicConfig(level=logging.INFO)
