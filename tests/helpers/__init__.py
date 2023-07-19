from .dataset import sample_dataset
from .model import ClassificationModel, SegmentationModel

import sys
import logging

# set for testing
logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def for_examples(*parameters):
    """
    Decorator to run tests for several examples
    """

    def tuplify(x):
        if not isinstance(x, tuple):
            return (x,)
        return x

    def decorator(method, parameters=parameters):
        for parameter in (tuplify(x) for x in parameters):

            def method_for_parameter(self, method=method, parameter=parameter):
                method(self, *parameter)

            args_for_parameter = ",".join(repr(v) for v in parameter)
            name_for_parameter = method.__name__ + "(" + args_for_parameter + ")"
            frame = sys._getframe(1)  # pylint: disable-msg=W0212
            frame.f_locals[name_for_parameter] = method_for_parameter
        return None

    return decorator
