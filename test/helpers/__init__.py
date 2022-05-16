import sys

from .model import Model


def for_examples(*parameters):
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
