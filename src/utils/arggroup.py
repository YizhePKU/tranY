import inspect
import types


def patch_arggroup(args):
    """Patch a Namespace object to add a group() method, which takes a class
    and returns a dict containing kwargs to initialize that class.
    """

    def group(self, cls):
        parameters = inspect.signature(cls.__init__).parameters.keys()
        return {key: getattr(self, key) for key in parameters if key in self}

    args.group = types.MethodType(group, args)
