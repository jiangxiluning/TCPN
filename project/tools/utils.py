import inspect
from argparse import ArgumentParser
import copy

from pytorch_lightning.utilities import parsing
from pytorch_lightning.utilities.argparse import parse_args_from_docstring, get_init_arguments_and_types, \
    _gpus_allowed_type, _gpus_arg_default, _int_or_float_type


def get_valid_arguments(cls, params):

    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    return kwargs

def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
    r"""Extends existing argparse by default `Trainer` attributes.

    Args:
        cls: Lightning class
        parent_parser:
            The custom cli arguments parser, which will be extended by
            the Trainer default arguments.

    Only arguments of the allowed types (str, float, int, bool) will
    extend the `parent_parser`.

    Examples:
        >>> import argparse
        >>> from pytorch_lightning import Trainer
        >>> parser = argparse.ArgumentParser()
        >>> parser = Trainer.add_argparse_args(parser)
        >>> args = parser.parse_args([])
    """
    parser = parent_parser


    blacklist = ['kwargs']
    depr_arg_names = cls.get_deprecated_arg_names() + blacklist

    allowed_types = (str, int, float, bool)

    args_help = parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__)
    for arg, arg_types, arg_default in (at for at in get_init_arguments_and_types(cls) if at[0] not in depr_arg_names):
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = parsing.str_to_bool
            elif str in arg_types:
                use_type = parsing.str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == 'gpus' or arg == 'tpu_cores':
            use_type = _gpus_allowed_type
            arg_default = _gpus_arg_default

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == 'track_grad_norm':
            use_type = float

        parser.add_argument(
            f'--{arg}',
            dest=arg,
            default=arg_default,
            type=use_type,
            help=args_help.get(arg),
            **arg_kwargs,
        )

    return parser