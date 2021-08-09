import argparse
import pathlib

from pytorch_lightning import Trainer
import anyconfig
import easydict

from project.tools.utils import add_argparse_args
from project.tools.train import train as _train
from project.tools.test import test as _test


def test(args: argparse.Namespace):
    config = anyconfig.load(args.config, ac_parser='yaml')
    config = easydict.EasyDict(config)

    ckpt_path = args.ckpt
    hparams_path = args.hparams
    command_opts = vars(args)
    _test(config=config,
          ckpt_path=ckpt_path,
          hparams_path=hparams_path,
          command_opts=command_opts)


def train(args: argparse.Namespace):
    config = anyconfig.load(args.config, ac_parser='yaml')
    config = easydict.EasyDict(config)

    command_opts = vars(args)
    _train(config=config,
           command_opts=command_opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main')

    sub_parsers = parser.add_subparsers(dest='name')
    train_parser = sub_parsers.add_parser(name='train', description='training utility')
    train_parser.add_argument('config',
                              help='config file')
    train_parser = add_argparse_args(Trainer, train_parser)

    test_parser = sub_parsers.add_parser(name='test', description='testing utility')
    test_parser.add_argument('config',
                             help='config file')
    test_parser.add_argument('ckpt',
                             help='test ckpt')
    test_parser.add_argument('hparams',
                             help='model hparams')
    test_parser = add_argparse_args(Trainer, test_parser)

    args = parser.parse_args()
    if args.name == 'train':
        train(args)

    if args.name == 'test':
        test(args)
