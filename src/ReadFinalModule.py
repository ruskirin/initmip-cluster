#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:10:19 2018

@author: rtwalker
"""
import numpy as np
from netCDF4 import Dataset
import yaml
from pathlib import Path
from collections import defaultdict, namedtuple

from ghub_utils import files


def ReadFinal(top_dir, experiment, infield, step, models):
    """
    Read initMIP netCDF4 datasets
    :param top_dir:
    :param experiment:
    :param infield:
    :param step:
    :param models: list of models to compare
    :return:
    """
    with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
        y = yaml.safe_load(f)
        model_paths = y['model_paths']

    Output = namedtuple('Output', 'xraw yraw outfield invalid')

    invalid = defaultdict(list)
    outfield = {}

    # I think that if I wanted to do this for a longer list of variables, I'd need a dict but could be inconvenient to access,
    # need to see tradeoff once I start processing
    for model in models:
        fpath = Path(top_dir) / model / experiment

        mname = model_paths[model].replace("/", "_")
        dpath = fpath / f'{infield}_GIS_{mname}_{experiment}.nc'
        if not dpath.is_file():
            # dataset doesn't exist for model, keep track
            invalid[infield].append((model, FileNotFoundError))
            continue

        U = Dataset(dpath)
        u = U.variables[infield][:]
        if experiment != 'init':
            u = u[step - 1, :, :]
        if u.shape[0] == 1:
            outfield[model] = np.squeeze(u, axis=0)
        else:
            outfield[model] = u

        # x and y are the same grid, so just get them once
        if model == models[0]:
            x = U.variables['x'][:]
            y = U.variables['y'][:]

    # easier to deal with in km
    x, y = x / 1000, y / 1000
    X, Y = np.meshgrid(x, y)

    return Output(X, Y, outfield, invalid)

# maybe add a save later after some analysis? right now, it's fast enough not to bother
# can do this with the save button on top of variable explorer


# if __name__ == '__main__':
#     from EOFandPlots2 import eof_and_plot
#
#     exp = 'ctrl'
#     # fields_all =
#     g_fields = {'orog': 'orog'}
#     b_fields = {'lithk': 'lithk', 'strbasemag': 'strbasemag'}
#     test = 'strbasemag'
#     # test = None
#
#     with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
#         y = yaml.safe_load(f)
#
#         models_all = list(y['model_paths'].keys())
#         # possible model exclusions
#         exclude_all = y['exclude']
#
#     models_exclude = []
#
#     models = sorted(list(set(models_all).difference(models_exclude)))
#
#     for f, feof in b_fields.items():
#         if test is not None:
#             tfield = b_fields[test]
#             eof_and_plot(models, [tfield,], [tfield,], exp, step=21)
#         else:
#             eof_and_plot(models, [f,], [feof,], exp, step=21)