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
import files


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

    outfield = {}

    # I think that if I wanted to do this for a longer list of variables, I'd need a dict but could be inconvenient to access,
    # need to see tradeoff once I start processing
    for model in models:
        fpath = Path(top_dir) / model / experiment

        mname = model_paths[model].replace("/", "_")
        dpath = fpath / f'{infield}_GIS_{mname}_{experiment}.nc'

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

    return X, Y, outfield

# maybe add a save later after some analysis? right now, it's fast enough not to bother
# can do this with the save button on top of variable explorer


if __name__ == '__main__':
    # experiment = 'ctrl'
    # fields = ['lithk']
    # step = 21
    #
    # with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
    #     y = yaml.safe_load(f)
    #
    #     models = list(y['model_paths'].keys())
    #     # possible model exclusions
    #     exclude = y['exclude']
    #
    # ReadFinal(
    #     files.DIR_SESS_DATA / 'models', experiment, fields[0], step, models
    # )
    import EOFandPlots
