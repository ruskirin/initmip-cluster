#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:10:19 2018

@author: rtwalker
"""
import os.path

import numpy as np
from netCDF4 import Dataset
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List

import files as mfiles
from ghub_utils import files as gfiles
import analysis


def ReadFinal(field: str, paths: List[Path], step):
    """
    Read netCDF4 datasets for a specific @field
    :param paths: list of paths to .nc files (must have specific name format)
    :param step:
    :return:
    """
    outfield = {}

    # Ryan: I think that if I wanted to do this for a longer list of variables,
    #   I'd need a dict but could be inconvenient to access, need to see
    #   tradeoff once I start processing
    for i, path in enumerate(paths):
        params = mfiles.netcdf_file_params(path)
        model = params.model
        experiment = params.exp

        path_abs = gfiles.DIR_PROJECT / path
        U = Dataset(path_abs)
        u = U.variables[field][:]

        if i == 0:
            # x and y are the same grid, so just get them once
            x = U.variables['x'][:] / 1000 # easier to deal with in km
            y = U.variables['y'][:] / 1000
            X, Y = np.meshgrid(x, y)

        if experiment != 'init':
            u = u[step - 1, :, :]
        if u.shape[0] == 1:
            outfield[model] = np.squeeze(u, axis=0)
        else:
            outfield[model] = u

    Output = namedtuple('Output', 'xraw yraw outfield')
    return Output(X, Y, outfield)
# def ReadFinal(top_dir, experiment, infield, step, models):
#     """
#     Read initMIP netCDF4 datasets
#     :param top_dir:
#     :param experiment:
#     :param infield:
#     :param step:
#     :param models: list of models to compare
#     :return:
#     """
#     with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
#         y = yaml.safe_load(f)
#         model_paths = y['model_paths']
#
#     Output = namedtuple('Output', 'xraw yraw outfield invalid')
#
#     invalid = defaultdict(list)
#     outfield = {}
#
#     # I think that if I wanted to do this for a longer list of variables, I'd need a dict but could be inconvenient to access,
#     # need to see tradeoff once I start processing
#     for model in models:
#         fpath = Path(top_dir) / model / experiment
#
#         mname = model_paths[model].replace("/", "_")
#         dpath = fpath / f'{infield}_GIS_{mname}_{experiment}.nc'
#         if not dpath.is_file():
#             # dataset doesn't exist for model, keep track
#             invalid[infield].append((model, FileNotFoundError))
#             continue
#
#         U = Dataset(dpath)
#         u = U.variables[infield][:]
#         if experiment != 'init':
#             u = u[step - 1, :, :]
#         if u.shape[0] == 1:
#             outfield[model] = np.squeeze(u, axis=0)
#         else:
#             outfield[model] = u
#
#         # x and y are the same grid, so just get them once
#         if model == models[0]:
#             x = U.variables['x'][:]
#             y = U.variables['y'][:]
#
#     # easier to deal with in km
#     x, y = x / 1000, y / 1000
#     X, Y = np.meshgrid(x, y)
#
#     return Output(X, Y, outfield, invalid)

# maybe add a save later after some analysis? right now, it's fast enough not to bother
# can do this with the save button on top of variable explorer


def read_data(paths: List[Path]):
    # variables will be a 2-level dict with field, model as keys
    variables = defaultdict(dict)
    # field: models map with any unaccounted exceptions
    exceptions = defaultdict(list)

    params = mfiles.union_netcdf_params(paths)
    eof_fields = analysis.group_fields(params.fields)

    for field in params.fields:
        # get only files for @field
        paths_field = mfiles.filter_paths_terms(
            paths, [field, ], mfiles.FileParams.FIELD
        )
        try:
            X, Y, variables[field] = ReadFinal(field, paths_field, step=21)

        except Exception as e:
            # other exception cases
            exceptions[field].append(analysis.format_data_exc(e))

    # # get {field: models} map of models with no/bad data
    # miss_data = analysis.test_valid_data(
    #     params.models, eof_fields, variables
    # )
    miss_data = analysis.test_valid_data(
        params.models, params.fields, variables
    )

    Output = namedtuple(
        'Output',
        'models fields eof_fields exps xraw yraw variables miss_data'
    )
    return Output(
        models=params.models,
        fields=list(params.fields),
        eof_fields=eof_fields,
        exps=list(params.exps),
        xraw=X,
        yraw=Y,
        variables=variables,
        miss_data=miss_data
    )
# def read_data(models, fields, experiment: str, step: int):
#     Output = namedtuple(
#         'Output',
#         'xraw yraw variables miss_fields miss_data'
#     )
#     # variables will be a 2-level dict with field, model as keys
#     variables = defaultdict(dict)
#     # mapping fields to exceptions
#     miss_fields = defaultdict(list) # keep track of models missing fields
#     # field: models map with any unaccounted exceptions
#     exceptions = defaultdict(list)
#
#     for field in fields:
#         try:
#             X, Y, variables[field], miss_data = ReadFinal(
#                 experiment, field, models, step
#             )
#             # TODO 8/1: move below to analysis.format_data_exc()
#             for f, m in miss_data.items():
#                 # keep track of bad models
#                 miss_fields[f].extend(m)
#
#         except Exception as e:
#             # other exception cases
#             exceptions[field].append(analysis.format_data_exc(e))
#
#     bm = set() # set of models with missing fields
#     for f, m in miss_fields.items():
#         bm.update({mi[0] for mi in m})
#
#     models_clean = set(models).difference(bm)
#     # get field: models map that has no data
#     miss_data = analysis.test_valid_data(models_clean, fields, variables)
#
#     return Output(X, Y, variables, miss_fields, miss_data)


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