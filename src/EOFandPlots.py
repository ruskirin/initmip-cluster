#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:09:38 2018

@author: rtwalker
"""

# imports #####################################################################

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from numpy.linalg import svd, norm
from collections import defaultdict, namedtuple
import seaborn as sns
import pickle
from functools import partial
from typing import List
from pathlib import Path, PosixPath

# from scipy.interpolate import griddata
import ReadFinalModule as read
from ghub_utils import files as gfiles
import analysis

# from sklearn.manifold import MDS


def eof_and_plot(
        fig_dist: plt.Figure,
        fig_weight: plt.Figure,
        paths: List[Path]
):
    """
    TODO 8/1: docs here
    :param fig_dist:
    :param fig_weight:
    :param paths:
    :return:
    """
    ### Read netCDF Files
    # X, Y is reinitialized, but doesn't matter because they should be the same
    #   geospatial coordinates
    # TODO 8/1: ^^^ perhaps remove potentially unnecessary calculations
    #   associated with having to do that?
    data = read.read_data(paths)
    step = 21

    models = data.models
    fields = data.fields
    eof_fields = data.eof_fields
    experiment = data.exps[0]

    X, Y = data.xraw, data.yraw
    variables = data.variables
    miss_data = data.miss_data

    # TODO 8/1: HERE is where you remove either the fields or the models
    #   using @miss_data
    # TODO 8/3: move cleaning/data validation to another function
    bm = set()
    for m in miss_data.values():
        bm.update({mi[0] for mi in m})

    # remove data associated with the bad models
    models = sorted(list(set(models).difference(bm)))
    for f in fields:
        for m in bm:
            try:
                del variables[f][m]
            except KeyError:
                continue

### Mask Invalid Model Outputs
    mask2d, xclean, yclean = analysis.mask_invalid(
        X, Y, models, variables, list(fields)
    )

    # this will become a dict with variables as keys and one matrix per variable as values
    matrix = defaultdict(partial(np.ndarray, 0))

    for field in fields:
        matrix[field] = np.empty((len(models), len(xclean)), float)

        for i, m in enumerate(models):
            # print(models[i])
            matrix[field][i, :] = ma.masked_array(
                variables[field][m],
                mask2d
            ).compressed()

    """------------------------Prepare for EOF Analysis--------------------------"""
    # this is because not all models reported velocity magnitude
    if {'uvelsurf', 'vvelsurf'}.issubset(set(fields)):
        matrix['uvelsurf'] *= 31557600  # m/a from m/s
        matrix['vvelsurf'] *= 31557600
        matrix['velsurf'] = np.sqrt(
            matrix['uvelsurf'] ** 2 + matrix['vvelsurf'] ** 2
        )
        fields.append('velsurf')
        # put into the dictionary for easier access post-run
        for i in range(len(models)):
            variables['velsurf'][models[i]] = matrix['velsurf'][i, :]

    if {'uvelbase', 'vvelbase'}.issubset(set(fields)):
        matrix['uvelbase'] *= 31557600  # m/a from m/s
        matrix['vvelbase'] *= 31557600
        matrix['velbase'] = np.sqrt(
            matrix['uvelbase'] ** 2 + matrix['vvelbase'] ** 2
            )
        fields.append('velbase')
        # put into the dictionary for easier access post-run
        for i in range(len(models)):
            variables['velbase'][models[i]] = matrix['velbase'][i, :]

    # scale the matrices
    anomaly_matrix = defaultdict(partial(np.ndarray, 0))
    scaled_matrix = defaultdict(partial(np.ndarray, 0))

    # eventually needs to include scales for all variables I use
    # scale = {'lithk': 500, 'orog': 500, 'uvelsurf': 1000, 'vvelsurf': 1000, 'velsurf': 1000,
    #         'acabf': 1, 'dlithkdt': 1, 'litempsnic': 1, 'litempbot': 1, 'strbasemag': 1}

    for field in eof_fields:
        # TODO 8/3 URGENT: why is normalization factor derived from models
        #   instead of observational data?
        std = np.std(matrix[field], axis=0) # normalizing factors
        matrix0 = matrix[field][:, std == 0]
        xclean0 = xclean[std == 0]
        yclean0 = yclean[std == 0]

        anomaly_matrix[field] = matrix[field] - np.mean(matrix[field], axis=0)
        scaled_matrix[field] = anomaly_matrix[field] / std

    """------------------------------EOF Analysis--------------------------------"""
    # create the matrix with all desired fields
    M = np.hstack([scaled_matrix[field] for field in eof_fields])

    # truncate the SVD because very many columns (one per point, ~ 60 K)
    # results are the same as with full matrices but much faster
    UU, svals, VT = svd(M, full_matrices=False)

    # fraction of variance explained
    var_frac = svals / np.sum(svals)

    # cumulative fraction of variance
    var_cum = np.cumsum(var_frac)

    # choose how many eof (i.e., cols) to keep (arbitrary cutoff)
    ncol = np.where(var_cum > 0.95)[0][0]

    # truncate the set of eofs
    UU = UU[:, :ncol]

    """---------------------------Find Intermodel Distances----------------------"""
    # nested dictionary easiest for lookup
    distance = defaultdict(dict)

    # matrix is useful for plotting
    distance_matrix = np.empty((len(models), len(models)))

    # find distances with euclidean norm of eof differences
    for i in range(len(models)):
        for j in range(len(models)):
            distance[models[i]][models[j]] = distance_matrix[i, j] = norm(
                UU[i, :] - UU[j, :]
            )

    """--------------------------------Find Weights------------------------------"""
    # set similarity radius
    num_std = [1.0, 2.0, 3.0]

    similarity_matrix = defaultdict(dict)
    weights = defaultdict(dict)

    for radius in num_std:
        similarity_radius = radius * np.std(distance_matrix[distance_matrix != 0])

        # similarity matrix
        similarity_matrix[radius] = np.exp(
            -(distance_matrix / similarity_radius) ** 2
        )

        # effective repetition of each model
        effective_repetition = np.zeros((len(models), 1))
        # formula starts with 1 +
        effective_repetition.fill(1.0)
        # sum over similarities to other models
        for i in range(len(models)):
            for j in range(len(models)):
                if j != i:
                    effective_repetition[i] += similarity_matrix[radius][i, j]

        # similarity weighting
        weights[radius] = 1.0 / effective_repetition

    """--------------------------------Save Arrays-------------------------------"""
    pickle_name = f'output-{eof_fields}-{experiment}.p'

    with open(gfiles.DIR_OUT / pickle_name, 'wb') as f:
        pickle.dump([distance, distance_matrix, weights], f)

    # this is how to load the variables again
    # with open(pickle_name, 'rb') as f:
    #    distance0, distance_matrix0, weights0 = pickle.load(f)

    """-----------------------------------Plot-----------------------------------"""
    # the distance matrix is symmetric, so just show lower half
    plotmask = np.zeros_like(distance_matrix)
    plotmask[np.triu_indices_from(plotmask)] = True

    # f1, ax1 = plt.subplots(figsize=(10, 10))
    fig_dist.clear()
    ax1 = fig_dist.add_subplot()

    # this limit is usually ~ 1, use +/- to center colors so median is white
    lim = 0.8 * np.max(np.abs(distance_matrix - np.median(distance_matrix)))
    ax1 = sns.heatmap(
        distance_matrix - np.median(distance_matrix), mask=plotmask,
        square=True, cmap='RdBu',
        linewidths=0.25, linecolor='white', vmin=-lim, vmax=lim,
        ax=ax1
    )

    # line up to have model names centered on each box
    ax1.xaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
    ax1.xaxis.set_ticklabels(models, rotation=90)
    ax1.yaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
    ax1.yaxis.set_ticklabels(models, rotation=0)

    # title depending on which data was used
    fields_string = ', '.join(eof_fields)
    fields_string = '(' + fields_string + ')'

    if experiment == 'init':
        time_string = ' at INIT'
    else:
        time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

    fig_dist.suptitle(
        'EOF inter-model distances vs median for fields ' + fields_string + time_string
    )

    if len(miss_data) > 0:
        cap_miss_d = [m[0] for m in list(miss_data.values())[0]]
    else:
        cap_miss_d = 'None'

    caption = f'Following models were excluded: '\
              f'\n- models with invalid data: {cap_miss_d}'
    fig_dist.text(
        x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
    )

    # make sure everything fits on the plot
    fig_dist.tight_layout()

    fname = f'distances-{fields_string}-{experiment.upper()}'
    fig_dist.savefig(gfiles.DIR_OUT / fname, bbox_inches='tight')

    display(fig_dist)
    # clear_output(wait=True)
    # plt.show()

    """---------------------------------Plot Weights-----------------------------"""
    # f2, ax2 = plt.subplots(figsize=(12, 8))
    fig_weight.clear()
    ax2 = fig_weight.add_subplot()

    for ix, key in enumerate(weights):
        ax2.plot(weights[key], marker='o', linestyle='none', alpha=0.75,
                 color=plt.cm.tab10(ix), label='R = ' + str(key) + r'$\sigma$'
                 )

    ax2.xaxis.set_ticks(np.arange(len(models)))
    ax2.xaxis.set_ticklabels(models, rotation=90)
    ax2.grid(which='both')

    fields_string = ', '.join(eof_fields)
    fields_string = '(' + fields_string + ')'

    if experiment == 'init':
        time_string = ' at INIT'
    else:
        time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

    fig_weight.suptitle(
        'Weights by similarity radius R for fields ' + fields_string + time_string
    )
    ax2.legend()

    fig_weight.text(
        x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
    )

    # plt.tight_layout()
    fig_weight.tight_layout()

    fname = f'weights-{fields_string}-{experiment.upper()}'
    fig_weight.savefig(gfiles.DIR_OUT / fname, bbox_inches='tight')

    display(fig_weight)
    clear_output(wait=True)
    # plt.show()

    """-------------------------------------MDS----------------------------------"""
    # # multidimensional scaling makes a 2d scatter plot from distances
    # mds_model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    # out = mds_model.fit_transform(distance_matrix)
    #
    # plt.figure(figsize=(10,10))
    # plt.scatter(out[:,0], out[:,1])
    # for i in range(len(models))    :
    #    plt.annotate(models[i], (out[i,0], out[i,1]), fontweight=16)
    #
    # # title depending on which data was used
    # fields_string = ', '.join(eof_fields)
    # plt.title('Multi-dimensional scaling of inter-model distances, fields: ' + fields_string)
    #
    # plt.show()
    #
    #
    #
    #
    #
    #
    #
    # # notes for other plots
    # # to plot any 2d variable, plt.imshow(np.flipud(bar), aspect='equal', interpolation='none') works well
    # # for plt.contourf(), first follow the griddata and mask procedure as for Tmean above


# def eof_and_plot(
#         fig_dist: plt.Figure,
#         fig_weight: plt.Figure,
#         models,
#         eof_fields: list,
#         experiment,
#         step
# ):
#     """
#     TODO 8/1: docs here
#     :param fig_dist:
#     :param fig_weight:
#     :param models:
#     :param eof_fields:
#     :param experiment:
#     :param step:
#     :return:
#     """
#     fields = analysis.get_field_pairs(eof_fields)
#
#     ### Read netCDF Files
#     # X, Y is reinitialized, but doesn't matter because they should be the same
#     #   geospatial coordinates
#     # TODO 8/1: ^^^ perhaps remove potentially unnecessary calculations
#     #   associated with having to do that?
#     X, Y, variables, miss_fields, miss_data = read_data(
#         models, fields, experiment, step
#     )
#
#     # TODO 8/1: HERE is where you remove either the fields or the models
#     #   using @miss_fields and @miss_data
#     # TODO 8/3: move cleaning/data validation to another function
#     bm = set()
#     for m in miss_fields.values():
#         bm.update({mi[0] for mi in m})
#     for m in miss_data.values():
#         bm.update({mi[0] for mi in m})
#
#     # remove data associated with the bad models
#     models = sorted(list(set(models).difference(bm)))
#     for f in fields:
#         for m in bm:
#             try:
#                 del variables[f][m]
#             except KeyError:
#                 continue
#
# ### Mask Invalid Model Outputs
#     mask2d, xclean, yclean = mask_invalid(X, Y, models, variables, fields)
#
#     # this will become a dict with variables as keys and one matrix per variable as values
#     matrix = defaultdict(partial(np.ndarray, 0))
#
#     for field in fields:
#         matrix[field] = np.empty((len(models), len(xclean)), float)
#
#         for i, m in enumerate(models):
#             # print(models[i])
#             matrix[field][i, :] = ma.masked_array(
#                 variables[field][m],
#                 mask2d
#             ).compressed()
#
#     """------------------------Prepare for EOF Analysis--------------------------"""
#     # this is because not all models reported velocity magnitude
#     if {'uvelsurf', 'vvelsurf'}.issubset(set(fields)):
#         matrix['uvelsurf'] *= 31557600  # m/a from m/s
#         matrix['vvelsurf'] *= 31557600
#         matrix['velsurf'] = np.sqrt(
#             matrix['uvelsurf'] ** 2 + matrix['vvelsurf'] ** 2
#         )
#         fields.append('velsurf')
#         # put into the dictionary for easier access post-run
#         for i in range(len(models)):
#             variables['velsurf'][models[i]] = matrix['velsurf'][i, :]
#
#     if {'uvelbase', 'vvelbase'}.issubset(set(fields)):
#         matrix['uvelbase'] *= 31557600  # m/a from m/s
#         matrix['vvelbase'] *= 31557600
#         matrix['velbase'] = np.sqrt(
#             matrix['uvelbase'] ** 2 + matrix['vvelbase'] ** 2
#             )
#         fields.append('velbase')
#         # put into the dictionary for easier access post-run
#         for i in range(len(models)):
#             variables['velbase'][models[i]] = matrix['velbase'][i, :]
#
#     # scale the matrices
#     anomaly_matrix = defaultdict(partial(np.ndarray, 0))
#     scaled_matrix = defaultdict(partial(np.ndarray, 0))
#
#     # eventually needs to include scales for all variables I use
#     # scale = {'lithk': 500, 'orog': 500, 'uvelsurf': 1000, 'vvelsurf': 1000, 'velsurf': 1000,
#     #         'acabf': 1, 'dlithkdt': 1, 'litempsnic': 1, 'litempbot': 1, 'strbasemag': 1}
#
#     for field in eof_fields:
#         # TODO 8/3 URGENT: why is normalization factor derived from models
#         #   instead of observational data?
#         std = np.std(matrix[field], axis=0) # normalizing factors
#         matrix0 = matrix[field][:, std == 0]
#         xclean0 = xclean[std == 0]
#         yclean0 = yclean[std == 0]
#
#         anomaly_matrix[field] = matrix[field] - np.mean(matrix[field], axis=0)
#         scaled_matrix[field] = anomaly_matrix[field] / std
#
#     """------------------------------EOF Analysis--------------------------------"""
#     # create the matrix with all desired fields
#     M = np.hstack([scaled_matrix[field] for field in eof_fields])
#
#     # truncate the SVD because very many columns (one per point, ~ 60 K)
#     # results are the same as with full matrices but much faster
#     UU, svals, VT = svd(M, full_matrices=False)
#
#     # fraction of variance explained
#     var_frac = svals / np.sum(svals)
#
#     # cumulative fraction of variance
#     var_cum = np.cumsum(var_frac)
#
#     # choose how many eof (i.e., cols) to keep (arbitrary cutoff)
#     ncol = np.where(var_cum > 0.95)[0][0]
#
#     # truncate the set of eofs
#     UU = UU[:, :ncol]
#
#     """---------------------------Find Intermodel Distances----------------------"""
#     # nested dictionary easiest for lookup
#     distance = defaultdict(dict)
#
#     # matrix is useful for plotting
#     distance_matrix = np.empty((len(models), len(models)))
#
#     # find distances with euclidean norm of eof differences
#     for i in range(len(models)):
#         for j in range(len(models)):
#             distance[models[i]][models[j]] = distance_matrix[i, j] = norm(
#                 UU[i, :] - UU[j, :]
#             )
#
#     """--------------------------------Find Weights------------------------------"""
#     # set similarity radius
#     num_std = [1.0, 2.0, 3.0]
#
#     similarity_matrix = defaultdict(dict)
#     weights = defaultdict(dict)
#
#     for radius in num_std:
#         similarity_radius = radius * np.std(distance_matrix[distance_matrix != 0])
#
#         # similarity matrix
#         similarity_matrix[radius] = np.exp(
#             -(distance_matrix / similarity_radius) ** 2
#         )
#
#         # effective repetition of each model
#         effective_repetition = np.zeros((len(models), 1))
#         # formula starts with 1 +
#         effective_repetition.fill(1.0)
#         # sum over similarities to other models
#         for i in range(len(models)):
#             for j in range(len(models)):
#                 if j != i:
#                     effective_repetition[i] += similarity_matrix[radius][i, j]
#
#         # similarity weighting
#         weights[radius] = 1.0 / effective_repetition
#
#     """--------------------------------Save Arrays-------------------------------"""
#     pickle_name = f'output-{eof_fields}-{experiment}.p'
#
#     with open(files.DIR_SESS_RESULTS / pickle_name, 'wb') as f:
#         pickle.dump([distance, distance_matrix, weights], f)
#
#     # this is how to load the variables again
#     # with open(pickle_name, 'rb') as f:
#     #    distance0, distance_matrix0, weights0 = pickle.load(f)
#
#     """-----------------------------------Plot-----------------------------------"""
#     # the distance matrix is symmetric, so just show lower half
#     plotmask = np.zeros_like(distance_matrix)
#     plotmask[np.triu_indices_from(plotmask)] = True
#
#     # f1, ax1 = plt.subplots(figsize=(10, 10))
#     fig_dist.clear()
#     ax1 = fig_dist.add_subplot()
#
#     # this limit is usually ~ 1, use +/- to center colors so median is white
#     lim = 0.8 * np.max(np.abs(distance_matrix - np.median(distance_matrix)))
#     ax1 = sns.heatmap(
#         distance_matrix - np.median(distance_matrix), mask=plotmask,
#         square=True, cmap='RdBu',
#         linewidths=0.25, linecolor='white', vmin=-lim, vmax=lim,
#         ax=ax1
#     )
#
#     # line up to have model names centered on each box
#     ax1.xaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
#     ax1.xaxis.set_ticklabels(models, rotation=90)
#     ax1.yaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
#     ax1.yaxis.set_ticklabels(models, rotation=0)
#
#     # title depending on which data was used
#     fields_string = ', '.join(eof_fields)
#     fields_string = '(' + fields_string + ')'
#
#     if experiment == 'init':
#         time_string = ' at INIT'
#     else:
#         time_string = ' at ' + experiment.upper() + ' time step ' + str(step)
#
#     fig_dist.suptitle(
#         'EOF inter-model distances vs median for fields ' + fields_string + time_string
#     )
#
#     if len(miss_fields) > 0:
#         cap_miss_f = [m[0] for m in list(miss_fields.values())[0]]
#     else:
#         cap_miss_f = 'None'
#
#     if len(miss_data) > 0:
#         cap_miss_d = [m[0] for m in list(miss_data.values())[0]]
#     else:
#         cap_miss_d = 'None'
#
#     caption = f'Following models were excluded: '\
#               f'\n- models missing fields: {cap_miss_f}'\
#               f'\n- models with invalid data: {cap_miss_d}'
#     fig_dist.text(
#         x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
#     )
#
#     # make sure everything fits on the plot
#     fig_dist.tight_layout()
#
#     fname = f'distances-{fields_string}-{experiment.upper()}'
#     fig_dist.savefig(files.DIR_SESS_RESULTS / fname, bbox_inches='tight')
#
#     display(fig_dist)
#     # clear_output(wait=True)
#     # plt.show()
#
#     """---------------------------------Plot Weights-----------------------------"""
#     # f2, ax2 = plt.subplots(figsize=(12, 8))
#     fig_weight.clear()
#     ax2 = fig_weight.add_subplot()
#
#     for ix, key in enumerate(weights):
#         ax2.plot(weights[key], marker='o', linestyle='none', alpha=0.75,
#                  color=plt.cm.tab10(ix), label='R = ' + str(key) + r'$\sigma$'
#                  )
#
#     ax2.xaxis.set_ticks(np.arange(len(models)))
#     ax2.xaxis.set_ticklabels(models, rotation=90)
#     ax2.grid(which='both')
#
#     fields_string = ', '.join(eof_fields)
#     fields_string = '(' + fields_string + ')'
#
#     if experiment == 'init':
#         time_string = ' at INIT'
#     else:
#         time_string = ' at ' + experiment.upper() + ' time step ' + str(step)
#
#     fig_weight.suptitle(
#         'Weights by similarity radius R for fields ' + fields_string + time_string
#     )
#     ax2.legend()
#
#     fig_weight.text(
#         x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
#     )
#
#     # plt.tight_layout()
#     fig_weight.tight_layout()
#
#     fname = f'weights-{fields_string}-{experiment.upper()}'
#     fig_weight.savefig(files.DIR_SESS_RESULTS / fname, bbox_inches='tight')
#
#     display(fig_weight)
#     clear_output(wait=True)
#     # plt.show()
#
#     """-------------------------------------MDS----------------------------------"""
#     # # multidimensional scaling makes a 2d scatter plot from distances
#     # mds_model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
#     # out = mds_model.fit_transform(distance_matrix)
#     #
#     # plt.figure(figsize=(10,10))
#     # plt.scatter(out[:,0], out[:,1])
#     # for i in range(len(models))    :
#     #    plt.annotate(models[i], (out[i,0], out[i,1]), fontweight=16)
#     #
#     # # title depending on which data was used
#     # fields_string = ', '.join(eof_fields)
#     # plt.title('Multi-dimensional scaling of inter-model distances, fields: ' + fields_string)
#     #
#     # plt.show()
#     #
#     #
#     #
#     #
#     #
#     #
#     #
#     # # notes for other plots
#     # # to plot any 2d variable, plt.imshow(np.flipud(bar), aspect='equal', interpolation='none') works well
#     # # for plt.contourf(), first follow the griddata and mask procedure as for Tmean above


if __name__ == '__main__':
    paths = [
        PosixPath('data/models/ILTSPIK-SICOPOLIS/ctrl/vvelsurf_GIS_ILTSPIK_SICOPOLIS_ctrl.nc'),
        PosixPath('data/models/ILTSPIK-SICOPOLIS/ctrl/uvelsurf_GIS_ILTSPIK_SICOPOLIS_ctrl.nc'),
        PosixPath('data/models/ILTS-SICOPOLIS/ctrl/vvelsurf_GIS_ILTS_SICOPOLIS_ctrl.nc'),
        PosixPath('data/models/ILTS-SICOPOLIS/ctrl/uvelsurf_GIS_ILTS_SICOPOLIS_ctrl.nc'),
        PosixPath('data/models/JPL-ISSM/ctrl/vvelsurf_GIS_JPL_ISSM_ctrl.nc'),
        PosixPath('data/models/JPL-ISSM/ctrl/uvelsurf_GIS_JPL_ISSM_ctrl.nc'),
        PosixPath('data/models/DMI-PISM2/uvelsurf_GIS_DMI_PISM2_ctrl.nc'),
        PosixPath('data/models/DMI-PISM2/vvelsurf_GIS_DMI_PISM2_ctrl.nc'),
        PosixPath('data/models/DMI-PISM2/ctrl/uvelsurf_GIS_DMI_PISM2_ctrl.nc'),
        PosixPath('data/models/DMI-PISM2/ctrl/vvelsurf_GIS_DMI_PISM2_ctrl.nc'),
        PosixPath('data/models/VUB-GISM2/ctrl/vvelsurf_GIS_VUB_GISM2_ctrl.nc'),
        PosixPath('data/models/VUB-GISM2/ctrl/uvelsurf_GIS_VUB_GISM2_ctrl.nc'),
        PosixPath('data/models/DMI-PISM5/uvelsurf_GIS_DMI_PISM5_ctrl.nc'),
        PosixPath('data/models/DMI-PISM5/vvelsurf_GIS_DMI_PISM5_ctrl.nc'),
        PosixPath('data/models/DMI-PISM5/ctrl/uvelsurf_GIS_DMI_PISM5_ctrl.nc'),
        PosixPath('data/models/DMI-PISM5/ctrl/vvelsurf_GIS_DMI_PISM5_ctrl.nc'),
        PosixPath('data/models/IGE-ELMER1/ctrl/vvelsurf_GIS_IGE_ELMER1_ctrl.nc'),
        PosixPath('data/models/IGE-ELMER1/ctrl/uvelsurf_GIS_IGE_ELMER1_ctrl.nc'),
        PosixPath('data/models/DMI-PISM1/vvelsurf_GIS_DMI_PISM1_ctrl.nc'),
        PosixPath('data/models/DMI-PISM1/uvelsurf_GIS_DMI_PISM1_ctrl.nc'),
        PosixPath('data/models/DMI-PISM1/ctrl/vvelsurf_GIS_DMI_PISM1_ctrl.nc'),
        PosixPath('data/models/DMI-PISM1/ctrl/uvelsurf_GIS_DMI_PISM1_ctrl.nc'),
        PosixPath('data/models/UAF-PISM5/ctrl/vvelsurf_GIS_UAF_PISM5_ctrl.nc'),
        PosixPath('data/models/UAF-PISM5/ctrl/uvelsurf_GIS_UAF_PISM5_ctrl.nc'),
        PosixPath('data/models/VUB-GISM1/ctrl/vvelsurf_GIS_VUB_GISM1_ctrl.nc'),
        PosixPath('data/models/VUB-GISM1/ctrl/uvelsurf_GIS_VUB_GISM1_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM2/uvelsurf_GIS_AWI_ISSM2_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM2/vvelsurf_GIS_AWI_ISSM2_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM2/ctrl/uvelsurf_GIS_AWI_ISSM2_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM2/ctrl/vvelsurf_GIS_AWI_ISSM2_ctrl.nc'),
        PosixPath('data/models/UAF-PISM3/ctrl/uvelsurf_GIS_UAF_PISM3_ctrl.nc'),
        PosixPath('data/models/UAF-PISM3/ctrl/vvelsurf_GIS_UAF_PISM3_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM1/uvelsurf_GIS_AWI_ISSM1_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM1/vvelsurf_GIS_AWI_ISSM1_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM1/ctrl/uvelsurf_GIS_AWI_ISSM1_ctrl.nc'),
        PosixPath('data/models/AWI-ISSM1/ctrl/vvelsurf_GIS_AWI_ISSM1_ctrl.nc'),
        PosixPath('data/models/MIROC-ICIES2/ctrl/vvelsurf_GIS_MIROC_ICIES2_ctrl.nc'),
        PosixPath('data/models/MIROC-ICIES2/ctrl/uvelsurf_GIS_MIROC_ICIES2_ctrl.nc'),
        PosixPath('data/models/UAF-PISM2/ctrl/vvelsurf_GIS_UAF_PISM2_ctrl.nc'),
        PosixPath('data/models/UAF-PISM2/ctrl/uvelsurf_GIS_UAF_PISM2_ctrl.nc'),
        PosixPath('data/models/MPIM-PISM/ctrl/vvelsurf_GIS_MPIM_PISM_ctrl.nc'),
        PosixPath('data/models/MPIM-PISM/ctrl/uvelsurf_GIS_MPIM_PISM_ctrl.nc'),
        PosixPath('data/models/DMI-PISM4/uvelsurf_GIS_DMI_PISM4_ctrl.nc'),
        PosixPath('data/models/DMI-PISM4/vvelsurf_GIS_DMI_PISM4_ctrl.nc'),
        PosixPath('data/models/DMI-PISM4/ctrl/uvelsurf_GIS_DMI_PISM4_ctrl.nc'),
        PosixPath('data/models/DMI-PISM4/ctrl/vvelsurf_GIS_DMI_PISM4_ctrl.nc'),
        PosixPath('data/models/IGE-ELMER2/ctrl/vvelsurf_GIS_IGE_ELMER2_ctrl.nc'),
        PosixPath('data/models/IGE-ELMER2/ctrl/uvelsurf_GIS_IGE_ELMER2_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE3/ctrl/vvelsurf_GIS_IMAU_IMAUICE3_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE3/ctrl/uvelsurf_GIS_IMAU_IMAUICE3_ctrl.nc'),
        PosixPath('data/models/UAF-PISM6/ctrl/vvelsurf_GIS_UAF_PISM6_ctrl.nc'),
        PosixPath('data/models/UAF-PISM6/ctrl/uvelsurf_GIS_UAF_PISM6_ctrl.nc'),
        PosixPath('data/models/ARC-PISM/uvelsurf_GIS_ARC_PISM_ctrl.nc'),
        PosixPath('data/models/ARC-PISM/vvelsurf_GIS_ARC_PISM_ctrl.nc'),
        PosixPath('data/models/ARC-PISM/ctrl/uvelsurf_GIS_ARC_PISM_ctrl.nc'),
        PosixPath('data/models/ARC-PISM/ctrl/vvelsurf_GIS_ARC_PISM_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE2/ctrl/uvelsurf_GIS_IMAU_IMAUICE2_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE2/ctrl/vvelsurf_GIS_IMAU_IMAUICE2_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES3/vvelsurf_GIS_BGC_BISICLES3_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES3/uvelsurf_GIS_BGC_BISICLES3_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES3/ctrl/vvelsurf_GIS_BGC_BISICLES3_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES3/ctrl/uvelsurf_GIS_BGC_BISICLES3_ctrl.nc'),
        PosixPath('data/models/LANL-CISM/ctrl/vvelsurf_GIS_LANL_CISM_ctrl.nc'),
        PosixPath('data/models/LANL-CISM/ctrl/uvelsurf_GIS_LANL_CISM_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES2/uvelsurf_GIS_BGC_BISICLES2_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES2/vvelsurf_GIS_BGC_BISICLES2_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES2/ctrl/uvelsurf_GIS_BGC_BISICLES2_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES2/ctrl/vvelsurf_GIS_BGC_BISICLES2_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE1/ctrl/uvelsurf_GIS_IMAU_IMAUICE1_ctrl.nc'),
        PosixPath('data/models/IMAU-IMAUICE1/ctrl/vvelsurf_GIS_IMAU_IMAUICE1_ctrl.nc'),
        PosixPath('data/models/MIROC-ICIES1/ctrl/vvelsurf_GIS_MIROC_ICIES1_ctrl.nc'),
        PosixPath('data/models/MIROC-ICIES1/ctrl/uvelsurf_GIS_MIROC_ICIES1_ctrl.nc'),
        PosixPath('data/models/ULB-FETISH2/ctrl/vvelsurf_GIS_ULB_FETISH2_ctrl.nc'),
        PosixPath('data/models/ULB-FETISH2/ctrl/uvelsurf_GIS_ULB_FETISH2_ctrl.nc'),
        PosixPath('data/models/LSCE-GRISLI/ctrl/uvelsurf_GIS_LSCE_GRISLI_ctrl.nc'),
        PosixPath('data/models/LSCE-GRISLI/ctrl/vvelsurf_GIS_LSCE_GRISLI_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES1/vvelsurf_GIS_BGC_BISICLES1_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES1/uvelsurf_GIS_BGC_BISICLES1_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES1/ctrl/vvelsurf_GIS_BGC_BISICLES1_ctrl.nc'),
        PosixPath('data/models/BGC-BISICLES1/ctrl/uvelsurf_GIS_BGC_BISICLES1_ctrl.nc'),
        PosixPath('data/models/UAF-PISM4/ctrl/vvelsurf_GIS_UAF_PISM4_ctrl.nc'),
        PosixPath('data/models/UAF-PISM4/ctrl/uvelsurf_GIS_UAF_PISM4_ctrl.nc'),
        PosixPath('data/models/UCIJPL-ISSM/ctrl/uvelsurf_GIS_UCIJPL_ISSM_ctrl.nc'),
        PosixPath('data/models/UCIJPL-ISSM/ctrl/vvelsurf_GIS_UCIJPL_ISSM_ctrl.nc'),
        PosixPath('data/models/DMI-PISM3/vvelsurf_GIS_DMI_PISM3_ctrl.nc'),
        PosixPath('data/models/DMI-PISM3/uvelsurf_GIS_DMI_PISM3_ctrl.nc'),
        PosixPath('data/models/DMI-PISM3/ctrl/vvelsurf_GIS_DMI_PISM3_ctrl.nc'),
        PosixPath('data/models/DMI-PISM3/ctrl/uvelsurf_GIS_DMI_PISM3_ctrl.nc'),
        PosixPath('data/models/UAF-PISM1/ctrl/uvelsurf_GIS_UAF_PISM1_ctrl.nc'),
        PosixPath('data/models/UAF-PISM1/ctrl/vvelsurf_GIS_UAF_PISM1_ctrl.nc'),
        PosixPath('data/models/ULB-FETISH1/ctrl/uvelsurf_GIS_ULB_FETISH1_ctrl.nc'),
        PosixPath('data/models/ULB-FETISH1/ctrl/vvelsurf_GIS_ULB_FETISH1_ctrl.nc')
    ]
    eof_and_plot(
        plt.Figure(figsize=(10,10)),
        plt.Figure(figsize=(10,10)),
        paths
    )