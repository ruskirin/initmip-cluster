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
from numpy.linalg import svd, norm
from collections import defaultdict
import seaborn as sns
import yaml
import pickle

# from scipy.interpolate import griddata
from ReadFinalModule import ReadFinal
from ghub_utils import files, widgets
from functools import partial

# from sklearn.manifold import MDS


"""---------------------------------Model List-------------------------------"""
with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
    y = yaml.safe_load(f)

    models = list(y['model_paths'].keys())
    # possible model exclusions
    exclude = y['exclude']

models_exclude = []
# models_exclude = exclude['thermal']
# models_exclude = exclude['libmassbf']
# models_exclude = exclude['thermal'] + exclude['ghf']
# models_exclude = exclude['dhdt']

# set difference is OK because no duplicate items
models = sorted(list(set(models).difference(models_exclude)))
"""------------------------------Read netCDF Files--------------------------"""
# fields to read
# thermal = ['litempsnic', 'litempbot']
# strbasemag is missing somewhere in asmb and ctrl runs
dynamic_read = ['uvelsurf', 'vvelsurf', 'uvelbase', 'vvelbase', 'strbasemag']
dynamic_eof = ['velsurf', 'velbase', 'strbasemag']
# mass = ['lithk', 'orog', 'acabf']
#
# fields = dynamic_read + ['orog', 'acabf']
# eof_fields = dynamic_eof + ['orog', 'acabf']
# fields = dynamic_read + ['orog', 'dlithkdt']
# eof_fields = dynamic_eof + ['orog', 'dlithkdt']
# fields = dynamic_read + ['orog', 'dlithkdt', 'acabf']
# eof_fields = dynamic_eof + ['orog', 'dlithkdt', 'acabf']
# fields = dynamic_read + ['orog']# + ['lithk'] #['orog']
# eof_fields = dynamic_eof + ['orog'] # + ['lithk'] # + ['orog']
# fields = dynamic_read
# eof_fields = dynamic_eof
#
#
# fields = ['uvelsurf', 'vvelsurf', 'orog']
# eof_fields = ['velsurf', 'orog']
#
# fields = ['uvelsurf', 'vvelsurf', 'orog', 'acabf']
# eof_fields = ['velsurf', 'orog', 'acabf']
#
# use dhdt_exclude
# fields = ['uvelsurf', 'vvelsurf', 'orog', 'acabf', 'dlithkdt']
# eof_fields = ['velsurf', 'orog', 'acabf', 'dlithkdt']
#
# # use thermal_exclude
# fields = ['uvelsurf', 'vvelsurf', 'orog', 'acabf', 'litempsnic']
# eof_fields = ['velsurf', 'orog', 'acabf', 'litempsnic']
#
# fields = ['uvelbase', 'vvelbase', 'strbasemag']
# eof_fields = ['velbase', 'strbasemag']
#
# # use thermal_exclude
# fields = ['uvelbase', 'vvelbase', 'strbasemag', 'litempbot']
# eof_fields = ['velbase', 'strbasemag', 'litempbot']
#
# # use libmassbf_exclude
# fields = ['libmassbf']
# eof_fields = ['libmassbf']
# fields = ['uvelbase', 'vvelbase', 'strbasemag', 'libmassbf']
# eof_fields = ['velbase', 'strbasemag', 'libmassbf']
#
# # use thermal_exclude and libmassbf_exclude
# fields = ['uvelbase', 'vvelbase', 'strbasemag', 'litempbot', 'libmassbf']
# eof_fields = ['velbase', 'strbasemag', 'litempbot', 'libmassbf']
#
# # use libmassbf_exclude, dhdt_exclude
# fields = ['acabf', 'libmassbf', 'dlithkdt']
# eof_fields = ['acabf', 'libmassbf', 'dlithkdt']
#
# # use dhdt_exclude
# fields = ['acabf', 'dlithkdt']
# eof_fields = ['acabf', 'dlithkdt']
#
# # use thermal_exclude, bhf_exclude
# fields = ['litempsnic', 'litempbot', 'hfgeoubed']
# eof_fields = ['litempsnic', 'litempbot', 'hfgeoubed']
#
# # for all
# fields = ['lithk', 'uvelsurf', 'vvelsurf', 'acabf', 'uvelbase', 'vvelbase', 'strbasemag']
# eof_fields = ['lithk', 'velsurf', 'acabf', 'velbase', 'strbasemag']


# fields = ['orog']
# fields = ['uvelsurf', 'vvelsurf', 'lithk']
fields = ['lithk']
# fields = ['uvelsurf', 'vvelsurf']
# fields = ['uvelbase', 'vvelbase']

# fields = ['acabf']
# fields = ['dlithkdt']
# fields = ['litempsnic']
# fields = ['strbasemag']
# fields = ['topg']
# fields = ['libmassbf']
# fields to analyze
# # note that velsurf will be created later from uvelsurf, vvelsurf
# eof_fields = ['velsurf', 'lithk']
# eof_fields = ['orog']
eof_fields = ['lithk']
# eof_fields = ['velsurf']
# eof_fields = ['velbase']
# eof_fields = ['topg']
# eof_fields = ['libmassbf']
# eof_fields = ['acabf']
# eof_fields = ['dlithkdt']
# eof_fields = ['litempsnic']
# eof_fields = ['strbasemag']

# experiment = 'init'
experiment = 'ctrl'
# experiment = 'asmb'
# size of the first dimension of the data
step = 21  # only applies when experiment is ctrl or asmb

# variables will be a 2-level dict with field, model as keys
variables = defaultdict(dict)

for field in fields:
    X, Y, variables[field] = ReadFinal(
        files.DIR_SESS_DATA / 'models', experiment, field, step, models
    )

"""-------------------------------Mask Model Outputs-------------------------"""
mask2d = np.full(X.shape, False)

for model in models:
    # masked points should be True
    # due to some nan's, can't assume all fields have the same mask
    for i in range(0, len(fields)):
        mask_nan = np.isnan(variables[fields[i]][model])
        mask_missing = variables[fields[i]][model] > 9e36
        mask2d = mask2d | mask_nan | mask_missing

"""-------------------------------Use the Mask------------------------------"""
# meshgrid to coordinate lists
x = ma.masked_array(X, mask2d).compressed()
y = ma.masked_array(Y, mask2d).compressed()

# this will become a dict with variables as keys and one matrix per variable as values
matrix = defaultdict(partial(np.ndarray, 0))

for field in fields:
    matrix[field] = np.empty((len(models), len(x)), float)

    for i in range(0, len(models)):
        # print(models[i])
        matrix[field][i, :] = ma.masked_array(
            variables[field][models[i]],
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
    anomaly_matrix[field] = matrix[field] - np.mean(matrix[field], axis=0)
    scaled_matrix[field] = anomaly_matrix[field] / np.std(matrix[field], axis=0)

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

with open(files.DIR_SESS_RESULTS / pickle_name, 'wb') as f:
    pickle.dump([distance, distance_matrix, weights], f)

# this is how to load the variables again
# with open(pickle_name, 'rb') as f:
#    distance0, distance_matrix0, weights0 = pickle.load(f)

"""-----------------------------------Plot-----------------------------------"""
# the distance matrix is symmetric, so just show lower half
plotmask = np.zeros_like(distance_matrix)
plotmask[np.triu_indices_from(plotmask)] = True

plt.figure(figsize=(10, 10))

# this limit is usually ~ 1, use +/- to center colors so median is white
lim = 0.8 * np.max(np.abs(distance_matrix - np.median(distance_matrix)))
ax = sns.heatmap(distance_matrix - np.median(distance_matrix), mask=plotmask,
                 square=True, cmap='RdBu',
                 linewidths=0.25, linecolor='white', vmin=-lim, vmax=lim
                 )

# line up to have model names centered on each box
ax.set_xticks(np.arange(0.5, len(models) + 0.5, 1))
ax.set_xticklabels(models, rotation=90)
ax.set_yticks(np.arange(0.5, len(models) + 0.5, 1))
ax.set_yticklabels(models, rotation=0)

# title depending on which data was used
fields_string = ', '.join(eof_fields)
fields_string = '(' + fields_string + ')'

if experiment == 'init':
    time_string = ' at INIT'
else:
    time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

plt.title(
    'EOF inter-model distances vs median for fields ' + fields_string + time_string
    )
# make sure everything fits on the plot
plt.tight_layout()

fname = f'distances-{fields_string}-{experiment.upper()}'
plt.savefig(files.DIR_SESS_RESULTS / fname)

plt.show()

"""---------------------------------Plot Weights-----------------------------"""
plt.figure(figsize=(12, 8))

for ix, key in enumerate(weights):
    plt.plot(weights[key], marker='o', linestyle='none', alpha=0.75,
             color=plt.cm.tab10(ix), label='R = ' + str(key) + r'$\sigma$'
             )

plt.xticks(np.arange(len(models)), models, rotation=90)
plt.grid(which='both')

fields_string = ', '.join(eof_fields)
fields_string = '(' + fields_string + ')'

if experiment == 'init':
    time_string = ' at INIT'
else:
    time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

plt.title(
    'Weights by similarity radius R for fields ' + fields_string + time_string
    )
plt.legend(loc='best')
plt.tight_layout()

fname = f'weights-{fields_string}-{experiment.upper()}'
plt.savefig(files.DIR_SESS_RESULTS / fname)

plt.show()

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


if __name__ == '__main__':
    pass
