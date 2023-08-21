from typing import List
from pathlib import Path
import regex as re
from enum import Enum
from collections import namedtuple, defaultdict

from ghub_utils.types import FileType
from ghub_utils import files as gfiles


class FileParams(Enum):
    """netCDF file parameters"""
    MODEL = 'model'
    EXP = 'exp'
    FIELD = 'field'


def filter_paths_terms(paths: List[Path], search: list, param: FileParams) -> list:
    """
    Search for a list of terms @search in file names @paths
    :param paths:
    :param search: list of terms to search for
    :param param: parameter type of @search
    :return:
    """
    if param == FileParams.MODEL:
        search = map(lambda x: fr'_{x.replace("-", "_")}_', search)
    elif param == FileParams.EXP:
        search = map(lambda x: fr'_{x}.nc', search)
    elif param == FileParams.FIELD:
        search = map(lambda x: fr'\b{x}_', search)
    else:
        raise ValueError(f'Bad input for @param: {param}')

    search_str = '|'.join(search)

    filtered = []
    for p in paths:
        mat = re.search(search_str, p.name)
        if mat is not None:
            filtered.append(gfiles.get_path_relative_to(p, gfiles.DIR_PROJECT))

    return filtered


def intersect_netcdf_model_params(paths: List[Path]) -> namedtuple:
    """
    Extract all unique fields, experiments, and models found in .nc filenames

    :param paths: list of netCDF4 file paths;
      NOTE: must follow pattern: <field>_GIS_<modelA>_<modelB>_<experiment>.nc
    :return: namedtuple of sets of models, experiments, and fields
    """
    pat_file_params = r'(?P<field>\w+)_GIS_(?P<model>\w+_\w+)_(?P<exp>\w+).nc'

    fields = defaultdict(set)
    for p in paths:
        if p.suffix != '.nc':
            continue

        mat = re.search(pat_file_params, p.name)
        if mat is not None:
            model = mat.group('model').replace('_', '-')
            fields[mat.group('field')].add(model)

    models = set.intersection(*fields.values())
    return models


def union_netcdf_params(paths: List[Path]) -> namedtuple:
    """
    Extract all unique fields, experiments, and models found in .nc filenames

    :param paths: list of netCDF4 file paths;
      NOTE: must follow pattern: <field>_GIS_<modelA>_<modelB>_<experiment>.nc
    :return: namedtuple of sets of models, experiments, and fields
    """
    pat_file_params = r'(?P<field>\w+)_GIS_(?P<model>\w+_\w+)_(?P<exp>\w+).nc'

    models = set()
    exps = set()
    fields = set()

    for p in paths:
        if p.suffix != '.nc':
            continue

        mat = re.search(pat_file_params, p.name)
        if mat is not None:
            model = mat.group('model').replace('_', '-')

            models.add(model)
            exps.add(mat.group('exp'))
            fields.add(mat.group('field'))

    Params = namedtuple('Params', 'models exps fields')
    return Params(models, exps, fields)


def get_dirs_union(
        dirs: List[Path],
        ftype: FileType = None,
        regex: str = None
) -> list:
    """
    TODO 8/14 (1): a lot of repetition in code

    Get the union of directory elements in @dirs;
      optionally use regular expression @regex to group by part of filenames
    """
    elems_all = []

    for d in dirs:
        if not d.is_dir():
            continue

        elems = set()

        for elem in d.iterdir():
            if ftype == FileType.DIR:
                if not elem.is_dir():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            elif ftype == FileType.FILE:
                if not elem.is_file():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            else:
                raise NotImplementedError('Not implemented')

        elems_all.append(elems)

    return sorted(set.union(*elems_all))


def get_dirs_intersect(
        dirs: List[Path],
        ftype: FileType = None,
        regex: str = None
) -> list:
    """
    # TODO 8/14 (2): a lot of code repetition
    Get the union of directory elements in @dirs;
      optionally use regular expression @regex to group by part of filenames
    """
    elems_all = []

    for d in dirs:
        if not d.is_dir():
            continue

        elems = set()

        for elem in d.iterdir():
            if ftype == FileType.DIR:
                if not elem.is_dir():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            elif ftype == FileType.FILE:
                if not elem.is_file():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            else:
                raise NotImplementedError('Not implemented')

        elems_all.append(elems)

    return sorted(set.intersection(*elems_all))


if __name__ == '__main__':
    model_path = gfiles.DIR_SAMPLE_DATA / 'models'
    files = list(model_path.rglob('*.nc'))

    filtered = filter_paths_terms(files, ['lithk', 'acabf'], FileParams.FIELD)
    [print(f) for f in filtered]