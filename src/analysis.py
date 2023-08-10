import yaml
import regex as re
from pathlib import Path
from collections import defaultdict, namedtuple
from typing import List
import traceback
import numpy as np

from ghub_utils import files


with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
    y = yaml.safe_load(f)

    MODELS = list(y['model_paths'].keys())
    # possible model exclusions
    EXCLUDE = y['exclude']
    FIELD_PAIRS = y['field_pairs']
    EXPS = y['experiments']
    STEPS = y['steps']

    RUN_FIELDS = y['run_fields']


def get_field_pairs(eof_fields: List[str]):
    """
    Return a list of corresponding fields that make the @eof_fields
    :param eof_fields: list of eof fields
    """
    eof_fields = tuple(eof_fields)
    fields = []

    for eof in eof_fields:
        field = FIELD_PAIRS[eof]
        if isinstance(field, list):
            fields.extend(field)
        else:
            fields.append(field)

    return fields


def format_data_exc(e: BaseException):
    """Process exceptions during data reading and put into a legible format"""
    Exc = namedtuple('Exc', 'name full simple')

    if isinstance(e, FileNotFoundError):
        fname_pat = r'No such file .+: b\'(.+.nc)\'' # filename regex pattern

        exc = traceback.format_exc(limit=0)
        file = re.search(fname_pat, exc).group(1)
        file = Path(file)
        fname = file.name

        outcome = Exc(name=e.__class__.__name__, full=exc, simple=fname)
    else:
        exc = traceback.format_exc()
        outcome = Exc(name=e.__class__.__name__, full=exc, simple=None)

    return outcome


def test_valid_data(models, fields, variables):
    """Test for invalid data in @models"""
    invalid = defaultdict(list) # MINE

    for field in fields:
        for m in models:
            uniq = np.unique(variables[field][m].selection)
            num_uniq = len(uniq)
            if num_uniq <= 2:
                # data is invalid and only has nans and/or mask values
                invalid[field].append((m, ValueError))

    return invalid


if __name__ == '__main__':
    print(FIELD_PAIRS)
    for fields in RUN_FIELDS:
        print(f'fields: {fields}')
        f = get_field_pairs(fields)
        print(f'pairs: {f}')