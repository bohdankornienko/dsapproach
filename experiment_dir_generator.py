# Greek gods
# Planets +
# Animals
# Foods and meals

from datetime import datetime
from glob import glob
import numpy as np
import os


def make_exp_dir(exp_dir):
    existing_names = [f.split('_')[-1] for f in glob(exp_dir + '/*')]

    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    path_to_planets = os.path.join(os.path.dirname(__file__), 'planets.txt')
    with open(path_to_planets, 'r') as fp:
        planets = fp.readlines()

    planets = [planet.replace('\n', '') for planet in planets]

    names = list(set(planets) - set(existing_names))

    err = 'Oh boy! We ran out of names for experiments. Please cleanup the empty ones or expand experiment names list.'
    if len(names) == 0:
        raise NameError(err)

    name = np.random.choice(names)
    exp_name = '{}_{}'.format(prefix, name)
    full_exp_path = os.path.join(exp_dir, exp_name)

    os.mkdir(full_exp_path)

    return full_exp_path
