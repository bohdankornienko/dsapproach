import os
import argparse
import logging
import shutil

from glob import glob

logging.basicConfig(level=logging.INFO)


def clean_exp_dir(exp_dir):
    logging.info('Proceed to cleaning {}'.format(exp_dir))

    directories = [f for f in glob(exp_dir + '/*')]

    dirs_removed = 0

    for directory in directories:
        is_dir = os.path.isdir(directory)
        is_not_complete = not os.path.exists(os.path.join(directory, 'flag.complete'))

        to_remove = False

        if is_dir and is_not_complete:
            shutil.rmtree(directory)
            to_remove = True
            dirs_removed += 1

        logging.info('Experiment {}, to remove => {}'.format(os.path.split(directory)[1], to_remove))

    logging.info('Cleaning complete.')
    logging.info('Directories removed: {}'.format(dirs_removed))


if __name__ == "__main__":
    description = """
    The script will clean the directory from experiments which was not finished completely.
    The script has to have permission to write into the directory in order to successfully clean up.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--exp_dir', help='The path to the directory with experiments.')

    args = parser.parse_args()


    clean_exp_dir(args.exp_dir)
