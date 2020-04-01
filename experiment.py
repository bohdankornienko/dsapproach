import os
import logging

from datetime import datetime

from .experiment_dir_generator import make_exp_dir

logging.basicConfig(level=20)


class Experiment:
    def __init__(self, approach_factory, datagen_factory, **kwargs):
        self._sets = kwargs
        self._approach_factory = approach_factory
        self._datagen_factory = datagen_factory

        self._iteration = 0
        self._gen_train = None
        self._gen_val = None

        logging.info('-'*20)
        logging.info('Settings used for the experiment')
        logging.info(self._sets)
        logging.info('-'*20)

        self._initialize()

    def _initialize(self):
        logging.info('Experiment initialization started...')

        logging.info('Creating experiment directory...')
        self._this_exp_dir = make_exp_dir(self._sets['exp_dir'])
        logging.info('This experiment is being stored in the following directory: {}'.format(self._this_exp_dir))
        logging.info('Experiment directory has been created.')

        self._approach = self._approach_factory.create(**self._sets['approach'])

        self._gen_train = self._datagen_factory.create(**self._sets['datagen']['train'])
        self._gen_val = self._datagen_factory.create(**self._sets['datagen']['val'])

        logging.info('Experiment initialized.')

    def _finalize(self):
        logging.info('Experiment finalized.')

        # TODO: add total time for experiment
        # TODO: add last epoch, iteration (useful for aborted flag)

        with open(os.path.join(self._this_exp_dir, 'flag.complete'), 'w') as fp:
            fp.write('Experiment completed at: {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))

        # TODO: aborted flag

    def _train_loop(self):
        for e in range(self._sets['optimizer']['epochs']):
            logging.info('Epoch: {}'.format(e))

            for step in range(self._sets['optimizer']['iterations_per_epoch']):
                logging.info('Train step: [{}] {}'.format(e, step))

                x, y = self._gen_train.next_batch()
                self._approach.train_on_batch(x, y)

    def perform(self):
        self._train_loop()
        self._finalize()
