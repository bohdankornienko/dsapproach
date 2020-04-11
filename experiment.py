import os
import logging
import time
import yaml

from datetime import datetime

from mscocosol.utils.general import pretty_dict_as_string
from .experiment_dir_generator import make_exp_dir

# TODO: figure out it later
from torch.utils.data import DataLoader

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
        logging.info((pretty_dict_as_string(self._sets)))
        logging.info('-'*20)

        self._initialize()

    def _initialize(self):
        logging.info('Experiment initialization started...')

        logging.info('Creating experiment directory...')
        self._this_exp_dir = make_exp_dir(self._sets['exp_dir'])
        logging.info('This experiment is being stored in the following directory: {}'.format(self._this_exp_dir))
        logging.info('Experiment directory has been created.')

        with open(os.path.join(self._this_exp_dir, 'sets.yaml'), 'w') as fp:
            yaml.dump(self._sets, fp)

        self._sets['approach']['this_exp_dir'] = self._this_exp_dir

        self._approach = self._approach_factory.create(**self._sets['approach'])

        logging.info('Initialize data generators...')
        self._gen_train = self._datagen_factory.create(**self._sets['datagen']['train'])
        logging.info('Train data get summary: {}'.format(self._gen_train.get_summary()))
        self._gen_val = self._datagen_factory.create(**self._sets['datagen']['val'])
        logging.info('Validation data get summary: {}'.format(self._gen_val.get_summary()))
        logging.info('Initialization complete.')

        self._data_loaders = {
            'train': DataLoader(self._gen_train, batch_size=self._gen_train.batch_size, shuffle=True, num_workers=3),
            'val': DataLoader(self._gen_val, batch_size=self._gen_val.batch_size, shuffle=True, num_workers=3)
        }

        logging.info('Experiment initialized.')

    def _finalize(self):
        logging.info('Experiment finalized.')

        # TODO: time when experiment started
        # TODO: add total time for experiment
        # TODO: add last epoch, iteration (useful for aborted flag)

        with open(os.path.join(self._this_exp_dir, 'flag.complete'), 'w') as fp:
            fp.write('Experiment completed at: {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))

        # TODO: aborted flag file to the experiment directory

    def _train_loop(self):
        step = 0
        stop = False

        for epoch in range(1, self._sets['approach']['optimizer']['epochs'] + 1):
            if stop:
                break
            logging.info('Epoch: {}'.format(epoch))

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if stop:
                    break

                self._approach.set_mode(phase)

                for inputs, labels in self._data_loaders[phase]:
                    step += 1
                    logging.info('Train step: [{}] {}'.format(epoch, step))
                    if step == self._sets['approach']['optimizer']['stopping_step']:
                        stop = True
                        break
                    readings = self._approach.train_on_batch(inputs, labels)
                    # readings = '\n' + readings + '\n'
                    logging.info(readings)

                if phase == 'val':
                    self._approach.evaluate()
                    eval_res = self._approach.get_last_evaluation()

                    logging.info('-' * 20)
                    logging.info('Evaluation results for epoch: {}'.format(epoch))
                    logging.info(eval_res)
                    logging.info('-' * 20)

                    self._approach.save_checkpoint(epoch)

            time_elapsed = time.time() - since
            logging.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def perform(self):
        self._train_loop()
        self._finalize()
