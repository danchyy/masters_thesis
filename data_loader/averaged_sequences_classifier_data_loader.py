from base.base_data_loader import BaseDataLoader
from utils import constants
import os
from data_generators.sequence_data_generator import SequenceDataGenerator


class AveragedSequencesClassifierDataLoader(BaseDataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.trainer.batch_size
        self.train_split = self.config.trainer.train_split
        self.test_split = self.config.trainer.test_split

    def get_train_data(self):
        return SequenceDataGenerator(self.batch_size, self.train_split, self.config.exp.num_of_classes,
                                     should_average=True, should_subsample=False)

    def get_test_data(self):
        return SequenceDataGenerator(self.batch_size, self.test_split, self.config.exp.num_of_classes,
                                     should_average=True, should_subsample=False)
