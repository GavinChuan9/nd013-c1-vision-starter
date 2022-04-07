import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    trainPath = os.path.join(data_dir, 'train')
    valPath = os.path.join(data_dir, 'val')
    testPath = os.path.join(data_dir, 'test')
    
    tfrecords = glob.glob(os.path.join(data_dir,"training_and_validation", "*.tfrecord"))

    # Produces a 80%, 10%, 10% split for training, validation and test sets.
    trainFiles = random.sample(tfrecords, int(len(tfrecords)*0.8))
    for trainFile in trainFiles:
        os.popen('mv %s %s'%(trainFile, trainPath))
    valAndTestFiles = list(set(tfrecords) - set(trainFiles))

    valFiles = random.sample(valAndTestFiles, int(len(valAndTestFiles)*0.5))
    for valFile in valFiles:
        os.popen('mv %s %s'%(valFile, valPath))
    testFiles = list(set(valAndTestFiles) - set(valFiles))

    for testFile in testFiles:
        os.popen('mv %s %s'%(testFile, testPath))
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)