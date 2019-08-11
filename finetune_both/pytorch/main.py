import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from tqdm import tqdm
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, load_scalar, segment_prediction_to_clip_prediction, segment_feature_to_clip_feature,
	write_submission, get_feature_map)
from data_generator import DataGenerator, TestDataGenerator
from pseudo_utils import NonLinearGenerator

from models import (ArcFace_9layers, PReLULinearNoisy, PReLUTeacher, PReLUTeacherFT)

from losses import binary_cross_entropy, lovasz_binary, FocalLoss, RingLoss, ArcFaceLoss
# from evaluate import Evaluator, StatisticsContainer
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward_infer, forward_dist, Nadam, CosineLRWithRestarts
import config

DATASET_DIR = 'C:/Users/blade/Documents/kaggle/freesound/input/freesound-audio-tagging-2019'
WORKSPACE = 'C:/Users/blade/Documents/kaggle/freesound'

def train(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      holdout_fold: '1', '2', '3', '4' | 'none', set `none` for training 
          on all data without validation
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arugments & parameters

    dataset_dir = DATASET_DIR
    workspace = WORKSPACE
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    curated_proportion = args.curated_proportion
    valid_source = args.valid_source
    pretrained_dir = args.pretrained_dir
    pretrained_model = args.pretrained_model
    noisy_model = args.noisy_model
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    max_iteration = 500      # Number of mini-batches to evaluate on training data
    reduce_lr = False
    
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
                
    curated_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_curated.h5')
        
    noisy_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    curated_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_curated_cross_validation.csv')

    noisy_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_noisy_cross_validation.csv')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
    
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type)
    create_folder(checkpoints_dir)

    validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type)
    create_logging(logs_dir, 'w')

    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    ModelNoisy = eval(noisy_model)
    model_n = ModelNoisy(classes_num * 2)

    Model0 = eval(pretrained_model)
    model0 = Model0(classes_num * 2, model_n)
    model0.load_state_dict(torch.load(pretrained_dir)['model'])

    Model = eval(model_type)
    curated_size = int(batch_size * curated_proportion)
    model = Model(classes_num * 2, model0, curated_size=curated_size)
    del model0
    
    if cuda:
        model.cuda()

    # Data generator
    data_generator = NonLinearGenerator(
        curated_feature_hdf5_path=curated_feature_hdf5_path, 
        noisy_feature_hdf5_path=noisy_feature_hdf5_path, 
        curated_cross_validation_path=curated_cross_validation_path, 
        noisy_cross_validation_path=noisy_cross_validation_path,
        holdout_fold=holdout_fold, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size,
        model=model,
        curated_proportion=curated_proportion)

    # Calculate total iteration required for n_epoch
    iter_per_epoch = np.ceil(len(data_generator.train_noisy_segments_indexes) / batch_size).astype(int)
    total_iter = iter_per_epoch * n_epoch

    # Define Warm-up LR scheduler
    epoch_to_warm = 1
    epoch_to_flat = n_epoch
    def _warmup_lr(optimizer, iteration, iter_per_epoch, epoch_to_warm, min_lr=0, max_lr=0.0035):
        delta = (max_lr - min_lr) / iter_per_epoch / epoch_to_warm
        lr = min_lr + delta * iteration
        for p in optimizer.param_groups:
            p['lr'] = lr
        return lr

    # Optimizer
    criterion = FocalLoss(2)
    # metric_loss = RingLoss(type='auto', loss_weight=1.0)
    metric_loss = ArcFaceLoss()
    if cuda:
        metric_loss.cuda()
    optimizer = Nadam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, schedule_decay=4e-3)
    scheduler = CosineLRWithRestarts(optimizer, batch_size-curated_size, 
        len(data_generator.train_noisy_segments_indexes), restart_period=epoch_to_flat-epoch_to_warm+1, t_mult=1, verbose=True)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)

    # Valid source
    if valid_source == 'curated':
        target_sources = ['curated']
    elif valid_source == 'noisy':
        target_sources = ['noisy']
    elif valid_source == 'both':
        target_sources = ['curated', 'noisy']
    
    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)
    
    train_bgn_time = time.time()
    iteration = 0
    epoch = 0
    
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        if iteration % 2500 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            
            # Evaluate on partial of train data
            # logging.info('Train statistics:')
            
            # for target_source in target_sources:
            #     validate_curated_statistics = evaluator.evaluate(
            #         data_type='train', 
            #         target_source=target_source, 
            #         max_iteration=max_iteration, 
            #         verbose=False)
            
            # Evaluate on holdout validation data
            if holdout_fold != 'none':                
                logging.info('Validate statistics:')
                
                for target_source in target_sources:
                    validate_curated_statistics = evaluator.evaluate(
                        data_type='validate', 
                        target_source=target_source, 
                        max_iteration=None, 
                        verbose=False)
                        
                    validate_statistics_container.append(
                        iteration, target_source, validate_curated_statistics)
                    
                validate_statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 2500 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'mask', 'target']:
                batch_data_dict[key] = move_data_to_gpu(
                    batch_data_dict[key], cuda)
        
        # Train
        model.train()
        cosine_curated, output_curated, cosine_mixed, output_mixed = model(batch_data_dict['feature'], mode='train')
        
        # loss
        loss = criterion(output_mixed, batch_data_dict['target']) + metric_loss(cosine_mixed, batch_data_dict['target']) + \
            criterion(output_curated, batch_data_dict['target'][:curated_size]) + metric_loss(cosine_curated, batch_data_dict['target'][:curated_size])

        # Backward
        optimizer.zero_grad()
        
        # LR Warm up
        if iteration < epoch_to_warm * iter_per_epoch:
            cur_lr = _warmup_lr(optimizer, iteration, iter_per_epoch, epoch_to_warm=epoch_to_warm, min_lr=0, max_lr=0.0035)

        loss.backward()
        optimizer.step()

        if iteration >= epoch_to_warm * iter_per_epoch and iteration < epoch_to_flat * iter_per_epoch:
            if data_generator.noisy_pointer >= len(data_generator.train_noisy_segments_indexes):
                scheduler.step()
            scheduler.batch_step()

        # Show LR information
        if iteration % iter_per_epoch == 0 and iteration != 0:
            epoch += 1
            if epoch % 10 == 0:
                for p in optimizer.param_groups:
                    logging.info('Learning rate at epoch {:3d} / iteration {:5d} is: {:.6f}'.format(epoch, iteration, p['lr']))

        # Stop learning
        if iteration == total_iter:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            
            # Evaluate on holdout validation data
            if holdout_fold != 'none':                
                logging.info('Validate statistics:')
                
                for target_source in target_sources:
                    validate_curated_statistics = evaluator.evaluate(
                        data_type='validate', 
                        target_source=target_source, 
                        max_iteration=None, 
                        verbose=False)
                        
                    validate_statistics_container.append(
                        iteration, target_source, validate_curated_statistics)
                    
                validate_statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

            break
            
        iteration += 1

        if iteration == epoch_to_warm * iter_per_epoch:
            scheduler.step()

        if iteration == epoch_to_flat * iter_per_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5


def inference_validation(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      holdout_fold: '1', '2', '3', '4'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      iteration: int, load model of this iteration
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool, visualize the logmel spectrogram of segments
    '''
    
    # Arugments & parameters
    dataset_dir = DATASET_DIR
    workspace = WORKSPACE
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    iteration = args.iteration
    batch_size = args.batch_size
    resume = args.resume
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename
    
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
          
    curated_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_curated.h5')
        
    noisy_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    curated_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_curated_cross_validation.csv')

    noisy_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_noisy_cross_validation.csv')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')

    if not resume:
        checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, '{}_iterations.pth'.format(iteration))
    else:
        checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, 'resume', '{}_iterations.pth'.format(iteration))
        
    figs_dir = os.path.join(workspace, 'figures')
    create_folder(figs_dir)
        
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
        ''.format(segment_seconds, hop_seconds, pad_type), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    ModelNoisy = eval(noisy_model)
    model_n = ModelNoisy(classes_num * 2)

    Model0 = eval(pretrained_model)
    model0 = Model0(classes_num * 2, model_n)
    model0.load_state_dict(torch.load(pretrained_dir)['model'])

    Model = eval(model_type)
    curated_size = int(batch_size * curated_proportion)
    model = Model(classes_num * 2, model0, curated_size=curated_size)
    del model0, model_n
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        curated_feature_hdf5_path=curated_feature_hdf5_path, 
        noisy_feature_hdf5_path=noisy_feature_hdf5_path, 
        curated_cross_validation_path=curated_cross_validation_path, 
        noisy_cross_validation_path=noisy_cross_validation_path, 
        train_source=train_source, 
        holdout_fold=holdout_fold, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)

    # Evaluate
    for target_source in ['curated', 'noisy']:
        validate_curated_statistics = evaluator.evaluate(
            data_type='validate', 
            target_source=target_source, 
            max_iteration=None, 
            verbose=True)
        
        # Visualize
        if visualize:
            save_fig_path = os.path.join(figs_dir, 
                '{}_logmel.png'.format(target_source))
            
            validate_curated_statistics = evaluator.visualize(
                data_type='validate', 
                target_source=target_source, 
                save_fig_path=save_fig_path, 
                max_iteration=None, 
                verbose=False)
        
        
def inference_test(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      iteration: int, load model of this iteration
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool, visualize the logmel spectrogram of segments
    '''
    
    # Arugments & parameters
    dataset_dir = DATASET_DIR
    workspace = WORKSPACE
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    model_type = args.model_type
    pretrained_model = args.pretrained_model
    noisy_model = args.noisy_model
    iteration = args.iteration
    batch_size = args.batch_size
    resume = args.resume
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename

    holdout_fold = args.holdout_fold   # Use model trained on full data without validation
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    test_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'test.h5')

    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
    
    if not resume:
        checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, '{}_iterations.pth'.format(iteration))
            
        submission_path = os.path.join(workspace, 'submissions', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, '{}_iterations_submission.csv'
            ''.format(iteration))
        create_folder(os.path.dirname(submission_path))
    else:
        checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, 'resume', '{}_iterations.pth'.format(iteration))
            
        submission_path = os.path.join(workspace, 'submissions', filename, 
            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
            ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
            ''.format(holdout_fold), model_type, 'resume', '{}_iterations_submission.csv'
            ''.format(iteration))
        create_folder(os.path.dirname(submission_path))
        
    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    ModelNoisy = eval(noisy_model)
    model_n = ModelNoisy(classes_num * 2)

    Model0 = eval(pretrained_model)
    model0 = Model0(classes_num * 2, model_n)

    Model = eval(model_type)
    model = Model(classes_num * 2, model0, curated_size=None)
    del model0, model_n
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = TestDataGenerator(
        test_feature_hdf5_path=test_feature_hdf5_path, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size)
        
    generate_func = data_generator.generate_test()
    
    # Results of segments
    output_dict = forward_infer(
        model=model, 
        generate_func=generate_func, 
        cuda=cuda)
    
    # Results of audio recordings
    result_dict = segment_prediction_to_clip_prediction(
        output_dict, average='arithmetic')
    
    # Write submission
    write_submission(result_dict, submission_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    # parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    # parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_train.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_train.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_train.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', 'none'], required=True, help='Set `none` for training on all data without validation.')
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--n_epoch', type=int, required=True, help='E.g., 50')
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--curated_proportion', type=float, required=True)
    parser_train.add_argument('--valid_source', type=str, choices=['curated', 'noisy', 'both'], required=True, help='Set data evaluated during training phase')
    parser_train.add_argument('--pretrained_dir', type=str, required=True, help='Resume training from given pretrained model')
    parser_train.add_argument('--pretrained_model', type=str, required=True, help='model type of pretrained model')
    parser_train.add_argument('--noisy_model', type=str, required=True, help='model type of noisy model')
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Inference validation data
    parser_inference_validation = subparsers.add_parser('inference_validation')
    # parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    # parser_inference_validation.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_validation.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_inference_validation.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_inference_validation.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_inference_validation.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', 'none'], required=True, help='Set `none` for training on all data without validation.')
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--pretrained_model', type=str, required=True, help='model type of pretrained model')
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--resume', action='store_true', default=False, help='Sub folder under model_type')
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False, help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Inference test data
    parser_inference_validation = subparsers.add_parser('inference_test')
    # parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    # parser_inference_validation.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_validation.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_inference_validation.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_inference_validation.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_inference_validation.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', 'none'], required=True, help='Set `none` for training on all data without validation.')
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--pretrained_model', type=str, required=True, help='model type of pretrained model')
    parser_inference_validation.add_argument('--noisy_model', type=str, required=True, help='model type of noisy model')
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--resume', action='store_true', default=False, help='Sub folder under model_type')
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'get_train_features':
        get_train_features(args)

    elif args.mode == 'get_infer_features':
        get_infer_features(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_test':
        inference_test(args)

    else:
        raise Exception('Error argument!')