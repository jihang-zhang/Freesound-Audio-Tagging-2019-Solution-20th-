import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging
import pandas as pd
import random

from utilities import scale
import config


def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.3, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(round(freq_percentage * spec.shape[1]))
            f0 = np.random.uniform(low=0.0, high=spec.shape[1] - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_frames_to_mask = int(round(time_percentage * spec.shape[0]))
            t0 = np.random.uniform(low=0.0, high=spec.shape[0] - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0

        return spec

class Base(object):
    def __init__(self):
        '''Base class for train, validate and test data generator. 
        '''
        pass
        
    def load_hdf5(self, hdf5_path, cross_validation_path):
        '''Load hdf5 file. 
        
        Args:
          hdf5_path: string, path of hdf5 file
          cross_validation_path, string | 'none', path of cross validation csv 
              file
          
        Returns:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'target': (audios_num, classes_num), 
                      (if exist) 'fold': (audios_num,)}
        '''
        
        data_dict = {}
        
        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature'] = hf['feature'][:].astype(np.float32)
            data_dict['begin_index'] = hf['begin_index'][:].astype(np.int32)
            data_dict['end_index'] = hf['end_index'][:].astype(np.int32)
            
            if 'target' in hf.keys():
                data_dict['target'] = hf['target'][:].astype(np.float32)
        
        if cross_validation_path:
            df = pd.read_csv(cross_validation_path, sep=',')    
            folds = []
            
            for n, audio_name in enumerate(data_dict['audio_name']):
                index = df.index[df['fname'] == audio_name][0]
                folds.append(df['fold'][index])
            
            data_dict['fold'] = np.array(folds)

        return data_dict


    def get_segment_metadata_dict_train(self, data_dict, audio_indexes, 
        segment_frames, hop_frames, source):
        '''Get segments metadata for training or inference. Long audio 
        recordings are split to segments with the same duration. Each segment 
        inherit the label of the audio recording. 
        
        Args:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'target': (audios_num, classes_num), 
                      (if exist) 'fold': (audios_num,)}
          audio_indexes: (audios_num,)
          segment_frames: int, frames number of a segment
          hop_frames: int, hop frames between segments
          source: 'curated' | 'noisy' | None
          
        Returns:
          segment_metadata_dict: {'audio_name': (segments_num,), 
                                  'begin_index': (segments_num,), 
                                  'end_index': (segments_num,), 
                                  (if exist) 'target': (segments_num, classes_num), 
                                  (if exist) 'source': (segments_num)}
        '''
        
        segment_metadata_dict = {'audio_name': [], 'begin_index': [], 
            'end_index': []}
            
        has_target = 'target' in data_dict.keys()

        if has_target:
            segment_metadata_dict['target'] = []
            
        if source:
            segment_metadata_dict['source'] = []
        
        for audio_index in audio_indexes:
            audio_name = data_dict['audio_name'][audio_index]
            begin_index = data_dict['begin_index'][audio_index]
            end_index = data_dict['end_index'][audio_index]
            
            if has_target:
                target = data_dict['target'][audio_index]
            else:
                target = None
            
            # We random sample frames from the whole segments in DataGenerator
            segment_metadata_dict['begin_index'].append(begin_index)
            segment_metadata_dict['end_index'].append(end_index)
            
            self._append_to_meta_data(segment_metadata_dict, audio_name, 
                target, source)
                
            
        for key in segment_metadata_dict.keys():
            segment_metadata_dict[key] = np.array(segment_metadata_dict[key])
        
        return segment_metadata_dict

        
    def get_segment_metadata_dict_infer(self, data_dict, audio_indexes, 
        segment_frames, hop_frames, source):
        '''Get segments metadata for training or inference. Long audio 
        recordings are split to segments with the same duration. Each segment 
        inherit the label of the audio recording. 
        
        Args:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'target': (audios_num, classes_num), 
                      (if exist) 'fold': (audios_num,)}
          audio_indexes: (audios_num,)
          segment_frames: int, frames number of a segment
          hop_frames: int, hop frames between segments
          source: 'curated' | 'noisy' | None
          
        Returns:
          segment_metadata_dict: {'audio_name': (segments_num,), 
                                  'begin_index': (segments_num,), 
                                  'end_index': (segments_num,), 
                                  (if exist) 'target': (segments_num, classes_num), 
                                  (if exist) 'source': (segments_num)}
        '''
        
        segment_metadata_dict = {'audio_name': [], 'begin_index': [], 
            'end_index': []}
            
        has_target = 'target' in data_dict.keys()

        if has_target:
            segment_metadata_dict['target'] = []
            
        if source:
            segment_metadata_dict['source'] = []
        
        for audio_index in audio_indexes:
            audio_name = data_dict['audio_name'][audio_index]
            begin_index = data_dict['begin_index'][audio_index]
            end_index = data_dict['end_index'][audio_index]
            
            if has_target:
                target = data_dict['target'][audio_index]
            else:
                target = None
            
            # If audio recording shorter than a segment
            if end_index - begin_index < segment_frames:
                segment_metadata_dict['begin_index'].append(begin_index)
                segment_metadata_dict['end_index'].append(end_index)
                
                self._append_to_meta_data(segment_metadata_dict, audio_name, 
                    target, source)
                
            # If audio recording longer than a segment then split
            else:
                shift = 0
                while end_index - (begin_index + shift) > segment_frames:
                    segment_metadata_dict['begin_index'].append(
                        begin_index + shift)
                        
                    segment_metadata_dict['end_index'].append(
                        begin_index + shift + segment_frames)
                        
                    self._append_to_meta_data(segment_metadata_dict, 
                        audio_name, target, source)
                        
                    shift += hop_frames
                    
                # Append the last segment
                segment_metadata_dict['begin_index'].append(
                    end_index - segment_frames)
                    
                segment_metadata_dict['end_index'].append(end_index)
                
                self._append_to_meta_data(segment_metadata_dict, audio_name, 
                    target, source)
            
        for key in segment_metadata_dict.keys():
            segment_metadata_dict[key] = np.array(segment_metadata_dict[key])
        
        return segment_metadata_dict
        
       
    def _append_to_meta_data(self, segment_metadata_dict, audio_name, target, 
        source):
        '''Append audio_name, target, source to segment_metadata_dict. 
        '''
        segment_metadata_dict['audio_name'].append(audio_name)
        
        if target is not None:
            segment_metadata_dict['target'].append(target)
            
        if source is not None:
            segment_metadata_dict['source'].append(source)
        
    def get_feature_mask_train(self, data_dict, begin_index, end_index, 
        segment_frames, pad_type, logmel_eps):
        '''Get logmel feature and mask of one segment. 
        
        Args:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'target': (audios_num, classes_num), 
                      (if exist) 'fold': (audios_num,)}
          begin_index: int, begin index of a segment
          end_index: int, end index of a segment
          segment_frames: int, frames number of a segment
          pad_type: string, 'constant' | 'repeat'
          logmel_eps: constant value to pad if pad_type == 'constant'
        '''
        
        isflip = False
        this_segment_frames = end_index - begin_index
        clip = data_dict['feature'][begin_index : end_index]
        
        # Flip ID
        if random.uniform(0, 1) > 0.5:
            clip = clip[::-1, :]
            isflip = True
        
        # If segment frames of this audio is fewer than the designed segment 
        # frames, then pad. 
        if this_segment_frames < segment_frames:
            if pad_type == 'constant':
                this_feature, this_mask = self.pad_constant(
                    clip, 
                    segment_frames, logmel_eps)
                    
            elif pad_type == 'repeat':
                this_feature, this_mask = self.pad_repeat(
                    clip, 
                    segment_frames)
                
            # this_mask = np.zeros(segment_frames)
            # this_mask[0 : this_segment_frames] = 1
            
        # If segment frames is equal to the designed segment frames, then load
        # data without padding. 
        else:
            # TODO random sample frames
            begin = np.random.randint(this_segment_frames - self.segment_frames + 1)
            end = begin + self.segment_frames
            this_feature = clip[begin : end]
            this_mask = np.ones(self.segment_frames)

        return this_feature, this_mask, isflip

    def get_feature_mask_infer(self, data_dict, begin_index, end_index, 
        segment_frames, pad_type, logmel_eps):
        '''Get logmel feature and mask of one segment. 
        
        Args:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'target': (audios_num, classes_num), 
                      (if exist) 'fold': (audios_num,)}
          begin_index: int, begin index of a segment
          end_index: int, end index of a segment
          segment_frames: int, frames number of a segment
          pad_type: string, 'constant' | 'repeat'
          logmel_eps: constant value to pad if pad_type == 'constant'
        '''
        
        isflip = False
        this_segment_frames = end_index - begin_index
        clip = data_dict['feature'][begin_index : end_index]
        flip_clip = clip[::-1, :]
        
        # If segment frames of this audio is fewer than the designed segment 
        # frames, then pad. 
        if this_segment_frames < segment_frames:
            if pad_type == 'constant':
                this_feature, this_mask = self.pad_constant(
                    clip, 
                    segment_frames, logmel_eps)
                flip_feature, this_mask = self.pad_constant(
                    flip_clip, 
                    segment_frames, logmel_eps)
                    
            elif pad_type == 'repeat':
                this_feature, this_mask = self.pad_repeat(
                    clip, 
                    segment_frames)
                flip_feature, this_mask = self.pad_repeat(
                    flip_clip, 
                    segment_frames)
                
            # this_mask = np.zeros(segment_frames)
            # this_mask[0 : this_segment_frames] = 1
            
        # If segment frames is equal to the designed segment frames, then load
        # data without padding. 
        else:
            this_feature = clip
            flip_feature = flip_clip
            this_mask = np.ones(self.segment_frames)
            
        return this_feature, flip_feature, this_mask
        
    def pad_constant(self, x, max_len, constant):
        '''Pad matrix with constant. 
        
        Args:
          x: (frames, mel_bins)
          max_len: int, legnth to be padded
          constant: float, value used for padding
        '''
        front = (max_len - x.shape[0]) // 2
        end = max_len - front - x.shape[0]
        pad_front = constant * np.ones((front, x.shape[1]))
        pad_end = constant * np.ones((end, x.shape[1]))
        padded_x = np.concatenate((pad_front, x, pad_end), axis=0)

        mask = np.zeros(max_len)
        mask[front:(front + x.shape[0])] = 1
        
        return padded_x, mask
        
    def pad_repeat(self, x, max_len):
        '''Repeat matrix to a legnth. 
        
        Args:
          x: (frames, mel_bins)
          max_len: int, length to be padded
        '''
        repeat_num = int(max_len / x.shape[0]) + 1
        repeated_x = np.tile(x, (repeat_num, 1))
        repeated_x = repeated_x[0 : max_len]

        mask = np.zeros(max_len)
        mask[:x.shape[0]] = 1
        
        return repeated_x, mask
        
    def transform(self, x):
        '''Transform data. 
        '''
        return scale(x, self.scalar['mean'], self.scalar['std'])


class DataGenerator(Base):
    
    def __init__(self, curated_feature_hdf5_path, noisy_feature_hdf5_path, 
        curated_cross_validation_path, noisy_cross_validation_path, train_source, 
        holdout_fold, segment_seconds, hop_seconds, pad_type, scalar, batch_size,
        seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          curated_feature_hdf5_path: string, path of hdf5 file
          noisy_feature_hdf5_path: string, path of hdf5 file
          curated_cross_validation_path: path of cross validation csv file
          noisy_cross_validation_path: path of cross validation csv file
          train_source: 'curated' | 'noisy' | 'curated_and_noisy'
          holdout_fold: '1', '2', '3', '4' | 'none', set `none` for training 
              on all data without validation
          segment_seconds: float, duration of audio recordings to be padded or split
          hop_seconds: float, hop seconds between segments
          pad_type: 'constant' | 'repeat'
          scalar: object, containing mean and std value
          batch_size: int
          seed: int
        '''

        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.segment_frames = int(segment_seconds * config.frames_per_second)
        self.hop_frames = int(hop_seconds * config.frames_per_second)
        self.pad_type = pad_type
        self.logmel_eps = config.logmel_eps
        
        # Load training data
        load_time = time.time()
        
        self.curated_data_dict = self.load_hdf5(
            curated_feature_hdf5_path, curated_cross_validation_path)
            
        self.noisy_data_dict = self.load_hdf5(
            noisy_feature_hdf5_path, noisy_cross_validation_path)
        
        # Get train and validate audio indexes
        (train_curated_audio_indexes, validate_curated_audio_indexes) = \
            self.get_train_validate_audio_indexes(
                self.curated_data_dict, holdout_fold)
            
        (train_noisy_audio_indexes, validate_noisy_audio_indexes) = \
            self.get_train_validate_audio_indexes(
                self.noisy_data_dict, holdout_fold)
        
        logging.info('Train curated audio num: {}'.format(
            len(train_curated_audio_indexes)))
        logging.info('Train noisy audio num: {}'.format(
            len(train_noisy_audio_indexes)))
        logging.info('Validate curated audio num: {}'.format(
            len(validate_curated_audio_indexes)))
        logging.info('Validate noisy audio num: {}'.format(
            len(validate_noisy_audio_indexes)))
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        
        # Get segment metadata for training
        self.train_curated_segment_metadata_dict = \
            self.get_segment_metadata_dict_train(
                self.curated_data_dict, train_curated_audio_indexes, 
                self.segment_frames, self.hop_frames, 'curated')
            
        self.train_noisy_segment_metadata_dict = self.get_segment_metadata_dict_train(
            self.noisy_data_dict, train_noisy_audio_indexes, 
            self.segment_frames, self.hop_frames, 'noisy')

        if train_source == 'curated':
            self.train_segment_metadata_dict = \
                self.train_curated_segment_metadata_dict
            
        elif train_source == 'noisy':
            self.train_segment_metadata_dict = \
                self.train_noisy_segment_metadata_dict
        
        elif train_source == 'curated_and_noisy':                
            self.train_segment_metadata_dict = \
                self.combine_curated_noisy_metadata_dict(
                    self.train_curated_segment_metadata_dict, 
                    self.train_noisy_segment_metadata_dict)

        # Get segment metadata for train feature map
        self.train_feature_map_curated_segment_metadata_dict = \
            self.get_segment_metadata_dict_infer(
                self.curated_data_dict, train_curated_audio_indexes, 
                self.segment_frames, self.hop_frames, 'curated')
            
        self.train_feature_map_noisy_segment_metadata_dict = \
            self.get_segment_metadata_dict_infer(
                self.noisy_data_dict, train_noisy_audio_indexes, 
                self.segment_frames, self.hop_frames, 'noisy')
        
        # Get segment metadata for validation
        self.validate_curated_segment_metadata_dict = \
            self.get_segment_metadata_dict_infer(
                self.curated_data_dict, validate_curated_audio_indexes, 
                self.segment_frames, self.hop_frames, 'curated')
            
        self.validate_noisy_segment_metadata_dict = \
            self.get_segment_metadata_dict_infer(
                self.noisy_data_dict, validate_noisy_audio_indexes, 
                self.segment_frames, self.hop_frames, 'noisy')
        
        # Print data statistics
        train_segments_num = len(self.train_segment_metadata_dict['audio_name'])
        
        validate_curated_segments_num = len(
            self.validate_curated_segment_metadata_dict['audio_name'])
            
        validate_noisy_segments_num = len(
            self.validate_noisy_segment_metadata_dict['audio_name'])
        
        logging.info('')
        logging.info('Total train segments num: {}'.format(train_segments_num))
        
        logging.info('Validate curated segments num: {}'.format(
            validate_curated_segments_num))
            
        logging.info('Validate noisy segments num: {}'.format(
            validate_noisy_segments_num))
                
        self.train_segments_indexes = np.arange(train_segments_num)
        self.random_state.shuffle(self.train_segments_indexes)
        self.pointer = 0
        
    def get_train_validate_audio_indexes(self, data_dict, holdout_fold):    
        '''Get train and validate audio indexes. 
        
        Args:
          data_dict: {'audio_name': (audios_num,), 
                      'feature': (dataset_total_frames, mel_bins), 
                      'target': (audios_num, classes_num), 
                      'begin_index': (audios_num,), 
                      'end_index': (audios_num,), 
                      (if exist) 'fold': (audios_num,)}
          holdout_fold: 'none' | int, if 'none' then validate indexes are empty
          
        Returns:
          train_audio_indexes: (train_audios_num,)
          validate_audio_indexes: (validate_audios_num)
        '''
        
        if holdout_fold == 'none':
            train_audio_indexes = np.arange(len(data_dict['audio_name']))
            validate_audio_indexes = np.array([])
            
        else:
            train_audio_indexes = np.where(
                data_dict['fold'] != int(holdout_fold))[0]
                
            validate_audio_indexes = np.where(
                data_dict['fold'] == int(holdout_fold))[0]
            
        return train_audio_indexes, validate_audio_indexes

    def combine_curated_noisy_metadata_dict(self, curated_metadata_dict, 
        noisy_metadata_dict):
        '''Combine curated and noisy segment metadata dict. 
        '''
                
        combined_metadata_dict = {}
        
        for key in curated_metadata_dict.keys():
            combined_metadata_dict[key] = np.concatenate(
                (curated_metadata_dict[key], noisy_metadata_dict[key]), axis=0)
            
        return combined_metadata_dict
       

    def _mixup(self, t1, t2, weights):
        return t1 * weights + t2 * (1 - weights)


    def generate_train(self, alpha=0.4):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: {'audio_name': (batch_size,), 
                            'feature': (batch_size, segment_frames, mel_bins), 
                            'mask': (batch_size, segment_frames), 
                            'target': (batch_size, classes_num), 
                            'source': (batch_size,)}
        '''
        
        while True:
            # Reset pointer
            if self.pointer >= len(self.train_segments_indexes):
                self.pointer = 0
                self.random_state.shuffle(self.train_segments_indexes)

            # Get batch segment indexes
            batch_segment_indexes = self.train_segments_indexes[
                self.pointer: self.pointer + self.batch_size]

            mixer_segment_indexes = np.random.choice(self.train_segments_indexes, self.batch_size)
                
            self.pointer += self.batch_size

            # Batch segment data
            batch_audio_name = self.train_segment_metadata_dict\
                ['audio_name'][batch_segment_indexes]
                
            batch_begin_index = self.train_segment_metadata_dict\
                ['begin_index'][batch_segment_indexes]
                
            batch_end_index = self.train_segment_metadata_dict\
                ['end_index'][batch_segment_indexes]
                
            batch_target = self.train_segment_metadata_dict\
                ['target'][batch_segment_indexes]
                
            batch_source = self.train_segment_metadata_dict\
                ['source'][batch_segment_indexes]

            # Mixer segment data
            # mixer_audio_name = self.train_segment_metadata_dict\
            #     ['audio_name'][mixer_segment_indexes]

            mixer_begin_index = self.train_segment_metadata_dict\
                ['begin_index'][mixer_segment_indexes]

            mixer_end_index = self.train_segment_metadata_dict\
                ['end_index'][mixer_segment_indexes]

            mixer_target = self.train_segment_metadata_dict\
                ['target'][mixer_segment_indexes]

            mixer_source = self.train_segment_metadata_dict\
                ['source'][mixer_segment_indexes]

            batch_feature = []
            batch_mask = []
            batch_flippid_target = []
            
            # Get logmel segments one by one, pad the short segments
            for n in range(len(batch_segment_indexes)):
                if batch_source[n] == 'curated':
                    data_dict = self.curated_data_dict
                elif batch_source[n] == 'noisy':
                    data_dict = self.noisy_data_dict
                else:
                    raise Exception('Incorrect source type!')

                if mixer_source[n] == 'curated':
                    mixer_dict = self.curated_data_dict
                elif mixer_source[n] == 'noisy':
                    mixer_dict = self.noisy_data_dict
                else:
                    raise Exception('Incorrect source type!')
                    
                (this_feature, this_mask, isflip) = self.get_feature_mask_train(
                    data_dict, batch_begin_index[n], batch_end_index[n], 
                    self.segment_frames, self.pad_type, self.logmel_eps)

                (mixer_feature, mixer_mask, mixer_isflip) = self.get_feature_mask_train(
                    mixer_dict, mixer_begin_index[n], mixer_end_index[n], 
                    self.segment_frames, self.pad_type, self.logmel_eps)

                if isflip:
                    flippid_target = np.concatenate((np.zeros(config.classes_num), batch_target[n]))
                else:
                    flippid_target = np.concatenate((batch_target[n], np.zeros(config.classes_num)))

                if mixer_isflip:
                    mixer_flippid_target = np.concatenate((np.zeros(config.classes_num), mixer_target[n]))
                else:
                    mixer_flippid_target = np.concatenate((mixer_target[n], np.zeros(config.classes_num)))

                # Mixup
                mixup_weight = np.random.beta(a=alpha, b=alpha)
                while mixup_weight <= 0.5:
                    mixup_weight = np.random.beta(a=alpha, b=alpha)
                
                this_feature = self._mixup(this_feature, mixer_feature, mixup_weight)
                this_feature = spec_augment(self.transform(this_feature))
                
                this_mask = ((this_mask + mixer_mask) > 0).astype(float)
                this_target = self._mixup(flippid_target, mixer_flippid_target, mixup_weight)
                    
                batch_feature.append(this_feature)
                batch_mask.append(this_mask)
                batch_flippid_target.append(this_target)
                
            batch_feature = np.array(batch_feature)
            
            batch_mask = np.array(batch_mask) 
            batch_flippid_target = np.array(batch_flippid_target) 
            
            batch_data_dict = {
                'audio_name': batch_audio_name, 
                'feature': batch_feature, 
                'mask': batch_mask, 
                'target': batch_flippid_target, 
                'source': batch_source}
            
            yield batch_data_dict
            
    def generate_validate(self, data_type, target_source, max_iteration=None):
        '''Generate mini-batch data for validation. 
        
        Returns:
          batch_data_dict: {'audio_name': (batch_size,), 
                            'feature': (batch_size, segment_frames, mel_bins), 
                            'mask': (batch_size, segment_frames), 
                            'target': (batch_size, classes_num)}
        '''

        assert(data_type in ['train_feature_map', 'validate'])
        assert(target_source in ['curated', 'noisy'])
        
        segment_metadata_dict = eval(
            'self.{}_{}_segment_metadata_dict'.format(data_type, target_source))
            
        data_dict = eval('self.{}_data_dict'.format(target_source))
        
        segments_num = len(segment_metadata_dict['audio_name'])
        segment_indexes = np.arange(segments_num)
        
        iteration = 0
        pointer = 0
      
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= segments_num:
                break

            # Get batch segment indexes
            batch_segment_indexes = segment_indexes[
                pointer: pointer + self.batch_size]                
                
            pointer += self.batch_size
            iteration += 1
            
            # Batch segment data
            batch_audio_name = segment_metadata_dict\
                ['audio_name'][batch_segment_indexes]
                
            batch_begin_index = segment_metadata_dict\
                ['begin_index'][batch_segment_indexes]
                
            batch_end_index = segment_metadata_dict\
                ['end_index'][batch_segment_indexes]
                
            batch_target = segment_metadata_dict\
                ['target'][batch_segment_indexes]
            
            batch_feature = []
            batch_flip_feature = []
            batch_mask = []

            # Get logmel segments one by one, pad the short segments
            for n in range(len(batch_segment_indexes)):
                (this_feature, flip_feature, this_mask) = self.get_feature_mask_infer(
                    data_dict, batch_begin_index[n], batch_end_index[n], 
                    self.segment_frames, self.pad_type, self.logmel_eps)
                
                batch_feature.append(this_feature)
                batch_flip_feature.append(flip_feature)
                batch_mask.append(this_mask)
                
            batch_feature = np.array(batch_feature)
            batch_feature = self.transform(batch_feature)
            batch_flip_feature = np.array(batch_flip_feature)
            batch_flip_feature = self.transform(batch_flip_feature)
            
            batch_mask = np.array(batch_mask)
            
            batch_data_dict = {
                'audio_name': batch_audio_name, 
                'feature': batch_feature, 
                'flip_feature': batch_flip_feature,
                'mask': batch_mask, 
                'target': batch_target}

            yield batch_data_dict
            

class TestDataGenerator(Base):
    def __init__(self, test_feature_hdf5_path, segment_seconds, hop_seconds, 
        pad_type, scalar, batch_size, seed=1234):
        '''Data generator for testing. 
        
        Args:
          test_feature_hdf5_path: string, path of hdf5 file
          segment_seconds: float, duration of audio recordings to be padded or split
          hop_seconds: float, hop seconds between segments
          pad_type: 'constant' | 'repeat'
          scalar: object, containing mean and std value
          batch_size: int
          seed: int
        '''
        
        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.segment_frames = int(segment_seconds * config.frames_per_second)
        self.hop_frames = int(hop_seconds * config.frames_per_second)
        self.pad_type = pad_type
        self.logmel_eps = config.logmel_eps
        
        # Load testing data
        self.test_data_dict = self.load_hdf5(
            test_feature_hdf5_path, cross_validation_path=None)
            
        audios_num = len(self.test_data_dict['audio_name'])
        test_audio_indexes = np.arange(audios_num)
            
        self.test_segment_metadata_dict = \
            self.get_segment_metadata_dict_infer(
                self.test_data_dict, test_audio_indexes, self.segment_frames, 
                self.hop_frames, source=None)
        
    def generate_test(self):
        '''Generate mini-batch data for test. 
        
        Returns:
          batch_data_dict: {'audio_name': (batch_size,), 
                            'feature': (batch_size, segment_frames, mel_bins), 
                            'mask': (batch_size, segment_frames)}
        '''
        
        segment_metadata_dict = self.test_segment_metadata_dict
        data_dict = self.test_data_dict
        
        segments_num = len(segment_metadata_dict['audio_name'])
        segment_indexes = np.arange(segments_num)
        
        iteration = 0
        pointer = 0
        
        while True:
            # Reset pointer
            if pointer >= segments_num:
                break

            # Get batch segment indexes
            batch_segment_indexes = segment_indexes[
                pointer: pointer + self.batch_size]
                
            pointer += self.batch_size
            iteration += 1
            
            # Batch segment data
            batch_audio_name = segment_metadata_dict\
                ['audio_name'][batch_segment_indexes]
                
            batch_begin_index = segment_metadata_dict\
                ['begin_index'][batch_segment_indexes]
                
            batch_end_index = segment_metadata_dict\
                ['end_index'][batch_segment_indexes]

            batch_feature = []
            batch_flip_feature = []
            batch_mask = []

            # Get logmel segments one by one, pad the short segments
            for n in range(len(batch_segment_indexes)):
                (this_feature, flip_feature, this_mask) = self.get_feature_mask_infer(
                    data_dict, batch_begin_index[n], batch_end_index[n], 
                    self.segment_frames, self.pad_type, self.logmel_eps)
                
                batch_feature.append(this_feature)
                batch_flip_feature.append(flip_feature)
                batch_mask.append(this_mask)
                    
            batch_feature = np.array(batch_feature)
            batch_feature = self.transform(batch_feature)
            batch_flip_feature = np.array(batch_flip_feature)
            batch_flip_feature = self.transform(batch_flip_feature)
            
            batch_mask = np.array(batch_mask)   
            
            batch_data_dict = {
                'audio_name': batch_audio_name, 
                'feature': batch_feature, 
                'flip_feature': batch_flip_feature,
                'mask': batch_mask} 

            yield batch_data_dict