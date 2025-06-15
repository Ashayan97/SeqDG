import torch
from torch.utils import data
import pandas as pd
import numpy as np
import random
from utils import narration_generator
import h5py


class epic_narrations(data.Dataset):
    def __init__(self,
                 window_len,
                 data_pickle,
                 masking=True
                 ):
        self.window_len = window_len
        self.data = pd.read_pickle(data_pickle)
        self.masking = masking

    def __getitem__(self, index):
        narration_id = self.data.iloc[index].name
        video_id = self.data.iloc[index]['video_id']
        df_sorted_video = self.data[self.data['video_id'] == video_id].sort_values('start_timestamp')
        idx = df_sorted_video.index.get_loc(narration_id)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_narration_ids = df_sorted_video.iloc[sequence_range].index.tolist()
        input_narrations = np.array(df_sorted_video.iloc[sequence_range]['narration'].values)
        if self.masking:
            input_narrations[self.window_len // 2] = '[MASK]'

        label = np.array(df_sorted_video.iloc[sequence_range]['narration'].values)[self.window_len // 2]
        final_input = narration_generator(input_narrations)

        return final_input, label, narration_id

    def __len__(self):
        return self.data.shape[0]


class EpicKitchens(data.Dataset):
    def __init__(self,
                 num_of_segments,
                 data_path,
                 labels_pickle,
                 text_feature_path='',
                 visual_feature_dim=1024,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 swapping=False,
                 center_classification=True,
                 text_feature_availablity=False,
                 random_swapping=False,
                 modalities = ['rgb'],
                 ):
        self.num_of_segments = num_of_segments
        self.dataset = None
        self.swapping = swapping
        self.data_path = data_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.window_len = window_len
        self.num_clips = num_clips
        self.modalities = modalities
        self.text_feature_availablity = text_feature_availablity
        if self.text_feature_availablity:
            self.test_features = pd.read_pickle(text_feature_path)
        self.center_classification = center_classification
        self.random_swapping = random_swapping
        assert clips_mode in ['all', 'random'], \
            "Labels mode not supported. Choose from ['all', 'random']"
        self.clips_mode = clips_mode
        self.dataset = pd.read_pickle(self.data_path)

    def find_swap_video(self, verb, noun, pid):
        new_id_swap = None

        mask = (self.df_labels['noun_class'] == int(noun)) & \
               (self.df_labels['verb_class'] == int(verb)) & \
               (self.df_labels['participant_id'] != pid)

        possible_samples = self.df_labels.loc[mask, ['noun_class', 'verb_class']].index.values

        if isinstance(possible_samples, str):
            possible_samples = [possible_samples]

        if len(possible_samples) != 0:
            indexs = [i for i in range(len(possible_samples))]
            chosen = random.sample(indexs, 1)
            new_id_swap = possible_samples[chosen][0]

        else:

            mask = (self.df_labels['noun_class'] == int(noun)) & (
                    self.df_labels['verb_class'] == int(verb))

            possible_samples = self.df_labels.loc[mask, ['noun_class', 'verb_class']].index.values

            if isinstance(possible_samples, str):
                possible_samples = [possible_samples]

            indexs = [i for i in range(len(possible_samples))]

            chosen = random.sample(indexs, 1)

            new_id_swap = possible_samples[chosen][0]

        return new_id_swap

    def __getitem__(self, index):

        num_clips = self.num_clips if self.clips_mode == 'all' else self.num_of_segments
        data = torch.zeros((len(self.modalities),self.window_len * num_clips, self.visual_feature_dim))

        narration_id = self.df_labels.iloc[index].name
        video_id = self.df_labels.iloc[index]['video_id']
        pid = self.df_labels.iloc[index]['participant_id']
        df_sorted_video = self.df_labels[self.df_labels['video_id'] == video_id].sort_values('start_timestamp')
        idx = df_sorted_video.index.get_loc(narration_id)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_narration_ids = df_sorted_video.iloc[sequence_range].index.tolist()

        if self.random_swapping:
            swapping_index = random.randint(0, self.window_len - 1)
        else:
            swapping_index = self.window_len // 2

        # # Center action
        verbs = torch.from_numpy(df_sorted_video.iloc[sequence_range]['verb_class'].values) \
            if 'verb_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
        nouns = torch.from_numpy(df_sorted_video.iloc[sequence_range]['noun_class'].values) \
            if 'noun_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
        # Concatenate the labels of the center action in the end to be classified by the summary embedding
        verbs = torch.cat([verbs, verbs[self.window_len // 2].unsqueeze(0)])
        nouns = torch.cat([nouns, nouns[self.window_len // 2].unsqueeze(0)])
        if self.swapping and bool( np.random.choice([1, 0], p=[0, 1])):
        #if self.swapping and bool(random.getrandbits(1)):
            sequence_narration_ids[swapping_index] = self.find_swap_video(verbs[swapping_index],
                                                                          nouns[swapping_index], pid)
        if self.center_classification:
            label = {'verb': verbs[-1].unsqueeze(0)[0], 'noun': nouns[-1].unsqueeze(0)[0]}
        else:
            label = {'verb': verbs, 'noun': nouns}

        narration_inputs = torch.zeros((self.window_len, 768))
        if self.clips_mode == 'random':
            for modal_idx in range(len(self.modalities)):
                for i in range(self.window_len):
                    clip_idxs = random.sample(range(0, self.num_clips), self.num_of_segments)
                    clip_idxs.sort()
                    for clip_idx in range(self.num_of_segments):
                        data[modal_idx][i * self.num_of_segments + clip_idx][:self.visual_feature_dim] = torch.from_numpy(
                            self.dataset[str(sequence_narration_ids[i])][self.modalities[modal_idx]][clip_idxs[clip_idx]])
            for i in range(self.window_len):
                if self.text_feature_availablity:
                    narration_inputs[i] = torch.tensor(self.test_features[str(sequence_narration_ids[i])])
                else:
                    narration_inputs[i] = torch.zeros(768)
        else:
            for modal_idx in range(len(self.modalities)):
                for i in range(self.window_len):
                        for j in range(self.num_clips):
                            data[modal_idx][i * self.num_clips + j][:self.visual_feature_dim] = torch.from_numpy(
                                self.dataset[str(sequence_narration_ids[i])][self.modalities[modal_idx]][j])
            for i in range(self.window_len):
                if self.text_feature_availablity:
                    narration_inputs[i] = self.test_features[str(sequence_narration_ids[i])]
                else:
                    narration_inputs[i] = ''

        return data, label, narration_inputs, narration_id

    def __len__(self):
        return self.df_labels.shape[0]


class egtea(data.Dataset):
    def __init__(self,
                 num_of_segments,
                 data_path,
                 labels_pickle,
                 text_feature_path='',
                 visual_feature_dim=2304,
                 audio_feature_dim=None,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 swapping=False,
                 center_classification=True,
                 text_feature_availablity=False,
                 random_swapping=False,
                 ):
        self.num_of_segments = num_of_segments
        self.dataset = None
        self.swapping = swapping
        self.data_path = data_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.window_len = window_len
        self.num_clips = num_clips
        self.text_feature_availablity = text_feature_availablity
        if self.text_feature_availablity:
            self.test_features = pd.read_pickle(text_feature_path)
        self.center_classification = center_classification
        self.random_swapping = random_swapping
        assert clips_mode in ['all', 'random'], \
            "Labels mode not supported. Choose from ['all', 'random']"
        self.clips_mode = clips_mode
        self.dataset = h5py.File(self.data_path, 'r')

    def find_swap_video(self, action, pid):
        new_id_swap = None

        mask = (self.df_labels['action_idx'] == int(action)) and (self.df_labels['video_name'] != pid)

        possible_samples = self.df_labels.loc[mask, ['clip_name']].values

        if isinstance(possible_samples, str):
            possible_samples = [possible_samples]

        if len(possible_samples) != 0:
            indexs = [i for i in range(len(possible_samples))]
            chosen = random.sample(indexs, 1)
            new_id_swap = possible_samples[chosen][0]

        else:

            mask = (self.df_labels['noun_class'] == int(action))

            possible_samples = self.df_labels.loc[mask, ['clip_name']].values

            if isinstance(possible_samples, str):
                possible_samples = [possible_samples]

            indexs = [i for i in range(len(possible_samples))]

            chosen = random.sample(indexs, 1)

            new_id_swap = possible_samples[chosen][0]

        return new_id_swap

    def __getitem__(self, index):

        num_clips = self.num_clips if self.clips_mode == 'all' else self.num_of_segments
        data = torch.zeros((self.window_len * num_clips, self.visual_feature_dim))

        clip_name = self.df_labels.iloc[index]['clip_name']
        video_name = self.df_labels.iloc[index]['video_name']
        df_idx = self.df_labels.iloc[index].name
        df_sorted_video = self.df_labels[self.df_labels['video_name'] == video_name].sort_values('start_frame')
        idx = df_sorted_video.index.get_loc(df_idx)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_clip_names = df_sorted_video.iloc[sequence_range]['clip_name'].tolist()

        if self.random_swapping:
            swapping_index = random.randint(0, self.window_len - 1)
        else:
            swapping_index = self.window_len // 2

        label = torch.from_numpy(df_sorted_video.iloc[sequence_range]['action_idx'].values)

        if self.swapping and bool(random.getrandbits(1)):
            sequence_clip_names[swapping_index] = self.find_swap_video(label[self.window_len // 2], video_name)

        if self.center_classification:
            label = label[self.window_len // 2]
        else:
            label = torch.cat([label, label[self.window_len // 2].unsqueeze(0)])
        narration_inputs = torch.zeros((self.window_len, 768))

        if self.clips_mode == 'random':
            for i in range(self.window_len):
                clip_idxs = random.sample(range(0, self.num_clips), self.num_of_segments)
                clip_idxs.sort()
                for clip_idx in range(self.num_of_segments):
                    data[i * self.num_of_segments + clip_idx][:self.visual_feature_dim] = torch.from_numpy(
                        self.dataset['visual_features/' + str(sequence_clip_names[i])][clip_idx])
                if self.text_feature_availablity:
                    narration_inputs[i] = torch.tensor(self.test_features[str(sequence_clip_names[i])])
                else:
                    narration_inputs[i] = torch.zeros(768)
        else:
            for i in range(self.window_len):
                for j in range(self.num_clips):
                    data[i * self.num_clips + j][:self.visual_feature_dim] = torch.from_numpy(
                        self.dataset['visual_features/' + sequence_clip_names[i]][j])
                if self.text_feature_availablity:
                    narration_inputs[i] = torch.tensor(self.test_features[str(sequence_clip_names[i])])
                else:
                    narration_inputs[i] = torch.zeros(768)
        # data, label, narration_inputs, narration_id
        return data, label, narration_inputs, clip_name

    def __len__(self):
        return self.df_labels.shape[0]



class EpicKitchensEndAction(data.Dataset):
    def __init__(self,
                 num_of_segments,
                 data_path,
                 labels_pickle,
                 text_feature_path='',
                 visual_feature_dim=1024,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 swapping=False,
                 center_classification=True,
                 text_feature_availablity=False,
                 random_swapping=False,
                 modalities = ['rgb'],
                 ):
        self.num_of_segments = num_of_segments
        self.dataset = None
        self.swapping = swapping
        self.data_path = data_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.window_len = window_len
        self.num_clips = num_clips
        self.modalities = modalities
        self.text_feature_availablity = text_feature_availablity
        if self.text_feature_availablity:
            self.test_features = pd.read_pickle(text_feature_path)
        self.center_classification = center_classification
        self.random_swapping = random_swapping
        assert clips_mode in ['all', 'random'], \
            "Labels mode not supported. Choose from ['all', 'random']"
        self.clips_mode = clips_mode
        self.dataset = pd.read_pickle(self.data_path)

    def find_swap_video(self, verb, noun, pid):
        new_id_swap = None

        mask = (self.df_labels['noun_class'] == int(noun)) & \
               (self.df_labels['verb_class'] == int(verb)) & \
               (self.df_labels['participant_id'] != pid)

        possible_samples = self.df_labels.loc[mask, ['noun_class', 'verb_class']].index.values

        if isinstance(possible_samples, str):
            possible_samples = [possible_samples]

        if len(possible_samples) != 0:
            indexs = [i for i in range(len(possible_samples))]
            chosen = random.sample(indexs, 1)
            new_id_swap = possible_samples[chosen][0]

        else:

            mask = (self.df_labels['noun_class'] == int(noun)) & (
                    self.df_labels['verb_class'] == int(verb))

            possible_samples = self.df_labels.loc[mask, ['noun_class', 'verb_class']].index.values

            if isinstance(possible_samples, str):
                possible_samples = [possible_samples]

            indexs = [i for i in range(len(possible_samples))]

            chosen = random.sample(indexs, 1)

            new_id_swap = possible_samples[chosen][0]

        return new_id_swap

    def __getitem__(self, index):

        num_clips = self.num_clips if self.clips_mode == 'all' else self.num_of_segments
        data = torch.zeros((len(self.modalities),self.window_len * num_clips, self.visual_feature_dim))

        narration_id = self.df_labels.iloc[index].name
        video_id = self.df_labels.iloc[index]['video_id']
        pid = self.df_labels.iloc[index]['participant_id']
        df_sorted_video = self.df_labels[self.df_labels['video_id'] == video_id].sort_values('start_timestamp')
        idx = df_sorted_video.index.get_loc(narration_id)
        start = idx - (self.window_len - 1)
        end = idx + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_narration_ids = df_sorted_video.iloc[sequence_range].index.tolist()

        if self.random_swapping:
            swapping_index = random.randint(0, self.window_len - 1)
        else:
            swapping_index = len(sequence_narration_ids) - 1

        # # Center action
        verbs = torch.from_numpy(df_sorted_video.iloc[sequence_range]['verb_class'].values) \
            if 'verb_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
        nouns = torch.from_numpy(df_sorted_video.iloc[sequence_range]['noun_class'].values) \
            if 'noun_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
        # Concatenate the labels of the center action in the end to be classified by the summary embedding
        verbs = torch.cat([verbs, verbs[-1].unsqueeze(0)])
        nouns = torch.cat([nouns, nouns[-1].unsqueeze(0)])
        if self.swapping and bool(random.getrandbits(1)):
            sequence_narration_ids[swapping_index] = self.find_swap_video(verbs[swapping_index],
                                                                          nouns[swapping_index], pid)
       
        #print(narration_id)
        #print(sequence_narration_ids)
        #print(verbs)
        #print(nouns)
        if self.center_classification:
            label = {'verb': verbs[-1].unsqueeze(0)[0], 'noun': nouns[-1].unsqueeze(0)[0]}
        else:
            label = {'verb': verbs, 'noun': nouns}
        #print(label)
        narration_inputs = torch.zeros((self.window_len, 768))
        if self.clips_mode == 'random':
            for modal_idx in range(len(self.modalities)):
                for i in range(self.window_len):
                    clip_idxs = random.sample(range(0, self.num_clips), self.num_of_segments)
                    clip_idxs.sort()
                    for clip_idx in range(self.num_of_segments):
                        data[modal_idx][i * self.num_of_segments + clip_idx][:self.visual_feature_dim] = torch.from_numpy(
                            self.dataset[str(sequence_narration_ids[i])][self.modalities[modal_idx]][clip_idxs[clip_idx]])
            for i in range(self.window_len):
                if self.text_feature_availablity:
                    narration_inputs[i] = torch.tensor(self.test_features[str(sequence_narration_ids[i])])
                else:
                    narration_inputs[i] = torch.zeros(768)
        else:
            for modal_idx in range(len(self.modalities)):
                for i in range(self.window_len):
                        for j in range(self.num_clips):
                            data[modal_idx][i * self.num_clips + j][:self.visual_feature_dim] = torch.from_numpy(
                                self.dataset[str(sequence_narration_ids[i])][self.modalities[modal_idx]][j])
            for i in range(self.window_len):
                if self.text_feature_availablity:
                    narration_inputs[i] = self.test_features[str(sequence_narration_ids[i])]
                else:
                    narration_inputs[i] = ''

        return data, label, narration_inputs, narration_id

    def __len__(self):
        return self.df_labels.shape[0]
