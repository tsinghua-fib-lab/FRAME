import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
import pandas as pd
import random
from torch.utils.data import Dataset

class CliprecDataset(Dataset):

    def __init__(self, num_frm=3, ensemble_n_clips=4,num_labels=4,
                datadir = './data_final2.csv', videodir = './micro_video2', visual_feature_dir = './fast_video_feature'):
        super(CliprecDataset, self).__init__()
        self.datadir = datadir
        self.videodir = videodir
        self.num_frm = num_frm
        self.ensemble_n_clips = ensemble_n_clips
        self.num_labels = num_labels
        self.interaction = pd.read_csv(self.datadir)


    def __len__(self):
        return np.array(self.interaction).shape[0]
    

    def __getitem__(self, index):
        inter = self.interaction.iloc[index,:]  # one interaction record
        vid_id = inter['photo_id']
        u_id = inter['user_id']
        playing_time = inter['playing_time']
        video_time = inter['duration_ms']
        ratio = playing_time/video_time
        clip_label=[]

        if ratio < 1/self.ensemble_n_clips:
            c = 0   
            total_label = 0
        if ratio>=1/self.ensemble_n_clips and ratio<2/self.ensemble_n_clips:
            c = 0.25
            total_label = 0
        if ratio>=2/self.ensemble_n_clips and ratio<3/self.ensemble_n_clips:
            c = 0.5
            total_label = 0
        if ratio>=3/self.ensemble_n_clips and ratio<4/self.ensemble_n_clips:
            c = 0.75
            total_label = 0
        if ratio>=4/self.ensemble_n_clips:
            c = 1
            total_label = 1
        

        for i in range(self.ensemble_n_clips):  
            clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 1)
            '''
            if ratio < 1/self.ensemble_n_clips:
                clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 0)
            if ratio>1/self.ensemble_n_clips and ratio<2/self.ensemble_n_clips:
                 clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 0.25)
            if ratio>2/self.ensemble_n_clips and ratio<3/self.ensemble_n_clips:
                 clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 0.5)
            if ratio>3/self.ensemble_n_clips and ratio<self.ensemble_n_clips:
                 clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 0.75)
            if ratio>=self.ensemble_n_clips:
                 clip_label.append(0 if ratio<(i+1)/self.ensemble_n_clips else 1)
            '''

            
        #vid_frm_array = self._load_video(vid_id)
        #print(vid_id)
        clip_feature = torch.tensor(np.load('./fast_video_feature/'+str(int(vid_id))+'.npy'))

        return dict(
            visual_inputs = clip_feature.float().cuda(),
            uid = u_id,
            vid = vid_id,
            label = clip_label,
            val_label = total_label,
            label_weight = c
        )
