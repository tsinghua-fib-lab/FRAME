import os
import time
# import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
import pandas as pd
import math
import csv
import torch.nn.functional as F

class Cliprec(nn.Module):
    def __init__(self,clip_em,pos_A,neg_A,u_num,video_num,total_u_id_list,total_v_id_list,step_train=True):
        super(Cliprec, self).__init__()
        self.num_frm = 3
        self.num_clips = 4
        self.user_num = u_num
        self.step_train = step_train
        #self.u_pos_interest = nn.Parameter(torch.tensor(u_pos_interest))
        #self.u_neg_interest = nn.Parameter(torch.tensor(u_neg_interest))
        #self.u_en_interest = nn.parameter(u_en_interest)
        self.total_u_id_list = total_u_id_list
        self.total_v_id_list = total_v_id_list
        self.video_num = video_num
        self.clip_total_num = self.num_clips*self.video_num
        self.other_dim = 128
        self.pos_weight = 0.8
        self.neg_weight = -0.2
        #self.en_interest_weight = 0.5
        self.pos_A = pos_A.cuda()
        self.neg_A = neg_A.cuda()
        self.clip_v_em = nn.Parameter(clip_em)
        self.gcn_w1 = nn.Parameter(torch.randn(self.other_dim,self.other_dim))  
        nn.init.xavier_normal_(self.gcn_w1)
        self.gcn_w2 = nn.Parameter(torch.randn(self.other_dim,self.other_dim))  
        nn.init.xavier_normal_(self.gcn_w2)
		#self.fc2 = nn.Linear(self.other_dim ,self.other_dim/2,bias=False)
        #self.H1 = nn.Parameter(torch.randn(self.clip_total_num,self.other_dim))
        self.item_size = 128
        self.dnn_size = 256
        #self.w1 = nn.Parameter(torch.randn(self.item_size*2, self.dnn_size))
        self.w1 = nn.Parameter(torch.randn(self.item_size*2, self.dnn_size))
        nn.init.xavier_normal_(self.w1)
        self.b1 = nn.Parameter(torch.zeros(self.dnn_size))
        self.w2 = nn.Parameter(torch.randn(self.dnn_size, 1))
        nn.init.xavier_normal_(self.w2)
        self.b2 = nn.Parameter(torch.zeros(1))
        #self.w3 = nn.Parameter(torch.randn(self.item_size*2, self.dnn_size))
        self.w3 = nn.Parameter(torch.randn(self.item_size*2, self.dnn_size))
        nn.init.xavier_normal_(self.w3)
        self.b3 = nn.Parameter(torch.zeros(self.dnn_size))
        self.w4 = nn.Parameter(torch.randn(self.dnn_size, 1))
        nn.init.xavier_normal_(self.w4)
        self.b4 = nn.Parameter(torch.zeros(1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans_w = nn.Parameter(torch.randn(self.item_size, self.item_size))
        nn.init.xavier_normal_(self.trans_w)
        self.agg_w = torch.tensor([0.2,0.2,0.6,0.1]).cuda()
        self.bsz =2048

    def user_lookup(self,u_id_list):
        a = []
        for i in range(len(u_id_list)):
            a.append(torch.tensor(np.where(self.total_u_id_list == u_id_list.numpy()[i])[0][0]).unsqueeze(0))
        return torch.cat(a,dim=0)
    '''
    def get_u_pos_interest(self,u_id_list):  #lookup
        u = self.user_lookup(u_id_list)
        u_pos_emb = self.pos_A[u].mm(self.clip_v_em)
        return u_pos_emb
        
        u_pos_interest_em = []
        for i in range(len(u_id_list)):
            #print(self.u_pos_interest.shape)
            #print(self.total_u_id_list.shape)
            #print(u_id_list.shape)
            u_pos_interest_em.append(self.u_pos_interest[np.where(self.total_u_id_list == u_id_list.numpy()[i])[0]].unsqueeze(0))
        u_pos_interest_em = torch.cat(u_pos_interest_em,dim=0)
        return u_pos_interest_em   #tensor:[batch_size,dim]
        
    def get_u_neg_interest(self,u_id_list):  #lookup
        u = self.user_lookup(u_id_list)
        u_neg_emb = self.neg_A[u].mm(self.clip_v_em)
        return u_neg_emb
    

        u_neg_interest_em = []
        for i in range(len(u_id_list)):
            u_neg_interest_em.append(self.u_neg_interest[np.where(self.total_u_id_list == u_id_list.numpy()[i])[0]].unsqueeze(0))
        u_neg_interest_em = torch.cat(u_neg_interest_em,dim=0)
        return u_neg_interest_em   #tensor:[batch_size,dim]
    '''


    def forward(self, batch, train_step):
 
        #clip_feature = batch['visual_inputs']

        #u_pos_em = self.get_u_pos_interest(batch['uid'])
        #u_neg_em = self.get_u_neg_interest(batch['uid'])
        u = self.user_lookup(batch['uid'])
        clip_em = torch.matmul(self.clip_v_em,self.trans_w)
        u_pos_emb = F.relu(torch.matmul(torch.matmul(self.pos_A,clip_em),self.gcn_w1))
        u_neg_emb = F.relu(torch.matmul(torch.matmul(self.neg_A,clip_em),self.gcn_w1))

        #w_pos = F.relu(torch.matmul(torch.matmul(self.pos_A,clip_em),self.gcn_w1))
        w_pos = F.relu(torch.matmul(torch.matmul(self.pos_A.t(),u_pos_emb),self.gcn_w2))
        #w_neg = F.relu(torch.matmul(torch.matmul(self.neg_A,clip_em),self.gcn_w1))
        w_neg = F.relu(torch.matmul(torch.matmul(self.neg_A.t(),u_neg_emb),self.gcn_w2))
        w = (w_pos+w_neg)/2

        #u_pos_emb = F.relu(torch.matmul(torch.matmul(self.pos_A,w_pos),self.gcn_w2))
        #u_neg_emb = F.relu(torch.matmul(torch.matmul(self.neg_A,w_neg),self.gcn_w2))

        u_pos_em = u_pos_emb[u]
        u_neg_em = u_neg_emb[u]


        #u_en_em = self.get_u_en_interest(batch['uid'])

        if train_step:
            for i in range(self.num_clips):
                a = np.array(batch['label'][i])
                if i  == 0:
                    label = a.reshape(-1,1)
                else:
                    label = np.hstack((label,a.reshape(-1,1)))
            num_valid_clips = []
            reshape_label = []

            for j in range(label.shape[0]):
                index = np.where(label[j] == 0)[0]
                if len(index) == 0:
                    num_valid_clips.append(4)
                    for n in range(self.num_clips):
                        reshape_label.append(torch.tensor(1).unsqueeze(0))
                else:
                    num_valid_clips.append(index[0]+1)
                    for p in range(int(index[0])):
                        reshape_label.append(torch.tensor(1).unsqueeze(0))
                    reshape_label.append(torch.tensor(0).unsqueeze(0))
            reshape_label = torch.cat(reshape_label,dim=0)           
  
            u_pos_matrix = []
            u_neg_matrix = []
            #clip_matrix = [] 
            clip_gcn = []
            loss_weight = []  
            for k in range(len(num_valid_clips)):
                video_id =  int(batch['vid'][k])
                for m in range(num_valid_clips[k]):
                    u_pos_matrix.append(u_pos_em[k].unsqueeze(0))
                    u_neg_matrix.append(u_neg_em[k].unsqueeze(0))
                    #u_en_matrix.append(u_en_em[k].unsqueeze(0))
                    #clip_matrix.append(clip_feature[k,m,:].unsqueeze(0))
                    clip_gcn.append(w[np.where(self.total_v_id_list == video_id)[0][0]*self.num_clips+m,:].unsqueeze(0))
                    loss_weight.append(batch['label_weight'][k].unsqueeze(0))
            u_pos_matrix = torch.cat(u_pos_matrix,dim=0).squeeze()
            u_neg_matrix = torch.cat(u_neg_matrix,dim=0).squeeze()
            #u_en_matrix = torch.cat(u_en_matrix,dim=0).squeeze()
            #clip_matrix = torch.cat(clip_matrix,dim=0)
            clip_gcn = torch.cat(clip_gcn,dim=0).squeeze()
            loss_weight = torch.cat(loss_weight,dim=0)

            #clip_mul_matrix = torch.cat([clip_matrix.float(),clip_gcn.float()],1)
            #pos_output = torch.matmul(torch.cat([clip_matrix.float(),u_pos_matrix.float()],1),self.w1)+self.b1
            pos_output = torch.matmul(torch.cat([clip_gcn.float(),u_pos_matrix.float()],1),self.w1)+self.b1
            pos_output = F.relu(pos_output)
            #neg_output = torch.matmul(torch.cat([clip_matrix.float(),u_neg_matrix.float()],1),self.w3)+self.b3
            neg_output = torch.matmul(torch.cat([clip_gcn.float(),u_neg_matrix.float()],1),self.w3)+self.b3
            neg_output = F.relu(neg_output)
            #en_output = torch.matmul(torch.cat([clip_matrix.float(),u_en_matrix.float()],1),self.w1)+self.b1   
            #en_output = F.relu(en_output)
            pos_output=torch.matmul(pos_output,self.w2)+self.b2
            neg_output=torch.matmul(neg_output,self.w4)+self.b4
            #en_output=torch.matmul(en_output,self.w2)+self.b2

            #output = self.pos_weight*pos_output+self.neg_weight*neg_output
            output = pos_output

            # print(output)
            output = torch.sigmoid(output)

            u_pos_matrix2 = []
            u_neg_matrix2 = [] 
            clip_gcn_pos2 = []
            clip_gcn_neg2 = []
            for k in range(len(num_valid_clips)):
                video_id2 =  int(batch['vid'][k])
                if num_valid_clips[k]<2:
                    continue
                index2 = np.where(label[k] == 0)[0]
                if len(index2) == 0:
                    continue  
                u_pos_matrix2.append(u_pos_em[k].unsqueeze(0))
                u_neg_matrix2.append(u_neg_em[k].unsqueeze(0))
                clip_gcn_pos2.append(torch.mean(w[np.where(self.total_v_id_list == video_id2)[0][0]*self.num_clips:np.where(self.total_v_id_list == video_id2)[0][0]*self.num_clips+num_valid_clips[k]-1,:],dim=0,keepdim=True).unsqueeze(0))
                clip_gcn_neg2.append(w[np.where(self.total_v_id_list == video_id2)[0][0]*self.num_clips+num_valid_clips[k]-1,:].unsqueeze(0))
            
            u_pos_matrix2 = torch.cat(u_pos_matrix2,dim=0).squeeze()
            u_neg_matrix2 = torch.cat(u_neg_matrix2,dim=0).squeeze()
            #u_en_matrix = torch.cat(u_en_matrix,dim=0).squeeze()
            #clip_matrix = torch.cat(clip_matrix,dim=0)
            clip_gcn_pos2 = torch.cat(clip_gcn_pos2,dim=0).squeeze()
            clip_gcn_neg2 = torch.cat(clip_gcn_neg2,dim=0).squeeze()
            #loss_weight = torch.cat(loss_weight,dim=0)
            #clip_mul_matrix = torch.cat([clip_gcn.float(),clip_gcn.float()],1)

            # print(clip_gcn_pos2.size())
            # print(u_pos_matrix2.size())
            pos_output2 = torch.matmul(torch.cat([clip_gcn_pos2.float(),u_pos_matrix2.float()],1),self.w1)+self.b1
            pos_output2 = F.relu(pos_output2)
            neg_output2 = torch.matmul(torch.cat([clip_gcn_pos2.float(),u_neg_matrix2.float()],1),self.w3)+self.b3
            neg_output2 = F.relu(neg_output2)
            #en_output = torch.matmul(torch.cat([clip_matrix.float(),u_en_matrix.float()],1),self.w1)+self.b1
            #en_output = F.relu(en_output)
            pos_output2=torch.matmul(pos_output2,self.w2)+self.b2
            neg_output2=torch.matmul(neg_output2,self.w4)+self.b4
            #en_output=torch.matmul(en_output,self.w2)+self.b2

            #output2 = self.pos_weight*pos_output2+self.neg_weight*neg_output2
            output2 = pos_output2

            output2 = torch.sigmoid(output2)

            pos_output3 = torch.matmul(torch.cat([clip_gcn_neg2.float(),u_pos_matrix2.float()],1),self.w1)+self.b1
            pos_output3 = F.relu(pos_output3)
            neg_output3 = torch.matmul(torch.cat([clip_gcn_neg2.float(),u_neg_matrix2.float()],1),self.w3)+self.b3
            neg_output3 = F.relu(neg_output3)
            #en_output = torch.matmul(torch.cat([clip_matrix.float(),u_en_matrix.float()],1),self.w1)+self.b1
            #en_output = F.relu(en_output)
            pos_output3=torch.matmul(pos_output3,self.w2)+self.b2
            neg_output3=torch.matmul(neg_output3,self.w4)+self.b4
            #en_output=torch.matmul(en_output,self.w2)+self.b2

            #output3 = self.pos_weight*pos_output3+self.neg_weight*neg_output3
            output3 = pos_output3

            output3 = torch.sigmoid(output3)
            
            #return output,loss_weight,reshape_label
            return output,loss_weight,reshape_label,output2,output3

        else:
            for i in range(self.num_clips):
                a = np.array(batch['label'][i])
                if i  == 0:
                    label = a.reshape(-1,1)
                else:
                    label = np.hstack((label,a.reshape(-1,1)))
            num_valid_clips = []
            reshape_label = []
            reshape_uid = []
            for j in range(label.shape[0]):
                index = np.where(label[j] == 0)[0]
                if len(index) == 0:  #完播
                    num_valid_clips.append(4)
                    for n in range(self.num_clips):
                        reshape_label.append(torch.tensor(1).unsqueeze(0))
                        reshape_uid.append(batch['uid'][j].unsqueeze(0))
                else:
                    num_valid_clips.append(4)
                    for p in range(self.num_clips):
                        reshape_label.append(torch.tensor(label[j][p]).unsqueeze(0))
                        reshape_uid.append(batch['uid'][j].unsqueeze(0))
                    #reshape_label.append(torch.tensor(0).unsqueeze(0))
            reshape_label = torch.cat(reshape_label,dim=0)
            reshape_uid = torch.cat(reshape_uid,dim=0)

            u_pos_matrix = []
            u_neg_matrix = []
            #clip_matrix = [] 
            clip_gcn = []
            #loss_weight = [] 
            #A_normed = self.normalize(self.A,True)
            #w_pos = F.relu(torch.matmul(torch.matmul(self.pos_A,clip_em),self.gcn_w1))
            #w_pos = F.relu(torch.matmul(torch.matmul(self.pos_A.t(),w_pos),self.gcn_w2))
            #w_neg = F.relu(torch.matmul(torch.matmul(self.neg_A,clip_em),self.gcn_w1))
            #w_neg = F.relu(torch.matmul(torch.matmul(self.neg_A.t(),w_neg),self.gcn_w2))
            #w = (w_pos+w_neg)/2
            for k in range(len(num_valid_clips)):
                video_id =  int(batch['vid'][k])
                for m in range(num_valid_clips[k]):
                    u_pos_matrix.append(u_pos_em[k].unsqueeze(0))
                    u_neg_matrix.append(u_neg_em[k].unsqueeze(0))
                    #u_en_matrix.append(u_en_em[k].unsqueeze(0))
                    #clip_matrix.append(clip_feature[k,m,:].unsqueeze(0))
                    clip_gcn.append(w[np.where(self.total_v_id_list == video_id)[0]*self.num_clips+m,:].unsqueeze(0))
                    #loss_weight.append(batch['label_weight'][k].unsqueeze(0))
            u_pos_matrix = torch.cat(u_pos_matrix,dim=0).squeeze()
            u_neg_matrix = torch.cat(u_neg_matrix,dim=0).squeeze()
            #u_en_matrix = torch.cat(u_en_matrix,dim=0).squeeze()
            #clip_matrix = torch.cat(clip_matrix,dim=0)
            clip_gcn = torch.cat(clip_gcn,dim=0).squeeze()
            #loss_weight = torch.cat(loss_weight,dim=0)
            #clip_mul_matrix = torch.cat([clip_gcn.float(),clip_gcn.float()],1)
            pos_output = torch.matmul(torch.cat([clip_gcn.float(),u_pos_matrix.float()],1),self.w1)+self.b1
            pos_output = F.relu(pos_output)
            neg_output = torch.matmul(torch.cat([clip_gcn.float(),u_neg_matrix.float()],1),self.w3)+self.b3
            neg_output = F.relu(neg_output)
            #en_output = torch.matmul(torch.cat([clip_matrix.float(),u_en_matrix.float()],1),self.w1)+self.b1
            #en_output = F.relu(en_output)
            pos_output=torch.matmul(pos_output,self.w2)+self.b2
            neg_output=torch.matmul(neg_output,self.w4)+self.b4
            #en_output=torch.matmul(en_output,self.w2)+self.b2

            #output = self.pos_weight*pos_output+self.neg_weight*neg_output
            output = pos_output

            output = output.squeeze().float()
            #print(type(output))
            eva_output = []
            for a in range(len(batch['val_label'])):
                temp = torch.tensor(0).float().cuda()
                for b in range(self.num_clips):
                    #print(output[self.num_clips*a+b])
                    #print(self.agg_w[b].size())
                    temp += output[self.num_clips*a+b]*self.agg_w[b]
                #temp = temp/self.num_clips
                eva_output.append(temp.unsqueeze(0))
            eva_output = torch.cat(eva_output,dim=0)
            eva_output = torch.sigmoid(eva_output)
        
            return eva_output
            #return output,reshape_label,reshape_uid,output2,output3
        '''
        else:                
            u_pos_matrix = []
            u_neg_matrix = []
            #u_en_matrix = []
            clip_matrix = []     
            for k in range(len(batch['val_label'])):
                for m in range(self.num_clips):
                    u_pos_matrix.append(u_pos_em[k].unsqueeze(0))
                    u_neg_matrix.append(u_neg_em[k].unsqueeze(0))
                    #u_en_matrix.append(u_en_em[k].unsqueeze(0))
                    clip_matrix.append(clip_feature[k,m,:].unsqueeze(0))
    
            u_pos_matrix = torch.cat(u_pos_matrix,dim=0).squeeze()
            u_neg_matrix = torch.cat(u_neg_matrix,dim=0).squeeze()
            #u_en_matrix = torch.cat(u_en_matrix,dim=0)  #tensor:[N,dim]
            clip_matrix = torch.cat(clip_matrix,dim=0)
            #print(clip_matrix.size())
            #print(u_pos_matrix.size())
            pos_output = torch.matmul(torch.cat([clip_matrix.float(),u_pos_matrix.float()],1),self.w1)+self.b1
            pos_output = F.relu(pos_output)
            neg_output = torch.matmul(torch.cat([clip_matrix.float(),u_neg_matrix.float()],1),self.w3)+self.b3
            neg_output = F.relu(neg_output)
            #en_output = torch.matmul(torch.cat([clip_matrix.float(),u_en_matrix.float()],1),self.w1)+self.b1   #同一个网络？
            #en_output = F.relu(en_output)
            pos_output=torch.matmul(pos_output,self.w2)+self.b2
            neg_output=torch.matmul(neg_output,self.w4)+self.b4
            #en_output=torch.matmul(en_output,self.w2)+self.b2

            output = self.pos_weight*pos_output+self.neg_weight*neg_output
            output = output.squeeze().float()
            #print(type(output))
            eva_output = []
            temp = torch.tensor(0).float()
            for a in range(len(batch['val_label'])):
                for b in range(self.num_clips):
                    temp += output[self.num_clips*a+b]
                    temp = temp/self.num_clips
                eva_output.append(temp.unsqueeze(0))
            eva_output = torch.cat(eva_output,dim=0)
            eva_output = torch.sigmoid(eva_output)
        
            return eva_output   
        '''
    def freeze_cnn_backbone(self):
        for n, p in self.cnn.named_parameters():
            p.requires_grad = False