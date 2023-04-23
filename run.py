import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
import pandas as pd
import setproctitle
from torch.utils.data import DataLoader
from modeling_gcn import Cliprec
from dataset_gcn import CliprecDataset
from torch import neg, optim
from metric import *
from sklearn.metrics import roc_auc_score, log_loss
import time
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy import sparse


def logloss_cal(label,prediction):
    #loss = -np.mean(np.multiply(label,np.log2(prediction+0.00005)))-np.mean(np.multiply(1-label,np.log2(1-prediction+0.00005)))
    loss = -np.mean(np.multiply(label,np.log(prediction+0.00005)))-np.mean(np.multiply(1-label,np.log(1-prediction+0.00005)))
    return loss

def pair_wise_loss(out1,out2):
    pos_value = out1.cpu()
    neg_value = out2.cpu()
    loss = torch.zeros(1)
    for i  in range(len(pos_value)):
            loss += -torch.log(torch.sigmoid(pos_value[i].squeeze()-neg_value[i].squeeze()))
    #final_loss = torch.mean(loss)
    final_loss = loss/(i)
    return final_loss

def point_wise_loss(prediction,label,weight_label):
    #weigthed_label = torch.mul(label,weight_label)
    #loss = -0.3*torch.mean(torch.mul(weight_label.float(),torch.mul(label.float(),torch.log(prediction.cpu()+0.00005))).float())
    loss = -torch.mean(torch.mul(weight_label.float(),torch.mul(label.float(),torch.log(prediction.cpu()+0.00005))).float())
    #print('loss1:',loss)
    loss += -torch.mean(torch.mul((1-label).float(),torch.log(1-prediction.cpu()+0.0005)).float())
    #print('loss2:',loss)
    return loss

def normalize(A):
        '''
        A = A + torch.eye(A.size(0))
        d = A.sum(1)
        D = torch.diag(torch.pow(d , -0.5))
        '''
        D = torch.sum(A, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        A = A.float()/D_sqrt.float().t()
        A = A.float()/D_sqrt.float().t()
        return A

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#user_pos_interest =  np.load('./fast_u_pos_interest.npy')
#user_neg_interest = np.load('./fast_u_neg_interest.npy')
#user_en_interest = np.load('./u_en_interest.npy')
total_u_id_list = np.load('fast_total_u_id_list.npy')
interaction = pd.read_csv('data_final2.csv')
total_v_id_list = np.unique(np.array(interaction['photo_id']))  
video_num = np.unique(np.array(interaction['photo_id'])).shape[0]
bsz = 1024
clip_num = 4
u_num = len(total_u_id_list)
'''
u_num = len(total_u_id_list)
c_num = clip_num*video_num
#node_num = u_num+c_num
u_pos = []
c_pos = []
u_neg = []
c_neg = []
cnt_pos = 0
cnt_neg = 0
for i in range(len(train)):
    user = int(train.iloc[i,:]['user_id'])
    video = train.iloc[i,:]['photo_id']
    ratio = train.iloc[i,:]['playing_time']/train.iloc[i,:]['duration_ms']
    if ratio>1:
        ratio = 1
    for j in range(int(ratio/0.25)):
        u_pos.append(np.where(total_u_id_list == user)[0][0])
        c_pos.append(np.where(total_v_id_list == video)[0][0]*clip_num+j)
        if ratio<1:
            u_neg.append(np.where(total_u_id_list == user)[0][0])
            c_neg.append(np.where(total_v_id_list == video)[0][0]*clip_num+int(ratio/0.25))
            cnt_neg = cnt_neg+1
        #u.append(u_num+np.where(total_v_id_list == video)[0][0]*clip_num+j)
        #c.append(np.where(total_u_id_list == user)[0][0])
        cnt_pos = cnt_pos+1
data_pos = np.ones(cnt_pos)
data_neg = np.ones(cnt_neg)
pos_a = csr_matrix((data_pos, (u_pos,c_pos)), shape=(u_num,c_num))
neg_a = csr_matrix((data_neg, (u_neg,c_neg)), shape=(u_num,c_num))
z_pos = scipy_sparse_mat_to_torch_sparse_tensor(pos_a)
pos_a_normed = normalize(z_pos.to_dense())
z_neg = scipy_sparse_mat_to_torch_sparse_tensor(neg_a)
neg_a_normed = normalize(z_neg.to_dense())
'''

visual_dir = './fast_video_feature/'

'''
clip_em = []
for i in range(len(total_v_id_list)):
    #print(visual_dir+str(total_v_id_list[i])+'.npy')
    for j in range(clip_num):
        clip_em.append(torch.tensor(np.load(visual_dir+str(int(total_v_id_list[i]))+'.npy')[j].astype(np.float32)).unsqueeze(0))
clip_em = torch.cat(clip_em,dim=0)
'''

clip_em = torch.tensor(np.load('./clip_em.npy'))
pos_a_normed = sparse.csr_matrix(np.load('./pos_a_normed.npy'))
neg_a_normed = sparse.csr_matrix(np.load('./neg_a_normed.npy'))
z_pos = scipy_sparse_mat_to_torch_sparse_tensor(pos_a_normed)
z_neg = scipy_sparse_mat_to_torch_sparse_tensor(neg_a_normed)

model = Cliprec(clip_em,z_pos,z_neg,u_num,video_num,total_u_id_list,total_v_id_list,step_train=True).cuda()
model.load_state_dict(torch.load('model0.pth'))
dataset_val = CliprecDataset(num_frm=3,ensemble_n_clips=4,num_labels=4,datadir='./fast_test_samples1.csv',videodir='./micro_video2', visual_feature_dir = './fast_video_feature')
val_loader = DataLoader(dataset_val,batch_size=bsz,shuffle=False)

# best_val_auc = 0
# best_recall3 = 0
# best_recall5 = 0
# best_ndcg3 = 0
# best_ndcg5 = 0
# best_logloss = 100
for epoch in range(1):
    if epoch>-1:
        with torch.no_grad():        
            p_matrix = np.zeros((0,3))
            #result = []
            #result_ans = []
            #uid = []
            for step, batch in enumerate(val_loader):
                num_clips = 4
                num_frm = 3
                result=[]
                result_ans=[]
                uid =[]
                val_output,val_label,val_uid = model(batch,False)
                result.extend(val_output.cpu().numpy())
                result_ans.extend(val_label)
                uid.extend(val_uid)
                m = np.hstack((np.array([t.numpy() for t in uid]).reshape(-1,1),np.array(result).reshape(-1,1)))
                n = np.hstack((m,np.array(result_ans).reshape(-1,1)))
                
                p_matrix = np.vstack((p_matrix,n))
            val_auc = roc_auc_score(p_matrix[:,-1],p_matrix[:,-2])
            logloss = logloss_cal(np.array(p_matrix[:,-1]),np.array(p_matrix[:,-2])+0.00005)
            ndcg3 = 0
            ndcg5 = 0
            recall3 = 0
            recall5 = 0
            for i in range(int(len(os.listdir('./test_samples')))):
                dataset_val1 = CliprecDataset(num_frm=3,ensemble_n_clips=4,num_labels=4,datadir='./test_samples/'+os.listdir('./test_samples')[i],videodir='./micro_video2', visual_feature_dir = './fast_video_feature')
                val_loader1 =  DataLoader(dataset_val1,batch_size=26,shuffle=False)

                for step, batch in enumerate(val_loader1):
                    num_clips = 4
                    num_frm = 3
                    result=[]
                    result_ans=[]
                    uid =[]
                    p_matrix = np.zeros((0,3))
                    val_output,val_label,val_uid = model(batch,False) 
                    result.extend(val_output.cpu().numpy())
                    result_ans.extend(val_label)
                    uid.extend(val_uid)
                    m = np.hstack((np.array([t.numpy() for t in uid]).reshape(-1,1),np.array(result).reshape(-1,1)))
                    n = np.hstack((m,np.array(result_ans).reshape(-1,1)))
                    p_matrix = np.vstack((p_matrix,n))
                    recall3_temp,ndcg3_temp = metric_cal(p_matrix,3)
                    recall5_temp,ndcg5_temp = metric_cal(p_matrix,5)

                    ndcg3 = ndcg3 + ndcg3_temp
                    ndcg5 = ndcg5 + ndcg5_temp
                    recall3 = recall3 + recall3_temp
                    recall5 = recall5 + recall5_temp

            ndcg3 = ndcg3/i
            ndcg5 = ndcg5/i
            recall3 = recall3/i
            recall5 = recall5/i
            print('val_auc:',val_auc,'Recall@3',recall3,'Recall@5',recall5,'NDCG@3',ndcg3,'NDCG@5',ndcg5,'logloss',logloss)
            
