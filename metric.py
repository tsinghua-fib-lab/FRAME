import numpy as np

def metric_cal(ground_truth,k): 
    user_watch={}
    K = k
    for i in range(len(ground_truth)):
        user_id = ground_truth[i][0]
        watch = ground_truth[i][2]
        probability = ground_truth[i][1]
        if(user_id not in user_watch):
            user_watch[user_id]=[]
        user_watch[user_id].append((watch,probability))
        
    
    for key in user_watch:
        recall = 0.0
        ndcg = 0.0
        user_watch[key].sort(key=lambda x:-x[1])
        
        watch_all=0
        predicate_correct=0
        for i in range(len(user_watch[key])):
            if(user_watch[key][i][0]==1):
                watch_all=watch_all+1

        for i in range(K):
            if(user_watch[key][i][0]==1):
                predicate_correct=predicate_correct+1
            ndcg = ndcg+user_watch[key][i][0]/np.log2(i+2)
        
        recall=float(predicate_correct)/watch_all

        return recall,ndcg

