from typing import List, Tuple

import minerl
import numpy as np

def get_key_actions(env: str,path: str) -> Tuple[np.ndarray, List[str]]:
    '''
    :param env: Name of the environment
    :param path: Extra path in case we don't want default value

    :returns:
        - key_actions: Numpy array of actions. Actions that always happen
                       that affect agent's inventory, as seen on demonstrations.
                       (i.e crafting actions)
        - Name of good trajectories
    '''
    
    score_percent = 0.9
    agreement_percent = 0.2
    
    data = minerl.data.make(env,path)
    
    traj_names = np.array(data.get_trajectory_names())
    data_dict = {}
    demo_rewards = []
    
    for i in range(len(traj_names)):
        reward = 0.0
        tmp_actions = []
        tmp_s = []
        tmp_ns = []
        for s,a,r,ns,d in data.load_data(traj_names[i]):
            reward = reward + r
            tmp_actions.append(a['vector'])
            tmp_s.append(s['vector'])
            tmp_ns.append(ns['vector'])
        actions = []
        _,indexs = np.unique(np.array(tmp_actions),return_index=True,axis=0)
        for t in range(len(tmp_actions)):
            if not(np.all(tmp_s[t] == tmp_ns[t])) and t in indexs:
                actions.append(tmp_actions[t])
        demo_rewards.append(reward)
        data_dict[traj_names[i]] = actions
        
    demo_rewards = np.array(demo_rewards)
    
    min_score = score_percent * np.max(demo_rewards)
    
    good_demos = traj_names[demo_rewards > min_score]
    traj_num = len(good_demos)
    
    actions = data_dict[good_demos[0]]
    
    for i in range(1,len(good_demos)):
        actions = np.concatenate((actions,data_dict[good_demos[i]]),axis=0)
    
    unique_actions,counts = np.unique(actions,axis=0,return_counts=True)
    
    min_agreement = agreement_percent * traj_num
    
    key_actions = unique_actions[counts > min_agreement]
    
    return key_actions,good_demos

def get_good_demo_kmeans_clustering(env,path,demo_names,n_clusters):
    data = minerl.data.make(env,path)
    actions = []
    for i in range(len(demo_names)):
        for _,act,_,_,_ in data.load_data(demo_names[i]):
            actions.append(act['vector'])
    
    actions = np.array(actions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(actions)
    return kmeans.cluster_centers_

def generate_action_parser(action_set):
    
    def action_parser(action):
        dis = pairwise_distances(action_set,action['vector'].reshape(1, -1))
        discrete_action = np.argmin(dis)
        return discrete_action
    
    return action_parser
