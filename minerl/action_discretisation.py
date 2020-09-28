from typing import Callable, List, Dict, Union

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

import minerl


def get_good_demo_names(env:str,path:str,score_percent:float) -> np.ndarray:
    '''
    :param env: Name of the environment
    :param path: Extra path in case we don't want default value
    :param score_percent: Percent of max score required to be considered a good demo

    :returns:
        - good_demos: Numpy array of trajectories names that get reward requirement
    '''

    data = minerl.data.make(env,path)

    traj_names = np.array(data.get_trajectory_names())

    demo_rewards = []

    for i in range(len(traj_names)):
        total_reward = 0.0
        for _,_,r,_,_ in data.load_data(traj_names[i]):
            total_reward = total_reward + r
        demo_rewards.append(total_reward)

    demo_rewards = np.array(demo_rewards)
    min_score = score_percent * np.max(demo_rewards)
    good_demos = traj_names[demo_rewards > min_score]

    return good_demos


def get_inventory_actions(env:str,path:str,trajectory_names:np.ndarray,agreement_percent:float) -> np.ndarray:
    '''
    :param env: Name of the environment
    :param path: Extra path in case we don't want default value
    :param trajectory_names: Numpy array of trajectories names to use
    :param agreement_percent: Percent of trajectories that need to contain an action for it to be considered key

    :returns:
        - key_inventory_actions: Numpy array of actions that affect the inventory and occur across X% of trajectories
    '''

    data = minerl.data.make(env,path)

    inventory_actions = []

    for j in range(len(trajectory_names)):
        tmp_actions = []
        inventory_change = []
        for s,a,_,ns,_ in data.load_data(trajectory_names[j]):
            tmp_actions.append(a['vector'])
            inventory_change.append(not(np.all(s['vector'] == ns['vector'])))
        _,indexs = np.unique(np.array(tmp_actions),return_index=True,axis=0)
        for i in range(len(tmp_actions)):
            if inventory_change[i] and i in indexs:
                inventory_actions.append(tmp_actions[i])

    min_agreement = agreement_percent * len(trajectory_names)
    inventory_actions = np.array(inventory_actions)

    unique_inventory_actions,counts = np.unique(inventory_actions,axis=0,return_counts=True)

    key_inventory_actions = unique_inventory_actions[counts > min_agreement]

    return key_inventory_actions


def get_kmeans_actions(env:str,path:str,trajectory_names:np.ndarray,n_clusters:int) -> np.ndarray:
    '''
    :param env: Name of the environment
    :param path: Extra path in case we don't want default value
    :param trajectory_names: Numpy array of trajectories names to use
    :param n_clusters: Number of clusters/actions to find

    :returns:
        - kmeans_actions: Numpy array of actions found by kmeans
    '''

    data = minerl.data.make(env,path)
    actions = []
    for i in range(len(trajectory_names)):
        for _,a,_,_,_ in data.load_data(trajectory_names[i]):
            actions.append(a['vector'])

    actions = np.array(actions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(actions)
    kmeans_actions = kmeans.cluster_centers_

    return kmeans_actions


def get_action_set(env:str,path:str,n_clusters:int,score_percent:float=0.9,agreement_percent:float=0.2) -> np.ndarray:
    '''
    :param env: Name of the environment
    :param path: Extra path in case we don't want default value
    :param n_clusters: Number of clusters/actions to find
    :param score_percent: Percent of max score required to be considered a good demo
    :param agreement_percent: Percent of trajectories that need to contain an action for it to be considered key


    :returns:
        - actions_set: Numpy array of actions found by kmeans and inventory actions
    '''

    good_demos = get_good_demo_names(env,path,score_percent)
    inventory_actions = get_inventory_actions(env,path,good_demos,agreement_percent)
    kmeans_actions = get_kmeans_actions(env,path,good_demos,n_clusters)

    if len(inventory_actions) > 0:
        action_set = np.concatenate((kmeans_actions,inventory_actions),axis=0)
    else:
        action_set = kmeans_actions

    return action_set


def generate_action_parser(action_set) -> Callable[[Dict[str, np.ndarray]], int]:
    def action_parser(action: Union[Dict, np.ndarray]):
        true_action = action['vector'] if isinstance(action, dict) else action
        dis = pairwise_distances(action_set, true_action.reshape(1, -1))
        discrete_action = np.argmin(dis)
        return discrete_action
    return action_parser
