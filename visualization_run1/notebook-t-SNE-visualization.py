#!/usr/bin/env python
# coding: utf-8

# # T-SNE Visualization Tool

# ## 1) Collect trajectories

# ### a) Create basis trajectories:

# In[9]:


import numpy as np
import os
import sys
import pickle
import pandas as pd
sys.path.append(os.path.abspath('../'))

from environments.gym_parser import parse_gym_environment
from rl_algorithms import rockAgent, paperAgent, scissorsAgent, randomAgent

def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')

def RPSTask(RPSenv):
    return parse_gym_environment(RPSenv)


# In[10]:


from tqdm import tqdm
from multiagent_loops import simultaneous_action_rl_loop


def collect_basis_trajectories_for(env, agents, fixed_opponents, nbr_episodes_matchup):
    trajs = {'agent':[],
                'opponent':[],
                'trajectory':[]
                }
    
    progress_bar = tqdm(range(len(fixed_opponents)))
    for e in progress_bar:
        fixed_opponent = fixed_opponents[e]
        for agent in agents:
            trajectories = simulate(env, agent, fixed_opponent, episodes=nbr_episodes_matchup, training=False)
            for t in trajectories:
                trajs['agent'].append( fixed_opponent.name)
                trajs['opponent'].append( agent.name)
                trajs['trajectory'].append( t)
        progress_bar.set_description(f'Collecting trajectories: {agent.name} against {fixed_opponent.name}.')
    return trajs

def simulate(env, agent, fixed_opponent, episodes, training):
    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    mode = 'Training' if training else 'Inference'
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        trajectory = simultaneous_action_rl_loop.run_episode(env, agent_vector, training=training)
        trajectories.append(trajectory)
        progress_bar.set_description(f'{mode} {agent.name} against {fixed_opponent.name}')
    return trajectories


# In[14]:


trajectories = collect_basis_trajectories_for(RPSenv(), 
                                              [randomAgent],
                                             [rockAgent, paperAgent, scissorsAgent],
                                             nbr_episodes_matchup=10000)


# In[15]:


trajectories.keys()


# In[16]:


nbr_basis_trajectories = len(trajectories['trajectory'])
print(nbr_basis_trajectories)


# ### b) Collect trajectories from training:

# In[17]:


def all_files_in_directory(directory):
    return [os.path.join(directory, f)
            for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def all_folders_in_directory(directory):
    return [os.path.join(directory, f)
            for f in os.listdir(directory) if not( os.path.isfile(os.path.join(directory, f))) ]

def all_files_in_directory(directory):
    return [os.path.join(directory, f)
            for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) ]

def get_agent_name_from_full_path(filename):
    return os.path.splitext(filename)[0].split('/')[-1]

def get_episode_number_from_full_path(filename):
    return os.path.splitext(filename)[0].split('/')[-1]

def collect_trajectories_from(run_dir, with_policies=False):
    menageries_dir = os.path.join(run_dir,'menageries')
    trajs = {'agent':[],
             'opponent':[],
             'trajectory':[],
             'episode':[]
                }
    policies = []
    
    for folder in all_folders_in_directory(menageries_dir):
        agent_name = get_agent_name_from_full_path(folder)
        policies_files = all_files_in_directory(folder)
        
        if with_policies:
            progress_bar = tqdm(range(len(policies_files)))
            for e in progress_bar:
                f = files[e]
                file_name = get_file_name_from_full_path(f)
                policy = AgentHook.unhook(path=f)
                policies.append(policy)
        
        trajectory_folder = os.path.join(folder, 'trajectories')
        trajectory_files = all_files_in_directory(trajectory_folder)
        progress_bar = tqdm(range(len(trajectory_files)))
        for e in progress_bar:
            t = trajectory_files[e]
            episode_number = get_episode_number_from_full_path(t)
            traj = pickle.load(open(t, 'rb'))
            trajs['agent'].append(agent_name)
            trajs['opponent'].append(agent_name)
            trajs['trajectory'].append(traj)
            trajs['episode'].append(episode_number)
    
    return trajs, policies
        


# In[19]:


#source_dir = "/home/kevin/Development/git/Generalized-RL-Self-Play-Framework/experiment/experiment-Naive-TrajTest-CH1e3/"
source_dir = '/home/kevin/experiment/'
run_dir = os.path.join(source_dir, "run-0")
#run-0/menageries/NaiveSP-ppo_h64_mlp/trajectories
trajs, policies = collect_trajectories_from( run_dir, with_policies=False)


# In[20]:


len(trajs['episode'])


# ### Add those to the current basis trajectories:

# In[21]:


for k in trajs.keys():
    for idx in range(len(trajs[k])):
        if k not in trajectories: 
            trajectories[k] = [None]*nbr_basis_trajectories
        trajectories[k].append( trajs[k][idx])


# ## 2) Encode trajectories

# In[22]:


import copy
ts = copy.deepcopy(trajectories['trajectory'])
print(f'Nbr traj: {len(ts)} // Steps per traj: {len(ts[0])} // Elements per steps: {len(ts[0][0])}')


# In[23]:


a0 = ts[0][0][1]
print(a0)
s0 = ts[0][0][0][0][-1]
print(s0)

oh_a0 = [ ]


# In[24]:


#actions = [ [step[1] for idx, step in enumerate(t) if idx<3]for t in ts]
#actions = [ [step[1] for idx, step in enumerate(t) if idx<10]for t in ts]

actions = [ [step[0][0][-1] for idx, step in enumerate(t) if idx<10 and idx>0]for t in ts]


# In[25]:


actions = np.asarray(actions)
actions.shape


# In[26]:


x_actions = copy.deepcopy(actions)
y_agents = np.asarray( copy.deepcopy(trajectories['agent']) )
print(x_actions.shape, y_agents.shape)


# In[27]:


data_dir = './data'
x_actions_dir = './x_actions'
y_agents_dir = './y_agents'
pickle.dump( trajectories, open(data_dir, 'wb'))
pickle.dump( x_actions, open(x_actions_dir, 'wb'))
pickle.dump( y_agents, open(y_agents_dir, 'wb'))


# In[28]:


data_dir = './data'
x_actions_dir = './x_actions'
y_agents_dir = './y_agents'
ttrajectories = pickle.load( open(data_dir, 'rb'))
tx_actions = pickle.load( open(x_actions_dir, 'rb'))
ty_agents = pickle.load( open(y_agents_dir, 'rb'))


# In[63]:


"""
def encode_trajectory(data):
    traj = data['trajectory']
    actions = np.asarray( [ [step[1] for idx, step in enumerate(t) if idx<10]for t in traj] )
    agents = np.asarray( data['agent'])
    return actions, agents
"""


# ## 3) Create t-SNE

# In[64]:


from sklearn.manifold import TSNE


# In[65]:


n_dims = 2
shuffle = False
if shuffle:
    p = np.random.permutation(len(x_actions))
    x_actions = x_actions[p]
    y_agents = y_agents[p]

X_sample_flat = np.reshape(x_actions, [x_actions.shape[0], -1])
perplexities = [5, 50, 100,200,300,500]
#embeddings = TSNE(n_components=n_dims, init='pca', random_state=17, verbose=2, perplexity=perplexities[1]).fit_transform(X_sample_flat)
embeddings = []
for perplexity in perplexities:
    embeddings.append( 
        TSNE(n_components=n_dims, 
                  init='pca',
                  #init='random', 
                  random_state=17, 
                  verbose=2, 
                  learning_rate=300,
                  n_iter=1000,
                  perplexity=perplexity
                 ).fit_transform(X_sample_flat)
    )


# In[66]:


embeddings_dir = './embedding'
pickle.dump(embeddings, open(embeddings_dir, "wb"))


# In[5]:


embeddings_dir = './embedding'
embeddings = pickle.load( open(embeddings_dir, 'rb'))


# In[35]:


X_sample_flat = np.reshape(x_actions, [x_actions.shape[0], -1])
perplexities = [5, 50, 100,200,300,500]


# ## Plot

# In[29]:


trajectories['episode'] = list(map( lambda ep: int(ep.split('_')[-1]) if ep is not None else ep, trajectories['episode']) )
trajectories['index'] = [idx for idx in range(len(trajectories['episode']))]
for idx, embedding in enumerate(embeddings):
    trajectories[f'emb{idx}'] = [ embedding[i] for i in range(embedding.shape[0]) ]


# In[69]:


df = pd.DataFrame(trajectories)


# In[71]:


training_df = df[ df.episode != None]


# In[72]:


df_dir = './dataframe.df'
pickle.dump(df, open(df_dir, 'wb') )


# In[30]:


df_dir = './dataframe.df'
df = pickle.load(open(df_dir, 'rb') )


# In[32]:


training_df = df[ df.episode != None]
sorted_training_df = training_df.sort_values(by=['episode'])


# In[74]:


sorted_training_df


# In[ ]:





# In[87]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter

num_classes = len(np.unique(y_agents))
labels = np.unique(y_agents)
y_sample = copy.deepcopy(y_agents)

# plot the 2D data points
labels.sort()
labels = list(reversed(labels))
print(labels)

for idx, perplexity in enumerate(perplexities):
    plot_dir = f'./plot-t-sne_per={perplexity}'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #colors = cm.Spectral(np.linspace(0, 1, num_classes))
    
    #s = np.linspace(2, 3, 10)
    #cmap = sns.cubehelix_palette(start=s[5], light=1, as_cmap=True)
    #colors = sns.cubehelix_palette(start=s[5], light=1, as_cmap=False)
    colors = ["red","yellow","orange", "blue", "green"]    
    
    xx = embeddings[idx][:, 0]
    yy = embeddings[idx][:, 1]
    
    for idx, label in enumerate(labels):
        xl = xx[y_sample==label]
        yl = yy[y_sample==label]
        #ax.scatter(xl, yl, color=colors[idx], label=label, s=10)
        sns.kdeplot(xl, yl, color=colors[idx], n_levels=5, shade=False, cut=1, ax=ax, alpha=0.5, cbar=False)


    #ax.xaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('off')
    plt.axis('tight')
    #plt.legend(loc='best', scatterpoints=1, fontsize=10)
    plt.savefig(plot_dir+'-multi.pdf', format='pdf', dpi=1000, transparent=True)
    plt.savefig(plot_dir+'-multi.png', format='png', dpi=1000, transparent=True)
    plt.show()
    plt.close(fig)


# In[89]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import seaborn as sns

num_classes = len(np.unique(y_agents))
labels = np.unique(y_agents)
y_sample = copy.deepcopy(y_agents)

# plot the 2D data points
labels.sort()
labels = list(reversed(labels))
print(labels)
always_on_labels = [labels[0], labels[1], labels[2]]
#agent_labels = ['NaiveSP-ppo_h64_rnn']
agent_labels = list( set(labels) - set(always_on_labels) )
per_agent_labels_list = [ (agent_label, always_on_labels+[agent_label]) for agent_label in agent_labels]
print(agent_labels)

for idx, perplexity in enumerate(perplexities):
    for idx_label, pa_labels in enumerate(per_agent_labels_list):
        agent_label = pa_labels[0]
        pa_labels_list = pa_labels[1]
        plot_dir = f'./plot-t-sne_{agent_label}_per={perplexity}'

        xx = embeddings[idx][:, 0]
        yy = embeddings[idx][:, 1]
        
        # Create a cubehelix colormap to use with kdeplot
        s = np.linspace(0, 3, 10)[5]
        cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        # Draw the background:
        xl = xx[y_sample==agent_label]
        yl = yy[y_sample==agent_label]
        sns.kdeplot(xl, yl, cmap=cmap, shade=False, cut=1, ax=ax, alpha=0.8)

        for idx_label, label in enumerate(pa_labels_list):
            xl = xx[y_sample==label]
            yl = yy[y_sample==label]
            ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.6)

        #ax.xaxis.set_major_formatter(NullFormatter())
        #ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('off')
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=10)
        plt.savefig(plot_dir+'.eps', format='eps', dpi=1000, transparent=True)
        plt.savefig(plot_dir+'.png', format='png', dpi=1000, transparent=True)
        print(plot_dir)
        plt.show()
        plt.close(fig)
        
        
        # SHADE ON :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        # Draw the background:
        xl = xx[y_sample==agent_label]
        yl = yy[y_sample==agent_label]
        sns.kdeplot(xl, yl, cmap=cmap, shade=True, cut=1, ax=ax, alpha=0.5, cbar=True)

        for idx_label, label in enumerate(pa_labels_list):
            xl = xx[y_sample==label]
            yl = yy[y_sample==label]
            ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.8)

        #ax.xaxis.set_major_formatter(NullFormatter())
        #ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('off')
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=10)
        plt.savefig(plot_dir+'_shaded.eps', format='eps', dpi=1000, transparent=True)
        plt.savefig(plot_dir+'_shaded.png', format='png', dpi=1000, transparent=True)
        print(plot_dir)
        #plt.show()
        plt.close(fig)


# ## Plot the average evolution of the menagerie through time:

# In[103]:


import numpy as np
import matplotlib.pyplot as plt


def find_centroid(points):
    '''
    Finds centroid from set of points
    :param points: np.array of points
    '''
    #length = points.shape[0]
    #sum_x = np.sum(points[:, 0])
    #sum_y = np.sum(points[:, 1])
    #return np.array([sum_x/length, sum_y/length])
    
    #return np.mean(points, axis=0)
    return np.median(points, axis=0)

def compute_std(points):
    return np.std(points, axis=0)

def divide_points(points, divisions):
    '''
    Divides :param: points into sub np.arrays
    :param points: np.array of points
    :param divisions: int, number of sub np.arrays we wish to divide the original :param: points
    '''
    return np.array_split(points, divisions)


def link_points(points, ax, colour_map, alpha):
    '''
    Plots :param: points as points linked by lines on :param: ax Axes.
    The colour of the points and the lines is a 'smooth' colour transition from
    both colour extremes of the :param: colour_map
    '''
    number_of_points = points.shape[0]
    colours = [colour_map(1.*i/(number_of_points-1)) for i in range(number_of_points-1)]
    point_pairs = zip(points, points[1:]) # Creates a pairs of points (p_0, p_1), (p_1, p_2)...
    for point_pair, colour in zip(point_pairs, colours):
        numpy_point_pair = np.array(point_pair)
        ax.plot(numpy_point_pair[:, 0], numpy_point_pair[:, 1], '-o', color=colour, alpha=alpha)


shape='full'
head_starts_at_zero=True
arrow_h_offset = 0.1  # data coordinates, empirically determined
max_arrow_width=0.25
max_arrow_length = 1 - 1.2 * arrow_h_offset
max_head_width = 4.5 * max_arrow_width
max_head_length = 5 * max_arrow_width
arrow_params = {'length_includes_head': True, 'shape': shape,
                'head_starts_at_zero': head_starts_at_zero}

def draw_arrow(ax, point_a, point_b, alpha, color):
    # set the length of the arrow
    length = max_arrow_length
    width = max_arrow_width
    head_width = max_head_width
    head_length = max_head_length
    
    point_a = np.reshape( point_a, (-1))
    point_b = np.reshape( point_b, (-1))
    delta = point_b-point_a
    dx, dy = delta[0], delta[1]
    x_a, y_a = point_a[0], point_a[1] 
    ax.arrow(x_a, y_a, dx, dy,
              fc=color, ec=color, alpha=alpha, width=width,
              head_width=head_width, head_length=head_length,
              **arrow_params
             )
    
def point_at_points(points, stds, ax, colour_map, alpha, until):
    '''
    Plots :param: points as points linked by lines on :param: ax Axes.
    The colour of the points and the lines is a 'smooth' colour transition from
    both colour extremes of the :param: colour_map
    '''
    number_of_points = points.shape[0]
    colours = [colour_map(1.*i/(number_of_points-1)) for i in range(number_of_points-1)]
    point_pairs = list(zip(points, points[1:])) # Creates a pairs of points (p_0, p_1), (p_1, p_2)...
    if until is None:
        until = len(colours)
    for idx, (point_pair, colour) in enumerate(zip(point_pairs[:until], colours[:until])):
        numpy_point_pair = np.array(point_pair)
        if idx==0: ax.plot(numpy_point_pair[0, 0], numpy_point_pair[0, 1], '-o', color=colour, alpha=alpha)
        draw_arrow(ax, numpy_point_pair[0, :], numpy_point_pair[1, :], alpha=alpha, color=colour)
        if idx==until-1: 
            #ax.scatter(numpy_point_pair[1, 0], numpy_point_pair[1, 1], marker='x', color=colour, alpha=alpha, s=10)
            #ax.scatter(numpy_point_pair[1, 0], numpy_point_pair[1, 1], marker='o', color=colour, alpha=0.5, s=10*np.sqrt( np.square(stds[idx]).sum()) )
            ax.scatter(numpy_point_pair[1, 0], numpy_point_pair[1, 1], marker='o', color=colour, alpha=0.5, s=150 )
            

def plot_trajectory_evolution_in_embedded_space(points, divisions, ax, colour_map, alpha):
    divided_points = divide_points(points, divisions=divisions)
    centroids = np.array([find_centroid(sub_points) for sub_points in divided_points])
    link_points(centroids, ax, colour_map, alpha)

def plot_trajectory_evolution_in_embedded_space_with_arrows(points, divisions, ax, colour_map, alpha, until=None):
    divided_points = divide_points(points, divisions=divisions)
    centroids = np.array([find_centroid(sub_points) for sub_points in divided_points])
    stds = np.array([compute_std(sub_points) for sub_points in divided_points])
    point_at_points(centroids, stds, ax, colour_map, alpha, until=until)


# In[118]:


agent_sorted_training_df = sorted_training_df[ sorted_training_df.agent == agent_label]
points = list(agent_sorted_training_df[f'emb{idx}'].values)
points = [ list(p) for p in points]
print(type(points) )
print(type(points[0]))
points = np.asarray(points)
print(points.shape)


# In[243]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import seaborn as sns

num_classes = len(np.unique(y_agents))
labels = np.unique(y_agents)
y_sample = copy.deepcopy(y_agents)

# plot the 2D data points
labels.sort()
labels = list(reversed(labels))
print(labels)
always_on_labels = [labels[0], labels[1], labels[2]]
#agent_labels = ['NaiveSP-ppo_h64_rnn']
agent_labels = list( set(labels) - set(always_on_labels) )
per_agent_labels_list = [ (agent_label, always_on_labels+[agent_label]) for agent_label in agent_labels]
print(agent_labels)

# Rainbow colour map:
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

colour_map = matplotlib.colors.LinearSegmentedColormap('rainbow_colormap',cdict,256)

for idx, perplexity in enumerate(perplexities[:]):
    for idx_label, pa_labels in enumerate(per_agent_labels_list):
        agent_label = pa_labels[0]
        pa_labels_list = pa_labels[1]
        plot_dir = f'./plot-t-sne_{agent_label}_per={perplexity}_evolution_through_time'

        xx = embeddings[idx][:, 0]
        yy = embeddings[idx][:, 1]
        
        # Create a cubehelix colormap to use with kdeplot
        s = np.linspace(0, 3, 10)[5]
        cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
        #colour_map = plt.get_cmap('winter')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        # Draw the background:
        xl = xx[y_sample==agent_label]
        yl = yy[y_sample==agent_label]
        sns.kdeplot(xl, yl, cmap=cmap, shade=False, cut=1, ax=ax, alpha=0.8, linestyles='dashed')

        for idx_label, label in enumerate(pa_labels_list):
            xl = xx[y_sample==label]
            yl = yy[y_sample==label]
            ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.6)
        
        agent_sorted_training_df = sorted_training_df[ sorted_training_df.agent == agent_label]
        points = list(agent_sorted_training_df[f'emb{idx}'].values)
        points = [ list(p) for p in points]
        points = np.asarray(points)
        #plot_trajectory_evolution_in_embedded_space(points, divisions=20, ax=ax, colour_map=colour_map, alpha=0.5)
        plot_trajectory_evolution_in_embedded_space_with_arrows(points, divisions=20, ax=ax, colour_map=colour_map, alpha=1.0)
        
        
        #ax.xaxis.set_major_formatter(NullFormatter())
        #ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('off')
        
        plt.axis('tight')
        #plt.legend(loc='best', scatterpoints=1, fontsize=10)
        plt.savefig(plot_dir+'.eps', format='eps', dpi=1000, transparent=True)
        plt.savefig(plot_dir+'.png', format='png', dpi=1000, transparent=False)
        print(plot_dir)
        plt.show()
        plt.close(fig)
        
        """
        # SHADE ON :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        # Draw the background:
        xl = xx[y_sample==agent_label]
        yl = yy[y_sample==agent_label]
        sns.kdeplot(xl, yl, cmap=cmap, shade=True, cut=1, ax=ax, alpha=0.5, cbar=True)

        for idx_label, label in enumerate(pa_labels_list):
            xl = xx[y_sample==label]
            yl = yy[y_sample==label]
            ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.8)
        
        plot_trajectory_evolution_in_embedded_space(points, divisions=20, ax=ax, colour_map=colour_map)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=10)
        plt.savefig(plot_dir+'_shaded.pdf', format='pdf', dpi=1000, transparent=True)
        plt.savefig(plot_dir+'_shaded.png', format='png', dpi=1000, transparent=True)
        print(plot_dir)
        #plt.show()
        plt.close(fig)
        """
        


# ### Plot evolution till T=t...

# In[104]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
import seaborn as sns

num_classes = len(np.unique(y_agents))
labels = np.unique(y_agents)
y_sample = copy.deepcopy(y_agents)

# plot the 2D data points
labels.sort()
labels = list(reversed(labels))
print(labels)
always_on_labels = [labels[0], labels[1], labels[2]]
#agent_labels = ['NaiveSP-ppo_h64_rnn']
agent_labels = list( set(labels) - set(always_on_labels) )
per_agent_labels_list = [ (agent_label, always_on_labels+[agent_label]) for agent_label in agent_labels]
print(agent_labels)

# Rainbow colour map:
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

colour_map = matplotlib.colors.LinearSegmentedColormap('rainbow_colormap',cdict,256)

for idx, perplexity in enumerate(perplexities[-2:]):
    for idx_label, pa_labels in enumerate(per_agent_labels_list[:]):
        agent_label = pa_labels[0]
        pa_labels_list = pa_labels[1]
        plot_dir = f'./plot-t-sne_{agent_label}_per={perplexity}_evolution_till'

        xx = embeddings[idx][:, 0]
        yy = embeddings[idx][:, 1]
        
        # Create a cubehelix colormap to use with kdeplot
        s = np.linspace(0, 3, 10)[6]
        cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
        color_data = 'b'
        num_labels = len(pa_labels_list)
        colors = cm.Spectral(np.linspace(0, 2, num_labels))

        agent_sorted_training_df = sorted_training_df[ sorted_training_df.agent == agent_label]
        points = list(agent_sorted_training_df[f'emb{idx}'].values)
        points = [ list(p) for p in points]
        points = np.asarray(points)
        nbr_divisions = 20
        n_levels = 5
        size_per_division = points.shape[0]//nbr_divisions 
        
        for until in range(nbr_divisions-1):
            from_l = (until)*size_per_division
            until_l = (until+1)*size_per_division
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Draw the background:
            xl = xx[y_sample==agent_label][from_l:until_l]
            yl = yy[y_sample==agent_label][from_l:until_l]
            sns.kdeplot(xl, yl, cmap=cmap, shade=False, n_levels=n_levels, 
                        cut=1, ax=ax, alpha=0.8, linestyles='dashed', 
                        linewidths=[(i+1)*0.75 for i in range(n_levels)],
                        cbar=False)

            for idx_label, label in enumerate(pa_labels_list):
                if label != agent_label:
                    xl = xx[y_sample==label]
                    yl = yy[y_sample==label]
                    ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.6)
                else:
                    xl = xx[y_sample==agent_label][from_l:until_l]
                    yl = yy[y_sample==agent_label][from_l:until_l]
                    ax.scatter(xl, yl, color=color_data, label='_nolegend_', s=3, alpha=0.3)
            
            plot_trajectory_evolution_in_embedded_space_with_arrows(points, divisions=nbr_divisions, ax=ax, colour_map=colour_map, alpha=1.0, until=until+1)


            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('off')

            plt.axis('tight')
            #plt.legend(loc='best', scatterpoints=1, fontsize=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.0),
                     ncol=3, fancybox=True, shadow=False, markerscale=5.0)
            plt.savefig(plot_dir+f'_T={until_l}.pdf', format='pdf', dpi=1000, transparent=True)
            plt.savefig(plot_dir+f'_T={until_l}.svg', format='svg', dpi=1000, transparent=True)
            print(plot_dir)
            plt.show()
            plt.close(fig)
            
            #
            # WIth kde from beginning:
            #
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Draw the background:
            xl = xx[y_sample==agent_label][:until_l]
            yl = yy[y_sample==agent_label][:until_l]
            sns.kdeplot(xl, yl, cmap=cmap, shade=False, n_levels=n_levels, 
                        cut=1, ax=ax, alpha=0.8, linestyles='dashed', 
                        linewidths=[(i+1)*0.75 for i in range(n_levels)],
                        cbar=False)

            for idx_label, label in enumerate(pa_labels_list):
                if label != agent_label:
                    xl = xx[y_sample==label]
                    yl = yy[y_sample==label]
                    ax.scatter(xl, yl, color=colors[idx_label], label=label, s=3, alpha=0.6)
                else:
                    xl = xx[y_sample==agent_label][from_l:until_l]
                    yl = yy[y_sample==agent_label][from_l:until_l]
                    ax.scatter(xl, yl, color=color_data, label='_nolegend_', s=3, alpha=0.3)
            
            plot_trajectory_evolution_in_embedded_space_with_arrows(points, divisions=nbr_divisions, ax=ax, colour_map=colour_map, alpha=1.0, until=until+1)


            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('off')

            plt.axis('tight')
            #plt.legend(loc='best', scatterpoints=1, fontsize=10)
            #ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.0),
            #         ncol=3, fancybox=True, shadow=False, markerscale=5.0)
            plt.savefig(plot_dir+f'_T={until_l}_with_KDE_from_T=0.pdf', format='pdf', dpi=1000, transparent=True)
            plt.savefig(plot_dir+f'_T={until_l}_with_KDE_from_T=0.svg', format='svg', dpi=1000, transparent=True)
            print(plot_dir)
            plt.show()
            plt.close(fig)


# In[ ]:




