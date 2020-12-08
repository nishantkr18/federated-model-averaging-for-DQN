import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(y1, y2, x1, ENV_NAME, NO_OF_TRAINERS, ROLLING, HOW_MANY_VALUES = None):
    if(HOW_MANY_VALUES == None):
        HOW_MANY_VALUES = len(y1)
    x1 = x1[:HOW_MANY_VALUES]
    y1 = y1[:HOW_MANY_VALUES]
    y2 = y2[:HOW_MANY_VALUES]
    plt.figure(figsize=[12, 9])
    plt.subplot(1, 1, 1)
    plt.title(ENV_NAME)
    plt.xlabel('Steps:')
    plt.ylabel('Avg Reward after 3 runs')
    plt.plot(x1, y1, color='lightgreen')
    plt.plot(x1, y2, color='pink')
    plt.plot(x1, pd.DataFrame(y1)[0].rolling(ROLLING).mean(), color='green',  marker='.', label='aggregated_agent({})'.format(NO_OF_TRAINERS))
    plt.plot(x1, pd.DataFrame(y2)[0].rolling(ROLLING).mean(), color='red',  marker='.', label='single_agent')
    plt.grid()
    plt.legend()

    # plt.show()
    plt.savefig('plots/'+ENV_NAME+'_'+str(NO_OF_TRAINERS)+'plot.png')
    plt.close()

if __name__ == "__main__":
    # ENV_NAME = 'Acrobot-v1'
    # ENV_NAME = 'CartPole-v0'
    ENV_NAME = 'LunarLander-v2'
    y1 = np.loadtxt('arrays/scores_global_agent_'+ENV_NAME+'.csv')
    y2 = np.loadtxt('arrays/scores_single_agent_'+ENV_NAME+'.csv')
    x1 = np.loadtxt('arrays/steps_'+ENV_NAME+'.csv')
    plot_graph(y1, y2, x1, ENV_NAME, 3, 10, HOW_MANY_VALUES =100)