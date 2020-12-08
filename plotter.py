import pandas as pd
import matplotlib.pyplot as plt
import numpy as numpy

def plot_graph(y1, y2, x1, ENV_NAME, NO_OF_TRAINERS, ROLLING):
    plt.figure(figsize=[12, 9])
    plt.subplot(1, 1, 1)
    plt.title(ENV_NAME)
    plt.xlabel('Steps:')
    plt.ylabel('Avg Reward after 3 runs')
    plt.plot(x1, pd.DataFrame(y1)[0].rolling(ROLLING).mean(), color='green',  marker='.', label='aggregated_agent({})'.format(NO_OF_TRAINERS))
    plt.plot(x1, y1, color='lightgreen')
    plt.plot(x1, pd.DataFrame(y2)[0].rolling(ROLLING).mean(), color='red',  marker='.', label='single_agent')
    plt.plot(x1, y2, color='pink')
    plt.grid()
    plt.legend()

    # plt.show()
    plt.savefig('plots/'+ENV_NAME+'_'+str(NO_OF_TRAINERS)+'plot.png')
    plt.close()