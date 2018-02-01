import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['font.size'] = 15

def svdplot(where):
    path = where + 'svds.npy'
    f = np.load(path).item()

    dis = [
        'D.conv1d.weight',
        'D.block.4.res_block.3.weight',
        'D.block.4.res_block.1.weight',
        'D.block.3.res_block.3.weight',
        'D.block.3.res_block.1.weight',
        'D.block.2.res_block.3.weight',
        'D.block.2.res_block.1.weight',
        'D.block.1.res_block.3.weight',
        'D.block.1.res_block.1.weight',
        'D.block.0.res_block.3.weight',
        'D.block.0.res_block.1.weight',
        'D.linear.weight']

    gen = [
        'G.conv1.weight',
        'G.block.4.res_block.3.weight',
        'G.block.4.res_block.1.weight',
        'G.block.3.res_block.3.weight',
        'G.block.3.res_block.1.weight',
        'G.block.2.res_block.3.weight',
        'G.block.2.res_block.1.weight',
        'G.block.1.res_block.3.weight',
        'G.block.1.res_block.1.weight',
        'G.block.0.res_block.3.weight',
        'G.block.0.res_block.1.weight',
        'G.fc1.weight']

    print('Plotting Gen ...')
    fig = plt.figure(figsize=np.array([40, 40]))
    plt.rcParams['lines.linewidth'] = 1.0
    for i, name in enumerate(gen):
            svs = np.array(f[name])
            ax = fig.add_subplot(12, 2, 2 * i + 1)
            ax.plot(range(len(svs)), svs)
            ax.set_title(name)
            if i != 11:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlim([0, (len(svs) - 1)])
            # ax.set_ylim([0, 15])
    print('Plotting Dis ...')
    for i, name in enumerate(dis):
            svs = np.array(f[name])
            ax = fig.add_subplot(12, 2, 2 * i + 2)
            ax.plot(range(len(svs)), svs)
            ax.set_title(name)
            if i != 11:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlim([0, (len(svs) - 1)])
            # ax.set_ylim([0, 15])

    plt.savefig('/results/1plot.png', dpi=150)
    plt.close()
