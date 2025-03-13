import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def myplot(date, name, my_ylim, n_tick, add_legend=True):
    file_name = "./output/"+name+"_"+date+"_avg20.csv"
    # df 
    da = pd.read_csv(file_name)
    tmp = list(da.columns)
    tmp[0] = 'c'
    da.columns = tmp
    df = da

    # Figure and subplot setup
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(3.5, 3.5), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
    # plt.rcParams.update({'font.size': 14})  # Font size adjustment for readability

    # Plotting
    df_low = df[df['c'] <= 10]
    df_high = df[df['c'] >= 20]

    # colors = {'z=1000': 'black', 'z=10': 'orange', 'z=1': 'red', 'z=1e-1': 'blue', 'z=1e-5': 'green'}
    colors = {'z=1000': 'green', 'z=10': 'blue', 'z=1': 'red', 'z=1e-1': 'orange', 'z=1e-5': 'black'}

    # Plotting data
    for column in df.columns[1:]:
        ax1.plot(df_low['c'], df_low[column], label=column, color=colors[column])
        ax3.plot(df_high['c'], df_high[column], color=colors[column])

    # Setting x and y axes limits and ticks
    ax1.set_xlim(0, 10)
    ax1.set_xticks([0, 2, 5, 10])
    ax3.set_xlim(20, 100)
    ax3.set_xticks([20, 50, 100])
    ax1.axvline(x=1, color='grey', linestyle='--')  # Vertical dashed line at x=1

    # Y-axis settings
    y_ticks = np.linspace(0, my_ylim, n_tick)
    ax1.set_ylim(0, my_ylim)
    ax3.set_ylim(0, my_ylim)
    ax1.set_yticks(y_ticks)
    ax1.tick_params(axis='y', which='both', length=5)  # Adding y-axis ticks back to the left panel
    ax3.tick_params(axis='y', which='both', length=0)  # Ensuring right panel has no y-axis ticks

    # Adding 45-degree lines
    def add_diag_line(ax, pos, top):
        d = 0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        if top:
            ax.plot((pos-d, pos+d), (1-d, 1+d), **kwargs)  # Top diagonal line
        else:
            ax.plot((pos-d, pos+d), (-d, +d), **kwargs)  # Bottom diagonal line

    add_diag_line(ax1, 1, top=True)
    add_diag_line(ax1, 1, top=False)
    add_diag_line(ax3, 0, top=True)
    add_diag_line(ax3, 0, top=False)

    # Legend configuration
    if add_legend:
        handles, labels = ax1.get_legend_handles_labels()
        print('### check ###')
        print(labels)
        labels = ['$\gamma$=1000', '$\gamma$=10', '$\gamma$=1', '$\gamma$=1e-1', '$\gamma$=1e-5']
        print('### check ###')
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.9, 0.7))
    else:
        0
    
    # Frame settings
    for ax in [ax1, ax3]:
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # Grid and plot spacing
    ax1.grid(False)
    ax3.grid(False)
    plt.subplots_adjust(wspace=0.3)

    fig.text(0.5, -0.02, 'c = P/T', ha='center')  # Adjust vertical position with the second parameter if needed

    plt.savefig('./output/'+name+'_'+date+'_avg20_random_N2_py.pdf', format='pdf', bbox_inches='tight')

    # Display the figure
    # plt.show()
    plt.close()

date = '20240607'
myplot(date, 'mn', 0.6, 7, add_legend=False)
myplot(date, 'sm', 1.5, 6, add_legend=False)
myplot(date, 'sn', 1.5, 6, add_legend=False)
myplot(date, 'sr', 4.5, 10, add_legend=False)
myplot(date, 'hj', 2.5, 6, add_legend=True)