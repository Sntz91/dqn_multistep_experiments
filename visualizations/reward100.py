import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import glob, os   

#data
i = 1

def read_csv_idx(file, nstep):
    print(file)
    df_ = pd.read_csv(file)
    
    df_['filenr'] = pd.Series(np.repeat(i, len(df_)))
    df_['nstep'] = pd.Series(np.repeat(nstep, len(df_)))
    df_['help_mark'] = pd.Series(np.repeat(1, len(df_)))

    return df_

def df_fct(n): return pd.concat(map(lambda file: read_csv_idx(file, n), glob.glob(os.path.join('', "../data/old/"+str(n)+"/*.csv"))))
df_old = pd.concat((df_fct(n) for n in [1, 3, 5]))
df_old = df_old.reset_index()

print(df_old)

plot_df = df_old

#axvcol = sns.color_palette('crest', 3)[2]
axvcol = 'black'

colors = sns.color_palette('rocket', 4)

mpl.rcParams['text.usetex']=True
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

fig, ax1 = plt.subplots(figsize=(12,8))


labels = ['n=1', 'n=3', 'n=5']
for n, i in zip((1, 3, 5), range(3)):
    ax1.scatter(plot_df[plot_df.nstep==n].Step.iloc[-1:].values, plot_df[plot_df.nstep==n].Value.iloc[-1:].values, color=colors[i])
    ax1.plot(plot_df[plot_df.nstep==n].Step.values, plot_df[plot_df.nstep==n].Value.values, color=colors[i], label=labels[i])

ax1.axhline(16, color=axvcol, linestyle='--')


ax1.text(0.13e6, 15.5, '$m_{100}$=16', size=14, color='black',
            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.2))


#ax1.text(1.1e6, -18, 'avg $m_{100}$ for n=1', size=14, color=axvcol,
#            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.2))

plt.minorticks_on()
ax1.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
ax1.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
ax1.set_xlim(0, 2e6)
ax1.set_ylim(-21, 22)

xticks = np.arange(0, 2.1e6, 0.5e6)
yticks = np.arange(-20, 22, 10)

plt.xticks(xticks)
plt.yticks(yticks)

ax1.set(xlabel='frames')
ax1.set_ylabel('$m_{100}$')


ax1.xaxis.set_label_position('bottom') 
ax1.yaxis.set_label_position('right')

ax1.legend()




plt.tight_layout()

plt.savefig('reward100.png', dpi=400)

plt.show()
