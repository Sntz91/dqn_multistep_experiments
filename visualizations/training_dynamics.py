import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle

#data
df = pickle.load(open("df.p", "rb"))

df1 = df[df.nstep==1]
df5 = df[df.nstep==5]
df15 = df[df.nstep==15]
df30 = df[df.nstep==30]

steps_needed_n1 = [df1[df1.filenr==i+1].iloc[-1:, 2].values for i in range(10)]
steps_needed_n5 = [df5[df5.filenr==i+1].iloc[-1:, 2].values for i in range(10)]
steps_needed_n15 = [df15[df15.filenr==i+1].iloc[-1:, 2].values for i in range(10)]
steps_needed_n30 = [df30[df30.filenr==i+1].iloc[-1:, 2].values for i in range(10)]
steps_needed = (steps_needed_n1, steps_needed_n5, steps_needed_n15, steps_needed_n30)

steps_needed_df = pd.DataFrame()
steps_needed_df = steps_needed_df.append([[x[0], 1] for x in steps_needed_n1])
steps_needed_df = steps_needed_df.append([[x[0], 5] for x in steps_needed_n5])
steps_needed_df = steps_needed_df.append([[x[0], 15] for x in steps_needed_n15])
steps_needed_df = steps_needed_df.append([[x[0], 30] for x in steps_needed_n30])
steps_needed_df.columns = ["nr_of_frames", "nstep"]


plot_df = df1

#axvcol = sns.color_palette('crest', 3)[2]
axvcol = 'black'

colors = sns.color_palette('rocket', 10)

mpl.rcParams['text.usetex']=True
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

fig, ax1 = plt.subplots(figsize=(5,5))


for i in range(10):
    ax1.scatter(plot_df[plot_df.filenr==i].Step.iloc[-1:].values, plot_df[plot_df.filenr==i].Value.iloc[-1:].values, color=colors[i])
    ax1.plot(plot_df[plot_df.filenr==i].Step.values, plot_df[plot_df.filenr==i].Value.values, color=colors[i])

ax1.axvline(np.mean(steps_needed_n1), color=axvcol, linestyle='--')


ax1.text(0.13e6, 17.5, 'step size n=1', size=14, color='black')


ax1.text(1.1e6, -18, 'avg $m_{100}$ for n=1', size=14, color=axvcol,
            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.2))

plt.minorticks_on()
ax1.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
ax1.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
ax1.set_xlim(0, 2e6)
ax1.set_ylim(-21, 21)

xticks = np.arange(0, 2.1e6, 0.5e6)
yticks = np.arange(-20, 21, 10)

plt.xticks(xticks)
plt.yticks(yticks)

ax1.set(xlabel='frames', ylabel='$m_{100}$')



ax1.xaxis.set_label_position('bottom') 
ax1.yaxis.set_label_position('right')




plt.tight_layout()

plt.savefig('training_dynamics_n1.png', dpi=400)

plt.show()
