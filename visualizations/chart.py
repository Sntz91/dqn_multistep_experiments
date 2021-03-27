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


plot_df = df30

#axvcol = sns.color_palette('crest', 3)[2]
axvcol = 'black'

colors = sns.color_palette('crest', 10)

#mpl.rcParams['text.usetex']=True
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey=True, figsize=(11,11))


tmp = df1
for i in range(10):
    ax1.scatter(tmp[tmp.filenr==i].Step.iloc[-1:].values, tmp[tmp.filenr==i].Value.iloc[-1:].values, color=colors[i])
    ax1.plot(tmp[tmp.filenr==i].Step.values, tmp[tmp.filenr==i].Value.values, color=colors[i])
    
tmp = df5
for i in range(10):
    ax2.scatter(tmp[tmp.filenr==i].Step.iloc[-1:].values, tmp[tmp.filenr==i].Value.iloc[-1:].values, color=colors[i])
    ax2.plot(tmp[tmp.filenr==i].Step.values, tmp[tmp.filenr==i].Value.values, color=colors[i])

tmp = df15
for i in range(10):
    ax3.scatter(tmp[tmp.filenr==i].Step.iloc[-1:].values, tmp[tmp.filenr==i].Value.iloc[-1:].values, color=colors[i])
    ax3.plot(tmp[tmp.filenr==i].Step.values, tmp[tmp.filenr==i].Value.values, color=colors[i])

tmp = df30
for i in range(10):
    ax4.scatter(tmp[tmp.filenr==i].Step.iloc[-1:].values, tmp[tmp.filenr==i].Value.iloc[-1:].values, color=colors[i])
    ax4.plot(tmp[tmp.filenr==i].Step.values, tmp[tmp.filenr==i].Value.values, color=colors[i])

ax1.axvline(np.mean(steps_needed_n1), color=axvcol, linestyle='--')
ax2.axvline(np.mean(steps_needed_n1), color=axvcol, linestyle='--')
ax3.axvline(np.mean(steps_needed_n1), color=axvcol, linestyle='--')
ax4.axvline(np.mean(steps_needed_n1), color=axvcol, linestyle='--')

ax1.text(0.13e6, 17.5, 'step size n=1', size=14, color='black')
ax2.text(0.13e6, 17.5, 'step size n=5', size=14, color='black')
ax3.text(0.13e6, 17.5, 'step size n=15', size=14, color='black')
ax4.text(0.13e6, 17.5, 'step size n=30', size=14, color='black')

ax1.text(1.1e6, -18, 'avg $m_{100}$ for n=1', size=14, color=axvcol,
            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.2))

plt.minorticks_on()

for ax in (ax1, ax2, ax3, ax4):
    ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
    ax.set_xlim(0, 2e6)
    ax.set_ylim(-21, 21)

xticks = np.arange(0, 2.1e6, 0.5e6)
yticks = np.arange(-20, 21, 5)

plt.xticks(xticks)
plt.yticks(yticks)

ax1.set(ylabel='$m_{100}$')
ax3.set(xlabel='frames', ylabel='$m_{100}$')
ax4.set(xlabel='frames')

plt.tight_layout()

plt.savefig('training_dynamics.png', dpi=400)

plt.show()
