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

colors = sns.color_palette('rocket', 5)

mpl.rcParams['text.usetex']=True
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

fig, ax1 = plt.subplots(figsize=(10,7))

ax1.yaxis.tick_right()
#ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top') 
plt.minorticks_on()
ax1.tick_params(direction='in', which='minor', length=5, bottom=False, top=False, left=True, right=True)
ax1.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
ax1.set_ylim(0, 2e6)
ax1.set(xlabel='stepsize', ylabel='frames')

yticks = np.arange(0, 2.1e6, 0.5e6)
plt.yticks(yticks)

#g = sns.boxplot(y='nr_of_frames', x='nstep', data=steps_needed_df, palette='rocket')

tmp = steps_needed_df
data = [tmp.nr_of_frames[tmp.nstep==1], tmp.nr_of_frames[tmp.nstep==5], tmp.nr_of_frames[tmp.nstep==15], tmp.nr_of_frames[tmp.nstep==30]]

medianprops = dict(linewidth=1.5, linestyle='-', color='black')

box = ax1.boxplot(data, labels=[1, 5, 15, 30], medianprops=medianprops)
ax1.scatter(np.repeat(1, len(tmp.nr_of_frames[tmp.nstep==1])), tmp.nr_of_frames[tmp.nstep==1], color=colors[1], alpha=0.8)
ax1.scatter(np.repeat(2, len(tmp.nr_of_frames[tmp.nstep==5])), tmp.nr_of_frames[tmp.nstep==5], color=colors[2], alpha = 0.8)
ax1.scatter(np.repeat(3, len(tmp.nr_of_frames[tmp.nstep==15])), tmp.nr_of_frames[tmp.nstep==15], color=colors[3], alpha = 0.8)
ax1.scatter(np.repeat(4, len(tmp.nr_of_frames[tmp.nstep==30])), tmp.nr_of_frames[tmp.nstep==30], color=colors[4], alpha = 0.8)

#for patch, color in zip(box['boxes'], colors[1:]):
#    patch.set_facecolor(color)



#statistische signifikanz!
def stars(p):
    if p < 0.0001:
        return '****'
    elif (p < 0.001):
        return '***'
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"

ax1.plot([3, 3, 4, 4], [1.3e6, 1.4e6, 1.4e6, 1.3e6], lw=1.0, c='#355C7D')
ax1.text((7)*.5, 1.41e6, "not significant", ha='center', va='bottom', color='#355C7D')

ax1.plot([1, 1, 2, 2], [0.45e6, 0.35e6, 0.35e6, 0.45e6], lw=1.0, c='#355C7D')
ax1.text((3)*.5, 0.325e6, "significant", ha='center', va='top', color='#355C7D')

plt.savefig('boxplot.png', dpi=400)

plt.show()
