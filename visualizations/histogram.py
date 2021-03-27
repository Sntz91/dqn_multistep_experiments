import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

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

s1 = steps_needed_df[steps_needed_df.nstep==1].nr_of_frames.values
s5 = steps_needed_df[steps_needed_df.nstep==5].nr_of_frames.values
s15 = steps_needed_df[steps_needed_df.nstep==15].nr_of_frames.values
s30 = steps_needed_df[steps_needed_df.nstep==30].nr_of_frames.values

N = 1000000

def permutation_test(a, b):
    n_ab = 20
    ab=np.append(a, b)
    np.random.shuffle(ab)

    idx_a, idx_b = np.random.choice(range(n_ab), n_ab, replace=False).reshape(2, 10)
    A = ab[idx_a]
    B = ab[idx_b]
    return B.mean() - A.mean()
#res = [permutation_test(s1, s5) for i in range(N)]
#res2 = [permutation_test(s15, s30) for i in range(N)]

#pickle.dump(res, open('res.p', 'wb'))
#pickle.dump(res2, open('res2.p', 'wb'))

res = pickle.load(open("res.p", "rb"))
res2 = pickle.load(open("res2.p", "rb"))
low_c2 = np.quantile(res2, 0.975)
high_c2 = np.quantile(res2, 0.025)
low_c = np.quantile(res, 0.975)
high_c = np.quantile(res, 0.025)

colors = sns.color_palette('rocket', 10)
mpl.rcParams['text.usetex']=True
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

fig, ax = plt.subplots(figsize=(5,5))
n, bins, patches = ax.hist(res, 30, color=colors[6], edgecolor='black', linewidth=1.2)
ax.xaxis.set_label_position('bottom') 
ax.yaxis.set_label_position('right') 
plt.minorticks_on()
#ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
#ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
#ax.set_ylim(0, 2e6)

ax.axvline(s5.mean() - s1.mean(), color=colors[1], lw=2)
ax.axvline(low_c, color=colors[8], ls = '-.', lw=2)
ax.axvline(high_c, color=colors[8], ls = '-.', lw=2)
ax.text(s5.mean() - s1.mean() -1.35e5, 5e4, "observation", color=colors[1], size=14,
            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.1))

ax.text(low_c-7.5e5, 9.55e4, "confidence intervall 95\%", color='white', size=14,
            bbox=dict(boxstyle="square", facecolor=colors[8], edgecolor='none', pad=0.34))

ax.fill_betweenx([0, 10.1e4], low_c, high_c, color=colors[8], alpha=0.2) 

ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
ax.set_xlabel('differences in means')
ax.set_ylabel('frames needed for $m_{100}$=16')
ax.set_ylim(0, 10.1e4)

fig.tight_layout()

#plt.savefig('permutationtest_1_5.png', dpi=400)
#plt.show()

#---------------------------2----------------------------------------

res = res2
low_c = low_c2
high_c = high_c2

fig, ax = plt.subplots(figsize=(5,5))
n, bins, patches = ax.hist(res, 30, color=colors[6], edgecolor='black', linewidth=1.2)
ax.xaxis.set_label_position('bottom') 
ax.yaxis.set_label_position('right') 
plt.minorticks_on()
#ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
#ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
#ax.set_ylim(0, 2e6)

ax.axvline(s30.mean() - s15.mean(), color=colors[1], lw=2)
ax.axvline(low_c, color=colors[8], ls = '-.', lw=2)
ax.axvline(high_c, color=colors[8], ls = '-.', lw=2)
ax.text(s30.mean() - s15.mean() -0.95e5, 4.7e4, "observation", color=colors[1], size=14,
            bbox=dict(boxstyle="square", facecolor='white', edgecolor='none', pad=0.1))

ax.text(low_c-3.85e5, 9.55e4, "confidence intervall 95\%", color='white', size=14,
            bbox=dict(boxstyle="square", facecolor=colors[8], edgecolor='none', pad=0.34))

ax.fill_betweenx([0, 10.1e4], low_c, high_c, color=colors[8], alpha=0.2) 

ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
ax.set_xlabel('differences in means')
ax.set_ylabel('frames needed for $m_{100}$=16')
ax.set_ylim(0, 10.1e4)

fig.tight_layout()

plt.savefig('permutationtest_15_30.png', dpi=400)
plt.show()