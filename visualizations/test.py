import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

colors=sns.color_palette("rocket",3)
mpl.rcParams['font.family'] = 'Avenir'
mpl.rcParams['font.size']=12


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))

ax1.plot(t, s, color=colors[0])


#fig.set_title('simple plot')
ax3.set(xlabel='time(s)', ylabel='voltage (mv)',
        title='$n=1$')

plt.minorticks_on()

for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
        ax.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

plt.show()