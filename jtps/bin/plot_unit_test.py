import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

optimizer_names = ['GradientDescentOptimizer', 'AdagradOptimizer', 'AdadeltaOptimizer', 'FtrlOptimizer', 'AdamOptimizer']

df = pd.read_csv('learning_curves.csv')
print(df.shape)
print((np.arange(0, 50000, 50) + 1).shape)
df['iter'] = np.arange(0, 50000, 50) + 1

cm = plt.get_cmap('gist_rainbow')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.grid(b=True, which='major', axis='both', ls='--', lw=.5, c='k', alpha=.3)
plt.axhline(y=0, lw=1, c='gray', alpha=1)
plt.axvline(x=0, lw=1, c='gray', alpha=1)

x = df.iter

for i, opt in enumerate(optimizer_names):
    y_baseline = df[opt + 'Loss']
    y_jtps = df[opt + 'JTPSLoss']

    plt.plot(x, y_baseline, color=cm(i / len(optimizer_names)), label=opt, lw=2, alpha=0.8, linestyle='--', solid_capstyle='butt')
    plt.plot(x, y_jtps, color=cm(i / len(optimizer_names)), label=opt+' JTPS', lw=2, alpha=0.8, linestyle='-', solid_capstyle='butt')

plt.xlabel('Iteration', weight='bold')
plt.ylabel('Loss', weight='bold')

plt.legend(fancybox=True, framealpha=0.75, frameon=True, facecolor='white', edgecolor='gray')

plt.gcf().set_size_inches(7, 5)

plt.tight_layout()

plt.savefig('learning_curves.png', dpi=300)

