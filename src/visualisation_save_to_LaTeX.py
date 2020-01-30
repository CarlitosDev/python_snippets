# After some time...the solution is to use mactex, not basictex


# Troubleshooting:
# brew cask uninstall basictex
# brew cask install mactex
# MaxTex is more than a GB and BasicTex is around 90MBs.
'''
  
'''


# Create a figure in matplotlib and export it to LaTeX
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

### Create the figure

def shifted_Gompertz_distribution(t=np.linspace(0,5,50), b_scale = 0.4, eta_shape = 10):
  # https://www.wikiwand.com/en/Shifted_Gompertz_distribution
  e_bt = np.exp(-b_scale*t)
  return b_scale*e_bt*np.exp(-eta_shape*e_bt)*(1+eta_shape*(1-e_bt))

t = np.linspace(0,20,100)

# Use LaTex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()
b_scale = 0.3
eta_shape = 8
this_label = f'b={b_scale:3.1f}, eta={eta_shape:3.2f}'
g_dist_shifted = shifted_Gompertz_distribution(t, b_scale, eta_shape)
ax.plot(t, g_dist_shifted, label=this_label)

legend = ax.legend()
plt.xlabel(r'Discount $\mu$')
plt.ylabel(r'Sales response $f(t; b, \eta)$')
#plt.title('Sales-Price as a shifted Gompertz distribution for 3 products')
#plt.show()

# save as PDF > Works with mactex
file_name = os.path.join('figs', 'shifted_Gompertz.pdf')
fig.savefig(file_name, bbox_inches='tight')

#### 


matplotlib.use('pgf')

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})
file_name = os.path.join('figs', 'shifted_Gompertz4.pgf')
plt.savefig(file_name)




# Similar results
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
file_name = os.path.join('figs', 'shifted_Gompertz5.pgf')
plt.savefig(file_name)



# Exporting to PDF also works
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages
with PdfPages('multipage.pdf', metadata={'author': 'Me'}) as pdf:

    fig1, ax1 = plt.subplots()
    ax1.plot([1, 5, 3])
    pdf.savefig(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot([1, 5, 3])
    pdf.savefig(fig2)


########

import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})

plt.figure(figsize=(4.5, 2.5))
plt.plot(range(5))
plt.text(0.5, 3., "serif", family="serif")
plt.text(0.5, 2., "monospace", family="monospace")
plt.text(2.5, 2., "sans-serif", family="sans-serif")
plt.xlabel(r"µ is not $\mu$")
plt.tight_layout(.5)

plt.savefig("pgf_texsystem.pdf")
plt.savefig("pgf_texsystem.png")


########
# Latexify from https://nipunbatra.github.io/blog/2014/latexify.html
###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from math import sqrt
SPINE_COLOR = 'gray'


def latexify(fig_width=None, fig_height=None, columns=1):
  """Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.

  Parameters
  ----------
  fig_width : float, optional, inches
  fig_height : float,  optional, inches
  columns : {1, 2}
  """

  # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
  # Width and max height in inches for IEEE journals taken from
  # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
  assert(columns in [1,2])

  if fig_width is None:
    fig_width = 3.39 if columns==1 else 6.9 # width in inches

  if fig_height is None:
    golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_height = fig_width*golden_mean # height in inches

  MAX_HEIGHT_INCHES = 8.0
  if fig_height > MAX_HEIGHT_INCHES:
    print("WARNING: fig_height too large:" + fig_height + \
    "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
    fig_height = MAX_HEIGHT_INCHES

  params = {'backend': 'ps',
    'text.latex.preamble':[
    r'\usepackage{amsmath}',
    r'\usepackage{gensymb}'],
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'text.fontsize': 8, # was 10
    'legend.fontsize': 8, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    'figure.figsize': [fig_width,fig_height],
    'font.family': 'serif'}
    
  matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax




df = pd.DataFrame(np.random.randn(10,2))
df.columns = ['Column 1', 'Column 2']

ax = df.plot()
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_title("Title")
plt.tight_layout()
plt.savefig("./figs/image1.pdf")



###

import matplotlib as mpl
import pathlib

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
        r"\usepackage{import}",
        f'\subimport{{{pathlib.Path.cwd().resolve()}/}}{{foo.tex}}']
})

import matplotlib.pyplot as plt
plt.figure(figsize=(4.5,2.5))
plt.plot(range(5))
plt.xlabel(r'\foo{}')
plt.savefig('foo.pgf')