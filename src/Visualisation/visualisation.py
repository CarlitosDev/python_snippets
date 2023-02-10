# Cool code to generate random colours
# Generate random colours
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))
    return color



# Remove gaps in the axis (tight)
plt.margins(0,0)


# Regular Matlab plot holding the axis
import matplotlib.pyplot as plt
import numpy as np
def shifted_Gompertz_distribution(t=np.linspace(0,5,50), b_scale = 0.4, eta_shape = 10):
  # https://www.wikiwand.com/en/Shifted_Gompertz_distribution
  e_bt = np.exp(-b_scale*t)
  return b_scale*e_bt*np.exp(-eta_shape*e_bt)*(1+eta_shape*(1-e_bt))

t = np.linspace(0,20,100)

fig, ax = plt.subplots()
b_scale = 0.4
eta_shape = 10
this_label = f'b={b_scale:3.1f}, eta={eta_shape:3.2f}'
g_dist_shifted = shifted_Gompertz_distribution(t, b_scale, eta_shape)
ax.plot(t, g_dist_shifted, label=this_label)


b_scale = 0.4
eta_shape = 6
this_label = f'b={b_scale:3.1f}, eta={eta_shape:3.2f}'
g_dist_shifted_2 = shifted_Gompertz_distribution(t, b_scale, eta_shape)
ax.plot(t, g_dist_shifted_2, label=this_label)


b_scale = 0.9
eta_shape = 2
this_label = f'b={b_scale:3.1f}, eta={eta_shape:3.2f}'
g_dist_shifted_3 = shifted_Gompertz_distribution(t, b_scale, eta_shape)
ax.plot(t, g_dist_shifted_3, label=this_label)


legend = ax.legend()
plt.xlabel('Discount')
plt.ylabel('Sales response')
plt.title('Sales-Price as a shifted Gompertz distribution for 4 products')
plt.show()






# Close all figures
import matplotlib.pyplot as plt
plt.close('all')


# Confusion matrix
import scikitplot as skpl
skplt.metrics.plot_confusion_matrix(y_valid, xgb.predict(X_valid))
plt.show()


# Heatmap keeping the lower corner (remove main diag plus upper corner)
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (12,12))
sns.heatmap(df.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()



## Basic scatterplot
y_hat = model.predict(X_test)
y_test

plt.scatter(y_hat, y_test)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()





# Histogram + pdf with seaborn:
# Let's plot the sales
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
# plot the log of the sales to see if they kind of distribute normally 
f, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
sns.distplot(np.log1p(dfTrain.sales), ax=axes[0]);





# Get test data
from mpl_toolkits.mplot3d import axes3d
X, Y, Z = axes3d.get_test_data(0.05)
numR, numC = X.shape



# Use seaborn to plot a heat map
import seaborn as sns
dfPurchasesFiltPerDay = pd.pivot_table(dfPurchasesFilt, values='productCount', 
    index=['productName'], columns=['hour'], aggfunc=np.mean)
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(dfPurchasesFiltPerDay, annot=True, linewidths=.5, ax=ax)
plt.show()



# Plot a quantity on the y axis ignoring the df index
import matplotlib.pyplot as plt
fig = plt.figure()
numSamples = 1000
plt.plot(range(0,numSamples), train_set['target_variable'].ix[0:numSamples])
plt.show()



# Colorbar depending on the current data:
data1 = df_corr.values
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
fig1.colorbar(heatmap1)



# Scatterplot colouring by a third variable
plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
# set the colour depending on 'median_house_value'
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())

# drop the axis by using _ 
_ = plt.plot()


# Get the axes of a figure:
fig = plt.gcf()
ax1 = fig.get_axes();




# Plot ROC
# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_test["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Subplots and Matplotlib

import matplotlib.pyplot as plt
x = range(10)
y = range(10)
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0][2].plot(x,y)
ax[1, 0].plot(x,y)
plt.show()



# %matplotlib notebook
fig = plt.figure(figsize=(20,10))
plt.plot(dfTotalsGBRFD, color='g')
plt.plot(dfTotalsGBRTPS)
plt.show()




# Cool SNS example where the visualisation is broken down by two parameters
import seaborn as sns
sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
fmri.head()

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)

plt.show()





# MOre Seaborn options
sns.lineplot(x="date", y="total_products",
             hue="productBand", markers=True, dashes=False, lw=2.5,
             data=dfBeamlyGRP)


# More on Seaborn (for calssification)
g = sns.FacetGrid(titanic, row='survived', col='class')
g.map(sns.distplot, "age")
plt.show()


#
sns.jointplot(data=titanic, x='age', y='fare', kind='reg', color='g')
plt.show()



# Plot a simple categorical heatmap
import catheat
import seaborn as sns
import matplotlib.pyplot as plt

# Get an example dataset from seaborn
tips = sns.load_dataset('tips')

# Plot the categorical columns as heatmap
ax = catheat.heatmap(tips[['sex','smoker','day','time','size']],
                      palette='Paired' )

plt.show()




# Example of figure combining histograms, pdf and annotations
fig = plt.figure()
ax1 = fig.gca()
sns.distplot(df_all_last_record[idx_2017 & idx_inactive].nummonthssincejoined, ax = ax1, label='Inactives joined 2017')
plt.plot(mu_inactive_2017, 0.06, linestyle='--', marker='o', color='b')
sns.distplot(df_all_last_record[idx_2018 & idx_inactive].nummonthssincejoined, ax = ax1, label='Inactives joined 2018')
plt.plot(mu_inactive_2018, 0.112, linestyle='--', marker='o', color='orange')

plt.xlabel('Number of months since joined')
plt.ylabel('Probability')

arrowprops = dict(arrowstyle = "->", connectionstyle="arc3")


ax1.annotate(f'Average {mu_inactive_2017:3.2f} months', 
xy=(mu_inactive_2017,0.06), xycoords='data', textcoords='data',
xytext=(3.0+mu_inactive_2017,0.08)
,arrowprops=arrowprops)


ax1.annotate(f'Average {mu_inactive_2018:3.2f} months', 
xy=(mu_inactive_2018,0.112), xycoords='data', textcoords='data',
xytext=(2.0+mu_inactive_2018,0.132)
,arrowprops=arrowprops)


ax1.legend()
ax1.tight=True

plt.show(block = True)
plt.close('all')



## Example of horizontal bar chart
import pandas
import matplotlib.pyplot as plt
import numpy as np

df = pandas.DataFrame(dict(graph=['Item one', 'Item two', 'Item three'],
                           n=[3, 5, 2], m=[6, 1, 3])) 

ind = np.arange(len(df))
width = 0.4

fig, ax = plt.subplots()
ax.barh(ind, df.n, width, color='red', label='N')
ax.barh(ind + width, df.m, width, color='green', label='M')

ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend()

plt.show()




def cool_heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)


# Read and plot images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('run_similarity_queries.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()





##### Manually create a histogram

#  create the figure
fig, ax1 = plt.subplots(figsize=(9, 7))
fig.subplots_adjust(left=0.115, right=0.88)
fig.canvas.set_window_title('UT3')

pos = np.arange(len(bins)-1)
hist, bin_edges = np.histogram(array_session_length, bins=bins, density=False)
bin_defs = [str(bins[idx_bin]) + '-' + str(bins[idx_bin+1]) for idx_bin in range(len(bins)-1)]

rects = ax1.barh(pos, hist,
                  align='center',
                  height=0.5,
                  tick_label=bin_defs)

ax1.set_title('ax title')
ax1.set_xlabel('Number of sessions')

#ax1.set_xlim([0, 100])
ax1.xaxis.grid(True, linestyle='--', which='major',
                color='grey', alpha=.25)

plt.show()




# Light effects (Matlab-like)
# Get lighting object for shading surface plots.
from matplotlib.colors import LightSource
# Get colormaps to use with lighting object.
from matplotlib import cm
light = LightSource(90, 45)
illuminated_surface = light.shade(Z, cmap=cm.coolwarm)
plt.facecolors=illuminated_surface





# Radar plot
# Spider plot
f, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(polar=True))

# number of variable
N = df_skills.shape[0]
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += [angles[0]]
# Initialise the spider plot
# If you want the first axis to be on top:
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], df_skills.StepTitle_inferred.tolist()) 
# Draw ylabels
ax.set_rlabel_position(0)
scores_grid = np.linspace(0,100,5)
scores_label = [f'{this_score:3.2f}' for this_score in scores_grid]
plt.yticks(scores_grid, scores_label, color="grey", size=7)
plt.ylim(0,100)

values=df_skills.score_p50.values.tolist() 
values += [values[0]]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
ax.fill(angles, values, 'b', alpha=0.1)
plt.show()




# bar h tricks
(news_train_df['headlineTag'].value_counts() / 1000)[:10].plot('barh')
plt.title('headlineTag counts (thousands)')