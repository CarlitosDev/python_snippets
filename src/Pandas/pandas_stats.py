# Groupby calculating mu and sigma for vars
# it works by setting the name of the variable to work on,
# the name of the resulting variables and the operations
aggregations = {
  'numberofclasses_agg': {
    'avg_numberofclasses_agg': 'mean',
    'std_numberofclasses_agg': 'std',
  }
}
varsToGroupBy = 'teachercenterparentname'

dfGrouped = df.groupby(varsToGroupBy, as_index=False).agg(aggregations).copy();



# The one aboves as a master
varsToGroupBy = 'teachercenterparentname'
aggregations = {}
for iVar in varsForStats:
  sub_aggregations = {}
  sub_aggregations['avg_' + iVar] = 'mean'
  sub_aggregations['std_' + iVar] = 'std'
  sub_aggregations['cnt_' + iVar] = 'count'
  aggregations[iVar] = sub_aggregations

dfGrouped = df.groupby(varsToGroupBy, as_index=False).agg(aggregations).copy();






# save the stats
dfStats = mainDF.describe();
minVals = dfStats.ix['min'];
maxVals = dfStats.ix['max'];

# stats for dates
dfStats = dfTrainPeriodA.date.describe();
minVals = dfStats.ix['first'];
maxVals = dfStats.ix['last'];
print('min date {} and max date {}'.format(minVals, maxVals));


Max of all values and max of each column:
	maxValue = df.values.max();
	xMax, yMax, zMax = df.max()

mainDFNorm = (mainDF - minVals)/(maxVals-minVals);


Bin a variable in pandas:
	def get_quantile_based_buckets(df, feature_name, num_buckets):
		boundaries = np.arange(1.0, num_buckets) / num_buckets;
		quantiles = df[feature_name].quantile(boundaries);
		[quantiles[q] for q in quantiles.keys()]
		bucketVar = 'bucket_' +  feature_name;
		df[bucketVar] = pd.cut(df[feature_name], quantiles, retbins=False, labels=False);
		return df;


Convert Pandas series to a list:
	[quantiles[q] for q in quantiles.keys()];
	Actually... quantiles.tolist()


Contingency matrix:

df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11}, \
			{'A':'bar', 'B':'blue', 'C': 20}, \
			{'A':'foo', 'B':'blue', 'C': 20}])

contingency = pd.crosstab(df.A, df.C, margins=True,  margins_name='Totals')
print(contingency)


Probability of a variable (df from above):

import pandas as pd
rating_probs = df.groupby('A').size().div(len(df))
print(rating_probs)

Conditional probablities > Prob(C|A):
cond_probs = df.groupby(['C', 'A']).size().div(len(df)).div(rating_probs, axis=0, level='A')
print(cond_probs)




# -     -       -       -
# crosstabs - more complex than above
# -     -       -       -

productType = 'Mascara_'

levelNumber = 1

fName = f'{productType}level_{levelNumber}' + '.xlsx'

baseFolder = '/Users/carlos.aguilar/Documents/Beamly/VMUA-Hollition/analysed_products'
xlsxPath   = os.path.join(baseFolder,  fName);
df        = pd.read_excel(xlsxPath, sheet_name = 'Sheet1');

colNames  = df.columns.tolist();
print(colNames)


indexes           = [df.eyeShape, df.finish];
contingency       = pd.crosstab(indexes, df.numProducts, margins=True,  margins_name='Totals')
contingency.Totals = 0;

for icol in list(set(contingency.columns.tolist()) - set(['Totals'])):
    idx = contingency[icol] > 0
    contingency[icol][idx] = icol
    contingency.Totals += contingency[icol]


baseFolder      = '/Users/carlos.aguilar/Documents/Beamly/VMUA-Hollition/contingency tables'
xlsxResultsPath = os.path.join(baseFolder,  f'level_{levelNumber}_{productType}' + '.xlsx');
cu.dataFrameToXLS(contingency, xlsxResultsPath, sheetName = 'Contingency', writeIndex = True);