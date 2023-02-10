
# Because I still think in a Matlab fashion...
df = pd.DataFrame([[10, 2, 10], [0, 8, 10], [20, 2, 3]],
                  columns=['A', 'B', 'C'])
df


# Let's say I want to calculate the average per column of the values that are > 4

idx = df > 4
idx

'''
|10| |10
|  |8|10
|20| |

'''

# As I cannot index it directly, I can take advantage of pandas applying nanmean by default
df.where(idx, np.nan).mean(axis=0)

# That gives me [15, 8, 10]