'''

  From https://www.pythonpool.com/knapsack-problem-python/


N stands for a number of items
wt stands for the weight of the item
VAL is the value of the item

I think it's better explained here
https://medium.com/swlh/dynamic-programming-0-1-knapsack-python-code-222e607a2e8

'''



'''
Dynamic Programming approach divides the problem to be solved into subproblems. 
The subproblems are further kept on dividing into smaller subproblems. 
Until you get subproblems that can be solved easily. 

The idea of Knapsack dynamic programming is to use a table to store the solutions of solved subproblems.

In the table, all the possible weights from ‘1’ to ‘W’ serve as the columns and weights are kept as the rows. 
The state DP[i][j] in the above example denotes the maximum value of ‘j-weight’ considering all values from ‘1 to ith’. 
So if we consider ‘wi’ (weight in ‘ith’ row) it is put in all columns which have ‘weight values > wi’. 
Two possibilities occur – to fill or not to fill ‘wi’ in the given column. 
If we do not fill ‘ith’ weight in ‘jth’ column then DP[i][j] state will be same as DP[i-1][j].
But if we fill the weight, DP[i][j] will be equal to the value of ‘wi’+ value of the column weighing ‘j-wi’ in the previous row. 
We therefore take the maximum of these two possibilities to fill the current state. 
'''
def knapSack(W, wt, val, n): 
    K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
   
    # Build table K[][] in bottom up manner 
    for i in range(n + 1): 
        for w in range(W + 1): 
            if i == 0 or w == 0: 
                K[i][w] = 0
            elif wt[i-1] <= w: 
                K[i][w] = max(val[i-1] 
                          + K[i-1][w-wt[i-1]],   
                              K[i-1][w]) 
            else: 
                K[i][w] = K[i-1][w] 
   
    return K[n][W] 
   
   
# Driver code 
val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 50
n = len(val) 
print(knapSack(W, wt, val, n))