'''
brew install glpk
brew tap coin-or-tools/coinor
brew install Xyz
pip3 install pulp cvxpy
'''

import pulp
pulp.pulpTestAll()


'''
	https://github.com/tirthajyoti/Optimization-Python/blob/master/Balanced_Diet_Problem_Complex.ipynb

	https://github.com/tirthajyoti/Optimization-Python

	Optimisation in Python
	https://medium.com/opex-analytics/optimization-modeling-in-python-pulp-gurobi-and-cplex-83a62129807a

	LP solvers
	https://github.com/stephane-caron/lpsolvers

	Cases of study
	https://pythonhosted.org/PuLP/CaseStudies/a_blending_problem.html


	I think this guy adds the variables
	https://blog.thinknewfound.com/2018/08/trade-optimization/

https://github.com/coin-or/
'''


# Indicator vars (binary) to avoid QP
food_chosen = LpVariable.dicts("Chosen",food_items,0,1,cat='Integer')

#Then we write a special code to link the usual food_vars and the binary food_chosen and add this constraint to the problem.
for f in food_items:
    prob += food_vars[f]>= food_chosen[f]*0.1
    prob += food_vars[f]<= food_chosen[f]*1e5

# If else conditions
# To incorporate the either/or condition of broccoli and iceberg lettuce, we just put a simple code,
prob += food_chosen['Frozen Broccoli']+food_chosen['Raw Iceberg Lettuce']<=1


#########

import pandas as pd
import pulp

_discount = 25
selling_price = 60 - _discount


df = pd.DataFrame([ \
	{'store_id': 1, 'discount': _discount, 'shelf_capacity': 12, 'shelf_replenisment_cost': 15,'forecast': 35}, \
	{'store_id': 1, 'discount': _discount, 'shelf_capacity': 8, 'shelf_replenisment_cost': 2,'forecast': 28}, \
	{'store_id': 1, 'discount': _discount, 'shelf_capacity': 4, 'shelf_replenisment_cost': 1, 'forecast': 8}, 
	{'store_id': 2, 'discount': _discount, 'shelf_capacity': 14, 'shelf_replenisment_cost': 4,'forecast': 28}, \
	{'store_id': 2, 'discount': _discount, 'shelf_capacity': 10, 'shelf_replenisment_cost': 2,'forecast': 18}, \
	{'store_id': 2, 'discount': _discount, 'shelf_capacity': 4, 'shelf_replenisment_cost': 1, 'forecast': 9}, 
	])

# Make the forecast implicit
df['replenishment'] = df['shelf_replenisment_cost']*df['forecast']/ df['shelf_capacity']

# Create a store_id for the optimiser
df['store_id_optimiser'] = 'str_' + df['store_id'].astype(str) + '_' + df['shelf_capacity'].astype(str)


#int_df['product_cost'] = 100 - 0.01*int_df['y_hat'].sum()

# Follow https://github.com/tirthajyoti/Optimization-Python/blob/master/Balanced_Diet_Problem_Complex.ipynb

store_list = df['store_id_optimiser'].tolist()
replenisment_cost = dict(zip(store_list, df['replenishment']))

store_frc = dict(zip(store_list, df['forecast']))

# Create a dictionary of food portion with lower bound 0 - these are the main optimization variables
store_vars = pulp.LpVariable.dicts("Portion",store_list,0,cat='Continuous')
# Create another set of variables for each food, with integer 0 or 1. These are indicator variables
stores_chosen = pulp.LpVariable.dicts("Chosen",store_vars,0,1,cat='Integer')

# Adding the objective function to the problem
# The objective function is added to 'prob' first


# Until I work out how to do it in the objective function...\
expected_units_sold = df.groupby('store_id')['forecast'].mean().sum()
product_cost = 100 - 0.01*expected_units_sold

selling_price = product_cost- _discount

# Create the 'prob' variable to contain the problem data
prob = pulp.LpProblem("Simple Stores Problem", pulp.LpMaximize)
#prob += product_cost-pulp.lpSum([replenisment_cost[i]*stores_chosen[i] for i in store_list]), "Objective"
# Break down into two lpSums: the profit and the costs
prob += pulp.lpSum([selling_price*store_frc[i]*stores_chosen[i] for i in store_list])-pulp.lpSum([replenisment_cost[i]*stores_chosen[i] for i in store_list]), "Objective"

# Constraint the stores
for f in store_list:
    prob += store_vars[f]>= stores_chosen[f]*0.1
    prob += store_vars[f]<= stores_chosen[f]*1e5

# Each store just once
prob += stores_chosen['str_1_12']+stores_chosen['str_1_8']+stores_chosen['str_1_4']<=1
prob += stores_chosen['str_2_14']+stores_chosen['str_2_10']+stores_chosen['str_2_4']<=1

prob.writeLP("SimpleStoresProblem.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", pulp.LpStatus[prob.status])

for v in prob.variables():
    if v.varValue>0 and v.name[0]=='P':
        print(v.name, "=", v.varValue)


print("The total profit of this balanced planning is: {}".format(round(pulp.value(prob.objective),2)))


total_units = df.iloc[0]['forecast'] + df.iloc[3]['forecast']
total_unit_price = 100 - 0.01*total_units

profit = total_unit_price*total_units - df['replenishment'].iloc[[0,3]].sum()



[ replenisment_cost[i]*stores_chosen[i] for i in store_list]
[ (store_frc[i]*stores_chosen[i])-replenisment_cost[i]*stores_chosen[i] for i in store_list]











######
import pulp
from pulp import *

ELEMENTS = ['Iron', 'Mercury', 'Silver']


Max_Per_Elem = {'Iron': 35, 
         'Mercury': 17, 
         'Silver': 28
               }

# A dictionary of the Iron percent in each of the CONCs
IronPercent = {'CONC_1': 20, 'CONC_2': 10, 'CONC_3': 25}

# A dictionary of the Hg percent in each of the CONCs
MercPercent = {'CONC_1': 15, 'CONC_2': 18, 'CONC_3': 12}

# A dictionary of the Silver percent in each of the CONCs
SilverPercent = {'CONC_1': 30,  'CONC_2': 40, 'CONC_3': 20}

CONCENTRATE_DIC = {'Iron': IronPercent,
              'Mercury': MercPercent,
              'Silver': SilverPercent              
              }

# Creates a list of Decision Variables
concs = ['CONC_1', 'CONC_2', 'CONC_3']

conc_vars = LpVariable.dicts("Util", concs, 0, 1.0)

# Create the 'prob' variable to contain the problem data
prob = LpProblem("Elements Concentration Problem", LpMaximize)

# The objective function
prob += lpSum([conc_vars[i] for i in concs]), "Total Utilization is maximized"

for elem in ELEMENTS:
    prob += lpSum([CONCENTRATE_DIC[elem][i]/Max_Per_Elem[elem] * conc_vars[i] for i in concs]) <= Max_Per_Elem[elem]/100, elem+"Percent"

prob.writeLP("ElemUtiliztionModel.lp")
prob.solve()
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
