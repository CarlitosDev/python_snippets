

'''

source ~/.bash_profile && python3 -m pip install causalnex --upgrade

pip3 install pygraphviz

'''

'''
This is the tutorial
https://causalnex.readthedocs.io/en/latest/03_tutorial/03_tutorial.html

# If Python 3.8
To run...
eval "$(pyenv init -)"
pyenv local 3.7.7
# 3. 
python3 -m venv env
source env/bin/activate
python3 -V
python3
deactivate
'''


# silence warnings
import warnings
warnings.filterwarnings("ignore")

# pygraphviz
import matplotlib.pyplot as plt
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas

data = pd.read_csv('./student/student-por.csv', delimiter=';')
data.head(5)

drop_col = ['school','sex','age','Mjob', 'Fjob','reason','guardian']
data = data.drop(columns=drop_col)
data.head(5)



struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)


le = LabelEncoder()
for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])

struct_data.head(5)



# This takes some time to run
# NOTEARS algorithm to learn the structure
sm = from_pandas(struct_data)

# Apply thresholding to the weaker edges.
sm.remove_edges_below_threshold(0.8)

viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)

viz.draw('file_A.png')


## Modifying the structure
sm.add_edge("failures", "G1")
sm.remove_edge("Pstatus", "G1")
sm.remove_edge("address", "G1")

viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)

viz.draw('file_B.png')


sm = sm.get_largest_subgraph()

viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)

viz.draw('file_C.png')


# We assume that we're happy with the structure from file_C
from causalnex.network import BayesianNetwork
bn = BayesianNetwork(sm)




discretised_data = data.copy()
data_vals = {col: data[col].unique() for col in data.columns}

failures_map = {v: 'no-failure' if v == [0]
            else 'have-failure' for v in data_vals['failures']}

studytime_map = {v: 'short-studytime' if v in [1,2]
                 else 'long-studytime' for v in data_vals['studytime']}

discretised_data["failures"] = discretised_data["failures"].map(failures_map)
discretised_data["studytime"] = discretised_data["studytime"].map(studytime_map)


# Discretising Numeric Features
from causalnex.discretiser import Discretiser

discretised_data["absences"] = Discretiser(method="fixed",
                          numeric_split_points=[1, 10]).transform(discretised_data["absences"].values)

discretised_data["G1"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G1"].values)

discretised_data["G2"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G2"].values)

discretised_data["G3"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G3"].values)



#Create Labels for Numeric Features

absences_map = {0: "No-absence", 1: "Low-absence", 2: "High-absence"}

G1_map = {0: "Fail", 1: "Pass"}
G2_map = {0: "Fail", 1: "Pass"}
G3_map = {0: "Fail", 1: "Pass"}

discretised_data["absences"] = discretised_data["absences"].map(absences_map)
discretised_data["G1"] = discretised_data["G1"].map(G1_map)
discretised_data["G2"] = discretised_data["G2"].map(G2_map)
discretised_data["G3"] = discretised_data["G3"].map(G3_map)

#Train / Test Split
# Split 90% train and 10% test
from sklearn.model_selection import train_test_split
train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1, random_state=7)

# Model Probability
bn = bn.fit_node_states(discretised_data)
bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
bn.cpds["G1"]

# Predict the State given the Input Data
discretised_data.loc[18, discretised_data.columns != 'G1']
predictions = bn.predict(discretised_data, "G1")

print('The ground truth is \'{truth}\''.format(truth=discretised_data.loc[18, 'G1']))

#Model Quality
from causalnex.evaluation import classification_report
classification_report(bn, test, "G1")

#ROC / AUC
from causalnex.evaluation import roc_auc
roc, auc = roc_auc(bn, test, "G1")
print(auc)

#Baseline Marginals
bn = bn.fit_cpds(discretised_data, method="BayesianEstimator", bayes_prior="K2")

# Inference marginals
from causalnex.inference import InferenceEngine
ie = InferenceEngine(bn)
marginals = ie.query()
marginals["G1"]

import numpy as np
labels, counts = np.unique(discretised_data["G1"], return_counts=True)
list(zip(labels, counts))


marginals_short = ie.query({"studytime": "short-studytime"})
marginals_long = ie.query({"studytime": "long-studytime"})
print("Marginal G1 | Short Studtyime", marginals_short["G1"])
print("Marginal G1 | Long Studytime", marginals_long["G1"])


# Do Calculus
# Updating a Node Distribution
print("distribution before do", ie.query()["higher"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})
print("distribution after do", ie.query()["higher"])

#Resetting a Node Distribution
ie.reset_do("higher")

#Effect of Do on Marginals
print("marginal G1", ie.query()["G1"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})
print("updated marginal G1", ie.query()["G1"])
