'''
	Implementation of a GBDT. Very well done in 200 lines of Python
	https://github.com/lancifollia/tinygbt/blob/master/tinygbt.py
'''

# to generate the test data
from utils.simulation_utils import generate_students_data
number_students = 50
df_students = generate_students_data(number_students)

inputVars = df_students.columns.tolist()
responseVar = 'score'
inputVars.remove(responseVar)
responseVar_vals = df_students[responseVar].values




# Adapted from the file to run some tests
def _calc_l2_gradient(response_var, scores):
    hessian = np.full(len(response_var), 2)
    if scores is None:
        grad = np.random.uniform(size=len(response_var))
    else:
        grad = np.array([2 * (response_var[i] - scores[i]) for i in range(len(response_var))])
    return grad, hessian

def _calc_split_gain(G, H, G_l, H_l, G_r, H_r, lambd):
    """
    Loss reduction
    (Refer to Eq7 of Reference[1])
    """
    def calc_term(g, h):
        return np.square(g) / (h + lambd)
    return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)




grad, hessian = _calc_l2_gradient(responseVar_vals, scores=None)

params = {'gamma': 0., 
'lambda': 1.,
'min_split_gain': 0.1,
'max_depth': 5,
'learning_rate': 0.3,
}


# Train on all instances
instances = X_train.values
G = np.sum(grad)
H = np.sum(hessian)
best_gain = 0.
best_feature_id = None
best_val = 0.
best_left_instance_ids = None
best_right_instance_ids = None
for feature_id in range(instances.shape[1]):
    G_l, H_l = 0., 0.
    sorted_instance_ids = instances[:,feature_id].argsort()
    for j in range(sorted_instance_ids.shape[0]):
        G_l += grad[sorted_instance_ids[j]]
        H_l += hessian[sorted_instance_ids[j]]
        G_r = G - G_l
        H_r = H - H_l
        current_gain = _calc_split_gain(G, H, G_l, H_l, G_r, H_r, params['lambda'])
        if current_gain > best_gain:
            best_gain = current_gain
            best_feature_id = feature_id
            best_val = instances[sorted_instance_ids[j]][feature_id]
            best_left_instance_ids = sorted_instance_ids[:j+1]
            best_right_instance_ids = sorted_instance_ids[j+1:]

print(f'Best split {inputVars[best_feature_id]} with threshold {best_val:3.2f}')



# Focus on timeSpent
feature_id = 'timeSpent'
instances = X_train[].values
G = np.sum(grad)
H = np.sum(hessian)
best_gain = 0.
best_feature_id = None
best_val = 0.
best_left_instance_ids = None
best_right_instance_ids = None
G_l, H_l = 0., 0.
sorted_instance_ids = instances.argsort()
for j in range(sorted_instance_ids.shape[0]):
    G_l += grad[sorted_instance_ids[j]]
    H_l += hessian[sorted_instance_ids[j]]
    G_r = G - G_l
    H_r = H - H_l
    current_gain = _calc_split_gain(G, H, G_l, H_l, G_r, H_r, params['lambda'])
    if current_gain > best_gain:
        best_gain = current_gain
        best_feature_id = feature_id
        best_val = instances[sorted_instance_ids[j]]
        best_left_instance_ids = sorted_instance_ids[:j+1]
        best_right_instance_ids = sorted_instance_ids[j+1:]

print(f'Best split {inputVars[best_feature_id]} with threshold {best_val:3.2f}')




















# Code from the file

scores = self._calc_training_data_scores(train_set, models)
grad, hessian = self._calc_gradient(train_set, scores)
learner = self._build_learner(train_set, grad, hessian, shrinkage_rate)



def _calc_training_data_scores(self, train_set, models):
    if len(models) == 0:
        return None
    X = train_set.X
    scores = np.zeros(len(X))
    for i in range(len(X)):
        scores[i] = self.predict(X[i], models=models)
    return scores

def _calc_l2_gradient(self, train_set, scores):
    labels = train_set.y
    hessian = np.full(len(labels), 2)
    if scores is None:
        grad = np.random.uniform(size=len(labels))
    else:
        grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
    return grad, hessian

def _calc_gradient(self, train_set, scores):
    """For now, only L2 loss is supported"""
    return self._calc_l2_gradient(train_set, scores)

def _build_learner(self, train_set, grad, hessian, shrinkage_rate):
    learner = Tree()
    learner.build(train_set.X, grad, hessian, shrinkage_rate, self.params)
    return learner


def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
    """
    Loss reduction
    (Refer to Eq7 of Reference[1])
    """
    def calc_term(g, h):
        return np.square(g) / (h + lambd)
    return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)

def _calc_leaf_weight(self, grad, hessian, lambd):
    """
    Calculate the optimal weight of this leaf node.
    (Refer to Eq5 of Reference[1])
    """
    return np.sum(grad) / (np.sum(hessian) + lambd)

def build(self, instances, grad, hessian, shrinkage_rate, depth, param):
    """
    Exact Greedy Algorithm for Split Finding
    (Refer to Algorithm1 of Reference[1])
    """
    assert instances.shape[0] == len(grad) == len(hessian)
    if depth > param['max_depth']:
        self.is_leaf = True
        self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
        return
    G = np.sum(grad)
    H = np.sum(hessian)
    best_gain = 0.
    best_feature_id = None
    best_val = 0.
    best_left_instance_ids = None
    best_right_instance_ids = None
    for feature_id in range(instances.shape[1]):
        G_l, H_l = 0., 0.
        sorted_instance_ids = instances[:,feature_id].argsort()
        for j in range(sorted_instance_ids.shape[0]):
            G_l += grad[sorted_instance_ids[j]]
            H_l += hessian[sorted_instance_ids[j]]
            G_r = G - G_l
            H_r = H - H_l
            current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature_id = feature_id
                best_val = instances[sorted_instance_ids[j]][feature_id]
                best_left_instance_ids = sorted_instance_ids[:j+1]
                best_right_instance_ids = sorted_instance_ids[j+1:]
    if best_gain < param['min_split_gain']:
        self.is_leaf = True
        self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
    else:
        self.split_feature_id = best_feature_id
        self.split_val = best_val

        self.left_child = TreeNode()
        self.left_child.build(instances[best_left_instance_ids],
                              grad[best_left_instance_ids],
                              hessian[best_left_instance_ids],
                              shrinkage_rate,
                              depth+1, param)

        self.right_child = TreeNode()
        self.right_child.build(instances[best_right_instance_ids],
                                grad[best_right_instance_ids],
                                hessian[best_right_instance_ids],
                                shrinkage_rate,
                                depth+1, param)








######
# Coarse aproximation to mse split
df_split_criterion = df_students[['timeSpent', responseVar]].sort_values(by='timeSpent')
total_split_cost = []
etha_values = df_split_criterion.timeSpent.astype(int).unique()

for current_pos in range(0, len(etha_values)):
  current_etha = etha_values[current_pos]

  mean_left  = df_split_criterion.score[0:current_pos].mean()
  mean_right = df_split_criterion.score[current_pos::].mean()

  split_cost_left = ((df_split_criterion.score[0:current_pos] - mean_left)**2).mean()
  split_cost_right =((df_split_criterion.score[current_pos::] - mean_right)**2).mean()

  total_split_cost.append(split_cost_left+split_cost_right)
  print(f'Current split {current_etha} ({current_pos}) is {total_split_cost[-1]:3.2f}')