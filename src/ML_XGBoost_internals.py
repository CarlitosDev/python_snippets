
model = xgb_model
original_model = model.get_booster()



xgb_loader = XGBTreeModelLoader(self.original_model)
self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
self.base_offset = xgb_loader.base_score
objective = objective_name_map.get(xgb_loader.name_obj, None)
tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
tree_limit = getattr(model, "best_ntree_limit", None)



model = xgb_model
original_model = model.get_booster()

buf = original_model.save_raw()
pos = 0

# load the model parameters
base_score = read('f')
num_feature = read('I')
num_class = read('i')
contain_extra_attrs = read('i')
contain_eval_metrics = read('i')
read_arr('i', 29) # reserved
name_obj_len = read('Q')
name_obj = read_str(name_obj_len)
name_gbm_len = read('Q')
name_gbm = read_str(name_gbm_len)

assert name_gbm == "gbtree", "Only the 'gbtree' model type is supported, not '%s'!" % name_gbm

# load the gbtree specific parameters
num_trees = read('i')
num_roots = read('i')
num_feature = read('i')
pad_32bit = read('i')
num_pbuffer_deprecated = read('Q')
num_output_group = read('i')
size_leaf_vector = read('i')
read_arr('i', 32) # reserved

# load each tree
num_roots = np.zeros(num_trees, dtype=np.int32)
num_nodes = np.zeros(num_trees, dtype=np.int32)
num_deleted = np.zeros(num_trees, dtype=np.int32)
max_depth = np.zeros(num_trees, dtype=np.int32)
num_feature = np.zeros(num_trees, dtype=np.int32)
size_leaf_vector = np.zeros(num_trees, dtype=np.int32)
node_parents = []
node_cleft = []
node_cright = []
node_sindex = []
node_info = []
loss_chg = []
sum_hess = []
base_weight = []
leaf_child_cnt = []



######
model = xgb_model
original_model = model.get_booster()
original_model.best_ntree_limit
original_model.booster
original_model.get_score()
original_model.get_split_value_histogram()
#original_model.get_split_value_histogram('x2')
df_trees = original_model.trees_to_dataframe()


# Model to JSON
json_trees = original_model.get_dump(with_stats=True, dump_format="json")
# this fixes a bug where XGBoost can return invalid JSON
json_trees = [t.replace(": inf,", ": 1000000000000.0,") for t in json_trees]
json_trees = [t.replace(": -inf,", ": -1000000000000.0,") for t in json_trees]

pyperclip.copy(json_trees[0])

# Check why Scott does this...
original_model.feature_names = None
json_trees = original_model.get_dump(with_stats=True, dump_format="json")