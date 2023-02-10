## Let's use sklearn pipelines to process the raw data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from mlToolbox import preprocessingUtils as pre


d = pre.infer_variable_type(df_closed_answers)
numericalVars = d['numerical']
categoricalVars = d['categorical']

inputVars = ['cefr', 'topic', 'skill', 'activity_type']
responseVar = 'is_correct'

input_cat_vars = [*set(categoricalVars).intersection(set(inputVars))]
input_num_vars = [*set(numericalVars).intersection(set(inputVars))]

X_train = df_closed_answers.loc[0:1000, inputVars]
y_train = df_closed_answers.loc[0:1000, responseVar]


X_test = df_closed_answers.loc[2000:2012, inputVars]
y_test = df_closed_answers.loc[2000:2012, responseVar]



cont_pipe = Pipeline([('scaler', StandardScaler()),
('imputer', SimpleImputer(strategy='median', add_indicator=True))])

cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore')),
('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))])

pre = ColumnTransformer([('categorical', cat_pipe, input_cat_vars),
('continuous', cont_pipe, input_num_vars)])


model = Pipeline([('preprocessing', pre), ('clf', LogisticRegression())])

param_grid = {'clf__C': np.logspace(-3, 3, 7)}
grid_search = GridSearchCV(model, param_grid=param_grid)
grid_search.fit(X_train, y_train)

prediction = grid_search.predict(X_test)