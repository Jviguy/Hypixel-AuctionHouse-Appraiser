import ast
import os
import pandas as pd
import shap
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from xgboost import XGBRFRegressor, XGBRegressor

df = pd.read_csv('auctions.csv')

# create item_count_on_market column
item_counts = df['item_id'].value_counts()
# Convert the Series to a DataFrame
item_counts_df = item_counts.reset_index()
item_counts_df.columns = ['item_id', 'item_count_on_market']
df = df.merge(item_counts_df, on='item_id', how='left')

# Fix missing reforges, or empty strings
df['reforge'] = df['reforge'].fillna('None')
# create lowest_bin column

# Get the lowest BIN for each item
lowest_bin = df.groupby('item_id')['price'].min()
lowest_price_per_item = df.groupby('item_id')['price'].min().reset_index()
lowest_price_per_item.rename(columns={'price': 'lowest_bin'}, inplace=True)
df = df.merge(lowest_price_per_item, on='item_id', how='left')
df = df.drop('item_id', axis=1)

# preprocess tier.
label_encoder = LabelEncoder()
df['tier'] = label_encoder.fit_transform(df['tier'])

# preprocess category.
one_hot_encoder = OneHotEncoder(sparse_output=False)
category_reshaped = df['category'].values.reshape(-1, 1)

# Fit and transform 'category'
category_encoded = one_hot_encoder.fit_transform(category_reshaped)

# Create a DataFrame from the encoded data
# Flatten the categories array to get the correct column names
category_encoded_df = pd.DataFrame(category_encoded, columns=one_hot_encoder.categories_[0], index=df.index)

# Drop the original 'category' column
df.drop('category', axis=1, inplace=True)

# Join the new DataFrame with the original one
df = df.join(category_encoded_df)

# preprocess reforge. (one hot)
one_hot_encoder = OneHotEncoder(sparse_output=False)
reforge_encoded = one_hot_encoder.fit_transform(df[['reforge']])
df.drop('reforge', axis=1, inplace=True)
# Create a DataFrame from the encoded data
# Use [0] to access the list of categories for the 'reforge' column
reforge_encoded_df = pd.DataFrame(reforge_encoded, columns=one_hot_encoder.categories_[0], index=df.index)
# Join the new DataFrame with the original one
df = df.join(reforge_encoded_df)

# preprocess enchantments through multilabel bin
mlb = MultiLabelBinarizer()
df['enchantments'] = df['enchantments'].apply(ast.literal_eval)
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('enchantments')),
                          columns=mlb.classes_,
                          index=df.index))

# preprocess unlocked_gem_slots through multilabel bin
mlb = MultiLabelBinarizer()
df['unlocked_gem_slots'] = df['unlocked_gem_slots'].apply(ast.literal_eval)
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('unlocked_gem_slots')),
                          columns=mlb.classes_,
                          index=df.index))

# preprocess slotted_gems through multilabel bin
mlb = MultiLabelBinarizer()
df['slotted_gems'] = df['slotted_gems'].apply(ast.literal_eval)
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('slotted_gems')),
                          columns=mlb.classes_,
                          index=df.index))
# Final data creation
y = df['price']
X = df.drop(['auction_id', 'price'], axis=1)
# train test split fn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
HYPERPARAMETER STUFF
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
}


def objective(params):
    params = {'n_estimators': int(params['n_estimators']),
              'learning_rate': params['learning_rate'],
              'max_depth': int(params['max_depth']),
              'min_child_weight': params['min_child_weight'],
              'gamma': params['gamma'],
              'subsample': params['subsample'],
              'colsample_bytree': params['colsample_bytree']}

    xgb_model = XGBRegressor(**params)
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return {'loss': mae, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)
'''
if os.path.isfile("model.json"):
    print("Loading Model...")
    model = XGBRegressor()
    model.load_model('model.json')
else:
    model = XGBRegressor(n_estimators=885, n_jobs=4, max_depth=10, subsample=0.9460876879182103,
                         colsample_bytree=0.5599455461165574,
                         learning_rate=0.14585998256990643, min_child_weight=1, gamma=0.21577976888055775)
# print(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean())
model.fit(X_train, y_train)
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

# Visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0], max_display=800)