import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import OneHotEncoder
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

# 1. 数据导入
df_shipment = pd.read_excel('附件1-商家历史出货量表.xlsx')
df_product = pd.read_excel('附件2-商品信息表.xlsx')
df_seller = pd.read_excel('附件3-商家信息表.xlsx')
df_warehouse = pd.read_excel('附件4-仓库信息表.xlsx')

# 2. 数据处理
# 数据合并 - 使用函数封装以提高可读性
def merge_dataframes(shipments, products, sellers, warehouses):
    df = shipments.merge(products, on='product_no', how='left')
    df = df.merge(sellers, on='seller_no', how='left')
    df = df.merge(warehouses, on='warehouse_no', how='left')
    return df

df = merge_dataframes(df_shipment, df_product, df_seller, df_warehouse)

# 数据处理 - 将异常值处理、时间特征提取、独热编码等封装为函数

def process_dates(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

def handle_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        if col not in ['year', 'month', 'day', 'weekday']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median())

def encode_categorical(df, categorical_columns):
    ohe = OneHotEncoder(sparse_output=False)
    df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_columns]))
    df_encoded.columns = ohe.get_feature_names_out(categorical_columns)
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    return df

# 调用上述函数处理数据
process_dates(df)
handle_outliers(df)
categorical_columns = ['category1', 'category2', 'category3', 'seller_category',
                       'inventory_category', 'seller_level', 'warehouse_category', 'warehouse_region'] # 确定分类列列表
df = encode_categorical(df, categorical_columns)


# # 处理日期数据
# df['date'] = pd.to_datetime(df['date'])
#
# # 异常值处理
# for col in df.select_dtypes(include=np.number).columns:
#     if col not in ['year', 'month', 'day', 'weekday']:  # 排除时间特征
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median())
#
# # 提取时间特征
# df['year'] = df['date'].dt.year
# df['month'] = df['date'].dt.month
# df['day'] = df['date'].dt.day
# df['weekday'] = df['date'].dt.weekday
#

# 特征工程 - 将时间滞后和滚动平均的创建封装为函数
def create_lag_features(df, group_cols, target, lags):
    for lag in lags:
        df[f'lag_{target}_{lag}'] = df.groupby(group_cols)[target].shift(lag)
        df[f'lag_{target}_{lag}'].fillna(df[f'lag_{target}_{lag}'].median(), inplace=True)

def create_rolling_features(df, group_cols, target, window):
    df[f'rolling_mean_{target}'] = df.groupby(group_cols)[target].rolling(window=window).mean().reset_index(level=group_cols, drop=True)
    df[f'rolling_mean_{target}'].fillna(df[f'rolling_mean_{target}'].median(), inplace=True)

create_lag_features(df, ['seller_no', 'product_no'], 'qty', [1])
create_rolling_features(df, ['seller_no', 'product_no'], 'qty', 3)




# # 数据处理部分，在提取时间特征后添加以下代码
# # 添加滞后特征
# df['lag_qty_1'] = df.groupby(['seller_no', 'product_no'])['qty'].shift(1)
#
# # 添加滚动平均特征
# df['rolling_mean_qty'] = df.groupby(['seller_no', 'product_no'])['qty'].rolling(window=3).mean().reset_index(level=[0,1], drop=True)
#
# # 确保处理了新加入特征的缺失值
# df['lag_qty_1'].fillna(df['lag_qty_1'].median(), inplace=True)
# df['rolling_mean_qty'].fillna(df['rolling_mean_qty'].median(), inplace=True)
#
# # 处理分类数据
# # 初始化独热编码器
# ohe = OneHotEncoder(sparse_output=False)
#
# # 选择要进行独热编码的列
# categorical_columns = ['category1', 'category2', 'category3', 'seller_category',
#                        'inventory_category', 'seller_level', 'warehouse_category', 'warehouse_region']
# df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_columns]))
#
# # 添加列名到独热编码的数据
# df_encoded.columns = ohe.get_feature_names_out(categorical_columns)
#
# # 删除原始的分类列，并添加编码后的列
# df = df.drop(categorical_columns, axis=1)
# df = pd.concat([df, df_encoded], axis=1)

df_train = df[df['date'] <= '2023-05-15']
df_predict = df[(df['date'] > '2023-05-15') & (df['date'] <= '2023-05-30')]

# 划分训练集和测试集
X = df.drop(['seller_no', 'product_no', 'warehouse_no', 'date', 'qty'], axis=1)
y = df['qty']
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, df['date'], test_size=0.2, shuffle=False)


# # 使用XGBoost的默认参数初始化模型进行特征重要性评估
# xg_reg_for_feature_selection = xgb.XGBRegressor(random_state=42)
# xg_reg_for_feature_selection.fit(X_train, y_train)
#
# # 创建SelectFromModel对象，使用median作为阈值来选择特征
# selector = SelectFromModel(xg_reg_for_feature_selection, threshold='median')
# selector.fit(X_train, y_train)
#
# # 选择特征
# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)
#

# 贝叶斯优化
space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 16, dtype=int)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 1000, 50)),
    'alpha': hp.choice('alpha', np.arange(0, 20, dtype=int)),
    # 贝叶斯优化部分，在space字典中添加
    'gamma': hp.uniform('gamma', 0.0, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10)
}

def objective(space):
    xg_reg = xgb.XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        alpha=int(space['alpha']),
        colsample_bytree=space['colsample_bytree'],
        gamma=space['gamma'],
        # eval_metric="rmse",
        subsample=space['subsample'],
        min_child_weight=space['min_child_weight'],
        objective='reg:squarederror',
        # early_stopping_rounds = 50  # 设置早停参数
    )

    # 使用交叉验证来评估模型的性能
    scores = cross_val_score(xg_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # evaluation = [(X_train, y_train), (X_test, y_test)]
    #
    # xg_reg.fit(X_train, y_train,
    #            eval_set=evaluation,
    #            verbose=False)




    # pred = xg_reg.predict(X_test)
    # 由于cross_val_score返回的是负MSE，我们取其负数以得到MSE
    mse = -scores.mean()
    print(f"Mean Squared Error: {mse}")
    return {'loss': mse, 'status': STATUS_OK}
    # mse = mean_squared_error(y_test, pred)
    # print(f"MSE: {mse}")
    # return {'loss': mse, 'status': STATUS_OK}


def convert_hyperparams(best_hyperparams):
    # Create a dictionary to hold the converted hyperparameters
    best_hyperparams_converted = {}

    # Convert indexes to actual parameter values
    best_hyperparams_converted['max_depth'] = np.arange(3, 16, dtype=int)[best_hyperparams['max_depth']]
    best_hyperparams_converted['n_estimators'] = np.arange(50, 1000, 50)[best_hyperparams['n_estimators']]
    best_hyperparams_converted['alpha'] = np.arange(0, 20, dtype=int)[best_hyperparams['alpha']]

    # The following parameters are not using hp.choice, so we can just copy the value
    best_hyperparams_converted['learning_rate'] = best_hyperparams['learning_rate']
    best_hyperparams_converted['colsample_bytree'] = best_hyperparams['colsample_bytree']
    best_hyperparams_converted['gamma'] = best_hyperparams['gamma']
    best_hyperparams_converted['subsample'] = best_hyperparams['subsample']
    best_hyperparams_converted['min_child_weight'] = best_hyperparams['min_child_weight']

    return best_hyperparams_converted


trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)

# Convert the best hyperparameters
best_hyperparams_converted = convert_hyperparams(best_hyperparams)

print("The best hyperparameters are: ", "\n")
print(best_hyperparams)





# 使用贝叶斯优化找到的最佳参数训练XGBoost模型
def train_xgboost(X_train, y_train, X_test, y_test, best_hyperparams):
    xg_reg = xgb.XGBRegressor(n_estimators=int(best_hyperparams['n_estimators']),
        max_depth=int(best_hyperparams['max_depth']),
        learning_rate=best_hyperparams['learning_rate'],
        alpha=int(best_hyperparams['alpha']),
        colsample_bytree=best_hyperparams['colsample_bytree'],
        gamma=best_hyperparams['gamma'],
        subsample=best_hyperparams['subsample'],
        min_child_weight=best_hyperparams['min_child_weight'],
        objective='reg:squarederror')
    xg_reg.fit(X_train, y_train)
    return xg_reg

xg_reg = train_xgboost(X_train, y_train, X_test, y_test, best_hyperparams_converted)




#####
# 打印特征排名
# importance = xg_reg.feature_importances_
# indices = np.argsort(importance)[::-1]
#
# print("Feature ranking:")
#
# for f in range(X_train.shape[1]):
#     print(f"{f + 1}. feature {indices[f]} ({importance[indices[f]]})")







# 学习曲线分析
# 模型评估 - 封装学习曲线和预测评估
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    # 添加绘制学习曲线的代码
    # ...

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

plot_learning_curve(xg_reg, X_train, y_train, 'Learning Curve for XGBoost')
evaluate_model(xg_reg, X_test, y_test)




# 预测并确保所有预测值都不为负
y_pred = xg_reg.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # 应用不允许负数的规则
mse = mean_squared_error(y_test, y_pred)
print(f"Final MSE: {mse}")



#####
# # 计算误差
# error = 1 - (np.sum(np.abs(y_test - y_pred)) / np.sum(y_test))
# print(f"Error: {error}")

#  #模型调优
# parameters = {
#     'max_depth': [7, 9, 11],
#     'colsample_bytree': [0.3, 0.5, 0.7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 300, 500],
#     'alpha': [5, 10, 15]
# }
# xg_reg = xgb.XGBRegressor(objective='reg:squarederror')
# grid_search = GridSearchCV(estimator=xg_reg, param_grid=parameters, cv=5, n_jobs=-1, verbose=2,
#                            scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
#
# # 使用最佳参数
# best_params = grid_search.best_params_
# print(f"Best parameters found: {best_params}")
# # 使用最佳参数训练XGBoost模型
# xg_reg = xgb.XGBRegressor(**grid_search.best_params_)
# xg_reg.fit(X_train, y_train)





# 使用Stacking集成方法替换原有的VotingRegressor
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gb', GradientBoostingRegressor(n_estimators=100)),
]

final_estimator = GradientBoostingRegressor(n_estimators=100, subsample=0.5, max_features='sqrt')

stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=xg_reg,
    cv=5
)

stacking_reg.fit(X_train, y_train)
y_pred_stacking = stacking_reg.predict(X_test)
error_stacking = 1 - (np.sum(np.abs(y_test - y_pred_stacking)) / np.sum(y_test))
print(f"Stacking model error: {error_stacking}")




#####
# # 集成学习
# rf_reg = RandomForestRegressor(n_estimators=100)
# gb_reg = GradientBoostingRegressor(n_estimators=100)
#
# ensemble_reg = VotingRegressor(estimators=[
#     ('xgb', xg_reg),
#     ('rf', rf_reg),
#     ('gb', gb_reg)
# ])
#
# ensemble_reg.fit(X_train, y_train)
#
# # 预测和评估
# y_pred_ensemble = ensemble_reg.predict(X_test)
# error_ensemble = 1 - (np.sum(np.abs(y_test - y_pred_ensemble)) / np.sum(y_test))
# print(f"Ensemble model error: {error_ensemble}")

# 绘图 - 使用函数封装绘图代码
# def plot_predictions(dates_test, y_test, y_pred, title):
#     plt.figure(figsize=(15, 5))
#     plt.scatter(dates_test, y_test, color='blue', label='Actual', alpha=0.5)
#     plt.scatter(dates_test, y_pred, color='red', label='Predicted', alpha=0.5)
#     plt.title(title)
#     plt.xlabel('Date')
#     plt.ylabel('Demand Quantity')
#     plt.legend()
#     plt.show()
#
# plot_predictions(dates_test, y_test, y_pred, 'Prediction vs Actual')