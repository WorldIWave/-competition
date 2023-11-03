import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder

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

# 特征工程 - 将时间滞后和滚动平均的创建封装为函数
#数据处理部分
def create_lag_features(df, group_cols, target, lags):
    for lag in lags:
        df[f'lag_{target}_{lag}'] = df.groupby(group_cols)[target].shift(lag)
        df[f'lag_{target}_{lag}'].fillna(df[f'lag_{target}_{lag}'].median(), inplace=True)

def create_rolling_features(df, group_cols, target, window):
    df[f'rolling_mean_{target}'] = df.groupby(group_cols)[target].rolling(window=window).mean().reset_index(level=group_cols, drop=True)
    df[f'rolling_mean_{target}'].fillna(df[f'rolling_mean_{target}'].median(), inplace=True)

create_lag_features(df, ['seller_no', 'product_no'], 'qty', [1])
create_rolling_features(df, ['seller_no', 'product_no'], 'qty', 3)

df_train = df[df['date'] <= '2023-05-15']
df_predict = df[(df['date'] > '2023-05-15') & (df['date'] <= '2023-05-30')]

# 划分训练集和测试集
X = df.drop(['seller_no', 'product_no', 'warehouse_no', 'date', 'qty'], axis=1)
y = df['qty']
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, df['date'], test_size=0.2, shuffle=False)

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
        subsample=space['subsample'],
        min_child_weight=space['min_child_weight'],
        objective='reg:squarederror'
    )

    # 使用交叉验证来评估模型的性能
    scores = cross_val_score(xg_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    print(f"Mean Squared Error: {mse}")
    return {'loss': mse, 'status': STATUS_OK}

def convert_hyperparams(best_hyperparams):
    best_hyperparams_converted = {}
    best_hyperparams_converted['max_depth'] = np.arange(3, 16, dtype=int)[best_hyperparams['max_depth']]
    best_hyperparams_converted['n_estimators'] = np.arange(50, 1000, 50)[best_hyperparams['n_estimators']]
    best_hyperparams_converted['alpha'] = np.arange(0, 20, dtype=int)[best_hyperparams['alpha']]
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

#将最优参数进行转化
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



# 学习曲线分析
# 模型评估 - 封装学习曲线和预测评估
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))


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

# 使用Stacking集成方法
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
