import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 1. 数据导入
df_shipment = pd.read_excel('附件1-商家历史出货量表.xlsx')
df_product = pd.read_excel('附件2-商品信息表.xlsx')
df_seller = pd.read_excel('附件3-商家信息表.xlsx')
df_warehouse = pd.read_excel('附件4-仓库信息表.xlsx')

# 2. 数据处理
# 合并数据集
df = df_shipment.merge(df_product, on='product_no', how='left')
df = df.merge(df_seller, on='seller_no', how='left')
df = df.merge(df_warehouse, on='warehouse_no', how='left')

assert 'warehouse_category' in df_warehouse.columns, "在仓库数据中未找到'warehouse_category'列。"
assert 'warehouse_region' in df_warehouse.columns, "在仓库数据中未找到'warehouse_region'列。"

# 处理日期数据
df['date'] = pd.to_datetime(df['date'])

# 提取时间特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# 移除目标变量（qty）中的异常值
Q1 = df['qty'].quantile(0.25)
Q3 = df['qty'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['qty'] >= (Q1 - 1.5 * IQR)) & (df['qty'] <= (Q3 + 1.5 * IQR))]

# 处理分类数据
# 初始化独热编码器
ohe = OneHotEncoder(sparse_output=False)  # 将 'sparse_output' 更正为 'sparse'

# 选择要进行独热编码的列
categorical_columns = ['category1', 'category2', 'category3', 'seller_category',
                       'inventory_category', 'seller_level', 'warehouse_category', 'warehouse_region']
df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_columns]))

# 添加列名到独热编码的数据
df_encoded.columns = ohe.get_feature_names_out(categorical_columns)

# 删除原始的分类列，并添加编码后的列
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, df_encoded], axis=1)

#特征工程 滞后特征（lag features）以捕捉时间相关的信息
df['previous_month_qty'] = df.groupby('product_no')['qty'].shift(1)

# 划分训练集和测试集
X = df.drop(['seller_no', 'product_no', 'warehouse_no', 'date', 'qty'], axis=1)
y = df['qty']
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, df['date'], test_size=0.2, shuffle=False)

imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 3. 应用随机森林模型进行训练
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)

rf_reg.fit(X_train, y_train)

# 4. 测试集预测并显示结果
y_pred_rf = rf_reg.predict(X_test)

# 计算误差
error = 1 - (np.sum(np.abs(y_test - y_pred_rf)) / np.sum(y_test))
print(error)