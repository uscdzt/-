#coding=gbk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.svm import SVR
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


dtrain = pd.read_csv('zhengqi_train.txt', sep='\t')
dtest = pd.read_csv('zhengqi_test.txt', sep='\t')
dfull = pd.concat([dtrain, dtest], ignore_index=True, sort=False)
print('训练集大小: ', np.shape(dtrain))
print('测试集大小: ', np.shape(dtest))

print('缺失值统计：')
print(dfull.apply(lambda x: sum(x.isnull())))

# 观察数据基本分布情况
plt.figure(figsize=(18, 8), dpi=100)
dfull.boxplot(sym='r^', patch_artist=True, notch=True)
plt.title('DATA-FULL')

# 绘制数据相关性热力图，查看数据相关性情况
def heatmap(df):
    plt.figure(figsize=(20, 16), dpi=100)
    cols = df.columns.tolist()
    mcorr = df[cols].corr(method='spearman')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.xticks(rotation=45)
    return mcorr
dtrain_mcorr = heatmap(dtrain)

# 删除dfull表中指定的列
def drop_var(var_lst):
   dfull.drop(var_lst, axis=1, inplace=True)


# 将dfull重新分割为dtrain核dtest
def split_dfull():
    dtrain = dfull[dfull['target'].notnull()]
    dtest = dfull[dfull['target'].isnull()]
    dtest.drop('target', axis=1, inplace=True)
    return dtrain, dtest

# 剔除训练集与测试集分布不均匀的特征变量
plt.figure(figsize=(20,50),dpi=100)
for i in range(38):
    plt.subplot(10,4,i+1)
    sns.distplot(dtrain.iloc[:,i], color='green')
    sns.distplot(dtest.iloc[:,i], color='red')
    plt.legend(['Train', 'Test'])

# 从各特征数据分布直方图发现：
# V5、V9、V11、V17、V20、V21、V22、V27、V28 特征训练集和测试集分布差异过大
# 因此为了减小数据分布不均对预测结果的影响，应将上述特征进行剔除
plt.tight_layout()
drop_var(['V5','V9','V11','V17','V20','V21','V22','V27','V28'])
dtrain, dtest = split_dfull()

# 删除无关变量
drop_var(['V14','V25','V26','V32','V33','V34','V35'])
dtrain, dtest = split_dfull()

# 对偏态数据进行正态化转换
# 分布呈明显左偏的特征
piantai = ['V0','V1','V6','V7','V8','V12','V16','V31']

# 创建函数——找到令偏态系数绝对值最小的对数转换的底
def find_min_skew(data):
    subs = list(np.arange(1.01,2,0.01))
    skews = []
    for x in subs:
        skew = abs(stats.skew(np.power(x,data)))
        skews.append(skew)
    min_skew = min(skews)
    i = skews.index(min_skew)
    return subs[i], min_skew

# 对训练集和测试集偏态特征同时进行对数转换
for col in piantai:
    sub = find_min_skew(dfull[col])[0]
    dfull[col] = np.power(sub, dfull[col])
dtrain, dtest = split_dfull()

# 采用 z-score标准化 方法
dfull.iloc[:,:-1] = dfull.iloc[:,:-1].apply(lambda x: (x-x.mean())/x.std())
dtrain, dtest = split_dfull()

# 训练模型

# 将训练数据分割为训练集与验证集

X = np.array(dtrain.iloc[:,:-1])
y = np.array(dtrain['target'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建模型评分函数

def score(y, y_pred):
    # 计算均方误差 MSE
    print('MSE = {0}'.format(mean_squared_error(y, y_pred)))
    # 计算模型决定系数 R2
    print('R2 = {0}'.format(r2_score(y, y_pred)))

    # 计算预测残差，找异常点
    y = pd.Series(y)
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z) > 3)

    # 图一：真实值vs预计值
    plt.figure(figsize=(18, 5), dpi=80)
    plt.subplot(131)
    plt.plot(y, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))

    # 图二：残差分布散点图
    plt.subplot(132)
    plt.plot(y, y - y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('resid')
    plt.ylim([-3, 3])
    plt.title('std_resid = {:.3f}'.format(std_resid))

    # 图三：残差z得分直方图
    plt.subplot(133)
    sns.distplot(z, bins=50)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))
    plt.tight_layout()

# 利用RidgeCV函数自动寻找最优参数
ridge = RidgeCV()
ridge.fit(X_train, y_train)
print('best_alpha = {0}'.format(ridge.alpha_))

# 开始模型训练前，利用岭回归模型预测，剔除异常样本
y_pred = ridge.predict(X_train)
score(y_train, y_pred)

# 找出异常样本点并剔除
resid = y_train - y_pred
resid = pd.Series(resid, index=range(len(y_train)))
resid_z = (resid-resid.mean()) / resid.std()
outliers = resid_z[abs(resid_z)>3].index
print(f'{len(outliers)} Outliers:')
print(outliers.tolist())

plt.figure(figsize=(14,6),dpi=60)

plt.subplot(121)
plt.plot(y_train, y_pred, '.')
plt.plot(y_train[outliers], y_pred[outliers], 'ro')
plt.title(f'MSE = {mean_squared_error(y_train,y_pred)}')
plt.legend(['Accepted', 'Outliers'])
plt.xlabel('y_train')
plt.ylabel('y_pred')

plt.subplot(122)
sns.distplot(resid_z, bins = 50)
sns.distplot(resid_z.loc[outliers], bins = 50, color = 'r')
plt.legend(['Accepted', 'Outliers'])
plt.xlabel('z')
plt.tight_layout()

# 开始进行模型训练

# 利用LassoCV自动选择最佳正则化参数
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
print('best_alpha = {0}'.format(lasso.alpha_))

pred_lasso = lasso.predict(X_valid)
score(y_valid, pred_lasso)

# 使用sklearn中的网格搜索方法 GridSearchCV 寻找SVR最优模型参数
# 创建GridSearchCV网格参数搜寻函数，评价标准为最小均方误差，采用K折交叉验证的检验方法
def gsearch(model, param_grid, scoring='neg_mean_squared_error', splits=5, repeats=1, n_jobs=-1):
    # p次k折交叉验证
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=0)
    model_gs = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=rkfold, verbose=1, n_jobs=-1)
    model_gs.fit(X_train, y_train)
    print('参数最佳取值: {0}'.format(model_gs.best_params_))
    print('最小均方误差: {0}'.format(abs(model_gs.best_score_)))
    return model_gs

# 缩小参数范围进行细调
svr = SVR()
cv_params = {'C': [1,2,5,10,15,20,30,50,80,100,150,200], 'gamma': [0.0001,0.0005,0.0008,0.001,0.002,0.003,0.005]}
svr = gsearch(svr, cv_params)

# 验证集预测
pred_svr = svr.predict(X_valid)
score(y_valid, pred_svr)

# XGBRegressor参数调优

# 初始参数值
params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

# 最佳迭代次数：n_estimators 得出最佳结果为500
cv_params = {'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1100,1200]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
# 更新参数
params['n_estimators'] = 500

# min_child_weight  以及 max_depth 最佳为4 ，4
cv_params = {'max_depth': [3,4,5,6,7,8,9],
            'min_child_weight': [1,2,3,4,5,6,7]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
params['max_depth'] = 4
params['min_child_weight'] = 7

# 后剪枝参数 gamma 最佳为0
cv_params = {'gamma': [0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
params['gamma'] = 0

# 样本采样subsample 和 列采样colsample_bytree 最佳为 0.8 0.8
cv_params = {'subsample': [0.6,0.7,0.8,0.9],
            'colsample_bytree': [0.6,0.7,0.8,0.9]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8

# L1正则项参数reg_alpha 和 L2正则项参数reg_lambda 最佳为0 1
cv_params = {'reg_alpha': [0,0.02,0.05,0.1,1,2,3],
             'reg_lambda': [0,0.02,0.05,0.1,1,2,3]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
params['reg_alpha'] = 0
params['reg_lambda'] = 1

# 最后是learning_rate 最佳为 0.04
cv_params = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2]}
xgb = XGBRegressor(**params)
xgb = gsearch(xgb, cv_params)
params['learning_rate'] = 0.04

# 参数调优完成，以验证集进行模型误差验证
pred_xgb = xgb.predict(X_valid)
score(y_valid, pred_xgb)

# 模型评估

# 训练集和验证集的准确率的变化曲线
models = [lasso, svr, xgb]
model_names = ['Lasso', 'SVR', 'XGB']
plt.figure(figsize=(20, 5))

for i, m in enumerate(models):
    train_sizes, train_scores, test_scores = learning_curve(m, X, y, cv=5, scoring='neg_mean_squared_error',
                                                            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.subplot(1, 3, i + 1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Train')
    plt.plot(train_sizes, test_scores_mean, '^-', label='Test')
    plt.xlabel('Train_size')
    plt.ylabel('Score')
    plt.ylim([0, 0.35])
    plt.title(model_names[i], fontsize=16)
    plt.legend()
    plt.grid()

plt.tight_layout()

# 模型加权融合
def model_mix(pred_1, pred_2, pred_3):
    result = pd.DataFrame(columns=['Lasso','SVR','XGB','Combine'])
    for a in range(5):
        for b in range(1,6):
            for c in range(5):
                y_pred = (a*pred_1 + b*pred_2 + c*pred_3) / (a+b+c)
                mse = mean_squared_error(y_valid, y_pred)
                result = result.append([{'Lasso':a, 'SVR':b, 'XGB':c, 'Combine':mse}], ignore_index=True)
    return result

model_combine = model_mix(pred_lasso, pred_svr, pred_xgb)
model_combine.sort_values(by='Combine', inplace=True)
model_combine.head()

# 模型预测
X_test = np.array(dtest)
ans_lasso = lasso.predict(X_test)
ans_svr = svr.predict(X_test)
ans_xgb = xgb.predict(X_test)
ans_mix = (ans_lasso + 5 * ans_svr + 2 * ans_xgb) / 8
pd.Series(ans_mix).to_csv('正态+标准化.txt', sep='\t', index=False)
print('finish')
