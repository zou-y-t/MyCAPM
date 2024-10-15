# 调包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 获取数据
name=[
    '国债指数',
    '浦发银行',
    '上证指数',
    '浙江龙盛',
]

df1=pd.read_csv('国债指数.csv',encoding='gbk')
df2=pd.read_csv('浦发银行.csv',encoding='gbk')
df3=pd.read_csv('上证指数.csv',encoding='gbk')
df4=pd.read_csv('浙江龙盛.csv',encoding='gbk')

df=[df1,df2,df3,df4,]

# 计算收益率
for i in range(4):
    df[i]['r']=df[i]['close']/df[i]['close'].shift(1)-1

# 计算累计收益率
for i in range(4):
    df[i]['R']=(1+df[i]['r']).cumprod()-1


# 拟合R
beta=[]
alpha=[]
R2=[]
for i in range(4):
    t=np.arange(1,len(df[i])).reshape(-1,1)
    R=df[i]['R'].values[1:].reshape(-1,1)

    model=LinearRegression()
    model.fit(t,R)

    beta.append(model.coef_[0][0])
    alpha.append(model.intercept_[0])
    R2.append(model.score(t,R))

# 输出结果
for i in range(4):
    print(name[i]+'的alpha,beta和R2分别为：',alpha[i],beta[i],R2[i])
# 可视化
plt.figure(figsize=(12,8))
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif']=['SimHei']

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(name[i]+'收益率')
    plt.plot(df[i]['date'],df[i]['r'],label=name[i])
    plt.xticks(range(0,len(df[i]),15),df[i]['date'][::15])
    plt.legend()

plt.show()

plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(name[i]+'累计收益率')
    plt.plot(df[i]['R'],label=name[i])
    # 绘制拟合直线图,截距和斜率分别为alpha[i]和beta[i]
    t=np.arange(1,len(df[i])).reshape(-1,1)
    plt.plot(t,alpha[i]+beta[i]*t,label='拟合线')
    plt.xticks(range(0,len(df[i]),15),df[i]['date'][::15])
    # 标记beta，alpha，R2在图例中
    plt.legend(title=f'alpha: {alpha[i]:.4f}\nbeta: {beta[i]:.4f}\nR2: {R2[i]:.4f}')

plt.show()


# 计算策略（持有）收益、年化收益率、最大回撤、夏普比率

for i in [1,3]:
    strategy=df[i]['r'].mean()# 策略收益率(月度)
    year_return=df[i]['R'].mean()# 年化收益率
    max_retracement=df[i]['r'].max()-df[i]['r'].min() # 最大回撤
    sharp=(year_return-df[0]['R'].mean())/df[i]['r'].std() # 夏普比率
    print(name[i]+'的策略收益率、年化收益率、最大回撤、夏普比率分别为：',strategy,year_return,max_retracement,sharp)

    mean_of_month=df[i]['r'].mean()
    std_of_month=df[i]['r'].std()
    variance_of_month=df[i]['r'].var()
    print(name[i]+'的月度收益率均值、标准差、方差分别为：',mean_of_month,std_of_month,variance_of_month)

# 删除包含 NaN 的行
df1_r_clean = df[1]['r'].dropna()
df3_r_clean = df[3]['r'].dropna()

# 计算协方差和相关系数
covariance = np.cov(df1_r_clean, df3_r_clean)[0][1]
correlation_coefficient = np.corrcoef(df1_r_clean, df3_r_clean)[0][1]
print('浦发银行和浙江龙盛的协方差和相关系数分别为：', covariance, correlation_coefficient)