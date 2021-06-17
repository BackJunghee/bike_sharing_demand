# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.2 64-bit (''miniforge3-4.9.2'': pyenv)'
#     language: python
#     name: python392jvsc74a57bd01e1316376a83551d13556d3d3320e9d66876ad560731359d20b0bf1660df0458
# ---

# +
##기본 코드 귀찮아서 복붙함

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

import missingno as msno
plt.style.use('seaborn')

import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['axes.unicode_minus'] = False

# %matplotlib inline




# +
#데이터 가져오기
import pandas as pd
import os 

#현재 경로 얻어오기
path = os.getcwd()

df_train = pd.read_csv("data/train.csv", parse_dates = ["datetime"])
df_test = pd.read_csv("data/test.csv", parse_dates = ["datetime"])


# -

#훈련 데이터 컬럼 정보
df_train.info()

#결측치 확인 
# => 이 작업은 test할 데이터 프레임에서도 해주어야 함 
# 훈련 데이터에는 없음 
for col in df_train.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msperc)

df_train.describe() # 각 컬럼별 요약 통계 지표 출력 



# +
# datetime 을 세부적으로 나누기 
# datetime 에서 년, 월, 일, 시간, 분,초 컬럼으로 변환 
# => 날짜에 영향을 많을 것을 예상함. 

df_train['year'] = df_train['datetime'].dt.year
df_train['month'] = df_train['datetime'].dt.month
df_train['day'] = df_train['datetime'].dt.day
df_train['hour'] = df_train['datetime'].dt.hour
df_train['minute'] = df_train['datetime'].dt.minute
df_train['second'] = df_train['datetime'].dt.second

# +
# 시각화 해보기 
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 10)

sns.barplot(data=df_train, x="year", y="count", ax=ax1)
sns.barplot(data=df_train, x="month", y="count", ax=ax2)
sns.barplot(data=df_train, x="day", y="count", ax=ax3)
sns.barplot(data=df_train, x="hour", y="count", ax=ax4)
sns.barplot(data=df_train, x="minute", y="count", ax=ax5)
sns.barplot(data=df_train, x="second", y="count", ax=ax6)

ax1.set(title="Rental amounts by year")
ax2.set(title="Rental amounts by month")
ax3.set(title="Rental amounts by day")
ax4.set(title="Rental amounts by hour");
# -

# minute, second는 모두 0으로 되어있으므로 활용 불가, 제거 
# 출,퇴근 시간이 peak time -> 

# +
# working day인 날중 출퇴근 시간을  peak 타임으로 설정

df_train['peak'] = df_train[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 13)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
# -

sns.set_style("darkgrid")
plt.figure(figsize=(15,10))
plt.suptitle('variables distribution')
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
for i, col in enumerate(df_train.columns[:12]):
    plt.subplot(3,4,i+1)
    if str(df_train[col].dtypes)[:3]=='int':
        if len(df_train[col].unique()) > 5:
            sns.distplot(df_train[col])
        else:
            sns.countplot(df_train[col])
    else:
        sns.distplot(df_train[col])
    plt.ylabel(col)
# +
# casual, registered, count 왼쪽으로 치우쳐진 그래프를 하고 있음 
# 정규분포의 형태에 가깝게 하기 위해 log화를 해준다. 

# logarithmic transformation of dependent cols
# adding 1 first so that 0 values don't become -inf
for col in ['casual', 'registered', 'count']:
    df_train['%s_log' % col] = np.log(df_train[col] + 1)

# +
for i, col in enumerate(df_train.columns[19:]):
    figure, axes = plt.subplots(nrows=1, ncols=1)
    
    #plt.subplot(3,4,i+1)
    if str(df_train[col].dtypes)[:3]=='int':
        if len(df_train[col].unique()) > 5:
            sns.distplot(df_train[casual_log])
        else:
            sns.countplot(df_train[col])
    else:
        sns.distplot(df_train[col])
    plt.ylabel(col)
    
# log화한 결과 조오금 정규분포 모양이 되긴 되었음... 
# -

df_train.columns

# +
#windspeed의 0은 정보가 없는 경우 0으로 들어간 듯함 
#평균 값으로 바꿔주면 좋을듯

#0을 제외했을 때의 평균값 
win_spd_mean = df_train["windspeed"].loc[df_train["windspeed"]!=0].mean()
df_train.loc[df_train["windspeed"]==0, "windspeed"] = win_spd_mean
# -

sns.distplot(df_train['windspeed'])

# +
#windspeed의 이상치 제거 할지 말지 모르겠음? 
df_train_without_outliers = df_train[df_train['windspeed'] - df_train['windspeed'].mean() < 3*df_train['windspeed'].std()]

print(df_train.shape)
print(df_train_without_outliers.shape)
# -

#피쳐간 상관관계 확인
g = sns.PairGrid(data=df_train, vars=['temp', 'atemp', 'casual_log', 'registered_log', 'humidity', 'windspeed', 'count_log'])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)


