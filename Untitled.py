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
for col in df_train.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msperc)




