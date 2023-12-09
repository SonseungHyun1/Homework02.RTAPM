# 데이터 셋 불러오기
import pandas as pd
data_path = r"..\data\Catalunya Accidents data.csv"
data = pd.read_csv(data_path)

print(data.head())
print(data. describe())

