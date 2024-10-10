import openpyxl
import pandas as pd
from openpyxl.styles import PatternFill

df = pd.read_excel('BRT.xlsx')

df = df[['sheet','User find','detection x','Out of bound','detection 한 이미지 수']]

df = df.dropna(subset=['sheet','User find','detection x','Out of bound','detection 한 이미지 수'])

df = df.sort_values(by='sheet')


df = df.groupby('sheet').agg({
    'detection x': 'sum',
    'User find' : 'sum',
    'Out of bound' : 'sum',
    'detection 한 이미지 수' :'sum',
}).reset_index()

df = df[['sheet','User find','detection 한 이미지 수','detection x','Out of bound']]
df['yolo'] = (df['detection 한 이미지 수'] / df['User find']) * 100
df['yolo'] = df['yolo'].round(0).astype(int)
df = df[['sheet','yolo','User find','detection 한 이미지 수','Out of bound','detection x']]
print(df.head(10))
df.columns=['Seat','yolo 인지율 (%)','총 이미지 수 (장)','detection 한 이미지 수 (장)','바운딩 박스 범위 벗어난 장수 (장)','detection 못한 이미지 (장)']

df.to_excel('result.xlsx',index=False)




