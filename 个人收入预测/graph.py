import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv('acc.csv')
line = go.Scatter(x=data['ID'], y=data['Value'])
fig = go.Figure(line)
fig.update_layout(
    title='训练30次时的正确率',
    title_x=0.5,
    xaxis_title='训练次数',
    yaxis_title='分类正确率'
)
fig.show()