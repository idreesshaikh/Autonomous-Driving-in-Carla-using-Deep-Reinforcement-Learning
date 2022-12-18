import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('logs/Town7/PPO_carla_1.csv')

fig = go.Figure(go.Scatter(x = df['timestep'], y = df['reward'],
                  name='PPO'))

fig.update_layout(title='Average Reward/Timestep',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()