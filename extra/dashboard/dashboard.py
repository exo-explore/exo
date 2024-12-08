import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import boto3
import json
from collections import defaultdict
import os

s3 = boto3.client('s3')
BUCKET_NAME = 'exo-benchmarks'

def load_mock_data():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  mock_data_path = os.path.join(current_dir, 'mock_data.json')
  with open(mock_data_path, 'r') as f:
    return json.load(f)

def load_data_from_s3():
  # For testing, use mock data if environment variable is set
  if os.getenv('USE_MOCK_DATA'):
    return load_mock_data()

  config_data = defaultdict(list)

  paginator = s3.get_paginator('list_objects_v2')
  for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get('Contents', []):
      key = obj['Key']
      key_parts = key.split('/')
      if len(key_parts) < 2:
        continue
      config_name = f"{key_parts[0]}/{key_parts[1]}"  # Include both config and model
      response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
      data = json.loads(response['Body'].read().decode('utf-8'))
      print(f"Processing object: {obj['Key']}: {data}")
      config_data[config_name].append({
        'timestamp': data.get('timestamp', obj['LastModified'].strftime('%Y-%m-%dT%H:%M:%S')),
        'prompt_tps': data.get('prompt_tps', 0),
        'generation_tps': data.get('generation_tps', 0),
        'commit': data.get('commit', ''),
        'run_id': data.get('run_id', '')
      })

  for config in config_data:
    config_data[config].sort(key=lambda x: x['timestamp'])

  return config_data

app = dash.Dash(__name__)

app.layout = html.Div([
  html.H1('Benchmark Performance Dashboard'),
  html.Div(id='graphs-container'),
  dcc.Interval(
    id='interval-component',
    interval=300000,  # Update every 5 minutes
    n_intervals=0
  )
])

@app.callback(
  Output('graphs-container', 'children'),
  Input('interval-component', 'n_intervals')
)
def update_graphs(n):
  config_data = load_data_from_s3()
  graphs = []

  for config_name, data in config_data.items():
    timestamps = [d['timestamp'] for d in data]
    prompt_tps = [d['prompt_tps'] for d in data]
    generation_tps = [d['generation_tps'] for d in data]
    commits = [d['commit'] for d in data]
    run_ids = [d['run_id'] for d in data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
      x=timestamps,
      y=prompt_tps,
      name='Prompt TPS',
      mode='lines+markers',
      hovertemplate='Commit: %{text}<br>TPS: %{y}<extra></extra>',
      text=commits,
      customdata=run_ids
    ))

    fig.add_trace(go.Scatter(
      x=timestamps,
      y=generation_tps,
      name='Generation TPS',
      mode='lines+markers',
      hovertemplate='Commit: %{text}<br>TPS: %{y}<extra></extra>',
      text=commits,
      customdata=run_ids
    ))

    fig.update_layout(
      title=f'Performance Metrics - {config_name}',
      xaxis_title='Timestamp',
      yaxis_title='Tokens per Second',
      hovermode='x unified',
      clickmode='event'
    )

    graphs.append(html.Div([
      dcc.Graph(
        figure=fig,
        id={'type': 'dynamic-graph', 'index': config_name},
        config={'displayModeBar': True}
      )
    ]))

  return graphs

@app.callback(
  Output('_', 'children'),
  Input({'type': 'dynamic-graph', 'index': dash.ALL}, 'clickData')
)
def handle_click(clickData):
  if clickData and clickData['points'][0].get('customdata'):
    run_id = clickData['points'][0]['customdata']
    url = f'https://github.com/exo-explore/exo/actions/runs/{run_id}'
    import webbrowser
    webbrowser.open_new_tab(url)
  return dash.no_update

if __name__ == '__main__':
  app.run_server(debug=True)
