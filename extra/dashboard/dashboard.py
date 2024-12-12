import dash
from dash import html, dcc, ctx
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import aioboto3
import asyncio
from aiohttp import ClientSession
import json
from collections import defaultdict
import os
import base64
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px

# Replace boto3 client with aioboto3 session
session = aioboto3.Session()

BUCKET_NAME = 'exo-benchmarks'

def load_mock_data():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  mock_data_path = os.path.join(current_dir, 'mock_data.json')
  with open(mock_data_path, 'r') as f:
    return json.load(f)

async def load_data_from_s3():
  # For testing, use mock data if environment variable is set
  if os.getenv('USE_MOCK_DATA'):
    return load_mock_data()

  config_data = defaultdict(list)

  async with session.client('s3') as s3:
    paginator = s3.get_paginator('list_objects_v2')
    objects_to_fetch = []

    # First, get all object keys
    async for page in paginator.paginate(Bucket=BUCKET_NAME):
      for obj in page.get('Contents', []):
        key = obj['Key']
        key_parts = key.split('/')
        if len(key_parts) < 2:
          continue
        objects_to_fetch.append((key, obj['LastModified'], f"{key_parts[0]}/{key_parts[1]}"))

    # Then fetch all objects in parallel
    async def fetch_object(key, last_modified, config_name):
      response = await s3.get_object(Bucket=BUCKET_NAME, Key=key)
      body = await response['Body'].read()
      data = json.loads(body.decode('utf-8'))
      print(f"Processing object: {key}: {data}")
      return {
        'config_name': config_name,
        'data': {
          'timestamp': data.get('timestamp', last_modified.strftime('%Y-%m-%dT%H:%M:%S')),
          'prompt_tps': data.get('prompt_tps', 0),
          'generation_tps': data.get('generation_tps', 0),
          'commit': data.get('commit', ''),
          'run_id': data.get('run_id', ''),
          'model': data.get('model', ''),
          'branch': data.get('branch', ''),
          'configuration': data.get('configuration', {}),
          'prompt_len': data.get('prompt_len', 0),
          'ttft': data.get('ttft', 0),
          'response_len': data.get('response_len', 0),
          'total_time': data.get('total_time', 0)
        }
      }

    # Create tasks for all objects
    tasks = [fetch_object(key, last_modified, config_name)
             for key, last_modified, config_name in objects_to_fetch]
    results = await asyncio.gather(*tasks)

    # Organize results into config_data
    for result in results:
      config_data[result['config_name']].append(result['data'])

    # Sort data by timestamp for each config
    for config in config_data:
      config_data[config].sort(key=lambda x: x['timestamp'])

    return config_data

async def get_best_benchmarks():
  config_data = await load_data_from_s3()
  best_results = {}

  for config_name, data in config_data.items():
    if not data:
      continue

    # Split config_name into config and model
    config, model = config_name.split('/')

    # Find the entry with the highest generation_tps
    best_result = max(data, key=lambda x: x['generation_tps'])

    # Create result dictionary with all original data plus config/model info
    result = dict(best_result)  # Make a copy of all data from the best run
    result.update({
      'config': config,
      'model': model,
    })

    best_results[config_name] = result

  return best_results

app = dash.Dash(__name__)

app.layout = html.Div([
  html.H1('Benchmark Performance Dashboard'),
  html.Button('Test Sound', id='test-sound-button', n_clicks=0),
  html.Div(id='graphs-container'),
  html.Audio(id='success-sound', src='assets/pokemon_evolve.mp3', preload="auto", style={'display': 'none'}),
  html.Audio(id='failure-sound', src='assets/gta5_wasted.mp3', preload="auto", style={'display': 'none'}),
  html.Audio(id='startup-sound', src='assets/pokemon_evolve.mp3', preload="auto", style={'display': 'none'}),
  html.Div(id='audio-trigger', style={'display': 'none'}),
  dcc.Store(id='previous-data', storage_type='memory'),
  dcc.Interval(
    id='interval-component',
    interval=15000,  # Update every 15 seconds
    n_intervals=0
  )
])

@app.callback(
  [Output('graphs-container', 'children'),
   Output('previous-data', 'data'),
   Output('audio-trigger', 'children')],
  [Input('interval-component', 'n_intervals')],
  [State('previous-data', 'data')]
)
def update_graphs(n, previous_data):
  # Run async operations synchronously
  config_data = asyncio.run(load_data_from_s3())
  graphs = []
  trigger_sound = None

  if previous_data:
    for config_name, data in config_data.items():
      if config_name in previous_data and data and previous_data[config_name]:
        current_generation_tps = data[-1]['generation_tps']
        previous_generation_tps = previous_data[config_name][-1]['generation_tps']

        # Add clear logging for TPS changes
        if current_generation_tps != previous_generation_tps:
          print("\n" + "="*50)
          print(f"Config: {config_name}")
          print(f"Previous Generation TPS: {previous_generation_tps}")
          print(f"Current Generation TPS: {current_generation_tps}")
          print(f"Change: {current_generation_tps - previous_generation_tps}")

        if current_generation_tps > previous_generation_tps:
          print("🔼 Generation TPS INCREASED - Should play success sound")
          trigger_sound = 'success'
        elif current_generation_tps < previous_generation_tps:
          print("🔽 Generation TPS DECREASED - Should play failure sound")
          trigger_sound = 'failure'

        if current_generation_tps != previous_generation_tps:
            print("="*50 + "\n")

  for config_name, data in config_data.items():
    timestamps = [d['timestamp'] for d in data]
    generation_tps = [d['generation_tps'] for d in data]
    commits = [d['commit'] for d in data]
    run_ids = [d['run_id'] for d in data]

    # Create subplot with 2 columns
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Performance Over Time', 'Generation TPS Distribution'),
                       column_widths=[0.7, 0.3])

    # Time series plot (left)
    fig.add_trace(go.Scatter(
      x=timestamps,
      y=generation_tps,
      name='Generation TPS',
      mode='lines+markers',
      hovertemplate='Commit: %{text}<br>TPS: %{y}<extra></extra>',
      text=commits,
      customdata=run_ids,
      line=dict(color='#2196F3', width=2),
      marker=dict(color='#2196F3')
    ), row=1, col=1)

    # Calculate statistics
    gen_tps_array = np.array(generation_tps)
    stats = {
      'Mean': np.mean(gen_tps_array),
      'Std Dev': np.std(gen_tps_array),
      'Min': np.min(gen_tps_array),
      'Max': np.max(gen_tps_array)
    }

    # Histogram plot (right)
    fig.add_trace(go.Histogram(
      x=generation_tps,
      name='Generation TPS Distribution',
      nbinsx=10,
      showlegend=False,
      marker=dict(color='#2196F3')
    ), row=1, col=2)

    # Add statistics as annotations
    stats_text = '<br>'.join([f'{k}: {v:.2f}' for k, v in stats.items()])
    fig.add_annotation(
      x=0.98,
      y=0.98,
      xref='paper',
      yref='paper',
      text=stats_text,
      showarrow=False,
      font=dict(size=12),
      align='left',
      bgcolor='rgba(255, 255, 255, 0.8)',
      bordercolor='black',
      borderwidth=1
    )

    fig.update_layout(
      title=f'Performance Metrics - {config_name}',
      height=500,
      showlegend=True,
      hovermode='x unified',
      clickmode='event'
    )

    # Update x and y axis labels
    fig.update_xaxes(title_text='Timestamp', row=1, col=1)
    fig.update_xaxes(title_text='Generation TPS', row=1, col=2)
    fig.update_yaxes(title_text='Tokens per Second', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=2)

    graphs.append(html.Div([
      dcc.Graph(
        figure=fig,
        id={'type': 'dynamic-graph', 'index': config_name},
        config={'displayModeBar': True}
      )
    ]))

  return graphs, config_data, trigger_sound

@app.callback(
  Output('graphs-container', 'children', allow_duplicate=True),
  Input({'type': 'dynamic-graph', 'index': dash.ALL}, 'clickData'),
  prevent_initial_call=True
)
def handle_click(clickData):
  # If you add any async operations here, wrap them with asyncio.run()
  if clickData and clickData[0] and clickData[0]['points'][0].get('customdata'):
    run_id = clickData[0]['points'][0]['customdata']
    url = f'https://github.com/exo-explore/exo/actions/runs/{run_id}'
    import webbrowser
    webbrowser.open_new_tab(url)
  return dash.no_update

app.clientside_callback(
  """
  function(trigger, test_clicks) {
    if (!trigger && !test_clicks) return window.dash_clientside.no_update;

    if (test_clicks > 0 && dash_clientside.callback_context.triggered[0].prop_id.includes('test-sound-button')) {
      console.log('Test button clicked');
      const audio = document.getElementById('startup-sound');
      if (audio) {
        audio.currentTime = 0;
        audio.play().catch(e => console.log('Error playing audio:', e));
      }
    } else if (trigger) {
      console.log('Audio trigger received:', trigger);
      if (trigger === 'success') {
        console.log('Playing success sound');
        const audio = document.getElementById('success-sound');
        if (audio) {
          audio.currentTime = 0;
          audio.play().catch(e => console.log('Error playing success sound:', e));
        }
      } else if (trigger === 'failure') {
        console.log('Playing failure sound');
        const audio = document.getElementById('failure-sound');
        if (audio) {
          audio.currentTime = 0;
          audio.play().catch(e => console.log('Error playing failure sound:', e));
        }
      }
    }
    return window.dash_clientside.no_update;
  }
  """,
  Output('audio-trigger', 'children', allow_duplicate=True),
  [Input('audio-trigger', 'children'),
   Input('test-sound-button', 'n_clicks')],
  prevent_initial_call=True
)

if __name__ == '__main__':
  import sys
  if '--generate' in sys.argv:
    async def generate_best():
      async with session.client('s3') as s3:
        best_benchmarks = await get_best_benchmarks()
        try:
          await s3.put_object(
            Bucket=BUCKET_NAME,
            Key='best.json',
            Body=json.dumps(best_benchmarks, indent=2),
            ContentType='application/json'
          )
          print("Successfully uploaded best.json to S3")
          print(f"Public URL: https://{BUCKET_NAME}.s3.amazonaws.com/best.json")
        except Exception as e:
          print(f"Error uploading to S3: {e}")

    asyncio.run(generate_best())
  else:
    app.run_server(debug=True)
