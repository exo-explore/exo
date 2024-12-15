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
import aiohttp
from datetime import datetime

# Replace boto3 client with aioboto3 session
session = aioboto3.Session()

BUCKET_NAME = 'exo-benchmarks'
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
CURSOR_KEY = 'last_processed_timestamp.txt'

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

async def get_previous_benchmark(config_data, config_name, current_timestamp):
  """Get the previous benchmark for a given configuration."""
  benchmarks = config_data.get(config_name, [])
  # Sort by timestamp and find the most recent benchmark before current_timestamp
  previous = None
  for b in sorted(benchmarks, key=lambda x: x['timestamp']):
    if b['timestamp'] < current_timestamp:
      previous = b
    else:
      break
  return previous

async def format_metric_comparison(current, previous, metric, format_str=".2f", lower_is_better=False):
  """Format a metric with trend indicator."""
  current_val = current.get(metric, 0)
  if not previous:
    return f"**{current_val:{format_str}}**"

  prev_val = previous.get(metric, 0)
  diff = current_val - prev_val

  # Invert the comparison logic if lower values are better
  if lower_is_better:
    diff = -diff  # This makes negative diffs good and positive diffs bad

  if diff > 0:
    return f"**{current_val:{format_str}}** 🟢↑ ({'-' if lower_is_better else '+'}{abs(current_val - prev_val):{format_str}})"
  elif diff < 0:
    return f"**{current_val:{format_str}}** 🔴↓ ({'+' if lower_is_better else '-'}{abs(current_val - prev_val):{format_str}})"
  else:
    return f"**{current_val:{format_str}}** ⚪"

async def send_discord_notification(benchmark_data, config_data):
  if not DISCORD_WEBHOOK_URL:
    print("Discord webhook URL not configured, skipping notification")
    return

  # Create a formatted message
  config_name = f"{benchmark_data['config']}/{benchmark_data['model']}"

  # Use the passed config_data instead of fetching again
  previous_benchmark = await get_previous_benchmark(
    config_data,
    f"{benchmark_data['config']}/{benchmark_data['model']}",
    benchmark_data['timestamp']
  )

  # Format metrics with comparisons
  gen_tps = await format_metric_comparison(benchmark_data, previous_benchmark, 'generation_tps')
  prompt_tps = await format_metric_comparison(benchmark_data, previous_benchmark, 'prompt_tps')
  ttft = await format_metric_comparison(
    {'ttft': benchmark_data['ttft'] * 1000},
    {'ttft': previous_benchmark['ttft'] * 1000} if previous_benchmark else None,
    'ttft',
    lower_is_better=True
  )
  prompt_len = await format_metric_comparison(benchmark_data, previous_benchmark, 'prompt_len', "d")
  response_len = await format_metric_comparison(benchmark_data, previous_benchmark, 'response_len', "d")

  # Create a simple JSON string of the topology
  topology = benchmark_data.get('configuration', {})
  topology_str = "```json\n" + json.dumps(topology, indent=2) + "\n```"

  message = (
    f"🚀 New Benchmark Result for **{config_name}**\n\n"
    f"📊 Performance Metrics:\n"
    f"• Generation TPS: {gen_tps}\n"
    f"• Prompt TPS: {prompt_tps}\n"
    f"• TTFT: {ttft}ms\n"
    f"• Prompt Length: {prompt_len}\n"
    f"• Response Length: {response_len}\n\n"
    f"🔍 Run Details:\n"
    f"• Commit: {benchmark_data['commit'][:7]}\n"
    f"• Branch: {benchmark_data['branch']}\n"
    f"• Run ID: [{benchmark_data['run_id']}](https://github.com/exo-explore/exo/actions/runs/{benchmark_data['run_id']})\n\n"
    f"{topology_str}"
  )

  async with aiohttp.ClientSession() as session:
    await session.post(DISCORD_WEBHOOK_URL, json={'content': message})

async def get_cursor():
  try:
    async with session.client('s3') as s3:
      response = await s3.get_object(Bucket=BUCKET_NAME, Key=CURSOR_KEY)
      body = await response['Body'].read()
      return body.decode('utf-8').strip()
  except:
    return "1970-01-01T00:00:00"  # Default to epoch if no cursor exists

async def update_cursor(timestamp):
  async with session.client('s3') as s3:
    await s3.put_object(
      Bucket=BUCKET_NAME,
      Key=CURSOR_KEY,
      Body=timestamp.encode('utf-8')
    )

async def generate_best():
  # Get the last processed timestamp
  last_processed = await get_cursor()
  print(f"Last processed timestamp: {last_processed}")

  async with session.client('s3') as s3:
    # Load all benchmark data once
    config_data = await load_data_from_s3()
    best_benchmarks = await get_best_benchmarks()

    # Check for new benchmarks in all data
    new_latest = last_processed
    for config_name, data_list in config_data.items():
      for benchmark in data_list:
        timestamp = benchmark['timestamp']

        # If this benchmark is newer than our last processed timestamp
        if timestamp > last_processed:
          print(f"Found new benchmark for {config_name} at {timestamp}")
          # Add config and model info to the benchmark data
          config, model = config_name.split('/')
          benchmark_with_info = dict(benchmark)
          benchmark_with_info.update({
            'config': config,
            'model': model,
          })
          # Pass the already loaded config_data to avoid refetching
          await send_discord_notification(benchmark_with_info, config_data)

          # Update the latest timestamp if this is the newest we've seen
          if timestamp > new_latest:
            new_latest = timestamp

    # Update the cursor if we found any new benchmarks
    if new_latest > last_processed:
      await update_cursor(new_latest)

    # Upload the best benchmarks as before
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

    # Create a list of unique branches for this config
    branches = list(set(d['branch'] for d in data))

    # Create subplot with 2 columns
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Performance Over Time', 'Generation TPS Distribution'),
                       column_widths=[0.7, 0.3])

    # Generate a color for each branch
    colors = px.colors.qualitative.Set1[:len(branches)]
    branch_colors = dict(zip(branches, colors))

    # Time series plot (left) - separate line for each branch
    for branch in branches:
        branch_data = [d for d in data if d['branch'] == branch]
        branch_timestamps = [d['timestamp'] for d in branch_data]
        branch_generation_tps = [d['generation_tps'] for d in branch_data]
        branch_commits = [d['commit'] for d in branch_data]
        branch_run_ids = [d['run_id'] for d in branch_data]

        fig.add_trace(go.Scatter(
            x=branch_timestamps,
            y=branch_generation_tps,
            name=f'{branch}',
            mode='lines+markers',
            hovertemplate='Branch: %{text}<br>Commit: %{customdata}<br>TPS: %{y}<extra></extra>',
            text=[branch] * len(branch_timestamps),
            customdata=branch_commits,
            line=dict(color=branch_colors[branch], width=2),
            marker=dict(color=branch_colors[branch])
        ), row=1, col=1)

    # Histogram plot (right) - stacked histogram by branch
    for branch in branches:
        branch_data = [d for d in data if d['branch'] == branch]
        branch_generation_tps = [d['generation_tps'] for d in branch_data]

        fig.add_trace(go.Histogram(
            x=branch_generation_tps,
            name=f'{branch}',
            nbinsx=10,
            marker=dict(color=branch_colors[branch]),
            opacity=0.75
        ), row=1, col=2)

    # Calculate statistics for all data
    gen_tps_array = np.array(generation_tps)
    stats = {
        'Mean': np.mean(gen_tps_array),
        'Std Dev': np.std(gen_tps_array),
        'Min': np.min(gen_tps_array),
        'Max': np.max(gen_tps_array)
    }

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
    asyncio.run(generate_best())
  else:
    app.run_server(debug=True)
