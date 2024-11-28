import os
import json
import logging
import asyncio
import aiohttp
import pandas as pd
import plotly.express as px
from typing import List, Dict, Optional
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import pygame.mixer
from datetime import datetime

class AsyncCircleCIClient:
    def __init__(self, token: str, project_slug: str):
        self.token = token
        self.project_slug = project_slug
        self.base_url = "https://circleci.com/api/v2"
        self.headers = {
            "Circle-Token": token,
            "Accept": "application/json"
        }
        self.logger = logging.getLogger("CircleCI")

    async def get_json(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Dict:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_recent_pipelines(
        self,
        session: aiohttp.ClientSession,
        org_slug: str = None,
        page_token: str = None,
        limit: int = None,
        branch: str = None
    ):
        """
        Get recent pipelines for a project with pagination support
        """
        params = {
            "branch": branch,
            "page-token": page_token
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        url = f"{self.base_url}/project/{self.project_slug}/pipeline"
        data = await self.get_json(session, url, params)
        pipelines = data["items"]

        next_page_token = data.get("next_page_token")

        # If we have a limit, check if we need more pages
        if limit and len(pipelines) >= limit:
            return pipelines

        # If there are more pages and we haven't hit the limit, recursively get them
        if next_page_token:
            next_pipelines = await self.get_recent_pipelines(
                session,
                org_slug,
                page_token=next_page_token,
                limit=limit - len(pipelines) if limit else None,  # Adjust limit for next page
                branch=branch
            )
            pipelines.extend(next_pipelines)

        return pipelines

    async def get_workflow_jobs(self, session: aiohttp.ClientSession, pipeline_id: str) -> List[Dict]:
        self.logger.debug(f"Fetching workflows for pipeline {pipeline_id}")
        url = f"{self.base_url}/pipeline/{pipeline_id}/workflow"
        workflows_data = await self.get_json(session, url)
        workflows = workflows_data["items"]

        # Fetch all jobs for all workflows in parallel
        jobs_tasks = []
        for workflow in workflows:
            url = f"{self.base_url}/workflow/{workflow['id']}/job"
            jobs_tasks.append(self.get_json(session, url))

        jobs_responses = await asyncio.gather(*jobs_tasks, return_exceptions=True)

        all_jobs = []
        for jobs_data in jobs_responses:
            if isinstance(jobs_data, Exception):
                continue
            all_jobs.extend(jobs_data["items"])

        return all_jobs

    async def get_artifacts(self, session: aiohttp.ClientSession, job_number: str) -> List[Dict]:
        url = f"{self.base_url}/project/{self.project_slug}/{job_number}/artifacts"
        data = await self.get_json(session, url)
        return data["items"]

class PackageSizeTracker:
    def __init__(self, token: str, project_slug: str, debug: bool = False):
        self.setup_logging(debug)
        self.client = AsyncCircleCIClient(token, project_slug)
        self.logger = logging.getLogger("PackageSizeTracker")
        self.last_data_hash = None
        self.debug = debug

        # Initialize pygame mixer
        pygame.mixer.init()

        # Sound file paths - can use MP3 files with pygame
        sounds_dir = Path(__file__).parent / "sounds"
        self.sounds = {
            'lines_up': sounds_dir / "gta5_wasted.mp3",
            'lines_down': sounds_dir / "pokemon_evolve.mp3",
            'tokens_up': sounds_dir / "pokemon_evolve.mp3",
            'tokens_down': sounds_dir / "gta5_wasted.mp3",
            'size_up': sounds_dir / "gta5_wasted.mp3",
            'size_down': sounds_dir / "pokemon_evolve.mp3"
        }

    def test_sound_effects(self):
        """Test all sound effects with a small delay between each"""
        self.logger.info("Testing sound effects...")
        for sound_key in self.sounds:
            self.logger.info(f"Playing {sound_key}")
            self._play_sound(sound_key)
            time.sleep(1)  # Wait 1 second between sounds

    def setup_logging(self, debug: bool):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def extract_commit_info(self, pipeline: Dict) -> Optional[Dict]:
        try:
            # Extract from github_app first (preferred)
            if 'trigger_parameters' in pipeline and 'github_app' in pipeline['trigger_parameters']:
                github_app = pipeline['trigger_parameters']['github_app']
                return {
                    'commit_hash': github_app.get('checkout_sha'),
                    'web_url': f"{github_app.get('repo_url')}/commit/{github_app.get('checkout_sha')}",
                    'branch': github_app.get('branch', 'unknown'),
                    'author': {
                        'name': github_app.get('commit_author_name'),
                        'email': github_app.get('commit_author_email'),
                        'username': github_app.get('user_username')
                    },
                    'message': github_app.get('commit_message')
                }

            # Fallback to git parameters
            if 'trigger_parameters' in pipeline and 'git' in pipeline['trigger_parameters']:
                git = pipeline['trigger_parameters']['git']
                return {
                    'commit_hash': git.get('checkout_sha'),
                    'web_url': f"{git.get('repo_url')}/commit/{git.get('checkout_sha')}",
                    'branch': git.get('branch', 'unknown'),
                    'author': {
                        'name': git.get('commit_author_name'),
                        'email': git.get('commit_author_email'),
                        'username': git.get('author_login')
                    },
                    'message': git.get('commit_message')
                }

            self.logger.warning(f"Could not find commit info in pipeline {pipeline['id']}")
            return None

        except Exception as e:
            self.logger.error(f"Error extracting commit info: {str(e)}")
            return None

    async def process_pipeline(self, session: aiohttp.ClientSession, pipeline: Dict) -> Optional[Dict]:
        try:
            commit_info = self.extract_commit_info(pipeline)
            if not commit_info:
                return None

            data_point = {
                "commit_hash": commit_info['commit_hash'],
                "commit_url": commit_info['web_url'],
                "timestamp": pipeline.get("created_at", pipeline.get("updated_at")),
                "pipeline_status": pipeline.get("state", "unknown"),
                "branch": commit_info['branch'],
                "author": commit_info['author'],
                "commit_message": commit_info['message']
            }

            jobs = await self.client.get_workflow_jobs(session, pipeline["id"])

            # Get package size data
            size_job = next(
                (j for j in jobs if j["name"] == "measure_pip_sizes" and j["status"] == "success"),
                None
            )

            # Get line count data
            linecount_job = next(
                (j for j in jobs if j["name"] == "check_line_count" and j["status"] == "success"),
                None
            )

            # Get benchmark data from runner job
            benchmark_job = next(
                (j for j in jobs if j["name"] == "runner" and j["status"] == "success"),
                None
            )

            # Return None if no relevant jobs found
            if not size_job and not linecount_job and not benchmark_job:
                self.logger.debug(f"No relevant jobs found for pipeline {pipeline['id']}")
                return None

            # Process benchmark data if available
            if benchmark_job:
                benchmark_artifacts = await self.client.get_artifacts(session, benchmark_job["job_number"])
                benchmark_report = next(
                    (a for a in benchmark_artifacts if a["path"].endswith("benchmark.json")),
                    None
                )
                if benchmark_report:
                    benchmark_data = await self.client.get_json(session, benchmark_report["url"])
                    data_point.update({
                        "tokens_per_second": benchmark_data["tokens_per_second"],
                        "time_to_first_token": benchmark_data.get("time_to_first_token", 0)
                    })
                    self.logger.info(
                        f"Processed benchmark data for pipeline {pipeline['id']}: "
                        f"commit {commit_info['commit_hash'][:7]}, "
                        f"tokens/s {benchmark_data['tokens_per_second']:.2f}"
                    )

            # Process size data if available
            if size_job:
                size_artifacts = await self.client.get_artifacts(session, size_job["job_number"])
                size_report = next(
                    (a for a in size_artifacts if a["path"].endswith("pip-sizes.json")),
                    None
                )
                if size_report:
                    size_data = await self.client.get_json(session, size_report["url"])
                    data_point.update({
                        "total_size_mb": size_data["total_size_mb"],
                        "packages": size_data["packages"]
                    })
                    self.logger.info(
                        f"Processed size data for pipeline {pipeline['id']}: "
                        f"commit {commit_info['commit_hash'][:7]}, "
                        f"size {size_data['total_size_mb']:.2f}MB"
                    )

            # Process linecount data if available
            if linecount_job:
                linecount_artifacts = await self.client.get_artifacts(session, linecount_job["job_number"])
                linecount_report = next(
                    (a for a in linecount_artifacts if a["path"].endswith("line-count-snapshot.json")),
                    None
                )
                if linecount_report:
                    linecount_data = await self.client.get_json(session, linecount_report["url"])
                    data_point.update({
                        "total_lines": linecount_data["total_lines"],
                        "total_files": linecount_data["total_files"],
                        "files": linecount_data["files"]
                    })
                    self.logger.info(
                        f"Processed line count data for pipeline {pipeline['id']}: "
                        f"commit {commit_info['commit_hash'][:7]}, "
                        f"lines {linecount_data['total_lines']:,}"
                    )

            return data_point

        except Exception as e:
            self.logger.error(f"Error processing pipeline {pipeline['id']}: {str(e)}")
            return None

    async def process_pipeline_batch(
        self,
        session: aiohttp.ClientSession,
        pipelines: List[Dict],
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Process a batch of pipelines with rate limiting.

        Args:
            session: aiohttp client session
            pipelines: List of pipelines to process
            batch_size: Number of pipelines to process in parallel

        Returns:
            List of processed pipeline data points
        """
        data_points = []

        for i in range(0, len(pipelines), batch_size):
            batch = pipelines[i:i + batch_size]

            # Process batch in parallel
            tasks = [self.process_pipeline(session, pipeline) for pipeline in batch]
            batch_results = await asyncio.gather(*tasks)

            # Filter out None results
            batch_data = [r for r in batch_results if r is not None]
            data_points.extend(batch_data)

            # Add delay between batches if there are more to process
            if i + batch_size < len(pipelines):
                await asyncio.sleep(1)  # 1 second delay between batches

        return data_points

    async def collect_data(self) -> List[Dict]:
        self.logger.info("Starting data collection...")
        async with aiohttp.ClientSession(headers=self.client.headers) as session:
            # Get pipelines from main branch
            main_pipelines = await self.client.get_recent_pipelines(
                session,
                org_slug=self.client.project_slug,
                limit=20,
                branch="main"
            )

            # Add delay between branch requests
            await asyncio.sleep(2)

            # Get pipelines from circleci branch
            circleci_pipelines = await self.client.get_recent_pipelines(
                session,
                org_slug=self.client.project_slug,
                limit=20,
                branch="circleci"
            )

            # Combine pipelines and sort by created_at date
            pipelines = main_pipelines + circleci_pipelines
            pipelines.sort(
                key=lambda x: datetime.fromisoformat(
                    x.get("created_at", x.get("updated_at")).replace('Z', '+00:00')
                ),
                reverse=True  # Most recent first
            )

            self.logger.info(f"Found {len(pipelines)} recent pipelines")

            # Process pipelines in batches
            data_points = await self.process_pipeline_batch(session, pipelines)

            # Sort by timestamp
            data_points.sort(
                key=lambda x: datetime.fromisoformat(
                    x.get("timestamp").replace('Z', '+00:00')
                ),
                reverse=True  # Most recent first
            )

        return data_points

    def generate_report(self, data: List[Dict], output_dir: str = "reports") -> Optional[str]:
        self.logger.info("Generating report...")
        if not data:
            self.logger.error("No data to generate report from!")
            return None

        # Get latest pipeline status based on errors
        latest_main_pipeline = next((d for d in data if d.get('branch') == 'main'), None)
        latest_pipeline_status = 'success' if latest_main_pipeline and not latest_main_pipeline.get('errors') else 'failure'

        # Log the pipeline status
        if latest_main_pipeline:
            self.logger.info(
                f"Latest main branch pipeline status: {latest_pipeline_status} "
                f"(commit: {latest_main_pipeline['commit_hash'][:7]})"
            )
        else:
            self.logger.warning("No pipeline data found for main branch")

        # Convert output_dir to Path object
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create separate dataframes for each metric
        df_size = pd.DataFrame([d for d in data if 'total_size_mb' in d])
        df_lines = pd.DataFrame([d for d in data if 'total_lines' in d])
        df_benchmark = pd.DataFrame([d for d in data if 'tokens_per_second' in d])

        # Create a single figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('', 'Package Size', '', 'Line Count', '', 'Tokens per Second'),
            vertical_spacing=0.2,
            column_widths=[0.2, 0.8],
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [None, {"type": "scatter"}],
                   [None, {"type": "scatter"}]]
        )

        # Add package size trace if we have data
        if not df_size.empty:
            df_size['timestamp'] = pd.to_datetime(df_size['timestamp'])
            df_size = df_size.sort_values('timestamp')

            fig.add_trace(
                go.Scatter(
                    x=df_size['timestamp'],
                    y=df_size['total_size_mb'],
                    mode='lines+markers',
                    name='Package Size',
                    customdata=df_size[['commit_hash', 'commit_url']].values,
                    hovertemplate="<br>".join([
                        "Size: %{y:.2f}MB",
                        "Date: %{x}",
                        "Commit: %{customdata[0]}",
                        "<extra></extra>"
                    ])
                ),
                row=1, col=2
            )
            fig.update_yaxes(title_text="Size (MB)", row=1, col=2)

        # Add line count trace if we have data
        if not df_lines.empty:
            df_lines['timestamp'] = pd.to_datetime(df_lines['timestamp'])
            df_lines = df_lines.sort_values('timestamp')

            fig.add_trace(
                go.Scatter(
                    x=df_lines['timestamp'],
                    y=df_lines['total_lines'],
                    mode='lines+markers',
                    name='Line Count',
                    customdata=df_lines[['commit_hash', 'commit_url']].values,
                    hovertemplate="<br>".join([
                        "Lines: %{y:,.0f}",
                        "Date: %{x}",
                        "Commit: %{customdata[0]}",
                        "<extra></extra>"
                    ])
                ),
                row=2, col=2
            )
            fig.update_yaxes(title_text="Total Lines", row=2, col=2)

        # Add tokens per second trace if we have data
        if not df_benchmark.empty:
            df_benchmark['timestamp'] = pd.to_datetime(df_benchmark['timestamp'])
            df_benchmark = df_benchmark.sort_values('timestamp')

            fig.add_trace(
                go.Scatter(
                    x=df_benchmark['timestamp'],
                    y=df_benchmark['tokens_per_second'],
                    mode='lines+markers',
                    name='Tokens/Second',
                    customdata=df_benchmark[['commit_hash', 'commit_url']].values,
                    hovertemplate="<br>".join([
                        "Tokens/s: %{y:.2f}",
                        "Date: %{x}",
                        "Commit: %{customdata[0]}",
                        "<extra></extra>"
                    ])
                ),
                row=3, col=2
            )
            fig.update_yaxes(title_text="Tokens per Second", row=3, col=2)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Package Metrics Dashboard",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            hovermode='x unified'
        )

        # Update the dashboard HTML with date range picker
        dashboard_html = f"""
        <html>
        <head>
            <title>Package Metrics Dashboard</title>
            <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.css" />
            <style>
                body {{
                    background-color: #f5f6fa;
                    margin: 0;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                }}

                .date-picker-container {{
                    background: white;
                    padding: 15px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px auto;
                    width: fit-content;
                }}

                #daterange {{
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    font-size: 14px;
                    width: 300px;
                    cursor: pointer;
                }}

                .quick-ranges {{
                    margin-top: 10px;
                    display: flex;
                    gap: 8px;
                    justify-content: center;
                }}

                .quick-ranges button {{
                    padding: 8px 16px;
                    border: 1px solid #e1e4e8;
                    border-radius: 8px;
                    background: white;
                    cursor: pointer;
                    font-size: 13px;
                    transition: all 0.2s ease;
                }}

                .quick-ranges button:hover {{
                    background: #f0f0f0;
                    transform: translateY(-1px);
                }}

                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: 300px 1fr;
                    gap: 20px;
                    margin-top: 20px;
                }}

                .chart-container {{
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    height: 350px;
                }}

                .chart-row {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }}

                .chart-row-full {{
                    grid-column: 2 / -1;
                }}

                .chart-box {{
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                }}

                .chart-title {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #2c3e50;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}

                .status-container {{
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    height: 350px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }}

                .traffic-light {{
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    margin: 20px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.2);
                    position: relative;
                }}

                .traffic-light.success {{
                    background: #2ecc71;  /* Bright green */
                    border: 8px solid #27ae60;  /* Darker green border */
                }}

                .traffic-light.failure {{
                    background: #e74c3c;  /* Bright red */
                    border: 8px solid #c0392b;  /* Darker red border */
                }}

                .status-text {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-top: 20px;
                    color: #2c3e50;
                }}

                /* Override Plotly's default margins */
                .js-plotly-plot .plotly {{
                    margin: 0 !important;
                }}
            </style>
        </head>
        <body>
            <div class="date-picker-container">
                <input type="text" id="daterange" />
                <div class="quick-ranges">
                    <button onclick="setQuickRange('1h')">Last Hour</button>
                    <button onclick="setQuickRange('6h')">Last 6 Hours</button>
                    <button onclick="setQuickRange('1d')">Last 24 Hours</button>
                    <button onclick="setQuickRange('7d')">Last 7 Days</button>
                    <button onclick="setQuickRange('30d')">Last 30 Days</button>
                    <button onclick="setQuickRange('all')">All Time</button>
                </div>
            </div>

            <div class="dashboard-grid">
                <div class="status-container">
                    <div class="chart-title">Pipeline Status</div>
                    <div class="traffic-light {'success' if latest_pipeline_status == 'success' else 'failure'}"></div>
                    <div class="status-text">
                        {'✓ Pipeline Passing' if latest_pipeline_status == 'success' else '✗ Pipeline Failing'}
                    </div>
                </div>
                <div class="chart-row">
                    <div class="chart-box">
                        <div class="chart-title">Package Size</div>
                        <div id="size-chart"></div>
                    </div>
                    <div class="chart-box">
                        <div class="chart-title">Line Count</div>
                        <div id="lines-chart"></div>
                    </div>
                </div>
                <div class="chart-row chart-row-full">
                    <div class="chart-box">
                        <div class="chart-title">Tokens per Second</div>
                        <div id="tokens-chart"></div>
                    </div>
                </div>
            </div>

            <script type="text/javascript" src="https://cdn.jsdelivr.net/jquery/latest/jquery.min.js"></script>
            <script type="text/javascript" src="https://cdn.jsdelivr.net/momentjs/latest/moment.min.js"></script>
            <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                let globalMinDate = null;
                let globalMaxDate = null;

                // Split the original figure into separate charts
                const originalData = {fig.to_json()};

                function initializeCharts() {{
                    // Create the size trend chart
                    const sizeTrace = originalData.data.find(trace => trace.name === 'Package Size');
                    if (sizeTrace) {{
                        Plotly.newPlot('size-chart',
                            [sizeTrace],
                            {{
                                showlegend: false,
                                height: 280,
                                margin: {{ t: 10, b: 40, l: 50, r: 20 }},
                                yaxis: {{ title: 'Size (MB)' }},
                                xaxis: {{
                                    type: 'date',
                                    title: null,
                                    range: [sizeTrace.x[0], sizeTrace.x[sizeTrace.x.length - 1]]
                                }}
                            }}
                        );
                    }}

                    // Create the line count chart
                    const lineTrace = originalData.data.find(trace => trace.name === 'Line Count');
                    if (lineTrace) {{
                        Plotly.newPlot('lines-chart',
                            [lineTrace],
                            {{
                                showlegend: false,
                                height: 280,
                                margin: {{ t: 10, b: 40, l: 50, r: 20 }},
                                yaxis: {{ title: 'Total Lines' }},
                                xaxis: {{
                                    type: 'date',
                                    title: null,
                                    range: [lineTrace.x[0], lineTrace.x[lineTrace.x.length - 1]]
                                }}
                            }}
                        );
                    }}

                    // Create the tokens per second chart
                    const tokensTrace = originalData.data.find(trace => trace.name === 'Tokens/Second');
                    if (tokensTrace) {{
                        Plotly.newPlot('tokens-chart',
                            [tokensTrace],
                            {{
                                showlegend: false,
                                height: 280,
                                margin: {{ t: 10, b: 40, l: 50, r: 20 }},
                                yaxis: {{ title: 'Tokens/Second' }},
                                xaxis: {{
                                    type: 'date',
                                    title: null,
                                    range: [tokensTrace.x[0], tokensTrace.x[tokensTrace.x.length - 1]]
                                }}
                            }}
                        );
                    }}

                    // Add debug logs to check axis names
                    console.log('Size Chart Layout:', document.getElementById('size-chart').layout);
                    console.log('Lines Chart Layout:', document.getElementById('lines-chart').layout);
                    console.log('Tokens Chart Layout:', document.getElementById('tokens-chart').layout);
                }}

                function setQuickRange(range) {{
                    let start, end = moment();

                    switch(range) {{
                        case '1h':
                            start = moment().subtract(1, 'hours');
                            break;
                        case '6h':
                            start = moment().subtract(6, 'hours');
                            break;
                        case '1d':
                            start = moment().subtract(1, 'days');
                            break;
                        case '7d':
                            start = moment().subtract(7, 'days');
                            break;
                        case '30d':
                            start = moment().subtract(30, 'days');
                            break;
                        case 'all':
                            start = moment(globalMinDate);
                            end = moment(globalMaxDate);
                            break;
                    }}

                    $('#daterange').data('daterangepicker').setStartDate(start);
                    $('#daterange').data('daterangepicker').setEndDate(end);
                    updatePlotRange(start.toISOString(), end.toISOString());
                }}

                function updatePlotRange(startDate, endDate) {{
                    console.log('Updating range:', startDate, endDate);

                    // Get the actual x-axis names from the chart layouts
                    const sizeChartLayout = document.getElementById('size-chart').layout;
                    const sizeXAxisName = Object.keys(sizeChartLayout).find(key => key.startsWith('xaxis'));

                    const linesChartLayout = document.getElementById('lines-chart').layout;
                    const linesXAxisName = Object.keys(linesChartLayout).find(key => key.startsWith('xaxis'));

                    const tokensChartLayout = document.getElementById('tokens-chart').layout;
                    const tokensXAxisName = Object.keys(tokensChartLayout).find(key => key.startsWith('xaxis'));

                    // Update the ranges
                    const sizeUpdateLayout = {{}};
                    sizeUpdateLayout[`{{sizeXAxisName}}.range`] = [startDate, endDate];

                    const linesUpdateLayout = {{}};
                    linesUpdateLayout[`{{linesXAxisName}}.range`] = [startDate, endDate];

                    const tokensUpdateLayout = {{}};
                    tokensUpdateLayout[`{{tokensXAxisName}}.range`] = [startDate, endDate];

                    // Update both charts
                    Plotly.relayout('size-chart', sizeUpdateLayout)
                        .catch(err => console.error('Error updating size chart:', err));

                    Plotly.relayout('lines-chart', linesUpdateLayout)
                        .catch(err => console.error('Error updating lines chart:', err));

                    Plotly.relayout('tokens-chart', tokensUpdateLayout)
                        .catch(err => console.error('Error updating tokens chart:', err));
                }}

                function findDateRange(data) {{
                    let minDate = null;
                    let maxDate = null;

                    data.forEach(trace => {{
                        if (trace.x && trace.x.length > 0) {{
                            const dates = trace.x.map(d => new Date(d));
                            const traceMin = new Date(Math.min(...dates));
                            const traceMax = new Date(Math.max(...dates));

                            if (!minDate || traceMin < minDate) minDate = traceMin;
                            if (!maxDate || traceMax > maxDate) maxDate = traceMax;
                        }}
                    }});

                    return {{ minDate, maxDate }};
                }}

                // Initialize everything when document is ready
                $(document).ready(function() {{
                    // Initialize charts
                    initializeCharts();

                    // Find date range from data
                    const {{ minDate, maxDate }} = findDateRange(originalData.data);
                    globalMinDate = minDate;
                    globalMaxDate = maxDate;

                    // Initialize daterangepicker
                    $('#daterange').daterangepicker({{
                        startDate: minDate,
                        endDate: maxDate,
                        minDate: minDate,
                        maxDate: maxDate,
                        timePicker: true,
                        timePicker24Hour: true,
                        timePickerIncrement: 1,
                        opens: 'center',
                        locale: {{
                            format: 'YYYY-MM-DD HH:mm',
                            applyLabel: "Apply",
                            cancelLabel: "Cancel",
                            customRangeLabel: "Custom Range"
                        }},
                        ranges: {{
                            'Last Hour': [moment().subtract(1, 'hours'), moment()],
                            'Last 6 Hours': [moment().subtract(6, 'hours'), moment()],
                            'Last 24 Hours': [moment().subtract(1, 'days'), moment()],
                            'Last 7 Days': [moment().subtract(7, 'days'), moment()],
                            'Last 30 Days': [moment().subtract(30, 'days'), moment()],
                            'All Time': [moment(minDate), moment(maxDate)]
                        }}
                    }});

                    // Update plots when date range changes
                    $('#daterange').on('apply.daterangepicker', function(ev, picker) {{
                        console.log('Date range changed:', picker.startDate.toISOString(), picker.endDate.toISOString());
                        updatePlotRange(picker.startDate.toISOString(), picker.endDate.toISOString());
                    }});

                    // Add click handlers for charts
                    ['size-chart', 'lines-chart', 'tokens-chart'].forEach(chartId => {{
                        const chart = document.getElementById(chartId);
                        if (chart) {{
                            chart.on('plotly_click', function(data) {{
                                const point = data.points[0];
                                if (point.customdata && point.customdata[1]) {{
                                    window.open(point.customdata[1], '_blank');
                                }}
                            }});
                        }}
                    }});

                    // Add debug logging for chart initialization
                    console.log('Size Chart:', document.getElementById('size-chart'));
                    console.log('Lines Chart:', document.getElementById('lines-chart'));
                    console.log('Tokens Chart:', document.getElementById('tokens-chart'));
                }});
            </script>
        </body>
        </html>
        """

        # Write the dashboard
        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)

        # Generate summary with available metrics
        latest_data = {}

        if not df_size.empty:
            latest = df_size.iloc[-1]
            previous = df_size.iloc[-2] if len(df_size) > 1 else latest
            size_change = float(latest['total_size_mb'] - previous['total_size_mb'])
            latest_data.update({
                'timestamp': latest['timestamp'].isoformat(),
                'commit_hash': latest['commit_hash'],
                'commit_url': latest['commit_url'],
                'total_size_mb': float(latest['total_size_mb']),
                'size_change_mb': size_change,
                'packages': latest.get('packages', [])
            })

        if not df_lines.empty:
            latest = df_lines.iloc[-1]
            previous = df_lines.iloc[-2] if len(df_lines) > 1 else latest
            linecount_change = int(latest['total_lines'] - previous['total_lines'])
            if not latest_data:  # Only add timestamp and commit info if not already added
                latest_data.update({
                    'timestamp': latest['timestamp'].isoformat(),
                    'commit_hash': latest['commit_hash'],
                    'commit_url': latest['commit_url'],
                })
            latest_data.update({
                'total_lines': int(latest['total_lines']),
                'linecount_change': linecount_change
            })

        if not df_benchmark.empty:
            latest = df_benchmark.iloc[-1]
            previous = df_benchmark.iloc[-2] if len(df_benchmark) > 1 else latest
            tokens_change = float(latest['tokens_per_second'] - previous['tokens_per_second'])
            if not latest_data:  # Only add timestamp and commit info if not already added
                latest_data.update({
                    'timestamp': latest['timestamp'].isoformat(),
                    'commit_hash': latest['commit_hash'],
                    'commit_url': latest['commit_url'],
                })
            latest_data.update({
                'tokens_per_second': float(latest['tokens_per_second']),
                'tokens_change': tokens_change
            })

        if latest_data:
            with open(output_dir / 'latest_data.json', 'w') as f:
                json.dump(latest_data, f, indent=2)

            self._print_summary(latest_data)
            self.logger.info(f"Report generated in {output_dir}")
            return str(output_dir)

        return None

    def _print_summary(self, latest_data: Dict):
        print("\n=== Package Size Summary ===")
        print(f"Timestamp: {latest_data['timestamp']}")
        print(f"Commit: {latest_data['commit_hash'][:7]}")

        if 'total_size_mb' in latest_data:
            print(f"Total Size: {latest_data['total_size_mb']:.2f}MB")
            change = latest_data['size_change_mb']
            change_symbol = "↓" if change <= 0 else "↑"
            print(f"Change: {change_symbol} {abs(change):.2f}MB")

            if latest_data.get('packages'):
                print("\nTop 5 Largest Packages:")
                sorted_packages = sorted(latest_data['packages'], key=lambda x: x['size_mb'], reverse=True)
                for pkg in sorted_packages[:5]:
                    print(f"- {pkg['name']}: {pkg['size_mb']:.2f}MB")

        if 'total_lines' in latest_data:
            print("\nLine Count Stats:")
            print(f"Total Lines: {latest_data['total_lines']:,}")
            change = latest_data['linecount_change']
            change_symbol = "↓" if change <= 0 else "↑"
            print(f"Change: {change_symbol} {abs(change):,}")

        if 'tokens_per_second' in latest_data:
            print("\nBenchmark Stats:")
            print(f"Tokens per Second: {latest_data['tokens_per_second']:.2f}")
            if 'time_to_first_token' in latest_data:
                print(f"Time to First Token: {latest_data['time_to_first_token']:.3f}s")

        print("\n")

    def _calculate_data_hash(self, data: List[Dict]) -> str:
        """Calculate a hash of the data to detect changes"""
        return hash(str(sorted([
            (d.get('commit_hash'), d.get('timestamp'))
            for d in data
        ])))

    def _play_sound(self, sound_key: str):
        """Play a specific notification sound using pygame"""
        try:
            sound_path = self.sounds.get(sound_key)
            if sound_path and sound_path.exists():
                sound = pygame.mixer.Sound(str(sound_path))
                sound.play()
                # Wait for the sound to finish playing
                pygame.time.wait(int(sound.get_length() * 1000))
            else:
                self.logger.warning(f"Sound file not found: {sound_key} at {sound_path}")
        except Exception as e:
            self.logger.error(f"Failed to play sound {sound_key}: {e}")

    def _check_metrics_changes(self, current_data: List[Dict], previous_data: List[Dict]):
        # Sort data by timestamp in descending order (most recent first)
        def sort_by_timestamp(data):
            return sorted(
                data,
                key=lambda x: x.get('timestamp', ''),
                reverse=True  # Most recent first
            )

        current_data = sort_by_timestamp(current_data)
        previous_data = sort_by_timestamp(previous_data)

        # Helper to find latest entry with a specific metric
        def find_latest_with_metric(data: List[Dict], metric: str) -> Optional[Dict]:
            return next((d for d in data if metric in d), None)

        # Check line count changes
        current_lines = find_latest_with_metric(current_data, 'total_lines')
        previous_lines = find_latest_with_metric(previous_data, 'total_lines')

        if current_lines and previous_lines:
            diff = current_lines['total_lines'] - previous_lines['total_lines']
            self.logger.debug(f"Lines of code diff: {diff}")
            if diff > 0:
                self.logger.info(f"Lines of code increased by {diff:,}")
                self._play_sound('lines_up')
            elif diff < 0:
                self.logger.info(f"Lines of code decreased by {abs(diff):,}")
                self._play_sound('lines_down')
        else:
            self.logger.debug("No lines of code data found")

        # Check tokens per second changes
        current_tokens = find_latest_with_metric(current_data, 'tokens_per_second')
        previous_tokens = find_latest_with_metric(previous_data, 'tokens_per_second')

        if current_tokens and previous_tokens:
            diff = current_tokens['tokens_per_second'] - previous_tokens['tokens_per_second']
            self.logger.debug(f"Tokens per second diff: {diff}")
            if diff > 0:
                self.logger.info(f"Tokens per second increased by {diff:.2f}")
                self._play_sound('tokens_up')
            elif diff < 0:
                self.logger.info(f"Tokens per second decreased by {abs(diff):.2f}")
                self._play_sound('tokens_down')
        else:
            self.logger.debug("No tokens per second data found")

        # Check package size changes
        current_size = find_latest_with_metric(current_data, 'total_size_mb')
        previous_size = find_latest_with_metric(previous_data, 'total_size_mb')

        if current_size and previous_size:
            diff = current_size['total_size_mb'] - previous_size['total_size_mb']
            self.logger.debug(f"Package size diff: {diff:.2f}MB")
            if diff > 0:
                self.logger.info(f"Package size increased by {diff:.2f}MB")
                self._play_sound('size_up')
            elif diff < 0:
                self.logger.info(f"Package size decreased by {abs(diff):.2f}MB")
                self._play_sound('size_down')
        else:
            self.logger.debug("No package size data found")

    async def run_dashboard(self, update_interval: int = 10):
        """Run the dashboard with periodic updates"""
        try:
            update_interval = float(update_interval)
            self.logger.debug(f"Update interval type: {type(update_interval)}, value: {update_interval}")
        except ValueError as e:
            self.logger.error(f"Failed to convert update_interval to float: {update_interval}")
            raise

        self.logger.info(f"Starting real-time dashboard with {update_interval}s updates")
        previous_data = None

        while True:
            try:
                start_time = time.time()

                # Collect new data
                current_data = await self.collect_data()
                if not current_data:
                    self.logger.warning("No data collected")
                    await asyncio.sleep(update_interval)
                    continue

                # Generate report
                report_path = self.generate_report(current_data)
                if report_path:
                    self.logger.info(
                        f"Dashboard updated at {datetime.now().strftime('%H:%M:%S')}"
                    )

                    print("Curr:", len(current_data))
                    print("Prev:", len(previous_data) if previous_data else "None")
                    if previous_data:
                        # Check for metric changes and play appropriate sounds
                        self.logger.debug(f"Checking metrics changes between {len(current_data)} current and {len(previous_data)} previous data points")
                        self._check_metrics_changes(current_data, previous_data)

                # Update previous data
                previous_data = current_data.copy()  # Make a copy to prevent reference issues

                # Calculate sleep time
                elapsed = float(time.time() - start_time)
                sleep_time = max(0.0, update_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}", exc_info=True)
                if self.debug:
                    raise
                await asyncio.sleep(update_interval)

async def main():
    token = os.getenv("CIRCLECI_TOKEN")
    project_slug = os.getenv("CIRCLECI_PROJECT_SLUG")
    debug = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")

    try:
        # Get update interval from environment or use default
        update_interval = float(os.getenv("UPDATE_INTERVAL", "10"))
        print(f"Update interval type: {type(update_interval)}, value: {update_interval}")  # Debug print
    except ValueError as e:
        print(f"Error converting UPDATE_INTERVAL to float: {os.getenv('UPDATE_INTERVAL')}")
        update_interval = 10.0

    if not token or not project_slug:
        print("Error: Please set CIRCLECI_TOKEN and CIRCLECI_PROJECT_SLUG environment variables")
        return

    tracker = PackageSizeTracker(token, project_slug, debug)

    try:
        await tracker.run_dashboard(update_interval)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        if debug:
            raise

if __name__ == "__main__":
    asyncio.run(main())
