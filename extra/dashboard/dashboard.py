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

        Args:
            session: aiohttp client session
            org_slug: Organization slug
            page_token: Token for pagination
            limit: Maximum number of pipelines to return
            branch: Specific branch to fetch pipelines from
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

        # If there are more pages and we haven't hit the limit, recursively get them
        if next_page_token and (limit is None or len(pipelines) < limit):
            next_pipelines = await self.get_recent_pipelines(
                session,
                org_slug,
                page_token=next_page_token,
                limit=limit,
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

    def setup_logging(self, debug: bool):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def extract_commit_info(self, pipeline: Dict) -> Optional[Dict]:
        try:
            if 'trigger_parameters' in pipeline:
                github_app = pipeline['trigger_parameters'].get('github_app', {})
                if github_app:
                    return {
                        'commit_hash': github_app.get('checkout_sha'),
                        'web_url': f"{github_app.get('repo_url')}/commit/{github_app.get('checkout_sha')}"
                    }

                git_params = pipeline['trigger_parameters'].get('git', {})
                if git_params:
                    return {
                        'commit_hash': git_params.get('checkout_sha'),
                        'web_url': f"{git_params.get('repo_url')}/commit/{git_params.get('checkout_sha')}"
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

            jobs = await self.client.get_workflow_jobs(session, pipeline["id"])

            # Add test status check
            test_job = next(
                (j for j in jobs if j["name"] == "test" and j["status"] in ["success", "failed"]),
                None
            )

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

            data_point = {
                "commit_hash": commit_info['commit_hash'],
                "commit_url": commit_info['web_url'],
                "timestamp": pipeline.get("created_at", pipeline.get("updated_at")),
                "tests_passing": test_job["status"] == "success" if test_job else None
            }

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

    async def collect_data(self) -> List[Dict]:
        self.logger.info("Starting data collection...")
        async with aiohttp.ClientSession(headers=self.client.headers, trust_env=True) as session:
            # Get pipelines from both main and circleci branches
            main_pipelines = await self.client.get_recent_pipelines(
                session,
                org_slug=self.client.project_slug,
                limit=20,
                branch="main"
            )
            circleci_pipelines = await self.client.get_recent_pipelines(
                session,
                org_slug=self.client.project_slug,
                limit=20,
                branch="circleci"
            )

            pipelines = main_pipelines + circleci_pipelines
            # Sort pipelines by created_at date
            pipelines.sort(key=lambda x: x.get("created_at", x.get("updated_at")), reverse=True)

            self.logger.info(f"Found {len(pipelines)} recent pipelines")

            # Process all pipelines in parallel
            tasks = [self.process_pipeline(session, pipeline) for pipeline in pipelines]
            results = await asyncio.gather(*tasks)

            # Filter out None results
            data_points = [r for r in results if r is not None]

        return data_points

    def generate_report(self, data: List[Dict], output_dir: str = "reports") -> Optional[str]:
        self.logger.info("Generating report...")
        if not data:
            self.logger.error("No data to generate report from!")
            return None

        # Create separate dataframes for each metric
        df_size = pd.DataFrame([d for d in data if 'total_size_mb' in d])
        df_lines = pd.DataFrame([d for d in data if 'total_lines' in d])
        df_benchmark = pd.DataFrame([d for d in data if 'tokens_per_second' in d])

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a single figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Test Status', 'Package Size', '', 'Line Count', '', 'Tokens per Second'),
            vertical_spacing=0.2,
            column_widths=[0.2, 0.8],
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [None, {"type": "scatter"}],
                   [None, {"type": "scatter"}]]
        )

        # Add test status indicator if we have data
        latest_test_status = next((d["tests_passing"] for d in reversed(data) if "tests_passing" in d), None)
        if latest_test_status is not None:
            fig.add_trace(
                go.Indicator(
                    mode="gauge",
                    gauge={
                        "shape": "bullet",
                        "axis": {"visible": False},
                        "bar": {"color": "green" if latest_test_status else "red"},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [0, 1], "color": "lightgray"}
                        ]
                    },
                    value=1,
                    title={"text": "Tests<br>Status"}
                ),
                row=1, col=1
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
                    justify-content: center;
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
                    <div class="chart-title">Test Status</div>
                    <div id="status-chart"></div>
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
                    // Create the status indicator
                    if (originalData.data[0].type === 'indicator') {{
                        Plotly.newPlot('status-chart',
                            [originalData.data[0]],
                            {{
                                ...originalData.layout,
                                margin: {{ t: 0, b: 0, l: 0, r: 0 }},
                                height: 280
                            }}
                        );
                    }}

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
                    sizeUpdateLayout[`${{sizeXAxisName}}.range`] = [startDate, endDate];

                    const linesUpdateLayout = {{}};
                    linesUpdateLayout[`${{linesXAxisName}}.range`] = [startDate, endDate];

                    const tokensUpdateLayout = {{}};
                    tokensUpdateLayout[`${{tokensXAxisName}}.range`] = [startDate, endDate];

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

async def main():
    token = os.getenv("CIRCLECI_TOKEN")
    project_slug = os.getenv("CIRCLECI_PROJECT_SLUG")
    debug = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")

    if not token or not project_slug:
        print("Error: Please set CIRCLECI_TOKEN and CIRCLECI_PROJECT_SLUG environment variables")
        return

    tracker = PackageSizeTracker(token, project_slug, debug)

    try:
        data = await tracker.collect_data()
        if not data:
            print("No data found!")
            return

        report_path = tracker.generate_report(data)
        if report_path:
            print(f"\nDetailed report available at: {report_path}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        if debug:
            raise

if __name__ == "__main__":
    asyncio.run(main())
