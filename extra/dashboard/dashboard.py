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

    async def get_recent_pipelines(self, session: aiohttp.ClientSession, limit: int = 100) -> List[Dict]:
        self.logger.info(f"Fetching {limit} recent pipelines...")
        url = f"{self.base_url}/project/{self.project_slug}/pipeline"
        params = {"limit": limit * 2}

        data = await self.get_json(session, url, params)
        pipelines = [
            p for p in data["items"]
            if p["state"] == "created"
            and p.get("trigger_parameters", {}).get("git", {}).get("branch") == "main"
        ][:limit]

        self.logger.info(f"Found {len(pipelines)} successful main branch pipelines")
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

            # Return None if no relevant jobs found
            if not size_job and not linecount_job:
                self.logger.debug(f"No relevant jobs found for pipeline {pipeline['id']}")
                return None

            data_point = {
                "commit_hash": commit_info['commit_hash'],
                "commit_url": commit_info['web_url'],
                "timestamp": pipeline.get("created_at", pipeline.get("updated_at")),
            }

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
        async with aiohttp.ClientSession(headers=self.client.headers) as session:
            # Get pipelines
            pipelines = await self.client.get_recent_pipelines(session, 100)

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

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a single figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Package Size Trend', 'Line Count Trend'),
            vertical_spacing=0.2
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
                row=1, col=1
            )
            fig.update_yaxes(title_text="Size (MB)", row=1, col=1)

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
                row=2, col=1
            )
            fig.update_yaxes(title_text="Total Lines", row=2, col=1)

        # Update layout
        fig.update_layout(
            height=800,  # Taller to accommodate both plots
            showlegend=False,
            title_text="Package Metrics Dashboard",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            hovermode='x unified',
            xaxis=dict(title_text="Date"),
            xaxis2=dict(title_text="Date")
        )

        # Add click event handling
        dashboard_html = f"""
        <html>
        <head>
            <title>Package Metrics Dashboard</title>
        </head>
        <body>
            <div id="dashboard">
                {fig.to_html(include_plotlyjs=True, full_html=False)}
            </div>
            <script>
                const plot = document.getElementById('dashboard').getElementsByClassName('plotly-graph-div')[0];
                plot.on('plotly_click', function(data) {{
                    const point = data.points[0];
                    const commitUrl = point.customdata[1];
                    window.open(commitUrl, '_blank');
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
