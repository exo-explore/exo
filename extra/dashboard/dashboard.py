import os
import json
import logging
import asyncio
import aiohttp
import pandas as pd
import plotly.express as px
from typing import List, Dict, Optional
from pathlib import Path

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

    async def get_recent_pipelines(self, session: aiohttp.ClientSession, limit: int = 50) -> List[Dict]:
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
            size_job = next(
                (j for j in jobs if j["name"] == "measure_pip_sizes" and j["status"] == "success"),
                None
            )

            if not size_job:
                self.logger.debug(f"No measure_pip_sizes job found for pipeline {pipeline['id']}")
                return None

            artifacts = await self.client.get_artifacts(session, size_job["job_number"])
            size_report = next(
                (a for a in artifacts if a["path"].endswith("pip-sizes.json")),
                None
            )

            if not size_report:
                self.logger.debug(f"No pip-sizes.json artifact found for job {size_job['job_number']}")
                return None

            json_data = await self.client.get_json(session, size_report["url"])
            data_point = {
                "commit_hash": commit_info['commit_hash'],
                "commit_url": commit_info['web_url'],
                "timestamp": pipeline.get("created_at", pipeline.get("updated_at")),
                "total_size_mb": json_data["total_size_mb"],
                "packages": json_data["packages"]
            }

            self.logger.info(
                f"Processed pipeline {pipeline['id']}: "
                f"commit {commit_info['commit_hash'][:7]}, "
                f"size {json_data['total_size_mb']:.2f}MB"
            )
            return data_point

        except Exception as e:
            self.logger.error(f"Error processing pipeline {pipeline['id']}: {str(e)}")
            return None

    async def collect_data(self) -> List[Dict]:
        self.logger.info("Starting data collection...")
        async with aiohttp.ClientSession(headers=self.client.headers) as session:
            # Get pipelines
            pipelines = await self.client.get_recent_pipelines(session, 50)

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

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        # commit_url is already in the data from process_pipeline

        # Create trend plot with updated styling
        fig = px.line(
            df,
            x='timestamp',
            y='total_size_mb',
            title='Package Size Trend',
            markers=True,
            hover_data={'commit_hash': True, 'timestamp': True, 'total_size_mb': ':.2f'},
            custom_data=['commit_hash', 'commit_url']
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Size (MB)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            title_x=0.5,
        )
        fig.update_traces(
            line=dict(width=2),
            marker=dict(size=8),
            hovertemplate="<br>".join([
                "Commit: %{customdata[0]}",
                "Size: %{y:.2f}MB",
                "Date: %{x}",
                "<extra>Click to view commit</extra>"
            ])
        )

        # Add JavaScript for click handling
        fig.update_layout(
            clickmode='event',
            annotations=[
                dict(
                    text="Click any point to view the commit on GitHub",
                    xref="paper", yref="paper",
                    x=0, y=1.05,
                    showarrow=False
                )
            ]
        )

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        plot_path = output_dir / "package_size_trend.html"
        fig.write_html(
            str(plot_path),
            include_plotlyjs=True,
            full_html=True,
            post_script="""
            const plot = document.getElementsByClassName('plotly-graph-div')[0];
            plot.on('plotly_click', function(data) {
                const point = data.points[0];
                const commitUrl = point.customdata[1];
                window.open(commitUrl, '_blank');
            });
            """
        )

        # Generate summary
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        size_change = latest['total_size_mb'] - previous['total_size_mb']

        latest_data = {
            'timestamp': latest['timestamp'].isoformat(),
            'commit_hash': latest['commit_hash'],
            'total_size_mb': latest['total_size_mb'],
            'size_change_mb': size_change,
            'packages': latest['packages']
        }

        with open(output_dir / 'latest_data.json', 'w') as f:
            json.dump(latest_data, f, indent=2)

        self._print_summary(latest_data)
        self.logger.info(f"Report generated in {output_dir}")
        return str(plot_path)

    def _print_summary(self, latest_data: Dict):
        print("\n=== Package Size Summary ===")
        print(f"Timestamp: {latest_data['timestamp']}")
        print(f"Commit: {latest_data['commit_hash'][:7]}")
        print(f"Total Size: {latest_data['total_size_mb']:.2f}MB")

        change = latest_data['size_change_mb']
        change_symbol = "↓" if change <= 0 else "↑"
        print(f"Change: {change_symbol} {abs(change):.2f}MB")

        print("\nTop 5 Largest Packages:")
        sorted_packages = sorted(latest_data['packages'], key=lambda x: x['size_mb'], reverse=True)
        for pkg in sorted_packages[:5]:
            print(f"- {pkg['name']}: {pkg['size_mb']:.2f}MB")
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
