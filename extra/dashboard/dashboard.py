import os
import json
import logging
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class CircleCIClient:
    def __init__(self, token: str, project_slug: str):
        self.token = token
        self.project_slug = project_slug
        self.base_url = "https://circleci.com/api/v2"
        self.headers = {
            "Circle-Token": token,
            "Accept": "application/json"
        }
        self.logger = logging.getLogger("CircleCI")

    def get_recent_pipelines(self, limit: int = 25) -> List[Dict]:
        self.logger.info(f"Fetching {limit} recent pipelines...")
        url = f"{self.base_url}/project/{self.project_slug}/pipeline"
        params = {"limit": limit * 2}  # Fetch extra to account for failed builds

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        pipelines = [p for p in response.json()["items"] if p["state"] == "created"]
        pipelines = pipelines[:limit]
        self.logger.info(f"Found {len(pipelines)} successful pipelines")

        # Fetch additional data for each pipeline
        detailed_pipelines = []
        for pipeline in pipelines:
            try:
                url = f"{self.base_url}/pipeline/{pipeline['id']}"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                detailed_pipelines.append(response.json())
            except Exception as e:
                self.logger.warning(f"Could not fetch details for pipeline {pipeline['id']}: {e}")
                continue

        return detailed_pipelines

    def get_workflow_jobs(self, pipeline_id: str) -> List[Dict]:
        self.logger.debug(f"Fetching workflows for pipeline {pipeline_id}")
        url = f"{self.base_url}/pipeline/{pipeline_id}/workflow"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        workflows = response.json()["items"]

        jobs = []
        for workflow in workflows:
            self.logger.debug(f"Fetching jobs for workflow {workflow['id']}")
            url = f"{self.base_url}/workflow/{workflow['id']}/job"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            jobs.extend(response.json()["items"])

        return jobs

    def get_artifacts(self, job_number: str) -> List[Dict]:
        self.logger.debug(f"Fetching artifacts for job {job_number}")
        url = f"{self.base_url}/project/{self.project_slug}/{job_number}/artifacts"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()["items"]

    def download_artifact(self, artifact_url: str) -> Dict:
        self.logger.debug(f"Downloading artifact from {artifact_url}")
        response = requests.get(artifact_url, headers=self.headers)
        response.raise_for_status()
        return response.json()

class PackageSizeTracker:
    def __init__(self, token: str, project_slug: str, debug: bool = False):
        self.setup_logging(debug)
        self.client = CircleCIClient(token, project_slug)
        self.logger = logging.getLogger("PackageSizeTracker")

    def setup_logging(self, debug: bool):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def extract_commit_info(self, pipeline: Dict) -> Optional[str]:
        """Extract commit hash from pipeline data structure"""
        try:
            # Try to get commit hash from trigger parameters
            if 'trigger_parameters' in pipeline:
                github_app = pipeline['trigger_parameters'].get('github_app', {})
                if github_app:
                    return github_app.get('commit_sha')

                # Fallback to git parameters if github_app is not available
                git_params = pipeline['trigger_parameters'].get('git', {})
                if git_params:
                    return git_params.get('checkout_sha')

            self.logger.warning(f"Could not find commit hash in pipeline {pipeline['id']}")
            return None

        except Exception as e:
            self.logger.error(f"Error extracting commit info: {str(e)}")
            return None

    def collect_data(self) -> List[Dict]:
        self.logger.info("Starting data collection...")
        pipelines = self.client.get_recent_pipelines(25)

        data_points = []
        for pipeline in pipelines:
            try:
                self.logger.debug(f"Processing pipeline {pipeline['id']}")

                # Extract commit hash
                commit_hash = self.extract_commit_info(pipeline)
                if not commit_hash:
                    continue

                jobs = self.client.get_workflow_jobs(pipeline["id"])

                size_job = next(
                    (j for j in jobs if j["name"] == "measure_pip_sizes" and j["status"] == "success"),
                    None
                )

                if size_job:
                    artifacts = self.client.get_artifacts(size_job["job_number"])
                    size_report = next(
                        (a for a in artifacts if a["path"].endswith("pip-sizes.json")),
                        None
                    )

                    if size_report:
                        json_data = self.client.download_artifact(size_report["url"])
                        data_point = {
                            "commit_hash": commit_hash,
                            "timestamp": pipeline.get("created_at", pipeline.get("updated_at")),
                            "total_size_mb": json_data["total_size_mb"],
                            "packages": json_data["packages"]
                        }
                        data_points.append(data_point)
                        self.logger.info(
                            f"Processed pipeline {pipeline['id']}: "
                            f"commit {commit_hash[:7]}, "
                            f"size {json_data['total_size_mb']:.2f}MB"
                        )
                    else:
                        self.logger.debug(f"No pip-sizes.json artifact found for job {size_job['job_number']}")
                else:
                    self.logger.debug(f"No measure_pip_sizes job found for pipeline {pipeline['id']}")
            except Exception as e:
                self.logger.error(f"Error processing pipeline {pipeline['id']}: {str(e)}")
                continue

        return data_points

    def generate_report(self, data: List[Dict], output_dir: str = "reports"):
        self.logger.info("Generating report...")
        if not data:
            self.logger.error("No data to generate report from!")
            return None

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Create trend plot
        fig = px.line(
            df,
            x='timestamp',
            y='total_size_mb',
            title='Package Size Trend'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Size (MB)",
            hovermode='x unified'
        )

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        plot_path = output_dir / "package_size_trend.html"
        fig.write_html(str(plot_path))

        # Generate summary
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        size_change = latest['total_size_mb'] - previous['total_size_mb']

        # Save latest data
        latest_data = {
            'timestamp': latest['timestamp'].isoformat(),
            'commit_hash': latest['commit_hash'],
            'total_size_mb': latest['total_size_mb'],
            'size_change_mb': size_change,
            'packages': latest['packages']
        }

        with open(output_dir / 'latest_data.json', 'w') as f:
            json.dump(latest_data, f, indent=2)

        # Print summary to console
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

def main():
    # Get configuration
    token = os.getenv("CIRCLECI_TOKEN")
    project_slug = os.getenv("CIRCLECI_PROJECT_SLUG")
    debug = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")

    if not token or not project_slug:
        print("Error: Please set CIRCLECI_TOKEN and CIRCLECI_PROJECT_SLUG environment variables")
        return

    # Initialize tracker
    tracker = PackageSizeTracker(token, project_slug, debug)

    try:
        # Collect data
        data = tracker.collect_data()
        if not data:
            print("No data found!")
            return

        # Generate report
        report_path = tracker.generate_report(data)
        if report_path:
            print(f"\nDetailed report available at: {report_path}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        if debug:
            raise

if __name__ == "__main__":
    main()
