#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone script to capture metrics from trtllm-serve /metrics endpoint.
This script runs as a background process during harness execution.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests


class MetricsCapture:
    """Captures metrics from trtllm-serve /metrics endpoint.

    Polls endpoints indefinitely, waiting for each response to complete
    before polling again at the specified interval.
    """

    def __init__(self,
                 endpoint_urls: List[str],
                 output_dir: Path,
                 poll_interval: float = 1.0):
        """
        Initialize metrics capture.

        Args:
            endpoint_urls: List of trtllm-serve endpoint URLs (e.g., ["localhost:8000"])
            output_dir: Directory to save metrics logs
            poll_interval: Time between polls in seconds
        """
        self.endpoint_urls = endpoint_urls
        self.output_dir = Path(output_dir)
        self.poll_interval = poll_interval
        self.running = True

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Setup HTTP session
        self.session = requests.Session()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self):
        """Configure logging to file."""
        log_file = self.output_dir / "metrics_capture.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _poll_metrics(self, endpoint_url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Poll metrics from a single endpoint. Waits indefinitely for response.

        Args:
            endpoint_url: The endpoint URL (e.g., "localhost:8000")

        Returns:
            Metrics data or None if request failed
        """
        try:
            url = f"http://{endpoint_url}/metrics"
            # No timeout - wait indefinitely for the server to return all data
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch metrics from {endpoint_url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON response from {endpoint_url}: {e}")
            return None

    def run(self):
        """Main loop to continuously poll metrics."""
        logging.info(f"Starting metrics capture for endpoints: {self.endpoint_urls}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Poll interval: {self.poll_interval}s")

        # Open metric files for each endpoint
        metric_files = {}
        for endpoint_url in self.endpoint_urls:
            # Create safe filename from endpoint URL
            safe_name = endpoint_url.replace(":", "_").replace(".", "_")
            metric_file = self.output_dir / f"metrics_{safe_name}.jsonl"
            metric_files[endpoint_url] = open(metric_file, 'w')
            logging.info(f"Writing metrics for {endpoint_url} to {metric_file}")

        try:
            poll_count = 0
            while self.running:
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

                # Poll each endpoint
                for endpoint_url in self.endpoint_urls:
                    metrics = self._poll_metrics(endpoint_url)

                    if metrics is not None:
                        # Add timestamp and write to file
                        for metric in metrics:
                            metric['timestamp'] = timestamp
                            metric['poll_count'] = poll_count

                        # Write as JSON lines
                        metric_file = metric_files[endpoint_url]
                        for metric in metrics:
                            json.dump(metric, metric_file)
                            metric_file.write('\n')
                        metric_file.flush()

                        # Log summary
                        if metrics:
                            latest = metrics[-1]
                            logging.debug(
                                f"[{endpoint_url}] Iter: {latest.get('iter', 'N/A')}, "
                                f"GPU Mem: {latest.get('gpuMemUsage', 0) / 1e9:.2f}GB, "
                                f"Latency: {latest.get('iterLatencyMS', 0):.2f}ms"
                            )

                poll_count += 1

                if self.running:
                    time.sleep(self.poll_interval)

        finally:
            # Close all metric files
            for metric_file in metric_files.values():
                metric_file.close()

            logging.info("Metrics capture stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture metrics from trtllm-serve endpoints")
    parser.add_argument(
        "--endpoints",
        type=str,
        help="Comma-separated list of endpoint URLs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='/tmp/trtllm_metrics/',
        help="Output directory for metrics logs"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Parse endpoints
    endpoint_urls = [url.strip() for url in args.endpoints.split(",") if url.strip()]

    if not endpoint_urls:
        logging.error("No endpoints specified")
        sys.exit(1)

    # Create and run metrics capture
    capture = MetricsCapture(
        endpoint_urls=endpoint_urls,
        output_dir=Path(args.output_dir),
        poll_interval=args.poll_interval,
    )

    try:
        capture.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
