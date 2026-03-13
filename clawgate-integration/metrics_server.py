#!/usr/bin/env python3
"""
Prometheus Metrics HTTP Server for ClawGate

Exposes /metrics endpoint for Prometheus scraping.

Usage:
    python3 metrics_server.py [--port 9090]

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse
import logging
from metrics import get_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Prometheus metrics"""

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            # Return Prometheus metrics
            metrics_output = get_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.end_headers()
            self.wfile.write(metrics_output)

        elif self.path == '/health':
            # Health check endpoint
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')

        else:
            # 404 for other paths
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))


def run_metrics_server(port: int = 9090):
    """
    Run the Prometheus metrics HTTP server

    Args:
        port: Port to listen on (default: 9090)
    """
    server_address = ('', port)
    httpd = HTTPServer(server_address, MetricsHandler)

    logger.info(f"Starting Prometheus metrics server on port {port}")
    logger.info(f"Metrics endpoint: http://localhost:{port}/metrics")
    logger.info(f"Health check: http://localhost:{port}/health")
    logger.info("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down metrics server")
        httpd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prometheus metrics server for ClawGate')
    parser.add_argument('--port', type=int, default=9090, help='Port to listen on (default: 9090)')
    args = parser.parse_args()

    run_metrics_server(port=args.port)
