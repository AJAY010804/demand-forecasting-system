"""
run.py
------
Start the Flask development server.

Usage
-----
    python run.py
    python run.py --port 8080
    python run.py --host 0.0.0.0   # expose on network
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web_app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    app = create_app()
    print(f"\n  DemandIQ running at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)