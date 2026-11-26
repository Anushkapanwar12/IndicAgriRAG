"""
Launcher for IndicAgriRAG Web Application
Run with:
    python run_ui.py
"""
import sys
import os
import socket
from pathlib import Path

# Ensure paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from web_server import make_app
import tornado.ioloop

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port=8000, max_attempts=20):
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch IndicAgriRAG Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    port = args.port
    if is_port_in_use(port):
        new_port = find_available_port(port + 1)
        print(f"Notice: Port {port} is busy. Switched to port {new_port}.")
        port = new_port

    app = make_app()
    app.listen(port)

    print("=" * 66)
    print("🌾  IndicAgriRAG · Multilingual Agricultural Question Answering")
    print(f"🚀  Web Application running at: http://localhost:{port}")
    print("=" * 66)
    print("• Dense Embeddings: BAAI/bge-m3 (1024 dimensions)")
    print("• Lexical Search:   BM25Okapi inverted index")
    print("• Evidence Corpus:  ICAR Kharif 2025 (2,813 passages)")
    print("• Supported:        English, हिंदी, Hinglish")
    print("Press Ctrl+C to terminate.")
    print("=" * 66)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
