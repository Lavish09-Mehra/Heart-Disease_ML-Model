#!/usr/bin/env bash
# Start the Heart Disease Risk Predictor (Streamlit) on http://127.0.0.1:8501
set -e
cd "$(dirname "$0")"
exec "$HOME/jupyter-env/bin/python" -m streamlit run server.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
