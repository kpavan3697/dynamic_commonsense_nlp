#!/bin/bash

# Activate your Python environment if needed
# source ~/.venv/bin/activate

# Set environment variables
export STREAMLIT_WATCHER_TYPE=none
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE 

# Run the Streamlit app
streamlit run app.py --server.runOnSave false