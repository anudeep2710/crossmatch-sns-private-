#!/bin/bash

# Run the Streamlit app with file watcher disabled
python3 -m streamlit run app.py --server.fileWatcherType none
