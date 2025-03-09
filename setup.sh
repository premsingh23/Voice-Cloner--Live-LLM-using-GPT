#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"magarpoon276@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt 