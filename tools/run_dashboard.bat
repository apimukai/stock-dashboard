@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements-dashboard.txt
python -m streamlit run stock_dashboard.py
pause
