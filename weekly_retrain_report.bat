@echo off
setlocal

cd /d "%~dp0"
python weekly_retrain_report.py --hours 6
