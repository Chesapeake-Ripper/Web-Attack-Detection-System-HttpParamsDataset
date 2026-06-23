@echo off
cd /d "%~dp0"
call conda activate python3_9
python app.py
pause
