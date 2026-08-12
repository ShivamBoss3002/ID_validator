@echo off
cd /d D:\AppData\IDvalidator

set LOGFILE=D:\AppData\IDvalidator\server_restart.log
echo ============================================== >> "%LOGFILE%"
echo Server Restart at %date% %time% >> "%LOGFILE%"
echo User: %USERNAME% >> "%LOGFILE%"
echo ============================================== >> "%LOGFILE%"

call .venv\Scripts\activate
echo Starting uvicorn... >> "%LOGFILE%"
uvicorn main:app --host 0.0.0.0 --port 9000 >> "%LOGFILE%" 2>&1