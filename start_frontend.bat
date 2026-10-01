@echo off
REM ============================================================
REM 一键启动 · 平陆运河前端
REM 每次重启自动清理 Qdrant 锁 + 释放 8501 端口
REM ============================================================
cd /d "%~dp0"

echo.
echo  ====================================================
echo    Zhi Hui Yun He - Streamlit Frontend Launcher
echo  ====================================================
echo.

REM ---- 1. 杀掉所有 python 进程（避免锁残留）----
echo  [1/3] Killing stale python processes...
taskkill /F /IM python.exe   >nul 2>&1
taskkill /F /IM pythonw.exe  >nul 2>&1
timeout /t 1 /nobreak >nul

REM ---- 2. 清理 Qdrant 锁文件 ----
echo  [2/3] Removing Qdrant lock file...
if exist "data\qdrant\.lock" (
    del /F "data\qdrant\.lock"
    echo        [OK] Deleted data\qdrant\.lock
) else (
    echo        [OK] No stale lock found
)

REM ---- 3. 启动 Streamlit ----
echo  [3/3] Starting Streamlit on port 8501...
echo.
echo  >>> Open browser: http://localhost:8501
echo  >>> Press Ctrl+C to stop
echo.

.\.venv\Scripts\streamlit.exe run frontend\app.py --server.port 8501 --server.headless false
