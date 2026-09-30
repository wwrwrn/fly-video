@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  py -3.11 -u "%~dp0setup_env.py" %*
) else (
  python -u "%~dp0setup_env.py" %*
)
set "install_status=%errorlevel%"
popd
echo.
if not "%install_status%"=="0" echo Installation stopped. Install Python 3.11 and check the message above.
pause
exit /b %install_status%
