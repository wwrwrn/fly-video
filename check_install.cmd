@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if not exist "%~dp0.venv-train\Scripts\python.exe" (
  echo First use: double-click setup_env.cmd.
  popd
  pause
  exit /b 2
)
"%~dp0.venv-train\Scripts\python.exe" -u "%~dp0check_install.py"
set "check_status=%errorlevel%"
popd
pause
exit /b %check_status%
