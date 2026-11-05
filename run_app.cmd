@echo off
setlocal
chcp 65001 >nul
set "project_dir=%~dp0"
pushd "%project_dir%"
if not exist "%project_dir%.venv-train\Scripts\python.exe" (
  echo First use: double-click setup_env.cmd to install the environment.
  popd
  pause
  exit /b 2
)
"%project_dir%.venv-train\Scripts\python.exe" -u "%project_dir%app_server.py"
popd
pause
