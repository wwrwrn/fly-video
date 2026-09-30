@echo off
setlocal
chcp 65001 >nul
set "project_dir=%~dp0"
set "source_video=%~f1"
pushd "%project_dir%"
if not exist "%project_dir%.venv-train\Scripts\python.exe" (
  echo First use: double-click setup_env.cmd to install the environment.
  popd
  pause
  exit /b 2
)
if "%~1"=="" (
  "%project_dir%.venv-train\Scripts\python.exe" -u "%project_dir%launch_single_fly.py"
) else (
  "%project_dir%.venv-train\Scripts\python.exe" -u "%project_dir%launch_single_fly.py" "%source_video%"
)
set "run_status=%errorlevel%"
popd
echo.
if not "%run_status%"=="0" echo 分析未完成，请看上面的错误信息。
pause
exit /b %run_status%
