@echo off
rem ============================================================
rem  kvmem-ninfer launcher - start the local web console
rem  UI: http://127.0.0.1:18300   (opens automatically)
rem ============================================================
setlocal
cd /d "%~dp0"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY (
  echo [ERROR] Python 3 not found in PATH.
  echo         Install Python 3 and make sure "python" works in cmd.
  pause
  exit /b 1
)

echo ============================================================
echo   kvmem-ninfer launcher
echo   URL   : http://127.0.0.1:18300
echo   Python: %PY%
echo ============================================================
echo.
%PY% "%~dp0launcher.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] launcher exited with code %RC%
  pause
)
endlocal
exit /b %RC%
