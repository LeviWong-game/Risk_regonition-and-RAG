@echo off
setlocal

cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Git is not installed or not in PATH.
  pause
  exit /b 1
)

for /f "delims=" %%i in ('git rev-parse --is-inside-work-tree 2^>nul') do set "IN_REPO=%%i"
if /i not "%IN_REPO%"=="true" (
  echo [ERROR] Current folder is not a Git repository.
  pause
  exit /b 1
)

for /f "delims=" %%i in ('git branch --show-current') do set "BRANCH=%%i"
if "%BRANCH%"=="" (
  echo [ERROR] Cannot detect current branch.
  pause
  exit /b 1
)

set "REMOTE=github"
git remote get-url "%REMOTE%" >nul 2>nul
if errorlevel 1 set "REMOTE=origin"

set "MSG=%*"
if "%MSG%"=="" set "MSG=chore: sync %date% %time%"

echo [INFO] Repository: %cd%
echo [INFO] Remote: %REMOTE%
echo [INFO] Branch: %BRANCH%
echo [INFO] Commit message: %MSG%

git add -A
if errorlevel 1 (
  echo [ERROR] git add failed.
  pause
  exit /b 1
)

git diff --cached --quiet
if not errorlevel 1 (
  echo [INFO] No changes detected. Nothing to commit.
  pause
  exit /b 0
)

git commit -m "%MSG%"
if errorlevel 1 (
  echo [ERROR] git commit failed.
  pause
  exit /b 1
)

git push "%REMOTE%" "%BRANCH%"
if errorlevel 1 (
  echo [ERROR] git push failed.
  pause
  exit /b 1
)

echo [SUCCESS] Sync completed.
pause
exit /b 0
