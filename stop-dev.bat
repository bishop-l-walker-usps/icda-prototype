@echo off
echo ============================================================
echo    ICDA Prototype - Stop All Development Servers
echo ============================================================
echo.

REM Kill processes on backend port 8000
echo [INFO] Checking port 8000 (FastAPI Backend)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo        Killing PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill processes on Vite port 5173
echo [INFO] Checking port 5173 (Vite Frontend)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    echo        Killing PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill processes on Vite preview port 4173
echo [INFO] Checking port 4173 (Vite Preview)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :4173 ^| findstr LISTENING') do (
    echo        Killing PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill any stray node processes related to vite
echo [INFO] Checking for stray Node.js processes...
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq ICDA Frontend" >nul 2>&1

REM Kill any stray python/uvicorn processes
echo [INFO] Checking for stray Python processes...
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq ICDA Backend" >nul 2>&1

echo.
echo ============================================================
echo [SUCCESS] All ICDA dev servers stopped!
echo ============================================================
echo.
echo Ports cleared: 8000, 5173, 4173
echo.
pause
