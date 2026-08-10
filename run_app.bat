@echo off
echo ============================================
echo   CropGuard AI - Starting Application
echo ============================================

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting CropGuard AI server...
echo Public portal:   http://127.0.0.1:5000/
echo Admin portal:    http://127.0.0.1:5000/admin/login
echo   (default admin user: host / CropGuard@2026 - change this after first login)
echo.

python app.py

pause
