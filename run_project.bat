@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 goto error
py code\ev_station_placement.py
if errorlevel 1 goto error
echo.
echo Finished. Open the outputs folder to view CSVs and charts.
pause
exit /b 0
:error
echo.
echo Something went wrong. Check Python installation and error messages above.
pause
exit /b 1
