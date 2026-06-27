@echo off
title Pacomixer
cd /d "%~dp0"

:: Buscar Anaconda/Miniconda en las ubicaciones más comunes
set CONDA_SCRIPT=

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat"     set CONDA_SCRIPT=%USERPROFILE%\anaconda3\Scripts\activate.bat
if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat"     set CONDA_SCRIPT=%USERPROFILE%\Anaconda3\Scripts\activate.bat
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat"    set CONDA_SCRIPT=%USERPROFILE%\miniconda3\Scripts\activate.bat
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat"    set CONDA_SCRIPT=%USERPROFILE%\Miniconda3\Scripts\activate.bat
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat"    set CONDA_SCRIPT=C:\ProgramData\anaconda3\Scripts\activate.bat
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat"    set CONDA_SCRIPT=C:\ProgramData\Anaconda3\Scripts\activate.bat
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat"   set CONDA_SCRIPT=C:\ProgramData\miniconda3\Scripts\activate.bat

if "%CONDA_SCRIPT%"=="" (
    echo.
    echo  No se encontro Anaconda ni Miniconda en las rutas habituales.
    echo  Abre Anaconda Prompt manualmente y ejecuta:
    echo.
    echo      conda activate Pacomixer
    echo      streamlit run app.py
    echo.
    pause
    exit /b 1
)

:: Activar entorno Pacomixer
call "%CONDA_SCRIPT%" Pacomixer

:: Lanzar Streamlit
streamlit run app.py
pause