@echo off
setlocal enabledelayedexpansion

:: Assicura che la directory di lavoro corrente sia sempre la cartella dello script
cd /d "%~dp0"

echo ======================================================================
echo    Dynamic Target Suite - Compilazione Eseguibile Standalone
echo ======================================================================
echo.

:: 1. Rilevamento interprete virtualenv locale
set "PYTHON_EXE="
set "PYINSTALLER_EXE="

if exist ".\venv\Scripts\pyinstaller.exe" (
    set "PYINSTALLER_EXE=.\venv\Scripts\pyinstaller.exe"
    set "PYTHON_EXE=.\venv\Scripts\python.exe"
    echo [INFO] Rilevato ambiente virtuale dedicato: .\venv
) else (
    where pyinstaller >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set "PYINSTALLER_EXE=pyinstaller"
        set "PYTHON_EXE=python"
        echo [WARN] .\venv non trovato: utilizzo di pyinstaller dal PATH di sistema.
    ) else (
        echo [ERROR] PyInstaller non trovato ne in .\venv\Scripts ne nel PATH di sistema!
        echo         Assicurati di installare le dipendenze nell'ambiente virtuale:
        echo         .\venv\Scripts\pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: 2. Supporto per opzioni di riga di comando (--clean, --no-pause)
set "CLEAN_FLAG="
set "NO_PAUSE="
for %%A in (%*) do (
    if /i "%%A"=="--clean" set "CLEAN_FLAG=--clean"
    if /i "%%A"=="/clean" set "CLEAN_FLAG=--clean"
    if /i "%%A"=="clean" set "CLEAN_FLAG=--clean"
    if /i "%%A"=="--no-pause" set "NO_PAUSE=1"
    if /i "%%A"=="-nopause" set "NO_PAUSE=1"
)

:: 3. Chiusura preventiva di eventuali istanze aperte per evitare WinError 5 (Accesso Negato)
taskkill /f /im MDO_Target_Suite.exe >nul 2>nul

if defined CLEAN_FLAG (
    echo [INFO] Richiesta pulizia preventiva cache build di PyInstaller...
    if exist "build\MDO_Target_Suite" rmdir /s /q "build\MDO_Target_Suite" 2>nul
)

:: 4. Esecuzione compilazione PyInstaller
echo.
echo [1/3] Avvio compilazione PyInstaller con MDO_Target_Suite.spec...
echo ----------------------------------------------------------------------
"!PYINSTALLER_EXE!" --noconfirm !CLEAN_FLAG! MDO_Target_Suite.spec

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ======================================================================
    echo [ERROR] Compilazione fallita con codice di errore !ERRORLEVEL!!
    echo         Verifica i messaggi di errore soprastanti.
    echo ======================================================================
    echo.
    if not defined NO_PAUSE pause
    exit /b !ERRORLEVEL!
)

:: 5. Sincronizzazione setups, tracciati e pneumatici aggiornati nella cartella dist
echo.
echo [2/3] Verifica e sincronizzazione configurazioni, tracciati e pneumatici...
if exist "setups" (
    if not exist "dist\MDO_Target_Suite\setups" mkdir "dist\MDO_Target_Suite\setups"
    xcopy /y /d "setups\*.json" "dist\MDO_Target_Suite\setups\" >nul 2>nul
    echo       Setups JSON sincronizzati in dist\MDO_Target_Suite\setups.
)
if exist "tire_tir" (
    if not exist "dist\MDO_Target_Suite\tire_tir" mkdir "dist\MDO_Target_Suite\tire_tir"
    xcopy /y /d "tire_tir\*.tir" "dist\MDO_Target_Suite\tire_tir\" >nul 2>nul
    echo       File pneumatici .TIR sincronizzati in dist\MDO_Target_Suite\tire_tir.
)
if exist "tracks" (
    if not exist "dist\MDO_Target_Suite\tracks" mkdir "dist\MDO_Target_Suite\tracks"
    xcopy /y /d /s "tracks\*" "dist\MDO_Target_Suite\tracks\" >nul 2>nul
    echo       Tracciati circuito sincronizzati in dist\MDO_Target_Suite\tracks.
)

:: 6. Smoke test headless dell'eseguibile appena generato
echo.
echo [3/3] Verifica integrita e avvio binario (Smoke Test Headless)...
if not exist "dist\MDO_Target_Suite\MDO_Target_Suite.exe" (
    echo [ERROR] L'eseguibile dist\MDO_Target_Suite\MDO_Target_Suite.exe non e stato trovato!
    if not defined NO_PAUSE pause
    exit /b 1
)

set "QT_QPA_PLATFORM=offscreen"
"!PYTHON_EXE!" -c "import subprocess, time, sys, os; exe = os.path.abspath('dist/MDO_Target_Suite/MDO_Target_Suite.exe'); env = os.environ.copy(); env['QT_QPA_PLATFORM'] = 'offscreen'; p = subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE); time.sleep(3.5); ret = p.poll(); (p.terminate(), sys.exit(0)) if ret is None else sys.exit(1)"

if !ERRORLEVEL! EQU 0 (
    echo       Smoke test SUPERATO: l'eseguibile si avvia regolarmente senza errori di import o DLL mancanti.
) else (
    echo [WARN] Lo smoke test non ha confermato l'avvio pulito. Si consiglia di testare manualmente.
)

echo.
echo ======================================================================
echo   [SUCCESS] Compilazione completata con successo!
echo   Eseguibile generato:
echo     dist\MDO_Target_Suite\MDO_Target_Suite.exe
echo ======================================================================
echo.
if not defined NO_PAUSE pause

