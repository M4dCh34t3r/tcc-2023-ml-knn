@echo off

echo.

IF exist venv (
    echo [92mAmbiente virtual detectado[0m
) ELSE (
    python -m venv ./venv
    echo [91mAmbiente virtual inexistente[0m
    echo.
    echo [93mAmbiente virtual criado na pasta 'venv'[0m
)

echo.
echo [92mInicializando ambiente virtual[0m
call .\venv\Scripts\activate.bat

echo.
echo [96mBuscando por atualizacoes do pip...[0m
python.exe -m pip install --upgrade pip

echo.
echo [96mInstalando lib sklearn...[0m
pip install scikit-learn

echo.
echo [96mInstalando lib matplotlib...[0m
pip install matplotlib

echo.
echo [96mInstalando lib seaborn...[0m
pip install seaborn

echo.
echo [92mLibs instaladas com sucesso[0m

pause
