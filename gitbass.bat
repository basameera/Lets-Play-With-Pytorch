@echo off
echo *** Git-Bass ***

set commitMsg=%~1
IF "%commitMsg%"=="" (
    echo ERROR: Need Commit Message as argument
    exit /B
)

echo # Commit message: %commitMsg%

echo.
echo # Git pull
echo.
git pull

echo.
echo # Checking git status
echo.
git status

echo.
echo # Git add
echo.
git add .

echo.
echo # Git commit
echo.
git commit -m %commitMsg%

echo.
echo # Git push
echo.
git push

echo.
echo # Checking git status
echo.
git status