@echo off
echo *** Git-Bass ***

set commitMsg=%~1
IF "%commitMsg%"=="" (
    echo ERROR: Need Commit Message as argument
    exit /B
)
echo %commitMsg%
:: check repo for pulls
git pull

:: lets push
git status
git add .
git commit -m %commitMsg%
git push
git status