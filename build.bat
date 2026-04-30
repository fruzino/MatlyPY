@echo off
set /p ver="Enter version number: "
set /p msg="Enter commit message: "

:: Add dist and egg-info to ignore if they aren't there
findstr /C:"dist/" .gitignore >nul 2>&1 || echo dist/ >> .gitignore
findstr /C:"*.egg-info/" .gitignore >nul 2>&1 || echo *.egg-info/ >> .gitignore

echo Syncing MatlyPy v%ver%...

git add .
git commit -m "Release v%ver%: %msg%"
git push origin main

echo.
echo Push successful. GitHub Actions will now build and publish to PyPI.
pause