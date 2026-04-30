@echo off
set /p ver="Enter version number(x.x.x): "
set /p msg="Enter commit message: "

echo Syncing version %ver%...

if exist dist (
    rmdir /s /q dist
)

git add .
git commit -m "Release v%ver%: %msg%"
git push origin main

echo.
echo Push complete. 
echo Ensure your config file matches v%ver%.
echo Monitor GitHub Actions for the PyPI upload.
pause