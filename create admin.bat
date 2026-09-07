cd .venv/scripts
call activate.bat
cd ../..

set /p USERNAME=Please enter the admin username:

set /p EMAIL=Please enter the admin email:

set /p PASSWORD=Please enter the admin password:

py main.py --create-admin --username "%USERNAME%" --email "%EMAIL%" --password "%PASSWORD%"
PAUSE

