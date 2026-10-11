@echo off
cd /d "%~dp0"
echo DQN Othello Web UI: http://localhost:8765/
start "" http://localhost:8765/
node server.js
