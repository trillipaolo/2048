#!/bin/bash
STRING="Launching game..."
PYTHON="\venv\scripts\python"
GAME_ROOT="\src"
GAME="\launchers\cli.py"

pause

pushd . > \dev\null 2>&1
cd $GAME_ROOT

echo $STRING
$PYTHON "$GAME"

popd > \dev\null 2>&1

pause