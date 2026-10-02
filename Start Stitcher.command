#!/bin/bash
# Double-click this file (macOS) to open the Whisky Webs Stitcher.
cd "$(dirname "$0")" || exit 1

pause_and_exit() {
    echo
    read -n 1 -s -r -p "Press any key to close this window."
    echo
    exit 1
}

# Find a Python 3.9+ with a working Tk 8.6 (the python.org installer has one).
# Apple's /usr/bin/python3 is skipped: it's only an installer stub or an old Tk.
PY=""
for cand in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
            /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) and tkinter.TkVersion >= 8.6 else 1)" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3 is not installed on this Mac (or it has no Tk support)."
    echo "Install it from https://www.python.org/downloads/macos/ and then"
    echo "double-click this file again."
    pause_and_exit
fi

# First run: a private Python environment inside this folder, so nothing is
# installed system-wide. Later runs reuse it.
VENV=".venv"
if ! "$VENV/bin/python" -c "import numpy, PIL, tifffile, imagecodecs, customtkinter" >/dev/null 2>&1; then
    echo "First run: installing the parts the stitcher needs. This only happens once..."
    "$PY" -m venv "$VENV" || { echo "Couldn't set up Python here."; pause_and_exit; }
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1
    if ! "$VENV/bin/python" -m pip install -r requirements.txt; then
        echo
        echo "Installation failed - please send a screenshot of this window to whoever set this up."
        pause_and_exit
    fi
fi

nohup "$VENV/bin/python" stitch_gui.py >"$VENV/stitcher.log" 2>&1 &
echo "The stitcher is opening. You can close this Terminal window."
