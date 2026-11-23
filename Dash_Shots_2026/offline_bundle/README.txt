Python dependency bundle for Dash_Shots_2026

Target: Windows x86-64, CPython 3.12.

Install while offline:
  python -m pip install --no-index --find-links=wheelhouse -r requirements-offline.txt

Then copy the Dash_Shots_2026 application folder to the target machine and run:
  python app.py

This archive contains the Python packages and their dependencies. The app loads MapLibre GL JavaScript from unpkg.com at runtime; that browser asset still needs network access unless separately mirrored into the app.
