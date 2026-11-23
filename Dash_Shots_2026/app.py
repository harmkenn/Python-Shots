import os
from copy import deepcopy

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.io as pio

# Initialize the Dash App with multi-page support enabled
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="SubApps",  # Replaces 'SubApps' folder
    external_stylesheets=[dbc.themes.DARKLY]
)

dark_blue_template = deepcopy(pio.templates["plotly_dark"])
dark_blue_template.layout.paper_bgcolor = "#0b1424"
dark_blue_template.layout.plot_bgcolor = "#101e32"
dark_blue_template.layout.font.color = "#e6efff"
dark_blue_template.layout.colorway = ["#4c9aff", "#37c5f0", "#f6bd60", "#a78bfa", "#34d399", "#fb7185"]
pio.templates["python_shots_dark_blue"] = dark_blue_template
pio.templates.default = "python_shots_dark_blue"

# Define Sidebar Styling
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE = {
    "margin-left": "19rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Sidebar Layout
sidebar = html.Div(
    [
        html.H3("Python Dash Shots v2026.2", className="display-8"),
        html.H6("by Ken Harmon", className="text-muted mb-4"),
        html.Hr(),
        # Generate navigation links dynamically from registered 
        dbc.Nav(
            [
                dbc.NavLink(
                    page["name"],
                    href=page["relative_path"],
                    active="exact",
                    className="mb-1"
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="app-sidebar",
    style=SIDEBAR_STYLE,
)

# Content area where sub-app views render dynamically
content = html.Div(dash.page_container, className="app-content", style=CONTENT_STYLE)

# Master Layout
app.layout = html.Div([sidebar, content], className="app-shell")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
    )
