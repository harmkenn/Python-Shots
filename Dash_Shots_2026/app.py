import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash App with multi-page support enabled
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="SubApps",  # Replaces 'SubApps' folder
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Define Sidebar Styling
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
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
    style=SIDEBAR_STYLE,
)

# Content area where sub-app views render dynamically
content = html.Div(dash.page_container, style=CONTENT_STYLE)

# Master Layout
app.layout = html.Div([sidebar, content])

if __name__ == "__main__":
    app.run(debug=True)
