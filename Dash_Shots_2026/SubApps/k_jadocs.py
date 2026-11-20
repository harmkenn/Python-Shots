"""Map and sort locations from an uploaded spreadsheet by distance to an impact point."""

import base64
import io
import math
import os

import dash
import folium
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
import dash_bootstrap_components as dbc
from folium import plugins

from SubApps import z_functions as zf


dash.register_page(__name__, path="/j_jadocs", name="JADOCS", order=10)
DEFAULT_IMPACT_MGRS = "29RPR6518104660"


def _impact_preview_map(mgrs):
    """Show the impact point immediately, before a spreadsheet is processed."""
    point = zf.MGRS2LL(mgrs)
    fmap = folium.Map(location=[point[1], point[2]], zoom_start=8)
    folium.TileLayer("OpenStreetMap", name="Open Street Map").add_to(fmap)
    folium.Marker(
        [point[1], point[2]],
        tooltip="Impact Point",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(fmap)
    return fmap.get_root().render()


layout = dbc.Container(
    [
        html.H2("JADOCS Location Mapper", className="mb-2"),
        html.P("Upload a spreadsheet of locations, choose its name and coordinate columns, and sort them by distance from an impact point."),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Impact point"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Impact Point (MGRS)"),
                                        dbc.Input(id="jd-impact-mgrs", value=DEFAULT_IMPACT_MGRS, type="text", className="mb-2"),
                                        html.Div(id="jd-impact-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Location spreadsheet"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="jd-upload",
                                            children=html.Div(["Drop an Excel or CSV file here, or ", html.A("browse")]),
                                            accept=".xlsx,.xls,.csv",
                                            multiple=False,
                                            style={"width": "100%", "padding": "24px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center"},
                                        ),
                                        html.Div(id="jd-upload-status", className="mt-2"),
                                        dbc.Label("Sheet (Excel files)"),
                                        dbc.Select(id="jd-sheet", options=[], value=None, className="mb-2"),
                                        dbc.Label("Name column"),
                                        dbc.Select(id="jd-name-column", options=[], value=None, className="mb-2"),
                                        dbc.Label("Description column"),
                                        dbc.Select(id="jd-description-column", options=[], value=None, className="mb-2"),
                                        dbc.Label("Type column"),
                                        dbc.Select(id="jd-type-column", options=[], value=None, className="mb-2"),
                                        dbc.Label("Latitude column"),
                                        dbc.Select(id="jd-latitude-column", options=[], value=None, className="mb-2"),
                                        dbc.Label("Longitude column"),
                                        dbc.Select(id="jd-longitude-column", options=[], value=None, className="mb-2"),
                                        dbc.Button("Process locations", id="jd-process", n_clicks=0, color="primary"),
                                    ]
                                ),
                            ]
                        ),
                    ], xs=12, lg=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Impact point and uploaded locations"),
                                dbc.CardBody(dcc.Loading(html.Iframe(id="jd-map", srcDoc=_impact_preview_map(DEFAULT_IMPACT_MGRS), title="Map of impact point and uploaded locations", style={"width": "100%", "height": "650px", "border": "none"}))),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Locations sorted by distance"),
                                dbc.CardBody(html.Div(id="jd-table-container")),
                            ]
                        ),
                    ], xs=12, lg=8,
                ),
            ], className="g-3",
        ),
    ], fluid=True,
)


def _decode_upload(contents, filename):
    if not contents:
        raise ValueError("Upload an Excel or CSV file to begin.")
    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)
        suffix = os.path.splitext(filename or "")[1].lower()
        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(raw))
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(raw))
        raise ValueError("Choose an .xlsx, .xls, or .csv file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded file: {exc}") from exc


def _decode_workbook(contents, filename):
    if not contents:
        raise ValueError("Upload an Excel or CSV file to begin.")
    _, encoded = contents.split(",", 1)
    raw = base64.b64decode(encoded)
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix == ".csv":
        return None, {"CSV": pd.read_csv(io.BytesIO(raw))}
    if suffix not in (".xlsx", ".xls"):
        raise ValueError("Choose an .xlsx, .xls, or .csv file.")
    workbook = pd.ExcelFile(io.BytesIO(raw))
    return workbook.sheet_names, {name: pd.read_excel(workbook, sheet_name=name) for name in workbook.sheet_names}


@callback(
    Output("jd-sheet", "options"), Output("jd-sheet", "value"),
    Output("jd-name-column", "options"), Output("jd-name-column", "value"),
    Output("jd-description-column", "options"), Output("jd-description-column", "value"),
    Output("jd-type-column", "options"), Output("jd-type-column", "value"),
    Output("jd-latitude-column", "options"), Output("jd-latitude-column", "value"),
    Output("jd-longitude-column", "options"), Output("jd-longitude-column", "value"),
    Output("jd-upload-status", "children"),
    Input("jd-upload", "contents"), State("jd-upload", "filename"),
)
def load_spreadsheet(contents, filename):
    empty = ([], None, [], None, [], None, [], None, [], None, [], None)
    if not contents:
        return (*empty, "Upload an Excel or CSV file.")
    try:
        sheets, frames = _decode_workbook(contents, filename)
        sheet_options = [{"label": name, "value": name} for name in sheets] if sheets else []
        selected_sheet = sheets[0] if sheets else None
        frame = frames[selected_sheet or "CSV"]
        columns = [str(column) for column in frame.columns]
        column_options = [{"label": column, "value": column} for column in columns]
        def guess(*terms):
            return next((column for column in columns if any(term in column.lower() for term in terms)), columns[0] if columns else None)
        return (
            sheet_options, selected_sheet,
            column_options, guess("name", "location", "site"),
            column_options, guess("description", "desc", "details"),
            column_options, guess("type", "category"),
            column_options, guess("latitude", "lat"),
            column_options, guess("longitude", "lon", "lng"),
            f"Loaded {filename or 'file'} ({len(frame)} rows). Select columns and process.",
        )
    except Exception as exc:
        return (*empty, dbc.Alert(str(exc), color="danger", className="mb-0"))


@callback(
    Output("jd-name-column", "options", allow_duplicate=True),
    Output("jd-description-column", "options", allow_duplicate=True),
    Output("jd-type-column", "options", allow_duplicate=True),
    Output("jd-latitude-column", "options", allow_duplicate=True),
    Output("jd-longitude-column", "options", allow_duplicate=True),
    Input("jd-sheet", "value"), State("jd-upload", "contents"), State("jd-upload", "filename"),
    prevent_initial_call=True,
)
def update_sheet_columns(sheet, contents, filename):
    if not contents:
        return [], [], [], [], []
    try:
        _, frames = _decode_workbook(contents, filename)
        frame = frames.get(sheet) if sheet else frames.get("CSV")
        options = [{"label": str(column), "value": str(column)} for column in frame.columns]
        return options, options, options, options, options
    except Exception:
        return [], [], [], [], []


@callback(
    Output("jd-impact-output", "children"),
    Input("jd-impact-mgrs", "value"),
)
def show_impact_point(mgrs):
    if not mgrs:
        return "Enter an impact point MGRS coordinate."
    try:
        point = zf.MGRS2LL(mgrs.strip())
        elevation = zf.elevation(point[1], point[2])
        return html.Div([
            html.P(f"Latitude: {point[1]:.5f} | Longitude: {point[2]:.5f}", className="mb-1"),
            html.P(f"Elevation: {elevation:.1f} meters", className="mb-0"),
        ])
    except Exception as exc:
        return dbc.Alert(f"Invalid MGRS coordinate: {exc}", color="danger", className="mb-0")


def _map_figure(frame, impact, name_col, description_col, type_col, lat_col, lon_col):
    impact_lat, impact_lon = impact
    fmap = folium.Map(location=[impact_lat, impact_lon], zoom_start=5)
    folium.TileLayer("OpenStreetMap", name="Open Street Map").add_to(fmap)
    for layer, name in (("m", "Google Maps"), ("s", "Google Satellite"), ("p", "Google Terrain")):
        folium.TileLayer(
            tiles=f"https://mt1.google.com/vt/lyrs={layer}&x={{x}}&y={{y}}&z={{z}}",
            attr="Google", name=name, overlay=False, control=True,
        ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.add_child(plugins.MiniMap(toggle_display=True))
    plugins.ScrollZoomToggler().add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)
    icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Icons")
    impact_path = os.path.join(icons_dir, "target.png")
    impact_icon = folium.features.CustomIcon(impact_path, icon_size=(25, 25)) if os.path.exists(impact_path) else folium.Icon(color="red", icon="crosshairs", prefix="fa")
    folium.Marker([impact_lat, impact_lon], popup="Impact Point", tooltip="Impact Point", icon=impact_icon).add_to(fmap)
    for _, row in frame.iterrows():
        label = str(row.get(name_col, "Location"))
        details = [f"{label}"]
        if description_col in frame.columns and pd.notna(row.get(description_col)):
            details.append(str(row[description_col]))
        if type_col in frame.columns and pd.notna(row.get(type_col)):
            details.append(f"Type: {row[type_col]}")
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])], radius=4,
            color="blue", fill=True, fill_color="blue", fill_opacity=0.8,
            tooltip=label, popup="<br>".join(details),
        ).add_to(fmap)
    plugins.Draw().add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    bounds = [(impact_lat, impact_lon)] + [
        (float(row[lat_col]), float(row[lon_col])) for _, row in frame.iterrows()
    ]
    fmap.fit_bounds(bounds, padding=(24, 24), max_zoom=14)
    return fmap.get_root().render()


@callback(
    Output("jd-map", "srcDoc"), Output("jd-table-container", "children"),
    Output("jd-upload-status", "children", allow_duplicate=True),
    Input("jd-process", "n_clicks"),
    State("jd-upload", "contents"), State("jd-upload", "filename"),
    State("jd-sheet", "value"), State("jd-impact-mgrs", "value"),
    State("jd-name-column", "value"), State("jd-description-column", "value"),
    State("jd-type-column", "value"), State("jd-latitude-column", "value"), State("jd-longitude-column", "value"),
    prevent_initial_call=True,
)
def process_locations(n_clicks, contents, filename, sheet, impact_mgrs, name_col, description_col, type_col, lat_col, lon_col):
    if not contents:
        return dash.no_update, dash.no_update, dbc.Alert("Upload a spreadsheet first.", color="warning", className="mb-0")
    try:
        sheets, frames = _decode_workbook(contents, filename)
        frame = frames[sheet] if sheets else frames["CSV"]
        if not all((name_col, description_col, type_col, lat_col, lon_col)):
            raise ValueError("Choose a value for each column selector.")
        required = [name_col, description_col, type_col, lat_col, lon_col]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError("Selected columns are not present in this sheet. Check the sheet and column selections.")
        impact_data = zf.MGRS2LL(impact_mgrs.strip())
        impact = float(impact_data[1]), float(impact_data[2])
        frame = frame.copy()
        frame[lat_col] = pd.to_numeric(frame[lat_col], errors="coerce")
        frame[lon_col] = pd.to_numeric(frame[lon_col], errors="coerce")
        frame = frame.dropna(subset=[lat_col, lon_col]).copy()
        frame = frame[frame[lat_col].between(-90, 90) & frame[lon_col].between(-180, 180)].copy()
        if frame.empty:
            raise ValueError("No rows contain valid latitude and longitude coordinates.")
        distances = [zf.LLDist(impact[0], impact[1], float(row[lat_col]), float(row[lon_col]))[0] for _, row in frame.iterrows()]
        frame["Distance (meters)"] = [round(float(distance), 1) for distance in distances]
        frame = frame.sort_values("Distance (meters)")
        table = dash_table.DataTable(
            data=frame.to_dict("records"),
            columns=[{"name": str(column), "id": str(column)} for column in frame.columns],
            sort_action="native", filter_action="native", page_action="native", page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "maxWidth": "260px", "whiteSpace": "normal"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
        )
        return _map_figure(frame, impact, name_col, description_col, type_col, lat_col, lon_col), table, f"Mapped {len(frame)} locations from {filename or 'file'}; invalid coordinate rows were skipped."
    except Exception as exc:
        return "", "", dbc.Alert(f"Could not process locations: {exc}", color="danger", className="mb-0")
