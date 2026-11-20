"""Map and sort locations from an uploaded spreadsheet by distance to an impact point."""

import base64
import io
import json
import os

import dash
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
import dash_bootstrap_components as dbc

from SubApps import z_functions as zf


dash.register_page(__name__, path="/j_jadocs", name="JADOCS", order=10)
DEFAULT_IMPACT_MGRS = "29RPR6518104660"


def _impact_preview_map(mgrs):
    """Show the impact point immediately, before a spreadsheet is processed."""
    point = zf.MGRS2LL(mgrs)
    return _render_map((float(point[1]), float(point[2])), [], preview=True)


def _script_json(value):
    """Serialize data safely for embedding in an inline script in the map iframe."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_map(impact, locations, preview=False):
    impact_lat, impact_lon = impact
    features = [
        {
            "type": "Feature",
            "properties": {"kind": "impact", "label": "Impact Point", "details": "Impact Point"},
            "geometry": {"type": "Point", "coordinates": [impact_lon, impact_lat]},
        }
    ]
    features.extend(
        {
            "type": "Feature",
            "properties": {"kind": "location", "label": item["label"], "details": item["details"]},
            "geometry": {"type": "Point", "coordinates": [item["lon"], item["lat"]]},
        }
        for item in locations
    )
    data_json = _script_json({"type": "FeatureCollection", "features": features})
    impact_json = _script_json([impact_lon, impact_lat])
    preview_js = "true" if preview else "false"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
html,body,#jd-map{{width:100%;height:100%;margin:0;overflow:hidden}}
.jd-attribution{{position:absolute;right:6px;bottom:4px;z-index:3;background:#ffffffdf;padding:4px 7px;font:11px sans-serif;color:#222}}
.jd-attribution a{{color:#1558a6;text-decoration:none}}
</style></head><body><div id="jd-map"></div>
<script type="module">
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';
const data = {data_json};
const impact = {impact_json};
const preview = {preview_js};
const map = new maplibregl.Map({{
  container:'jd-map', style:'https://tiles.openfreemap.org/styles/liberty',
  center:impact, zoom:preview ? 8 : 5, pitch:preview ? 30 : 40, maxPitch:85,
  dragRotate:true, touchZoomRotate:true
}});
map.addControl(new maplibregl.NavigationControl({{visualizePitch:true}}), 'top-right');
map.addControl(new maplibregl.ScaleControl({{maxWidth:120, unit:'metric'}}), 'bottom-left');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
map.once('style.load', () => {{
  map.addSource('jd-elevation', {{type:'raster-dem', url:'https://tiles.mapterhorn.com/tilejson.json'}});
  const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
  map.addLayer({{id:'jd-hillshade',type:'hillshade',source:'jd-elevation',paint:{{'hillshade-exaggeration':0.35}}}}, firstLabel && firstLabel.id);
  map.setTerrain({{source:'jd-elevation',exaggeration:1.0}});
  map.setSky({{'sky-color':'#081522','sky-horizon-blend':0.35,'atmosphere-blend':0.8}});
  map.addSource('jd-points', {{type:'geojson',data}});
  map.addLayer({{id:'jd-location-points',type:'circle',source:'jd-points',filter:['==',['get','kind'],'location'],paint:{{'circle-radius':5,'circle-color':'#1976d2','circle-opacity':0.9,'circle-stroke-color':'#fff','circle-stroke-width':1.25}}}});
  map.addLayer({{id:'jd-impact-point',type:'circle',source:'jd-points',filter:['==',['get','kind'],'impact'],paint:{{'circle-radius':9,'circle-color':'#d62828','circle-stroke-color':'#fff','circle-stroke-width':2}}}});
  const popup = new maplibregl.Popup({{closeButton:false,closeOnClick:false}});
  map.on('mouseenter','jd-location-points', (event) => {{
    map.getCanvas().style.cursor = 'pointer';
    const feature = event.features && event.features[0];
    if (feature) popup.setLngLat(event.lngLat).setText(feature.properties.label || 'Location').addTo(map);
  }});
  map.on('mousemove','jd-location-points', (event) => {{
    const feature = event.features && event.features[0];
    if (feature) popup.setLngLat(event.lngLat).setText(feature.properties.label || 'Location');
  }});
  map.on('mouseleave','jd-location-points', () => {{ map.getCanvas().style.cursor = ''; popup.remove(); }});
  map.on('click','jd-location-points', (event) => {{
    const feature = event.features && event.features[0];
    if (feature) new maplibregl.Popup().setLngLat(event.lngLat).setText(feature.properties.details || feature.properties.label || 'Location').addTo(map);
  }});
  map.on('click','jd-impact-point', (event) => new maplibregl.Popup().setLngLat(event.lngLat).setText('Impact Point').addTo(map));
  if (!preview && data.features.length > 1) {{
    const box = new maplibregl.LngLatBounds(impact, impact);
    for (const feature of data.features) {{
      if (feature.geometry.type !== 'Point') continue;
      const [lon, lat] = feature.geometry.coordinates;
      const wrappedLon = impact[0] + ((((lon - impact[0]) + 540) % 360) - 180);
      box.extend([wrappedLon, lat]);
    }}
    map.fitBounds(box, {{padding:45,maxZoom:14,pitch:40,duration:0}});
  }}
}});
</script>
<div class="jd-attribution">
<a href="https://maplibre.org/" target="_blank" rel="noopener">MapLibre</a> |
<a href="https://mapterhorn.com/attribution" target="_blank" rel="noopener">&copy; Mapterhorn</a> |
<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> |
<a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">&copy; OpenMapTiles</a> |
Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>
</div></body></html>"""


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
    locations = []
    for _, row in frame.iterrows():
        label = str(row.get(name_col, "Location")) if pd.notna(row.get(name_col)) else "Location"
        details = [f"{label}"]
        if description_col in frame.columns and pd.notna(row.get(description_col)):
            details.append(str(row[description_col]))
        if type_col in frame.columns and pd.notna(row.get(type_col)):
            details.append(f"Type: {row[type_col]}")
        details.append(f"Distance: {float(row['Distance (meters)']):,.1f} m")
        locations.append({
            "label": label,
            "details": "\n".join(details),
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
        })
    return _render_map(impact, locations)


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
