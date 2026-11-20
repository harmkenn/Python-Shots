"""AFATDS ballistic planning page for the Dash Shots app."""

import math
import json
import re

import dash
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from SubApps import z_functions as zf


dash.register_page(__name__, path="/d_afatds", name="AFATDS", order=4)


layout = dbc.Container(
    [
        html.H2("AFATDS Shot Planning", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Location lookup"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="af-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="af-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="af-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot coordinates"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Paste point coordinates"),
                                        dbc.Textarea(
                                            id="af-paste-point",
                                            placeholder="Latitude: 00.0000\nLongitude: 00.0000",
                                            style={"height": "68px"},
                                            className="mb-2",
                                        ),
                                        dbc.RadioItems(
                                            id="af-paste-destination",
                                            options=[
                                                {"label": "Convert to launch point", "value": "launch"},
                                                {"label": "Convert to impact point", "value": "impact"},
                                            ],
                                            value="launch",
                                            inline=True,
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Convert coordinates",
                                            id="af-convert-button",
                                            n_clicks=0,
                                            color="secondary",
                                            outline=True,
                                            className="mb-3",
                                        ),
                                        html.Div(id="af-convert-output", className="mb-2"),
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="af-launch-mgrs",
                                            type="text",
                                            value="36SYC6482056356",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Launch Altitude (meters)"),
                                        dbc.Input(id="af-launch-altitude", type="number", value=0, step="any", className="mb-2"),
                                        dbc.Label("Impact Point (MGRS)"),
                                        dbc.Input(
                                            id="af-impact-mgrs",
                                            type="text",
                                            value="36SYC6482066356",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Impact Altitude (meters)"),
                                        dbc.Input(id="af-impact-altitude", type="number", value=0, step="any", className="mb-2"),
                                        html.Div(id="af-coordinate-output", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Projectile parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Mass (lbs)"),
                                        dbc.Input(id="af-mass", type="number", value=103.5, min=0, step="any", className="mb-2"),
                                        dbc.Label("Initial Velocity (m/s)"),
                                        dbc.Input(id="af-velocity", type="number", value=568, min=0, step="any", className="mb-2"),
                                        dbc.Label("Azimuth of Lay (mils)"),
                                        dbc.Input(id="af-aol", type="number", value=2200, step="any", className="mb-2"),
                                        dbc.Label("Trajectory"),
                                        dbc.RadioItems(
                                            id="af-angle-mode",
                                            options=[{"label": "High angle", "value": "high"}, {"label": "Low angle", "value": "low"}],
                                            value="high",
                                            inline=True,
                                        ),
                                        html.Div(id="af-range-output", className="mt-3 fw-bold"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot solution"),
                                dbc.CardBody(html.Div(id="af-solution-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    xl=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot map"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Iframe(
                                            id="af-map-iframe",
                                            title="Map showing the launch point, impact point, and azimuth of lay",
                                            style={"width": "100%", "height": "560px", "border": "none"},
                                        )
                                    )
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Projectile trajectory"),
                                dbc.CardBody(dcc.Graph(id="af-trajectory-graph", figure=go.Figure())),
                            ],
                        ),
                    ],
                    xs=12,
                    xl=8,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


@callback(
    Output("af-lookup-output", "children"),
    Input("af-lookup-button", "n_clicks"),
    State("af-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search."
    try:
        place = zf.lookup(query.strip())
        altitude = zf.elevation(place[1], place[2])
        return html.Div(
            [
                html.P(place[0], className="mb-1 fw-bold"),
                html.P(f"Lat: {place[1]} | Lon: {place[2]}", className="mb-1"),
                html.P(f"MGRS: {place[3]}", className="mb-1"),
                html.P(f"Alt: {round(altitude, 2)} meters", className="mb-0"),
            ]
        )
    except Exception as exc:
        return dbc.Alert(f"Lookup error: {exc}", color="danger", className="mb-0")


@callback(
    Output("af-launch-mgrs", "value"),
    Output("af-impact-mgrs", "value"),
    Output("af-convert-output", "children"),
    Input("af-convert-button", "n_clicks"),
    State("af-paste-point", "value"),
    State("af-paste-destination", "value"),
    State("af-launch-mgrs", "value"),
    State("af-impact-mgrs", "value"),
    prevent_initial_call=True,
)
def convert_pasted_coordinates(n_clicks, pasted, destination, launch_mgrs, impact_mgrs):
    if not n_clicks:
        return launch_mgrs, impact_mgrs, ""
    if not pasted:
        return launch_mgrs, impact_mgrs, dbc.Alert("Paste latitude and longitude first.", color="warning")

    # Accept both the labeled two-line text used by the source app and a plain
    # pair of decimal coordinates separated by commas or whitespace.
    values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", pasted)
    if len(values) < 2:
        return launch_mgrs, impact_mgrs, dbc.Alert("Could not find latitude and longitude in the pasted text.", color="danger")
    try:
        latitude, longitude = map(float, values[:2])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Latitude or longitude is outside its valid range.")
        mgrs = zf.LL2MGRS(latitude, longitude)[1]
        if destination == "impact":
            return launch_mgrs, mgrs, f"Impact point converted to {mgrs}."
        return mgrs, impact_mgrs, f"Launch point converted to {mgrs}."
    except Exception as exc:
        return launch_mgrs, impact_mgrs, dbc.Alert(f"Coordinate conversion error: {exc}", color="danger")


@callback(
    Output("af-launch-altitude", "value"),
    Input("af-launch-mgrs", "value"),
)
def set_default_launch_altitude(mgrs):
    if not mgrs:
        return None
    try:
        point = zf.MGRS2LL(mgrs)
        return round(float(zf.elevation(point[1], point[2])), 2)
    except Exception:
        return None


@callback(
    Output("af-impact-altitude", "value"),
    Input("af-impact-mgrs", "value"),
)
def set_default_impact_altitude(mgrs):
    if not mgrs:
        return None
    try:
        point = zf.MGRS2LL(mgrs)
        return round(float(zf.elevation(point[1], point[2])), 2)
    except Exception:
        return None


@callback(
    Output("af-range-output", "children"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
)
def display_range(launch_mgrs, impact_mgrs):
    if not launch_mgrs or not impact_mgrs:
        return ""
    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        distance, _ = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        return f"Range: {distance:,.0f} m"
    except Exception as exc:
        return f"Range unavailable: {exc}"


@callback(
    Output("af-velocity", "value"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
    Input("af-mass", "value"),
)
def suggest_initial_velocity(launch_mgrs, impact_mgrs, mass):
    if not launch_mgrs or not impact_mgrs or mass is None:
        return None
    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        distance, _ = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        mass = float(mass)
        if distance <= 12000:
            velocity = -2.305882 * mass + 806.6588
        elif distance <= 18000:
            velocity = -2.105882 * mass + 899.9588
        else:
            velocity = -2.447059 * mass + 1043.271
        return round(velocity, 1)
    except Exception:
        return None


def _sample_map_line(start, bearing, distance, end=None, segments=80):
    """Return geodesic samples as GeoJSON [longitude, latitude] coordinates."""
    points = []
    for index in range(segments + 1):
        latitude, longitude = zf.vPolar(
            float(start[1]), float(start[2]), bearing, distance * index / segments
        )
        points.append([float(longitude), float(latitude)])
    if end is not None:
        points[-1] = [float(end[2]), float(end[1])]
    return points


def _make_map(launch, impact, azimuth_of_lay, range_m, bearing_deg):
    launch_lat, launch_lon = float(launch[1]), float(launch[2])
    impact_lat, impact_lon = float(impact[1]), float(impact[2])
    lay_bearing = float(azimuth_of_lay) * 180 / 3200

    route = _sample_map_line(launch, bearing_deg, range_m, impact)
    lay_line = _sample_map_line(launch, lay_bearing, 20000)
    ring = []
    for index in range(97):
        latitude, longitude = zf.vPolar(launch_lat, launch_lon, index * 360 / 96, 20000)
        ring.append([float(longitude), float(latitude)])

    overlays = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "gtl"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(route)}},
            {"type": "Feature", "properties": {"kind": "lay"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(lay_line)}},
            {"type": "Feature", "properties": {"kind": "range"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(ring)}},
            {"type": "Feature", "properties": {"kind": "launch", "label": "Launch Point"}, "geometry": {"type": "Point", "coordinates": [launch_lon, launch_lat]}},
            {"type": "Feature", "properties": {"kind": "impact", "label": "Impact Point"}, "geometry": {"type": "Point", "coordinates": [impact_lon, impact_lat]}},
        ],
    }
    launch_json = json.dumps([launch_lon, launch_lat])
    overlays_json = json.dumps(overlays)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
<style>html,body,#af-map{{width:100%;height:100%;margin:0;overflow:hidden}} .map-error{{position:absolute;z-index:2;left:12px;top:12px;padding:8px 12px;background:#fff;color:#222;font:13px sans-serif;border-radius:4px}}</style>
</head><body><div id="af-map"></div>
<script type="module">
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';
const overlays = {overlays_json};
const map = new maplibregl.Map({{
  container:'af-map', style:'https://tiles.openfreemap.org/styles/liberty',
  center:{launch_json}, zoom:9, pitch:40, maxPitch:85, dragRotate:true, touchZoomRotate:true
}});
map.addControl(new maplibregl.NavigationControl({{visualizePitch:true}}), 'top-right');
map.addControl(new maplibregl.ScaleControl({{maxWidth:120, unit:'metric'}}), 'bottom-left');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
map.once('style.load', () => {{
  map.setProjection({{type:'globe'}});
  map.addSource('af-elevation', {{type:'raster-dem', url:'https://tiles.mapterhorn.com/tilejson.json'}});
  const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
  map.addLayer({{id:'af-hillshade',type:'hillshade',source:'af-elevation',paint:{{'hillshade-exaggeration':0.35}}}}, firstLabel && firstLabel.id);
  map.setTerrain({{source:'af-elevation',exaggeration:3.0}});
  map.setSky({{'sky-color':'#081522','sky-horizon-blend':0.35,'atmosphere-blend':0.8}});
  map.addSource('af-overlays', {{type:'geojson',data:overlays}});
  map.addLayer({{id:'af-range',type:'line',source:'af-overlays',filter:['==',['get','kind'],'range'],paint:{{'line-color':'#dc2626','line-width':2,'line-opacity':0.75}}}});
  map.addLayer({{id:'af-lay',type:'line',source:'af-overlays',filter:['==',['get','kind'],'lay'],paint:{{'line-color':'#1976d2','line-width':3}}}});
  map.addLayer({{id:'af-gtl',type:'line',source:'af-overlays',filter:['==',['get','kind'],'gtl'],paint:{{'line-color':'#e32636','line-width':3}}}});
  map.addLayer({{id:'af-points',type:'circle',source:'af-overlays',filter:['in',['get','kind'],['literal',['launch','impact']]],paint:{{'circle-radius':8,'circle-color':['match',['get','kind'],'launch','#16803c','#d62828'],'circle-stroke-color':'#fff','circle-stroke-width':2}}}});
  map.addLayer({{id:'af-labels',type:'symbol',source:'af-overlays',filter:['in',['get','kind'],['literal',['launch','impact']]],layout:{{'text-field':['get','label'],'text-offset':[0,1.2],'text-size':13,'text-allow-overlap':true}},paint:{{'text-color':'#111827','text-halo-color':'#fff','text-halo-width':1.5}}}});
  const anchorLon = overlays.features.find((feature) => feature.properties.kind === 'launch').geometry.coordinates[0];
  const extent = overlays.features.filter((feature) => feature.geometry.type === 'MultiLineString').flatMap((feature) => {{
    let previousLon = anchorLon;
    return feature.geometry.coordinates.flat().map(([longitude, latitude]) => {{
      let adjustedLon = longitude;
      while (adjustedLon - previousLon > 180) adjustedLon -= 360;
      while (adjustedLon - previousLon < -180) adjustedLon += 360;
      previousLon = adjustedLon;
      return [adjustedLon, latitude];
    }});
  }});
  const bounds = extent.reduce((box, coordinate) => box.extend(coordinate), new maplibregl.LngLatBounds(extent[0], extent[0]));
  map.fitBounds(bounds, {{padding:55,maxZoom:12,pitch:45,duration:0}});
  map.on('click', (event) => new maplibregl.Popup().setLngLat(event.lngLat).setText(`${{event.lngLat.lat.toFixed(5)}}, ${{event.lngLat.lng.toFixed(5)}}`).addTo(map));
}});
map.on('error', (event) => {{
  if (document.querySelector('.map-error')) return;
  const message = document.createElement('div'); message.className = 'map-error';
  const source = event.sourceId || 'map';
  const detail = event.error && event.error.message ? `: ${{event.error.message}}` : '';
  message.textContent = `${{source}} could not load${{detail}}`;
  document.body.appendChild(message);
}});
</script>
<div style="position:absolute;right:6px;bottom:4px;z-index:3;background:#ffffffdd;padding:3px 6px;font:11px sans-serif;color:#222">
<a href="https://maplibre.org/" target="_blank" rel="noopener">MapLibre</a> |
<a href="https://mapterhorn.com/attribution" target="_blank" rel="noopener">© Mapterhorn</a> |
<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> |
<a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">© OpenMapTiles</a> |
Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>
</div></body></html>"""


def _missing_solver_message():
    expected = ("MLAHshot", "MLALshot", "projectile_motion_with_drag")
    missing = [name for name in expected if not callable(getattr(zf, name, None))]
    if not missing:
        return None
    return (
        "The map and range are available, but the ballistic solution and trajectory "
        "cannot be calculated because these functions are missing from "
        f"SubApps/z_functions.py: {', '.join(missing)}."
    )


@callback(
    Output("af-map-iframe", "srcDoc"),
    Output("af-solution-output", "children"),
    Output("af-trajectory-graph", "figure"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
    Input("af-launch-altitude", "value"),
    Input("af-impact-altitude", "value"),
    Input("af-mass", "value"),
    Input("af-velocity", "value"),
    Input("af-aol", "value"),
    Input("af-angle-mode", "value"),
)
def update_shot(launch_mgrs, impact_mgrs, launch_alt, impact_alt, mass, velocity, aol, angle_mode):
    empty_figure = go.Figure()
    if not launch_mgrs or not impact_mgrs or mass is None or velocity is None or aol is None:
        return "", "Enter valid coordinates and projectile parameters.", empty_figure

    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        launch_lat, launch_lon = float(launch[1]), float(launch[2])
        impact_lat, impact_lon = float(impact[1]), float(impact[2])
        range_m, azimuth_deg = zf.LLDist(launch_lat, launch_lon, impact_lat, impact_lon)
        map_html = _make_map(launch, impact, aol, range_m, azimuth_deg)
        launch_mils = azimuth_deg * 3200 / 180
        diagnostics = html.Div(
            [
                html.P(f"Range: {range_m:,.0f} m", className="mb-1"),
                html.P(f"Launch Bearing: {azimuth_deg:.2f}°", className="mb-1"),
                html.P(f"Launch Azimuth: {launch_mils:.2f} mils", className="mb-1"),
            ],
            className="mb-2",
        )

        missing_message = _missing_solver_message()
        if missing_message:
            return map_html, html.Div([diagnostics, dbc.Alert(missing_message, color="warning")]), empty_figure

        high_angle = angle_mode == "high"
        solver = zf.MLAHshot if high_angle else zf.MLALshot
        solution = solver(
            launch_lat,
            float(launch_alt),
            float(impact_alt),
            float(mass),
            launch_mils,
            float(aol),
            float(velocity),
            range_m,
        )
        mass_kg = float(mass) / 2.2
        diameter_mm = 155
        area = math.pi * (diameter_mm / 2000) ** 2
        spin_rate = float(velocity) / (20 * diameter_mm / 1000)
        spin_velocity = spin_rate * 2 * math.pi
        elevation_deg = float(solution[0]) / 3200 * 180
        motion = zf.projectile_motion_with_drag(
            float(velocity), elevation_deg, mass_kg, 0.25, area,
            launch_lat, azimuth_deg, float(launch_alt), float(launch_alt), spin_velocity,
        )
        time_of_flight, final_x, final_y, final_z, times, x_vals, y_vals, z_vals, vx, vy, vz, v_mag, qe = motion
        import pandas as pd

        trajectory = pd.DataFrame({"time": times, "distance": x_vals, "height": y_vals})
        trajectory["distance"] = trajectory["distance"] / max(trajectory["distance"]) * range_m
        trajectory["height"] = (
            trajectory["height"] - float(launch_alt)
        ) / max(trajectory["height"] - float(launch_alt)) * (float(solution[1]) - float(launch_alt)) + float(launch_alt)
        figure = px.line(
            trajectory,
            x="distance",
            y="height",
            title="Projectile trajectory",
            labels={"distance": "Distance (m)", "height": "Height (m)"},
        )
        figure.update_layout(height=350, margin={"l": 30, "r": 20, "t": 45, "b": 35})

        output = html.Div(
            [
                diagnostics,
                html.P(f"Max Ord: {int(solution[1])} m", className="mb-1"),
                html.P(f"Time of Flight: {int(solution[2])} s", className="mb-1"),
                html.P(f"QE: {int(solution[0])} mils", className="mb-1"),
                html.P(f"Deflection: {int(solution[4])} mils", className="mb-1"),
                html.P(f"Drift: {int(solution[3])} mils", className="mb-1"),
                html.P(f"Final Velocity: {int(v_mag[-10])} m/s", className="mb-1"),
                html.P(f"Final QE: {int(qe[-10])} mils", className="mb-0"),
            ]
        )
        return map_html, output, figure
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate shot: {exc}", color="danger", className="mb-0"), empty_figure
