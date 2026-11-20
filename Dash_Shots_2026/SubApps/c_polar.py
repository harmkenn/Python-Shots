"""Polar shot planning page for the Dash Shots app."""

import json
import math

import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from SubApps import z_functions as zf


dash.register_page(__name__, path="/c_polar", name="Polar Shot", order=3)


layout = dbc.Container(
    [
        html.H2("Polar Shot", className="mb-4"),
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
                                                    id="cp-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="cp-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="cp-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Launch parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="cp-launch-mgrs",
                                            type="text",
                                            value="12RWU1059022575",
                                            className="mb-2",
                                        ),
                                        html.Div(id="cp-launch-ll", className="text-muted mb-3"),
                                        dbc.Label("Azimuth (mils)"),
                                        dbc.Input(
                                            id="cp-azimuth",
                                            type="number",
                                            value=1600,
                                            step="any",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Distance (meters)"),
                                        dbc.Input(
                                            id="cp-distance",
                                            type="number",
                                            value=10000000,
                                            min=0,
                                            step="any",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Polar shot details"),
                                dbc.CardBody(html.Div(id="cp-details-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Trajectory map"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="cp-map-iframe",
                                    title="Map showing the polar shot trajectory",
                                    style={"width": "100%", "height": "650px", "border": "none"},
                                )
                            ),
                        ]
                    ),
                    xs=12,
                    lg=8,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


@callback(
    Output("cp-lookup-output", "children"),
    Input("cp-lookup-button", "n_clicks"),
    State("cp-lookup", "value"),
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
    Output("cp-launch-ll", "children"),
    Input("cp-launch-mgrs", "value"),
)
def display_launch_coordinates(mgrs):
    if not mgrs:
        return ""
    try:
        point = zf.MGRS2LL(mgrs)
        return f"Launch Point (LL): {point[1]:.5f}, {point[2]:.5f}"
    except Exception as exc:
        return f"Invalid MGRS: {exc}"


def _trajectory_points(launch, bearing, distance, impact, segments=100):
    """Sample the geodesic route as GeoJSON [longitude, latitude] coordinates."""
    if distance <= 0:
        return [[float(launch[2]), float(launch[1])], [float(impact[1]), float(impact[0])]]
    points = []
    for index in range(segments + 1):
        lat, lon = zf.vPolar(launch[1], launch[2], bearing, distance * index / segments)
        points.append([float(lon), float(lat)])
    points[-1] = [float(impact[1]), float(impact[0])]
    return points


def _map_html(route_data, center, zoom):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
html,body,#polar-map{{width:100%;height:100%;margin:0;overflow:hidden}}
.map-error{{position:absolute;z-index:2;left:12px;top:12px;padding:8px 12px;background:#fff;color:#222;font:13px sans-serif;border-radius:4px}}
</style></head><body><div id="polar-map"></div>
<script type="module">
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';
const shot = {json.dumps(route_data)};
const map = new maplibregl.Map({{
  container:'polar-map', style:'https://tiles.openfreemap.org/styles/liberty',
  center:{json.dumps(center)}, zoom:{zoom:.2f}, pitch:40, maxPitch:85,
  dragRotate:true, touchZoomRotate:true
}});
map.addControl(new maplibregl.NavigationControl({{visualizePitch:true}}), 'top-right');
map.addControl(new maplibregl.ScaleControl({{maxWidth:120, unit:'metric'}}), 'bottom-left');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
map.once('style.load', () => {{
  map.setProjection({{type:'globe'}});
  map.addSource('polar-elevation', {{type:'raster-dem', url:'https://tiles.mapterhorn.com/tilejson.json'}});
  const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
  map.addLayer({{
    id:'polar-hillshade', type:'hillshade', source:'polar-elevation',
    paint:{{'hillshade-exaggeration':0.35}}
  }}, firstLabel && firstLabel.id);
  map.setTerrain({{source:'polar-elevation', exaggeration:3.0}});
  map.setSky({{'sky-color':'#081522','sky-horizon-blend':0.35,'atmosphere-blend':0.8}});
  map.addSource('polar-shot', {{type:'geojson', data:shot}});
  map.addLayer({{
    id:'polar-trajectory', type:'line', source:'polar-shot',
    filter:['==',['get','kind'],'trajectory'],
    paint:{{'line-color':'#e32636','line-width':4,'line-opacity':0.95}}
  }});
  map.addLayer({{
    id:'polar-points', type:'circle', source:'polar-shot',
    filter:['!=',['get','kind'],'trajectory'],
    paint:{{'circle-radius':8,'circle-color':['match',['get','kind'],'launch','#16a34a','#dc2626'],'circle-stroke-color':'#fff','circle-stroke-width':2}}
  }});
  map.addLayer({{
    id:'polar-labels', type:'symbol', source:'polar-shot',
    filter:['!=',['get','kind'],'trajectory'],
    layout:{{'text-field':['get','label'],'text-offset':[0,1.2],'text-size':13,'text-allow-overlap':true}},
    paint:{{'text-color':'#111827','text-halo-color':'#fff','text-halo-width':1.5}}
  }});
}});
map.on('error', (event) => {{
  if (document.querySelector('.map-error')) return;
  const message = document.createElement('div');
  message.className = 'map-error';
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


@callback(
    Output("cp-map-iframe", "srcDoc"),
    Output("cp-details-output", "children"),
    Input("cp-launch-mgrs", "value"),
    Input("cp-azimuth", "value"),
    Input("cp-distance", "value"),
)
def update_map_and_details(launch_mgrs, azimuth_mils, distance_meters):
    if not launch_mgrs or azimuth_mils is None or distance_meters is None:
        return "", "Enter a launch point, azimuth, and distance."

    try:
        azimuth_mils = float(azimuth_mils)
        distance_meters = float(distance_meters)
        if distance_meters < 0:
            raise ValueError("Distance must be zero or greater.")

        launch = zf.MGRS2LL(launch_mgrs)
        launch_lat, launch_lon = float(launch[1]), float(launch[2])
        launch_bearing = azimuth_mils * 180 / 3200
        impact_lat, impact_lon = zf.vPolar(
            launch_lat, launch_lon, launch_bearing, distance_meters
        )
        _, impact_bearing = zf.LLDist(impact_lat, impact_lon, launch_lat, launch_lon)
        impact = zf.LL2MGRS(impact_lat, impact_lon)
        impact_mgrs = impact[1]
        impact_mils = impact_bearing * 3200 / 180

        # Center across the antimeridian using the shorter longitude interval.
        lon_delta = ((impact_lon - launch_lon + 180) % 360) - 180
        zoom = max(2, min(15, -1.36 * math.log(max(distance_meters, 1) / 1000) + 15))
        center = [((launch_lon + lon_delta / 2 + 180) % 360) - 180, (launch_lat + impact_lat) / 2]
        route_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"kind": "trajectory"},
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": zf.split_antimeridian_line(
                            _trajectory_points(
                                launch, launch_bearing, distance_meters, (impact_lat, impact_lon)
                            )
                        ),
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"kind": "launch", "label": "Launch Point", "mgrs": launch_mgrs},
                    "geometry": {"type": "Point", "coordinates": [launch_lon, launch_lat]},
                },
                {
                    "type": "Feature",
                    "properties": {"kind": "impact", "label": "Impact Point", "mgrs": impact_mgrs},
                    "geometry": {"type": "Point", "coordinates": [impact_lon, impact_lat]},
                },
            ],
        }
        map_html = _map_html(route_data, center, zoom)

        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance: "), f"{distance_meters:,.2f} meters"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Bearing: "), f"{launch_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Azimuth: "), f"{azimuth_mils:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Bearing: "), f"{impact_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Azimuth: "), f"{impact_mils:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Location (MGRS): "), impact_mgrs], xs=12),
                dbc.Col(
                    [html.Strong("Impact Point (LL): "), f"{impact_lat:.5f}, {impact_lon:.5f}"],
                    xs=12,
                ),
            ],
            className="g-2",
        )
        return map_html, details
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate polar shot: {exc}", color="danger", className="mb-0")
