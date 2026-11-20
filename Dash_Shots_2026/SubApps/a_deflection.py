import json

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Import helper functions module
from SubApps import z_functions as zf

# 1. Register page in Dash registry
dash.register_page(
    __name__,
    path="/",  # Default home route or change to '/deflection'
    name="Deflection",
    order=1
)

# 2. Page Layout definition
layout = dbc.Container(
    [
        html.H2("Deflection", className="mb-4"),
        dbc.Row(
            [
                # Left Column (Controls & Inputs)
                dbc.Col(
                    [
                        # Coordinates Block
                        dbc.Card(
                            [
                                dbc.CardHeader("Coordinates Lookup & Conversion"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Latitude"),
                                                        dbc.Input(id="def-lat", type="number", value=-30, step="any"),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Longitude"),
                                                        dbc.Input(id="def-lon", type="number", value=30, step="any"),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        html.Div(id="def-ll-converted-output", className="mb-3 text-muted"),
                                        html.Hr(),
                                        dbc.Label("Lookup Identifier"),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="def-lookup", type="text", placeholder="Enter a place or address..."),
                                                dbc.Button("Search", id="def-lookup-button", n_clicks=0),
                                            ]
                                        ),
                                        html.Div(id="def-lookup-output", className="mt-2 text-info"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Tactical Inputs Block
                        dbc.Card(
                            [
                                dbc.CardHeader("Tactical Parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Azimuth of Fire (mils):"),
                                        dbc.Input(id="def-aof", type="number", value=2000, className="mb-2"),
                                        
                                        dbc.Label("Start Point (MGRS):"),
                                        dbc.Input(id="def-lpmgrs", type="text", value="", className="mb-2"),
                                        html.Div(id="def-start-ll-output", className="text-muted mb-3"),
                                        
                                        dbc.Label("End Point (MGRS):"),
                                        dbc.Input(id="def-ipmgrs", type="text", value="", className="mb-3"),
                                        
                                        html.Div(id="def-tp-output", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    width=4,
                ),
                # Right Column (Map & Tactical Calculations)
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Interactive Tactical Map"),
                                dbc.CardBody(
                                    [
                                        html.Iframe(
                                            id="def-map-iframe",
                                            title="OpenStreetMap tactical map with terrain",
                                            style={"width": "100%", "height": "500px", "border": "none"},
                                        )
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Fire Control Calculations"),
                                dbc.CardBody(html.Div(id="def-details-output")),
                            ]
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

# Callback 1: Automatically set MGRS inputs when Lat/Lon changes
@callback(
    Output("def-ll-converted-output", "children"),
    Output("def-lpmgrs", "value"),
    Output("def-ipmgrs", "value"),
    Input("def-lat", "value"),
    Input("def-lon", "value"),
)
def update_initial_mgrs(lat, lon):
    if lat is None or lon is None:
        return "", "", ""
    try:
        back = zf.LL2MGRS(float(lat), float(lon))
        ip = zf.LL2MGRS(float(lat) + 0.1, float(lon) + 0.1)

        info_text = html.Div(
            [
                html.P(f"UTM: {back[0]}", className="mb-1"),
                html.P(f"MGRS: {back[1]}", className="mb-0"),
            ]
        )
        return info_text, back[1], ip[1]
    except Exception as e:
        return f"Conversion error: {e}", "", ""


@callback(
    Output("def-start-ll-output", "children"),
    Input("def-lpmgrs", "value"),
)
def display_start_point_coordinates(lpmgrs):
    if not lpmgrs:
        return ""
    try:
        start = zf.MGRS2LL(lpmgrs)
        return f"Start Point (LL): {round(start[1], 5)}, {round(start[2], 5)}"
    except Exception as e:
        return f"Invalid MGRS: {e}"


# Callback 2: Optional Lookup functionality
@callback(
    Output("def-lookup-output", "children"),
    Input("def-lookup-button", "n_clicks"),
    State("def-lookup", "value"),
)
def process_lookup(n_clicks, lu):
    if not n_clicks or not lu or not lu.strip():
        return ""
    try:
        where = zf.lookup(lu)
        alt = zf.elevation(where[1], where[2])
        return html.Div(
            [
                html.P(where[0], className="mb-0 fw-bold"),
                html.P(f"Lat: {where[1]} | Lon: {where[2]}", className="mb-0"),
                html.P(f"MGRS: {where[3]}", className="mb-0"),
                html.P(f"Alt: {round(alt, 2)} Meters", className="mb-0"),
            ]
        )
    except Exception as e:
        return f"Lookup error: {e}"


# Callback 3: Update Map & Fire Control Metrics
def _sample_geodesic(start, bearing, distance, end=None, segments=80):
    """Return [longitude, latitude] samples along a geodesic."""
    if distance <= 0:
        return [[float(start[2]), float(start[1])]]
    points = []
    for index in range(segments + 1):
        latitude, longitude = zf.vPolar(
            float(start[1]), float(start[2]), bearing, distance * index / segments
        )
        points.append([float(longitude), float(latitude)])
    if end is not None:
        points[-1] = [float(end[2]), float(end[1])]
    return points


def _make_map_html(launch, impact, launch_mgrs, impact_mgrs, azimuth_mils, distance, bearing):
    azimuth_bearing = float(azimuth_mils) * 180 / 3200
    route = _sample_geodesic(launch, bearing, distance, impact)
    azimuth = _sample_geodesic(launch, azimuth_bearing, 30000)
    ring = []
    for index in range(97):
        latitude, longitude = zf.vPolar(
            float(launch[1]), float(launch[2]), index * 360 / 96, 30000
        )
        ring.append([float(longitude), float(latitude)])

    overlays = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "azimuth"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(azimuth)}},
            {"type": "Feature", "properties": {"kind": "gtl"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(route)}},
            {"type": "Feature", "properties": {"kind": "range"}, "geometry": {"type": "MultiLineString", "coordinates": zf.split_antimeridian_line(ring)}},
            {"type": "Feature", "properties": {"kind": "start", "label": "Start Point", "mgrs": launch_mgrs}, "geometry": {"type": "Point", "coordinates": [float(launch[2]), float(launch[1])]}},
            {"type": "Feature", "properties": {"kind": "end", "label": "End Point", "mgrs": impact_mgrs}, "geometry": {"type": "Point", "coordinates": [float(impact[2]), float(impact[1])]}},
        ],
    }
    center = json.dumps([float(launch[2]), float(launch[1])])
    overlays_json = json.dumps(overlays)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
html,body,#def-map{{width:100%;height:100%;margin:0;overflow:hidden}}
.map-error{{position:absolute;z-index:2;left:12px;top:12px;padding:8px 12px;background:#fff;color:#222;font:13px sans-serif;border-radius:4px}}
</style></head><body><div id="def-map"></div>
<script type="module">
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';
const overlays = {overlays_json};
const map = new maplibregl.Map({{
  container:'def-map', style:'https://tiles.openfreemap.org/styles/liberty',
  center:{center}, zoom:9, pitch:45, maxPitch:85, dragRotate:true, touchZoomRotate:true
}});
map.addControl(new maplibregl.NavigationControl({{visualizePitch:true}}), 'top-right');
map.addControl(new maplibregl.ScaleControl({{maxWidth:120, unit:'metric'}}), 'bottom-left');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
map.on('style.load', () => {{
  map.setProjection({{type:'globe'}});
  map.addSource('def-elevation', {{type:'raster-dem', url:'https://tiles.mapterhorn.com/tilejson.json'}});
  const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
  map.addLayer({{
    id:'def-hillshade', type:'hillshade', source:'def-elevation',
    paint:{{'hillshade-exaggeration':0.35}}
  }}, firstLabel && firstLabel.id);
  map.setTerrain({{source:'def-elevation', exaggeration:3.0}});
  map.setSky({{'sky-color':'#081522', 'sky-horizon-blend':0.35, 'atmosphere-blend':0.8}});
  map.addSource('def-overlays', {{type:'geojson', data:overlays}});
  map.addLayer({{
    id:'def-range-ring', type:'line', source:'def-overlays',
    filter:['==',['get','kind'],'range'],
    paint:{{'line-color':'#14833b','line-width':2,'line-opacity':0.8}}
  }});
  map.addLayer({{
    id:'def-azimuth', type:'line', source:'def-overlays',
    filter:['==',['get','kind'],'azimuth'],
    paint:{{'line-color':'#1976d2','line-width':3}}
  }});
  map.addLayer({{
    id:'def-gtl', type:'line', source:'def-overlays',
    filter:['==',['get','kind'],'gtl'],
    paint:{{'line-color':'#e32636','line-width':3}}
  }});
  map.addLayer({{
    id:'def-points', type:'circle', source:'def-overlays',
    filter:['in',['get','kind'],['literal',['start','end']]],
    paint:{{'circle-radius':8,'circle-color':['match',['get','kind'],'start','#16803c','#d62828'],'circle-stroke-color':'#fff','circle-stroke-width':2}}
  }});
  map.addLayer({{
    id:'def-labels', type:'symbol', source:'def-overlays',
    filter:['in',['get','kind'],['literal',['start','end']]],
    layout:{{'text-field':['get','label'],'text-offset':[0,1.2],'text-size':13,'text-allow-overlap':true}},
    paint:{{'text-color':'#111827','text-halo-color':'#fff','text-halo-width':1.5}}
  }});
  const launchFeature = overlays.features.find((feature) => feature.properties.kind === 'start');
  const anchorLongitude = launchFeature.geometry.coordinates[0];
  const extent = overlays.features
    .filter((feature) => feature.geometry.type === 'MultiLineString')
    .flatMap((feature) => {{
      let previousLongitude = anchorLongitude;
      return feature.geometry.coordinates.flat().map(([longitude, latitude]) => {{
        let adjustedLongitude = longitude;
        while (adjustedLongitude - previousLongitude > 180) adjustedLongitude -= 360;
        while (adjustedLongitude - previousLongitude < -180) adjustedLongitude += 360;
        previousLongitude = adjustedLongitude;
        return [adjustedLongitude, latitude];
      }});
    }});
  const bounds = extent.reduce((box, coordinate) => box.extend(coordinate), new maplibregl.LngLatBounds(extent[0], extent[0]));
  map.fitBounds(bounds, {{padding:55, maxZoom:12, pitch:45, duration:0}});
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
</div>
</body></html>"""


@callback(
    Output("def-map-iframe", "srcDoc"),
    Output("def-tp-output", "children"),
    Output("def-details-output", "children"),
    Input("def-aof", "value"),
    Input("def-lpmgrs", "value"),
    Input("def-ipmgrs", "value"),
)
def update_map_and_metrics(aof, lpmgrs, ipmgrs):
    if not lpmgrs or not ipmgrs or aof is None:
        return "", "", ""

    try:
        aof = float(aof)
        lp = zf.MGRS2LL(lpmgrs)
        ip = zf.MGRS2LL(ipmgrs)

        # Tactical Point Info Output
        tp_info = html.Div(
            [
                html.P(f"UTM: {ip[0]}", className="mb-1"),
                html.P(f"End Point (LL): {round(ip[1], 5)}, {round(ip[2], 5)}", className="mb-0"),
            ]
        )

        # Calculation Details
        deets = zf.LLDist(lp[1], lp[2], ip[1], ip[2])
        bearing_deg = deets[1]
        map_html = _make_map_html(lp, ip, lpmgrs, ipmgrs, aof, deets[0], bearing_deg)
        azimuth_mils = bearing_deg * 3200 / 180
        diff = round(aof - azimuth_mils + 3200, 0)
        if diff < 0:
            diff += 6400

        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance:"), html.Span(f" {int(deets[0])} meters")], width=6),
                dbc.Col([html.Strong("Bearing:"), html.Span(f" {round(bearing_deg, 2)}°")], width=6),
                dbc.Col([html.Strong("Azimuth:"), html.Span(f" {round(azimuth_mils, 2)} mils")], width=6),
                dbc.Col([html.Strong("Deflection:"), html.Span(f" {diff} mils", className="text-danger fw-bold")], width=6),
            ],
            className="g-2",
        )

        return map_html, tp_info, details

    except Exception as e:
        return "", f"Error computing metrics: {e}", ""
