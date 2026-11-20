"""Interactive 3D world globe with point-to-point MGRS planning."""

import json

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from SubApps import z_functions as zf


dash.register_page(__name__, path="/b_points", name="Point to Point", order=2)

DEFAULT_LAUNCH_MGRS = "12RWU1059022575"
DEFAULT_IMPACT_MGRS = "31UDQ5248411718"

WORLD_GLOBE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
  <style>
    html, body, #world-map { width:100%; height:100%; margin:0; overflow:hidden; }
    .map-error {
      position:absolute; z-index:2; top:12px; left:12px; max-width:440px;
      padding:9px 12px; border-radius:4px; background:#fff; color:#222;
      font:14px sans-serif; box-shadow:0 1px 5px #0004;
    }
  </style>
</head>
<body>
  <div id="world-map"></div>
  <script type="module">
    import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';

    const routeData = __ROUTE_DATA__;
    const map = new maplibregl.Map({
      container: 'world-map',
      style: 'https://tiles.openfreemap.org/styles/liberty',
      center: [0, 15],
      zoom: 1.35,
      pitch: 0,
      dragRotate: true,
      touchZoomRotate: true,
      maxPitch: 85,
      attributionControl: true
    });

    map.addControl(new maplibregl.NavigationControl({visualizePitch: true}), 'top-right');
    map.addControl(new maplibregl.ScaleControl({maxWidth: 120, unit: 'metric'}), 'bottom-left');

    map.once('style.load', () => {
      map.setProjection({type: 'globe'});
      map.addSource('elevation-dem', {
        type: 'raster-dem',
        url: 'https://tiles.mapterhorn.com/tilejson.json'
      });
      const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
      map.addLayer({
        id: 'elevation-hillshade',
        type: 'hillshade',
        source: 'elevation-dem',
        paint: {'hillshade-exaggeration': 0.35}
      }, firstLabel && firstLabel.id);
      map.setTerrain({source: 'elevation-dem', exaggeration: 3.0});
      map.setSky({
        'sky-color': '#081522',
        'sky-horizon-blend': 0.35,
        'atmosphere-blend': 0.8
      });

      if (routeData) {
        map.addSource('point-to-point', {type: 'geojson', data: routeData});
        map.addLayer({
          id: 'point-to-point-route',
          type: 'line',
          source: 'point-to-point',
          filter: ['==', ['get', 'kind'], 'route'],
          paint: {'line-color': '#e32636', 'line-width': 4, 'line-opacity': 0.95}
        });
        map.addLayer({
          id: 'point-to-point-markers',
          type: 'circle',
          source: 'point-to-point',
          filter: ['!=', ['get', 'kind'], 'route'],
          paint: {
            'circle-radius': 7,
            'circle-color': ['match', ['get', 'kind'], 'launch', '#16a34a', '#dc2626'],
            'circle-stroke-color': '#fff',
            'circle-stroke-width': 2
          }
        });
        map.addLayer({
          id: 'point-to-point-labels',
          type: 'symbol',
          source: 'point-to-point',
          filter: ['!=', ['get', 'kind'], 'route'],
          layout: {
            'text-field': ['get', 'label'],
            'text-offset': [0, 1.2],
            'text-size': 13,
            'text-allow-overlap': true
          },
          paint: {'text-color': '#111827', 'text-halo-color': '#fff', 'text-halo-width': 1.5}
        });

        const route = routeData.features.find((feature) => feature.properties.kind === 'route');
        const rawCoordinates = route.geometry.coordinates.flat();
        const launchFeature = routeData.features.find((feature) => feature.properties.kind === 'launch');
        let previousLongitude = launchFeature.geometry.coordinates[0];
        const coordinates = rawCoordinates.map(([longitude, latitude]) => {
          let adjustedLongitude = longitude;
          while (adjustedLongitude - previousLongitude > 180) adjustedLongitude -= 360;
          while (adjustedLongitude - previousLongitude < -180) adjustedLongitude += 360;
          previousLongitude = adjustedLongitude;
          return [adjustedLongitude, latitude];
        });
        const bounds = coordinates.reduce(
          (current, coordinate) => current.extend(coordinate),
          new maplibregl.LngLatBounds(coordinates[0], coordinates[0])
        );
        map.fitBounds(bounds, {padding: 70, maxZoom: 12, pitch: 45, duration: 0});
      }
    });

    map.on('error', (event) => {
      if (!document.querySelector('.map-error')) {
        const message = document.createElement('div');
        message.className = 'map-error';
        const sourceId = event.sourceId || '';
        const detail = event.error && event.error.message ? ` ${event.error.message}` : '';
        message.textContent = sourceId === 'elevation-dem'
          ? `Terrain elevation tiles could not load.${detail}`
          : sourceId
            ? `Map source "${sourceId}" could not load.${detail}`
            : `A map resource could not load.${detail}`;
        document.body.appendChild(message);
      }
    });
  </script>
</body>
</html>"""


def _world_globe_html(route_data=None):
    return WORLD_GLOBE_TEMPLATE.replace("__ROUTE_DATA__", json.dumps(route_data))


layout = dbc.Container(
    [
        html.H2("Point to Point", className="mb-3"),
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
                                                    id="world-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="world-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="world-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Point coordinates"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(id="world-launch-mgrs", type="text", value=DEFAULT_LAUNCH_MGRS, className="mb-2"),
                                        html.Div(id="world-launch-ll", className="text-muted mb-3"),
                                        dbc.Label("Impact Point (MGRS)"),
                                        dbc.Input(id="world-impact-mgrs", type="text", value=DEFAULT_IMPACT_MGRS, className="mb-2"),
                                        html.Div(id="world-impact-ll", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Point-to-point details"),
                                dbc.CardBody(html.Div(id="world-details-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    xl=3,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("OpenStreetMap, terrain, and route"),
                            dbc.CardBody(
                                dcc.Loading(
                                    html.Iframe(
                                        id="world-globe-frame",
                                        srcDoc=_world_globe_html(),
                                        title="Interactive 3D globe and point-to-point route",
                                        style={"width": "100%", "height": "78vh", "border": "none"},
                                    )
                                )
                            ),
                        ]
                    ),
                    xs=12,
                    xl=9,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


@callback(
    Output("world-lookup-output", "children"),
    Input("world-lookup-button", "n_clicks"),
    State("world-lookup", "value"),
    prevent_initial_call=True,
)
def process_world_lookup(n_clicks, query):
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


def _route_points(start, bearing, distance, end, segments=100):
    """Sample the geodesic route from launch to impact."""
    if distance == 0:
        return [[float(start[2]), float(start[1])], [float(end[2]), float(end[1])]]
    points = []
    for index in range(segments + 1):
        latitude, longitude = zf.vPolar(
            float(start[1]), float(start[2]), bearing, distance * index / segments
        )
        points.append([float(longitude), float(latitude)])
    points[-1] = [float(end[2]), float(end[1])]
    return points


@callback(
    Output("world-globe-frame", "srcDoc"),
    Output("world-details-output", "children"),
    Output("world-launch-ll", "children"),
    Output("world-impact-ll", "children"),
    Input("world-launch-mgrs", "value"),
    Input("world-impact-mgrs", "value"),
)
def update_world_route(launch_mgrs, impact_mgrs):
    if not launch_mgrs or not impact_mgrs:
        return _world_globe_html(), "Enter both MGRS coordinates.", "", ""
    try:
        launch = zf.MGRS2LL(launch_mgrs.strip())
        impact = zf.MGRS2LL(impact_mgrs.strip())
        distance, launch_bearing = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        _, impact_bearing = zf.LLDist(impact[1], impact[2], launch[1], launch[2])
        launch_mils = launch_bearing * 3200 / 180
        impact_mils = impact_bearing * 3200 / 180
        route_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"kind": "route"},
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": zf.split_antimeridian_line(
                            _route_points(launch, launch_bearing, distance, impact)
                        ),
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"kind": "launch", "label": "Launch Point"},
                    "geometry": {"type": "Point", "coordinates": [float(launch[2]), float(launch[1])]},
                },
                {
                    "type": "Feature",
                    "properties": {"kind": "impact", "label": "Impact Point"},
                    "geometry": {"type": "Point", "coordinates": [float(impact[2]), float(impact[1])]},
                },
            ],
        }
        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance: "), f"{round(distance):,} meters"], xs=12),
                dbc.Col([html.Strong("Launch Bearing: "), f"{launch_bearing:.2f}°"], xs=12),
                dbc.Col([html.Strong("Launch Azimuth: "), f"{launch_mils:.2f} mils"], xs=12),
                dbc.Col([html.Strong("Impact Bearing: "), f"{impact_bearing:.2f}°"], xs=12),
                dbc.Col([html.Strong("Impact Azimuth: "), f"{impact_mils:.2f} mils"], xs=12),
            ],
            className="g-2",
        )
        launch_ll = f"Launch Point (LL): {float(launch[1]):.5f}, {float(launch[2]):.5f}"
        impact_ll = f"Impact Point (LL): {float(impact[1]):.5f}, {float(impact[2]):.5f}"
        return _world_globe_html(route_data), details, launch_ll, impact_ll
    except Exception as exc:
        return _world_globe_html(), dbc.Alert(f"Could not calculate route: {exc}", color="danger", className="mb-0"), "", ""
