"""Small shared MapLibre HTML renderer for Dash iframe maps."""

import json


def _script_json(value):
    """Serialize JSON safely for use in an inline script."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_geojson_map(features, center, bounds=None, zoom=5, pitch=40):
    """Render point and route GeoJSON over MapLibre terrain."""
    geojson = _script_json({"type": "FeatureCollection", "features": features})
    center_json = _script_json([float(center[1]), float(center[0])])
    bounds_json = _script_json(bounds or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
html,body,#map{{width:100%;height:100%;margin:0;overflow:hidden}}
.map-attribution{{position:absolute;right:6px;bottom:4px;z-index:3;background:#ffffffdf;padding:4px 7px;font:11px sans-serif;color:#222}}
.map-attribution a{{color:#1558a6;text-decoration:none}}
.map-error{{position:absolute;z-index:4;left:12px;top:12px;padding:8px 12px;background:#fff;color:#222;font:13px sans-serif;border-radius:4px}}
</style></head><body><div id="map"></div>
<script type="module">
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.19.0/dist/maplibre-gl.mjs';
const data = {geojson};
const center = {center_json};
const boundsCoordinates = {bounds_json};
const map = new maplibregl.Map({{
  container:'map', style:'https://tiles.openfreemap.org/styles/liberty', center, zoom:{int(zoom)}, pitch:{int(pitch)},
  maxPitch:85, dragRotate:true, touchZoomRotate:true
}});
map.addControl(new maplibregl.NavigationControl({{visualizePitch:true}}), 'top-right');
map.addControl(new maplibregl.ScaleControl({{maxWidth:120, unit:'metric'}}), 'bottom-left');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
map.once('style.load', () => {{
  map.addSource('map-elevation', {{type:'raster-dem', url:'https://tiles.mapterhorn.com/tilejson.json'}});
  const firstLabel = map.getStyle().layers.find((layer) => layer.type === 'symbol');
  map.addLayer({{id:'map-hillshade',type:'hillshade',source:'map-elevation',paint:{{'hillshade-exaggeration':0.35}}}}, firstLabel && firstLabel.id);
  map.setTerrain({{source:'map-elevation',exaggeration:1.0}});
  map.setSky({{'sky-color':'#081522','sky-horizon-blend':0.35,'atmosphere-blend':0.8}});
  map.addSource('map-features', {{type:'geojson', data}});
  map.addLayer({{id:'map-lines',type:'line',source:'map-features',filter:['==',['geometry-type'],'LineString'],paint:{{
    'line-color':['coalesce',['get','color'],'#e32636'],
    'line-width':['coalesce',['get','width'],3],
    'line-opacity':0.9
  }}}});
  map.addLayer({{id:'map-points',type:'circle',source:'map-features',filter:['==',['geometry-type'],'Point'],paint:{{
    'circle-radius':['coalesce',['get','radius'],8],
    'circle-color':['coalesce',['get','color'],'#1976d2'],
    'circle-stroke-color':'#fff','circle-stroke-width':2
  }}}});
  map.addLayer({{id:'map-labels',type:'symbol',source:'map-features',filter:['all',['==',['geometry-type'],'Point'],['has','label']],layout:{{
    'text-field':['get','label'],'text-offset':[0,1.2],'text-size':12,'text-allow-overlap':true
  }},paint:{{'text-color':'#111827','text-halo-color':'#fff','text-halo-width':1.5}}}});
  map.on('mouseenter','map-points', () => {{ map.getCanvas().style.cursor='pointer'; }});
  map.on('mouseleave','map-points', () => {{ map.getCanvas().style.cursor=''; }});
  map.on('click', (event) => {{
    const hits = map.queryRenderedFeatures(event.point, {{layers:['map-points','map-lines']}});
    const feature = hits.find((item) => item.properties && (item.properties.popup || item.properties.label));
    const text = feature ? (feature.properties.popup || feature.properties.label) : `${{event.lngLat.lat.toFixed(5)}}, ${{event.lngLat.lng.toFixed(5)}}`;
    new maplibregl.Popup().setLngLat(event.lngLat).setText(String(text)).addTo(map);
  }});
  if (boundsCoordinates.length > 1) {{
    const box = new maplibregl.LngLatBounds(center, center);
    for (const [lon, lat] of boundsCoordinates) {{
      const wrappedLon = center[0] + ((((lon - center[0]) + 540) % 360) - 180);
      box.extend([wrappedLon, lat]);
    }}
    map.fitBounds(box, {{padding:45,maxZoom:14,pitch:{int(pitch)},duration:0}});
  }}
}});
map.on('error', (event) => {{
  if (document.querySelector('.map-error')) return;
  const error = document.createElement('div'); error.className='map-error';
  error.textContent = `Map data could not load: ${{event.error && event.error.message ? event.error.message : 'unknown error'}}`;
  document.body.appendChild(error);
}});
</script>
<div class="map-attribution">
<a href="https://maplibre.org/" target="_blank" rel="noopener">MapLibre</a> |
<a href="https://mapterhorn.com/attribution" target="_blank" rel="noopener">&copy; Mapterhorn</a> |
<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> |
<a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">&copy; OpenMapTiles</a> |
Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>
</div></body></html>"""
