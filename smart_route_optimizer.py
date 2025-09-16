import streamlit as st
import folium
from streamlit_folium import st_folium
import networkx as nx
import heapq
import math
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import requests
import random
import json

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def get_coordinates(address):
    try:
        geolocator = Nominatim(user_agent="smart_route_optimizer")
        location = geolocator.geocode(address, timeout=10, language='en')
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None
    return None

def get_osrm_route(coord1, coord2):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['routes']:
            route = data['routes'][0]
            geometry = [[point[1], point[0]] for point in route['geometry']['coordinates']]
            return {
                "distance": route['distance'] / 1000,  # in km
                "duration": route['duration'] / 60,   # in minutes
                "geometry": geometry
            }
    except requests.exceptions.RequestException as e:
        st.error(f"OSRM API error: {e}")
        return None
    return None

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def create_graph(coords, use_osrm=False, optimize_by='Distance', add_traffic=False):
    G = nx.Graph()
    num_coords = len(coords)
    for i in range(num_coords):
        G.add_node(i, pos=coords[i])

    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            if use_osrm:
                route_data = get_osrm_route(coords[i], coords[j])
                if route_data:
                    weight = route_data['distance'] if optimize_by == 'Distance' else route_data['duration']
                    G.add_edge(i, j, weight=weight, distance=route_data['distance'], duration=route_data['duration'], geometry=route_data['geometry'])
                else:
                    # Fallback to Haversine if OSRM fails
                    dist = haversine(coords[i], coords[j])
                    G.add_edge(i, j, weight=dist, distance=dist, duration=dist * 1.5, geometry=[coords[i], coords[j]])
            else:
                dist = haversine(coords[i], coords[j])
                weight = dist * random.uniform(1.0, 2.0) if add_traffic else dist
                G.add_edge(i, j, weight=weight, distance=dist, geometry=[coords[i], coords[j]])
    return G

# --- Algorithms ---
def dijkstra(G, start_node_idx, end_node_idx):
    distances = {node: float('inf') for node in G.nodes}
    distances[start_node_idx] = 0
    previous_nodes = {node: None for node in G.nodes}
    priority_queue = [(0, start_node_idx)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]: continue
        if current_node == end_node_idx: break
        for neighbor in G.neighbors(current_node):
            weight = G[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    path, current = [], end_node_idx
    while current is not None:
        path.insert(0, current)
        current = previous_nodes.get(current)
    return (path, distances[end_node_idx]) if path and path[0] == start_node_idx else ([], float('inf'))

def a_star(G, start_node_idx, end_node_idx):
    start_node = Node(None, G.nodes[start_node_idx]['pos'])
    end_node = Node(None, G.nodes[end_node_idx]['pos'])
    start_node_map = {G.nodes[i]['pos']: i for i in G.nodes}
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)
        if current_node.position == end_node.position:
            path = []
            current = current_node
            while current is not None:
                path.append(start_node_map[current.position])
                current = current.parent
            return path[::-1], current_node.g
        current_idx = start_node_map[current_node.position]
        for neighbor_idx in G.neighbors(current_idx):
            neighbor_pos = G.nodes[neighbor_idx]['pos']
            if neighbor_pos in closed_set: continue
            neighbor = Node(current_node, neighbor_pos)
            neighbor.g = current_node.g + G[current_idx][neighbor_idx]['weight']
            neighbor.h = haversine(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h
            if any(n for n in open_list if n.position == neighbor.position and n.f < neighbor.f): continue
            heapq.heappush(open_list, neighbor)
    return [], float('inf')

def tsp_nearest_neighbor(G):
    nodes = list(G.nodes)
    if not nodes: return [], 0
    start_node = nodes[0]
    path = [start_node]
    total_weight = 0
    unvisited = set(nodes)
    unvisited.remove(start_node)
    current_node = start_node
    while unvisited:
        nearest_neighbor = min(unvisited, key=lambda node: G[current_node][node]['weight'])
        total_weight += G[current_node][nearest_neighbor]['weight']
        current_node = nearest_neighbor
        path.append(current_node)
        unvisited.remove(current_node)
    total_weight += G[path[-1]][start_node]['weight']
    path.append(start_node)
    return path, total_weight

st.set_page_config(page_title="Smart Route Optimizer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background-color: #F5F9FF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E6F0FF 0%, #D4E5FF 100%);
        border-right: 1px solid #B8D4FF;
    }
    
    .stButton > button, 
    .stDownloadButton > button,
    .stFormSubmitButton > button,
    div[data-testid="stForm"] button {
        background: linear-gradient(180deg, #4A86E8 0%, #3A76D8 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        padding: 6px 12px;    
        width: 90%;           
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(74, 134, 232, 0.3);
        margin: 6px auto;    
        text-align: center;
        height: 34px;          
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stButton > button:hover, 
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover,
    div[data-testid="stForm"] button:hover {
        background: linear-gradient(180deg, #3A76D8 0%, #2A66C8 100%);
        box-shadow: 0 4px 8px rgba(74, 134, 232, 0.4);
        color: white !important;
    }
    
    .stButton > button:active, 
    .stDownloadButton > button:active,
    .stFormSubmitButton > button:active,
    div[data-testid="stForm"] button:active {
        box-shadow: 0 1px 2px rgba(74, 134, 232, 0.3);
        color: white !important;
    }

    .stButton > button div,
    .stDownloadButton > button div,
    .stFormSubmitButton > button div,
    div[data-testid="stForm"] button div {
        color: white !important;
        font-weight: 600;
    }

    .stButton > button[kind="secondary"] {
        background: linear-gradient(180deg, #E6F0FF 0%, #D4E5FF 100%);
        color: #2A66C8 !important;
        border: 1px solid #B8D4FF;
        border-radius: 6px;
        font-weight: 600;
        height: 34px;        
        width: 90%;
        margin: 6px auto;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(180deg, #D4E5FF 0%, #C2D5FF 100%);
        color: #1A56B8 !important;
        border-color: #98C0FF;
    }

    div[data-testid="stForm"] {
        border: 1px solid #D4E5FF;
        border-radius: 10px;
        padding: 15px;
        background-color: #F0F7FF;
        margin-bottom: 15px;
    }

    div[data-testid="stForm"] button {
        margin: 5px 0;
    }

    div[data-testid="column"] {
        padding: 0 5px;
    }

    .stRadio > div {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .stRadio [role="radiogroup"] label > div:first-child {
    border: 2px solid #94A3B8;
    background: white;
    border-radius: 50%;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    transition: all 0.2s ease;
    }

    .stRadio [role="radiogroup"] label:hover > div:first-child {
        border-color: #2563EB;
        box-shadow: 0 0 4px rgba(37, 99, 235, 0.4);
    }

    .stRadio [role="radiogroup"] label > div:first-child::before {
        content: "";
        display: block;
        width: 10px;
        height: 10px;
        margin: 2px auto;
        border-radius: 50%;
        background-color: #2563EB;
    }

    .stRadio label {
        color: #334155;
        font-weight: 500;
        cursor: pointer;
        transition: color 0.2s ease;
    }

    .stRadio [role="radiogroup"] label:has(input[type="radio"]:checked) {
        color: #1E3A8A; 
        font-weight: 600;
    }

    .stCheckbox > div {
        background-color: #E6F0FF;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #D4E5FF;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F0F7FF 100%);
        border: 1px solid #B8D4FF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(74, 134, 232, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #2A66C8;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4A86E8;
    }

    h1 { 
        color: #2A66C8; 
        border-bottom: 2px solid #B8D4FF;
        padding-bottom: 10px;
    }
    
    h2 { 
        color: #3A76D8; 
        border-bottom: 2px solid #D4E5FF;
        padding-bottom: 8px;
    }
    
    h3 { 
        color: #4A86E8; 
    }

    .stAlert {
        background-color: #E6F0FF;
        border: 1px solid #B8D4FF;
        color: #2A66C8;
    }

    .stAlert [data-testid="stMarkdownContainer"] {
        color: #2A66C8;
    }

    hr {
        border-color: #D4E5FF;
    }

    .stTextInput input {
        background-color: #F0F7FF;
        border: 1px solid #D4E5FF;
        border-radius: 6px;
        padding: 8px 12px;
    }
    
    .stTextInput input:focus {
        border-color: #4A86E8;
        box-shadow: 0 0 0 1px #4A86E8;
    }
</style>
""", unsafe_allow_html=True)


st.title("Smart Route & Delivery Optimizer")
st.markdown("A sophisticated tool for calculating the shortest delivery path using classic and heuristic algorithms.")

if 'addresses' not in st.session_state: st.session_state.addresses = ["", ""]
if 'coords' not in st.session_state: st.session_state.coords = []
if 'results' not in st.session_state: st.session_state.results = None

with st.sidebar:
    st.header("Delivery Locations", divider="blue")
    with st.form(key='address_form'):
        for i in range(len(st.session_state.addresses)):
            st.session_state.addresses[i] = st.text_input(f"Address {i+1}", st.session_state.addresses[i], key=f"addr_{i}")
        form_cols = st.columns(2)
        with form_cols[0]:
            add_address_button = st.form_submit_button(label="Add Address")
        with form_cols[1]:
            remove_address_button = st.form_submit_button(label="Remove Last")
        
        geocode_button = st.form_submit_button(label="Update Locations", type="primary")

    if add_address_button:
        st.session_state.addresses.append("")
        st.session_state.results = None
        st.rerun()

    if remove_address_button and len(st.session_state.addresses) > 2:
        st.session_state.addresses.pop()
        st.session_state.results = None
        st.rerun()

    st.header("Optimization Settings", divider="blue")
    routing_mode = st.radio("Routing Mode", ["Straight Line", "Real-World (OSRM)"], horizontal=True)
    optimize_by = st.radio("Optimize For", ["Distance", "Time"], horizontal=True, disabled=(routing_mode != "Real-World (OSRM)"))
    algorithm = st.radio("Algorithm", ('Dijkstra (A to B)', 'A* (A to B)', 'TSP (Multi-stop)'))
    
    st.markdown("---")
    optimize_button = st.button("Optimize Route", type="primary", use_container_width=True)
    clear_button = st.button("Clear Route", type="secondary", use_container_width=True)

if clear_button:
    st.session_state.results = None
    st.rerun()
if geocode_button:
    st.session_state.coords = []
    st.session_state.results = None
    with st.spinner("Geocoding addresses..."):
        for address in st.session_state.addresses:
            if address:
                coord = get_coordinates(address)
                if coord: st.session_state.coords.append(coord)
                else: st.error(f"Could not geocode address: {address}. Please try again.")
    if st.session_state.coords: st.sidebar.success(f"{len(st.session_state.coords)} locations updated.")

if optimize_button and len(st.session_state.coords) >= 2:
    if algorithm.startswith('TSP') and len(st.session_state.coords) < 3:
        st.warning("TSP requires at least 3 addresses.")
    else:
        spinner_msg = "Fetching routes and calculating path..." if routing_mode == "Real-World (OSRM)" else "Calculating optimal route..."
        with st.spinner(spinner_msg):
            G = create_graph(st.session_state.coords, use_osrm=(routing_mode == "Real-World (OSRM)"), optimize_by=optimize_by)
            path_indices, total_weight = [], 0
            if algorithm.startswith('Dijkstra'): path_indices, total_weight = dijkstra(G, 0, len(st.session_state.coords) - 1)
            elif algorithm.startswith('A*'): path_indices, total_weight = a_star(G, 0, len(st.session_state.coords) - 1)
            elif algorithm.startswith('TSP'): path_indices, total_weight = tsp_nearest_neighbor(G)
            if path_indices: st.session_state.results = {"path": path_indices, "weight": total_weight, "graph": G, "mode": routing_mode, "optimize_by": optimize_by}
            else: st.error("Could not find a path. Check addresses or OSRM service status.")
elif optimize_button:
    st.warning("Please enter and update at least two locations first.")

st.header("Route Visualization")
map_tile = 'CartoDB positron'

if st.session_state.results:
    results = st.session_state.results
    G = results["graph"]
    path_indices = results["path"]
    
    result_map = folium.Map(location=st.session_state.coords[0], zoom_start=12, tiles=map_tile)

    for i in range(len(path_indices) - 1):
        u, v = path_indices[i], path_indices[i+1]
        if G.has_edge(u,v) and 'geometry' in G[u][v]:
            folium.PolyLine(G[u][v]['geometry'], color="#2563EB", weight=5, opacity=0.8).add_to(result_map)

    for i, node_idx in enumerate(path_indices):
        if i == len(path_indices) - 1 and not algorithm.startswith('TSP'): continue
        coord = st.session_state.coords[node_idx]
        address = st.session_state.addresses[node_idx]
        color = '#2A66C8' if i == 0 else ('#4A86E8' if i == len(path_indices) - 2 and not algorithm.startswith('TSP') else '#3A76D8')
        folium.Marker(location=coord, popup=f"<b>Stop {i+1}:</b><br>{address}", icon=folium.Icon(color=color, icon='truck', prefix='fa')).add_to(result_map)
        folium.Marker(location=coord, icon=folium.DivIcon(icon_size=(150,36), icon_anchor=(8,35), html=f'<div style="font-size: 1rem; background-color: {color}; color: white; border-radius: 50%; width: 2rem; height: 2rem; text-align: center; line-height: 2rem; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{i+1}</div>')).add_to(result_map)

    st_folium(result_map, width='100%', height=500)
else:
    map_center = st.session_state.coords[0] if st.session_state.coords else [28.6139, 77.2090]
    m = folium.Map(location=map_center, zoom_start=12 if st.session_state.coords else 5, tiles=map_tile)
    if st.session_state.coords:
        for i, (coord, address) in enumerate(zip(st.session_state.coords, st.session_state.addresses)):
            folium.Marker(location=coord, popup=f"<b>{i+1}:</b> {address}", icon=folium.Icon(color='#4A86E8', icon='map-marker', prefix='fa')).add_to(m)
    else:
        st.info("Enter addresses in the sidebar and click 'Update Locations' to see them on the map.")
    st_folium(m, width='100%', height=500)

if st.session_state.results:
    st.markdown("---")
    st.header("Performance & Route Details")
    results = st.session_state.results
    G, path_indices, optimize_by = results["graph"], results["path"], results["optimize_by"]
    
    total_dist = sum(G[path_indices[i]][path_indices[i+1]]['distance'] for i in range(len(path_indices)-1))
    total_time = sum(G[path_indices[i]][path_indices[i+1]]['duration'] for i in range(len(path_indices)-1)) if results['mode'] == 'Real-World (OSRM)' else total_dist * 1.5

    metric_cols = st.columns(3)
    metric_cols[0].metric(label="Total Distance", value=f"{total_dist:.2f} km")
    metric_cols[1].metric(label="Estimated Time", value=f"{total_time:.2f} min")
    
    st.subheader("Delivery Order")
    route_text = ""
    for i, node_idx in enumerate(path_indices):
        route_text += f"**{i+1}.** {st.session_state.addresses[node_idx]}\n"
    st.markdown(route_text)

    route_data = {"route": [{"stop": i+1, "address": st.session_state.addresses[node_idx]} for i, node_idx in enumerate(path_indices)]}
    st.download_button(label="Export Route as JSON", data=json.dumps(route_data, indent=4), file_name="route.json", mime="application/json", use_container_width=True, type="primary")
