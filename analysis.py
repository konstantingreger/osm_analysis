import io
import re
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.ops import nearest_points
from shapely.geometry import LineString
from sqlalchemy import create_engine
import folium

pd.set_option("display.max_columns",None)


def download_geojson_data(bundeslaender):
    """Download Bundesland data for provided records from GADM."""
    response = requests.get("https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_DEU_1.json")
    response.raise_for_status()
    gadm = gpd.read_file(io.BytesIO(response.content))
    return gadm[gadm["NAME_1"].isin(bundeslaender)].reset_index(drop=True)


def parse_voltage(voltage_str):
    """Convert voltage string to numeric V, apply only highest listed value."""
    if not voltage_str:
        return None
    values = []
    for part in str(voltage_str).split(";"):
        m = re.search(r"\d+", part)
        if m:
            v = int(m.group())
            if v < 1000:  # assume kV
                v *= 1000
            values.append(v)
    return max(values) if values else None


def parse_capacity(value):
    """Convert capacity string to numeric MW."""
    if not value:
        return 0
    # Extract numeric value
    m = re.search(r"[\d.]+", str(value))
    if m:
        cap = float(m.group())
        # Convert kW to MW if needed
        if "kw" in str(value).lower():
            cap /= 1000
        return cap
    return 0


def main():
    # load Bundesland geometries from GADM
    bundeslaender = download_geojson_data(["Brandenburg", "Sachsen"])
    
    # query OSM for each Bundesland
    substations_list = []
    generators_list = []
    for _, row in bundeslaender.iterrows():
        sub = ox.features_from_polygon(row["geometry"], {"power": "substation"})
        sub = sub.reset_index() # make sure osmid is a column
        sub["bundesland"] = row["NAME_1"]   # append Bundesland name
        sub["voltage"] = sub["voltage"].apply(parse_voltage)    # parse voltage to coherent format
        substations_list.append(sub)
        
        gen = ox.features_from_polygon(row["geometry"], {"power": ["generator", "plant"]})
        gen = gen.reset_index() # make sure osmid is a column
        gen["bundesland"] = row["NAME_1"]   # append Bundesland name
        gen["capacity_mw"] = gen["generator:output:electricity"].apply(parse_capacity)  # parse voltage to coherent format
        generators_list.append(gen)

    # union substations
    substations = gpd.GeoDataFrame(pd.concat(substations_list, ignore_index=False), crs=substations_list[0].crs)
    substations = substations[substations["voltage"] >= 110000] # filter high & highest voltage only
    substations = substations.to_crs(epsg=3857) # reproject to metric CRS
    substations["geometry"] = substations.geometry.centroid # convert ways & relations to nodes

    # union generators
    generators = gpd.GeoDataFrame(pd.concat(generators_list, ignore_index=False), crs=generators_list[0].crs)
    generators = generators.to_crs(epsg=3857)   # reproject to metric CRS
    generators["geometry"] = generators.geometry.centroid   # convert ways & relations to nodes

    # find nearest substation for each generator and compute distance in m
    generators = gpd.sjoin_nearest(generators, substations[["id", "geometry"]], how="left", distance_col="distance_to_substation")
    generators = generators.rename(columns={
        "id_left": "id",
        "id_right": "nearest_substation_id",
        "distance_to_substation": "nearest_substation_distance_m"
    })
    
    # aggregate and summarize
    grouped = generators.groupby("nearest_substation_id")

    num_generators = grouped.size().rename("num_generators")    # count number of nearby generators
    substations = substations.merge(num_generators, how="left", left_on="id", right_on="nearest_substation_id")
    substations["num_generators"] = substations["num_generators"].fillna(0).astype(int) # fill NaNs
    
    gen_type_breakdown = grouped["generator:method"].value_counts().unstack(fill_value=0)   # breakdown by generator type
    gen_type_breakdown = gen_type_breakdown.reset_index()
    substations = substations.merge(gen_type_breakdown, how="left", left_on="id", right_on="nearest_substation_id")
    breakdown_cols = gen_type_breakdown.columns.difference(["nearest_substation_id"])   # identify columns
    substations[breakdown_cols] = substations[breakdown_cols].fillna(0).astype(int) # fill NaNs

    total_capacity = grouped["capacity_mw"].sum().rename("total_capacity_mw")   # calculate sum of capacities
    substations = substations.merge(total_capacity, how="left", left_on="id", right_on="nearest_substation_id")
    substations["total_capacity_mw"] = substations["total_capacity_mw"].fillna(0).astype(int)   # fill NaNs


    # create connecting lines between generators (x) and substations (y)
    lines = generators[["id", "geometry", "capacity_mw", "nearest_substation_id"]].merge(substations[["id", "geometry"]], left_on="nearest_substation_id", right_on="id", how="left")
    lines["line_geometry"] = lines.apply(lambda row: LineString([row.geometry_x, row.geometry_y]), axis=1)
    lines = gpd.GeoDataFrame(lines[["id_x", "id_y", "capacity_mw"]], geometry=lines["line_geometry"], crs=substations.crs)


    # export result
    substations.to_file("./output/substations.geojson", driver="GeoJSON")
    generators.to_file("./output/generators.geojson", driver="GeoJSON")
    lines.to_file("./output/lines.geojson", driver="GeoJSON")


    # visualize result
    substations = substations.to_crs(epsg=4326)
    bounds = substations.total_bounds
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    popup_fields = ["id", "name", "num_generators"] + breakdown_cols.to_list() + ["total_capacity_mw", "bundesland"]
    folium.GeoJson(substations, 
                   name="Substations", 
                   marker=folium.CircleMarker(radius=5, fill_color="blue", fill_opacity=0.66, color="black", weight=1),
                   popup=folium.GeoJsonPopup(fields=popup_fields)
                   ).add_to(m)
    generators = generators.to_crs(epsg=4326)
    folium.GeoJson(generators,
                   name="Generators & Plants",
                   marker=folium.CircleMarker(radius=5, fill_color="orange", fill_opacity=0.66, color="black", weight=1),
                   popup=folium.GeoJsonPopup(fields=["id", "name", "generator:method", "capacity_mw", "bundesland"])
                   ).add_to(m)
    folium.GeoJson(lines,
                   name="Connections",
                   style_function=lambda x: {"color": "black", "weight": 1},
                   popup=folium.GeoJsonPopup(fields=["id_x", "id_y", "capacity_mw"])
                   ).add_to(m)

    m.save("./output/interactive_map.html")
    

if __name__ == "__main__":
    main()
