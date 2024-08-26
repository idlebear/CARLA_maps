# XML Map Parser
import argparse
import carla
import cv2
from enum import Enum
import json
import numpy as np
import os
from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
import dill

from groups import GroupData

MAP_MARGIN = 50.0
MAP_RESOLUTION = 0.1


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenDRIVE File Loader")

    # Carla parameters
    parser.add_argument(
        "--no-load", action="store_true", help="Render the current CARLA map instead of forcing a reload"
    )
    parser.add_argument("--town", type=str, default="Town03", help="CARLA town name to map")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server IP address")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port number")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for connecting to CARLA server")

    # Output parameters
    parser.add_argument("--prefix", type=str, default=None, help="Prefix on the output files")
    parser.add_argument("--output", type=str, default=".", help="Output directory for the generated files")
    parser.add_argument(
        "--no-render", action="store_true", help="Do not render the map.  Numpy arrays will still be saved."
    )
    parser.add_argument("--groups", type=str, default="groups.json", help="JSON file with group definitions")

    # XODR parameters
    parser.add_argument("--resolution", type=float, default=MAP_RESOLUTION, help="Resolution of the generated map")
    parser.add_argument("--xodr", type=str, default=None, help="XODR file to load (instead of connecting to CARLA)")

    args = parser.parse_args()
    return args


def parse_geometry(road):
    # parse the planview element and pull the geometry
    # and link elements
    planView = road.find("planView")

    min_x = float("inf")
    min_y = float("inf")
    max_x = -float("inf")
    max_y = -float("inf")

    geometry = []
    for geometry_data in planView:
        mode = list(geometry_data)[0]
        if mode.tag == "line":
            points = line_road_to_polyline(geometry_data)
        elif mode.tag == "arc":
            points = arc_road_to_polyline(geometry_data, float(mode.attrib["curvature"]))
        else:
            raise ValueError(f"Unknown geometry type: {mode.tag}")

        geometry.append(points)

    geometry = np.concatenate(geometry, axis=0)
    min_pts = np.min(geometry[:, 1:3], axis=0)
    max_pts = np.max(geometry[:, 1:3], axis=0)
    min_x = min(min_x, min_pts[0])
    min_y = min(min_y, min_pts[1])
    max_x = max(max_x, max_pts[0])
    max_y = max(max_y, max_pts[1])

    return geometry, (min_x, min_y), (max_x, max_y)


def parse_lanes(road):
    # parse the lanes element and pull the lane elements
    lanes_data = road.find("lanes")

    offsets = []
    for element in lanes_data:
        if element.tag == "laneOffset":
            s = float(element.get("s"))
            a = float(element.get("a"))
            b = float(element.get("b"))
            c = float(element.get("c"))
            d = float(element.get("d"))
            offset = [s, a, b, c, d]
            offsets.append(offset)
    else:
        offset = None
    offsets = np.array(offsets)

    def parse_lane(lane):
        id = int(lane.get("id"))
        type = lane.get("type")
        level = lane.get("level")  # don't care

        widths = []
        widths_data = lane.findall("width")
        for width_data in widths_data:
            s = float(width_data.get("sOffset"))
            a = float(width_data.get("a"))
            b = float(width_data.get("b"))
            c = float(width_data.get("c"))
            d = float(width_data.get("d"))
            width = [s, a, b, c, d]
            widths.append(width)
        widths = np.array(widths)

        predecessor, successor = get_lane_connections(lane)

        lane_dict = {
            "id": id,
            "type": type,
            "level": level,
            "widths": widths,
            "predecessor": predecessor,
            "successor": successor,
        }

        return lane_dict

    # get the section with lane data
    lane_sections_data = lanes_data.findall("laneSection")
    lane_sections = []
    for lane_section_data in lane_sections_data:

        s = float(lane_section_data.get("s"))

        # parse the lanes separately, starting with the left lanes
        left_lanes = []
        lane_data = lane_section_data.find("left")
        if lane_data is not None:
            for lane in lane_data:
                left_lanes.append(parse_lane(lane))
        left_lanes.sort(key=lambda x: abs(int(x["id"])))

        # TODO: Ignore the center lanes for now as they seem to be used for lane markings,
        #       not actual lanes
        # lane_data = lane_section_data.find("center")
        # if lane_data is not None:
        #     for lane in lane_data:
        #         lanes.append(parse_lane(lane))

        right_lanes = []
        lane_data = lane_section_data.find("right")
        if lane_data is not None:
            for lane in lane_data:
                right_lanes.append(parse_lane(lane))
        right_lanes.sort(key=lambda x: abs(int(x["id"])))

        lane_sections.append([s, {"left": left_lanes, "right": right_lanes}])

    return {"offsets": offsets, "lane_sections": lane_sections}


def parse_objects(road):
    # crosswalks are listed as objects in a road element
    objects_data = road.find("objects")
    if objects_data is None:
        return []

    objects = []
    for object_data in objects_data:
        ob_type = object_data.get("type")
        if ob_type == "crosswalk":
            s = float(object_data.get("s"))
            t = float(object_data.get("t"))
            heading = float(object_data.get("hdg"))
            length = float(object_data.get("length"))
            width = float(object_data.get("width"))
            orientation = object_data.get("orientation")

            outline_data = object_data.find("outline")
            points_data = outline_data.findall("cornerLocal")
            points = []
            for point_data in points_data:
                u = float(point_data.get("u"))
                v = float(point_data.get("v"))
                points.append([u, v])
            points = np.array(points)

            object = {
                "type": ob_type,
                "id": int(object_data.get("id")),
                "s": s,
                "t": t,
                "orientation": orientation,
                "heading": heading,
                "length": length,
                "width": width,
                "outline": points,
            }

            objects.append(object)

    return objects


def real_to_pixel(x, y, origin, resolution):
    return [int((x - origin[0]) / resolution), int((y - origin[1]) / resolution)]


def line_road_to_polyline(geometry, resolution=0.05):
    x1 = float(geometry.get("x"))
    y1 = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))
    num_points = max( int(length / resolution), 10)  # at least 10 points

    x2 = x1 + (length) * np.cos(heading)
    y2 = y1 + (length) * np.sin(heading)

    xs = np.linspace(x1, x2, num_points, endpoint=True)
    ys = np.linspace(y1, y2, num_points, endpoint=True)
    headings = np.full_like(xs, heading)

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    ds = np.zeros_like(xs)
    ds[1:] = np.sqrt(dx**2 + dy**2)
    cs = np.cumsum(ds) + s

    points = np.array([[s, x, y, heading] for s, x, y, heading in zip(cs, xs, ys, headings)])

    return points


def arc_road_to_polyline(geometry, curvature, resolution=0.05):
    x = float(geometry.get("x"))
    y = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))
    num_points = max( int(length / resolution), 10)  # at least 10 points

    # calculate arc parameters
    side = np.sign(curvature)
    radius = 1.0 / abs(curvature)
    angle = length / radius

    # find the center of the arc
    center_x = x + radius * np.cos(heading + np.pi / 2) * side
    center_y = y + radius * np.sin(heading + np.pi / 2) * side

    t = np.linspace(0, angle, num_points, endpoint=True)
    headings = heading + t * side
    angles = headings - np.pi / 2 * side
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    ds = np.zeros_like(xs)
    ds[1:] = np.sqrt(dx**2 + dy**2)
    cs = np.cumsum(ds) + s

    points = np.array([[s, x, y, heading] for s, x, y, heading in zip(cs, xs, ys, headings)])

    return points


def draw_road_polyline(img, points, origin, resolution, colour=(255, 255, 255)):
    polyline = np.array([real_to_pixel(x, y, origin, resolution) for x, y in points], dtype=np.int32).reshape(
        -1, 1, 2
    )
    cv2.polylines(img, [polyline], isClosed=False, color=colour, thickness=2)


def get_road_connections(road):
    link_data = road.find("link")
    if link_data is None:
        return None, None

    predecessor_data = link_data.find("predecessor")
    predecessor = None
    if predecessor_data is not None:
        if predecessor_data.get("elementType") == "road":
            predecessor = [ int(predecessor_data.get('elementId')), predecessor_data.get("contactPoint")]
        else:  # junction
            predecessor = int(predecessor_data.get('elementId'))

    successor_data = link_data.find("successor")
    successor = None
    if successor_data is not None:
        if successor_data.get("elementType") == "road":
            successor = [int (successor_data.get('elementId')), successor_data.get("contactPoint")]
        else:  # junction
            successor = int(successor_data.get('elementId'))

    return predecessor, successor


def get_lane_connections(road):
    link_data = road.find("link")
    if link_data is None:
        return None, None

    predecessor_data = link_data.find("predecessor")
    if predecessor_data is not None:
        predecessor = int(predecessor_data.get('id'))
    else:
        predecessor = None

    successor_data = link_data.find("successor")
    if successor_data is not None:
        successor = int(successor_data.get('id'))
    else:
        successor = None

    return predecessor, successor


def construct_inner_lane_edge(road, side="left"):
    road_id = road["id"]
    geometry = road["geometry"]
    offsets = road["lanes"]["offsets"]
    lane_sections = road["lanes"]["lane_sections"]

    # adjust the centerline
    center_line = np.zeros([len(geometry), 2])
    offset_index = 0
    s_0 = geometry[0][0]
    for index, (s, x, y, heading) in enumerate(geometry):
        if offset_index >= len(offsets) - 1:
            pass
        else:
            while s >= offsets[offset_index + 1, 0]:
                offset_index += 1
                s_0 = offsets[offset_index, 0]
                if offset_index == len(offsets) - 1:
                    break

        ds = s - s_0
        offset = (
            offsets[offset_index, 1]
            + offsets[offset_index, 2] * ds
            + offsets[offset_index, 3] * ds**2
            + offsets[offset_index, 4] * ds**3
        )
        center_line[index, 0] = x + offset * np.cos(heading + np.pi / 2)
        center_line[index, 1] = y + offset * np.sin(heading + np.pi / 2)

    return center_line


def construct_lane_geometry(road, side="left"):
    road_id = road["id"]

    first_inner_edge = construct_inner_lane_edge(road, side)

    left = construct_side_geometry(road, "left", first_inner_edge)
    right = construct_side_geometry(road, "right", first_inner_edge)

    for key, value in right.items():
        if key in left:
            left[key].update(value)
        else:
            left[key] = value

    # parse the objects
    objects = construct_object_geometry(road)

    return objects, left


def construct_side_geometry(road, side="left", first_inner_edge=None):
    sign = 1
    if side == "right":
        sign = -1

    geometry = road["geometry"]
    lane_sections = road["lanes"]["lane_sections"]

    lanes = {}
    geometry_start_index = 0
    for lane_section_index, lane_section in enumerate(lane_sections):
        lanes_data = lane_section[1][side]
        if len(lanes_data) == 0:
            continue

        if lane_section_index not in lanes:
            lanes[lane_section_index] = {}

        inner_lane_edge = first_inner_edge

        # skip this section if the 's' value is already past the end
        if lane_section_index < len(lane_sections) - 1:
            if geometry[geometry_start_index][0] >= lane_sections[lane_section_index + 1][0]:
                continue

        for lane_data in lanes_data:
            id = int(lane_data["id"])

            outer_lane_edge = np.zeros_like(inner_lane_edge)

            for index in range(geometry_start_index, len(geometry) + 1):
                if index == len(geometry):
                    break

                s = geometry[index][0]
                if lane_section_index < len(lane_sections) - 1:
                    if s >= lane_sections[lane_section_index + 1][0]:
                        break

                # x = geometry[index][1]
                # y = geometry[index][2]
                heading = geometry[index][3]

                width_index = 0
                widths = lane_data["widths"]
                if width_index >= len(widths) - 1:
                    pass
                else:
                    while s >= widths[width_index + 1, 0]:
                        width_index += 1
                        if width_index == len(widths) - 1:
                            break

                ds = s - widths[width_index, 0]
                width = (
                    widths[width_index, 1]
                    + widths[width_index, 2] * ds
                    + widths[width_index, 3] * ds**2
                    + widths[width_index, 4] * ds**3
                )
                outer_lane_edge[index, 0] = inner_lane_edge[index, 0] + sign * width * np.cos(heading + np.pi / 2)
                outer_lane_edge[index, 1] = inner_lane_edge[index, 1] + sign * width * np.sin(heading + np.pi / 2)

            if index - geometry_start_index < 2:
                print(f"Empty lane: {road['id']}, road_type: {lane_data['type']}")
                break

            # find the centers between the inner and outer edges
            centers = (
                inner_lane_edge[geometry_start_index:index, :] + outer_lane_edge[geometry_start_index:index, :]
            ) / 2

            points = np.array([[s, x, y, heading] for s, x, y, heading in zip(geometry[geometry_start_index:index, 0],
                                                                              centers[:,0],
                                                                              centers[:,1],
                                                                              geometry[geometry_start_index:index, 1])])


            lanes[lane_section_index][id] = {
                "polygon": np.concatenate(
                    [
                        inner_lane_edge[geometry_start_index:index, :],
                        np.flip(outer_lane_edge[geometry_start_index:index, :], axis=0),
                    ],
                    axis=0,
                ),
                "centers": points,
                "road_type": lane_data["type"],
                "predecessor": lane_data["predecessor"],
                "successor": lane_data["successor"],
            }

            # move the inner lane edge to the outer lane edge
            inner_lane_edge = outer_lane_edge

        # move the geometry start index to the next lane section
        geometry_start_index = index - 1

    return lanes


def construct_object_geometry(road, resolution=0.05):
    geometry = road["geometry"]
    objects_data = road["objects"]

    objects = {}
    for object_data in objects_data:
        s = object_data["s"]
        t = object_data["t"]
        heading = object_data["heading"]
        # orientation = object_data["orientation"]
        # width = object_data["width"]
        length = object_data["length"]

        # TODO: This works for almost all crosswalks but does not account for the orientation which
        #       may be the reason for the occasional misalignment.  Good enough for now.
        road_heading = np.interp(s, geometry[:, 0], geometry[:, 3])
        center_x = np.interp(s, geometry[:, 0], geometry[:, 1]) + t * np.cos(road_heading + np.pi / 2)
        center_y = np.interp(s, geometry[:, 0], geometry[:, 2]) + t * np.sin(road_heading + np.pi / 2)

        # rotate the object outline to match the road heading
        object_heading = road_heading + heading
        rotation_matrix = np.array(
            [[np.cos(object_heading), -np.sin(object_heading)], [np.sin(object_heading), np.cos(object_heading)]]
        )
        corner_points = np.dot(rotation_matrix, np.array(object_data["outline"]).T).T
        corner_points = corner_points + np.array([center_x, center_y])

        # find the centerline points for the object
        x1 = center_x + 0.51 * length * np.cos(object_heading)
        y1 = center_y + 0.51 * length * np.sin(object_heading)
        x2 = center_x - 0.51 * length * np.cos(object_heading)
        y2 = center_y - 0.51 * length * np.sin(object_heading)

        # sample the object center to the requested step size
        num_points = max( int(length / resolution), 10)  # at least 10 points
        xs = np.linspace(x1, x2, num_points, endpoint=True)
        ys = np.linspace(y1, y2, num_points, endpoint=True)

        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]
        ds = np.zeros_like(xs)
        ds[1:] = np.sqrt(dx**2 + dy**2)
        cs = np.cumsum(ds)  # always starts at zero

        centers = np.array([[s, x, y, object_heading] for s, x, y in zip(cs, xs, ys)])

        objects[object_data["id"]] = {
            "polygon": corner_points,
            "centers": centers,
            "road_type": object_data["type"],
            "heading": object_heading,
        }

    return objects


def render(img, polygons, road_types, elements, map_origin, map_resolution):

    for poly, road_type in zip(polygons, road_types):
        if road_type in elements:
            points = np.array([real_to_pixel(x, y, map_origin, map_resolution) for x, y in poly], dtype=np.int32)
            cv2.fillPoly(img, [points], color=255)

    return img


def insert_road_link(road_links, from_id, to_id):
    if from_id not in road_links:
        road_links[from_id] = [to_id]
    else:
        if to_id not in road_links[from_id]:
            road_links[from_id].append(to_id)
    return road_links

def get_crosswalk_links( road_map, categories, objects, threshold=1 ):
    object_links = {}
    for object_id, object in objects.items():
        if object['road_type'] == 'crosswalk':
            for road_id, lanes in road_map.items():
                for lane_section_id, lane_section in lanes.items():
                    for lane_id, lane in lane_section.items():
                        if lane['road_type'] in categories:
                            if np.linalg.norm(object['centers'][0, 1:3] - lane['centers'][0,1:3]) < threshold:
                                object_link_name = f"{object_id}-0"
                                lane_link_name= f"{road_id}-{lane_section_id}-{lane_id}-0"
                                insert_road_link(object_links, object_link_name, lane_link_name)
                                insert_road_link(object_links, lane_link_name, object_link_name)
                            elif np.linalg.norm(object['centers'][0, 1:3] - lane['centers'][-1,1:3]) < threshold:
                                object_link_name = f"{object_id}-0"
                                lane_link_name= f"{road_id}-{lane_section_id}-{lane_id}-{len(lane['centers'])-1}"
                                insert_road_link(object_links, object_link_name, lane_link_name)
                                insert_road_link(object_links, lane_link_name, object_link_name)

                            if np.linalg.norm(object['centers'][-1,1:3] - lane['centers'][0,1:3]) < threshold:
                                object_link_name = f"{object_id}-{len(object['centers'])-1}"
                                lane_link_name= f"{road_id}-{lane_section_id}-{lane_id}-0"
                                insert_road_link(object_links, object_link_name, lane_link_name)
                                insert_road_link(object_links, lane_link_name, object_link_name)
                            elif np.linalg.norm(object['centers'][-1,1:3] - lane['centers'][-1, 1:3]) < threshold:
                                object_link_name = f"{object_id}-{len(object['centers'])-1}"
                                lane_link_name= f"{road_id}-{lane_section_id}-{lane_id}-{len(lane['centers'])-1}"
                                insert_road_link(object_links, object_link_name, lane_link_name)
                                insert_road_link(object_links, lane_link_name, object_link_name)
    return object_links


def main():
    args = parse_arguments()

    prefix = ""
    if args.prefix is not None:
        prefix = args.prefix + "_"

    # load the groups data
    groups = GroupData(args.groups)

    if args.xodr is None:
        # Connect to the CARLA server and download the map
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        if not args.no_load:
            world = client.load_world(args.town)
        carla_map = world.get_map()
        map_data = carla_map.to_opendrive()

        # write the map data to a file for reference
        with open(os.path.join(args.output, f"{prefix}{args.town}.xodr"), "w") as f:
            f.write(map_data)
    else:
        map_data = open(args.xodr, "r").read()

    # Parse the map data
    root = ET.fromstring(map_data)
    roads_data = root.findall("road")

    # Extract and print all subelements for each road element
    roads = {}
    mins = []
    maxs = []
    for road_data in roads_data:
        id = int(road_data.get("id"))

        predecessor, successor = get_road_connections(road_data)

        name = road_data.get("name")
        length = float(road_data.get("length"))

        geometry, geo_mins, geo_maxs = parse_geometry(road_data)
        mins.append(geo_mins)
        maxs.append(geo_maxs)

        lanes = parse_lanes(road_data)
        objects = parse_objects(road_data)

        road = {
            "id": id,
            "name": name,
            "length": length,
            "geometry": geometry,
            "lanes": lanes,
            "objects": objects,
            "predecessor": predecessor,
            "successor": successor,
        }
        roads[id] = road

    min_x, min_y = np.min(np.array(mins), axis=0)
    max_x, max_y = np.max(np.array(maxs), axis=0)

    map_width = int((max_x - min_x + 2 * MAP_MARGIN) / args.resolution)
    map_height = int((max_y - min_y + 2 * MAP_MARGIN) / args.resolution)

    map_origin = [min_x - MAP_MARGIN, min_y - MAP_MARGIN]

    # # draw the roads
    # centers_img = np.zeros((map_height, map_width, 3), np.uint8)
    # for road in roads:
    #     draw_road_polyline(centers_img, road["geometry"], map_origin, args.resolution)

    # construct the road polygons
    polygons = []
    road_types = []

    road_map = {}
    object_map = {}
    forward_road_links = {}  # map of road-end-lane to road-end-lane connections
    backward_road_links = {}  # map of road-end-lane to road-end-lane connections

    for road_id, road in roads.items():

        objects, lanes = construct_lane_geometry(road)
        road_map[road_id] = lanes

        for lane_section_id, lane_section in lanes.items():

            for lane_id, lane in lane_section.items():
                polygons.append(lane["polygon"])
                road_types.append(lane["road_type"])

                # build the road connections map
                if lane["predecessor"] is not None:
                    link_name = f"{road_id}-{lane_section_id}-{lane_id}-0"
                    if not lane_section_id:
                        if road["predecessor"] is not None:
                            # there is a predecessor road and this is the first road section
                            predecessor_name = f"{road['predecessor'][0]}-{road['predecessor'][1]}-{lane['predecessor']}-{road['predecessor'][1]}"
                            insert_road_link( backward_road_links, link_name, predecessor_name )
                    else:
                        predecessor_name = f"{road_id}-{lane_section_id-1}-{lane['predecessor']}-end"
                        insert_road_link(backward_road_links, link_name, predecessor_name)

                if lane["successor"] is not None:
                    link_name = f"{road_id}-{lane_section_id}-{lane_id}-{len(lane['centers'])-1}"
                    if lane_section_id == len(lanes) - 1:
                        if road["successor"] is not None:
                            # there is a successor road and this is the last road section
                            successor_name = f"{road['successor'][0]}-{road['successor'][1]}-{lane['successor']}-{road['successor'][1]}"
                            insert_road_link( backward_road_links, link_name, successor_name )
                    else:
                        successor_name = f"{road_id}-{lane_section_id+1}-{lane['successor']}-0"
                        insert_road_link(backward_road_links, link_name, successor_name)

        for object_id, object in objects.items():
            object_map[object_id] = object
            polygons.append(object["polygon"])
            road_types.append(object["road_type"])

    # # draw the centerlines for 'sidewalk' roads and 'crosswalk' objects
    # centers_img = np.zeros((map_height, map_width, 3), np.uint8)
    # for road_id, lanes in road_map.items():
    #     for lane_section_id, lane_section in lanes.items():
    #         for lane_id, lane in lane_section.items():
    #             if lane["road_type"] == "sidewalk":
    #                 draw_road_polyline(centers_img, lane["centers"][:, 1:3], map_origin, args.resolution)
    # for object_id, object in object_map.items():
    #     if object["road_type"] == "crosswalk":
    #         draw_road_polyline(centers_img, object["centers"][:,1:3], map_origin, args.resolution, colour=(0, 255, 0))
    # filename = os.path.join(args.output, f"{prefix}{args.town}_sidewalks.png")
    # cv2.imwrite(filename, centers_img)

    # get the object links
    object_links = get_crosswalk_links(road_map, ["sidewalk"], object_map, threshold=2.5)

    # parse the junctions
    junctions_data = root.findall("junction")
    for junction_data in junctions_data:
        junction_id = int(junction_data.get("id"))
        if junction_id != 1820:
            continue

        connections_data = junction_data.findall("connection")
        for connection_data in connections_data:
            incoming_road = int(connection_data.get("incomingRoad"))
            try:
                road = roads[int(incoming_road)]
            except KeyError:
                continue

            connecting_road = connection_data.get("connectingRoad")
            contact_point = connection_data.get("contactPoint")

            if road['predecessor'] == junction_id:
                incoming_link = 'start'
                link_dict = backward_road_links
            else:
                incoming_link = 'end'
                link_dict = forward_road_links

            for lane_link in connection_data.findall("laneLink"):
                incoming_link_name = f"{incoming_road}-{incoming_link}-{lane_link.get('from')}"
                connection_link_name = f"{connecting_road}-{contact_point}-{lane_link.get('to')}"
                insert_road_link(link_dict, incoming_link_name, connection_link_name)  # forward

    # render the map
    for group_name in groups.get_group_names():

        num_layers = groups.get_num_layers(group_name)
        map_data = np.zeros([num_layers, map_height, map_width], dtype=np.uint8)

        for index, layer in enumerate(groups.get_layers(group_name)):
            render(map_data[index, ...], polygons, road_types, layer["elements"], map_origin, args.resolution)

        # write the map to a file

        # flip the map to match the reversed Carla coordinate system
        map_data = np.flip(map_data, axis=1)

        # convert the image format to 3 x X x Y - note the transpose to the X,Y from Y,X image format
        np_img = map_data.transpose(0, 2, 1)
        filename = os.path.join(args.output, f"{prefix}{args.town}_{group_name}_map.npy")
        np.save(filename, np_img)

        if not args.no_render:
            # write out the first three layers as an image
            img = map_data[:3, ...].transpose(1, 2, 0)
            filename = os.path.join(args.output, f"{prefix}{args.town}_{group_name}_map.png")
            cv2.imwrite(filename, img)

    # write the map parameters to a file
    # swap the map width and height to match numpy array format
    map_width, map_height = map_height, map_width

    # CARLA uses a left-handed coordinate system - flip the y values and
    # when we write the map, we will flip the image as well
    min_y, max_y = -max_y, -min_y
    print(f"Map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    map_origin = [min_x - MAP_MARGIN, min_y - MAP_MARGIN]

    parameters = {
        "town": args.town,
        "prefix": args.prefix,
        "map_origin": map_origin,
        "map_width": map_width,
        "map_height": map_height,
        "map_resolution": args.resolution,
        "groups": groups.get_group_names(),
    }
    with open(os.path.join(args.output, f"{prefix}{args.town}_map_parameters.json"), "w") as f:
        json.dump(parameters, f, indent=2)

    # write out the road map to a file adding the object links
    world_map = {
        "roads": road_map,
        "forward_road_links": forward_road_links,
        "backward_road_links": backward_road_links,
        "object_links": object_links,
    }
    with open(os.path.join(args.output, f"{prefix}{args.town}_links.dill"), "wb") as f:
        dill.dump(world_map, f)

    print("done")


if __name__ == "__main__":
    main()
