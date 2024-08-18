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

from groups import GroupData

MAP_MARGIN = 50.0
MAP_RESOLUTION = 0.1

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenDRIVE File Loader")

    # Carla parameters
    parser.add_argument("--no-load", action="store_true", help="Render the current CARLA map instead of forcing a reload")
    parser.add_argument("--town", type=str, default="Town03", help="CARLA town name to map")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server IP address")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port number")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for connecting to CARLA server")

    # Output parameters
    parser.add_argument("--prefix", type=str, default=None, help="Prefix on the output files")
    parser.add_argument("--output", type=str, default=".", help="Output directory for the generated files")
    parser.add_argument("--no-render", action="store_true", help="Do not render the map.  Numpy arrays will still be saved.")
    parser.add_argument("--groups", type=str, default="groups.json", help="JSON file with group definitions")

    # XODR parameters
    parser.add_argument("--resolution", type=float, default=0.1, help="Resolution of the generated map")
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

        lane_dict = {
            "id": id,
            "type": type,
            "level": level,
            "widths": widths,
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


def line_road_to_polyline(geometry):
    x1 = float(geometry.get("x"))
    y1 = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))

    x2 = x1 + (length) * np.cos(heading)
    y2 = y1 + (length) * np.sin(heading)

    xs = np.linspace(x1, x2, 500, endpoint=True)
    ys = np.linspace(y1, y2, 500, endpoint=True)
    headings = np.full_like(xs, heading)

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    ds = np.zeros_like(xs)
    ds[1:] = np.sqrt(dx**2 + dy**2)
    cs = np.cumsum(ds) + s

    points = np.array([[s, x, y, heading] for s, x, y, heading in zip(cs, xs, ys, headings)])

    return points


def arc_road_to_polyline(geometry, curvature):
    x = float(geometry.get("x"))
    y = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))

    # calculate arc parameters
    side = np.sign(curvature)
    radius = 1.0 / abs(curvature)
    angle = length / radius

    # find the center of the arc
    center_x = x + radius * np.cos(heading + np.pi / 2) * side
    center_y = y + radius * np.sin(heading + np.pi / 2) * side

    t = np.linspace(0, angle, 500)
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


def draw_road_polyline(img, points, origin, resolution):
    polyline = np.array([real_to_pixel(x, y, origin, resolution) for _, x, y, _ in points], dtype=np.int32).reshape(
        -1, 1, 2
    )
    cv2.polylines(img, [polyline], isClosed=False, color=(255, 255, 255), thickness=3)


def construct_polygon_array(road, side="left"):
    road_id = road["id"]
    geometry = road["geometry"]
    offsets = road["lanes"]["offsets"]
    lane_sections = road["lanes"]["lane_sections"]

    sign = 1
    if side == "right":
        sign = -1

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

        ds = (s - s_0)
        offset = (
            offsets[offset_index, 1]
            + offsets[offset_index, 2] * ds
            + offsets[offset_index, 3] * ds**2
            + offsets[offset_index, 4] * ds**3
        )
        center_line[index, 0] = x + offset * np.cos(heading + np.pi / 2)
        center_line[index, 1] = y + offset * np.sin(heading + np.pi / 2)

    polygons = []
    road_types = []

    geometry_start_index = 0
    for lane_section_index, lane_section in enumerate(lane_sections):
        lanes = lane_section[1][side]
        inner_lane_edge = center_line

        # skip this section if the 's' value is already past the end
        if lane_section_index < len(lane_sections) - 1:
            if geometry[geometry_start_index][0] >= lane_sections[lane_section_index + 1][0]:
                continue

        for lane_index, lane in enumerate(lanes):
            outer_lane_edge = np.zeros_like(inner_lane_edge)

            for index in range(geometry_start_index, len(geometry)+1):
                if index == len(geometry):
                    break

                s = geometry[index][0]
                if lane_section_index < len(lane_sections) - 1:
                    if s >= lane_sections[lane_section_index + 1][0]:
                        break

                x = geometry[index][1]
                y = geometry[index][2]
                heading = geometry[index][3]

                width_index = 0
                widths = lane["widths"]
                if width_index >= len(widths) - 1:
                    pass
                else:
                    while s >= widths[width_index + 1, 0]:
                        width_index += 1
                        if width_index == len(widths) - 1:
                            break

                ds = (s - widths[width_index, 0])
                width = (
                    widths[width_index, 1]
                    + widths[width_index, 2] * ds
                    + widths[width_index, 3] * ds**2
                    + widths[width_index, 4] * ds**3
                )
                outer_lane_edge[index, 0] = inner_lane_edge[index, 0] + sign * width * np.cos(heading + np.pi / 2)
                outer_lane_edge[index, 1] = inner_lane_edge[index, 1] + sign * width * np.sin(heading + np.pi / 2)

            polygon = np.concatenate(
                [
                    inner_lane_edge[geometry_start_index:index, :],
                    np.flip(outer_lane_edge[geometry_start_index:index, :], axis=0),
                ],
                axis=0,
            )

            if len(polygon) < 3:
                print(f"Empty polygon: {road_id}, road_type: {lane['type']}")
                break

            polygons.append(polygon)
            road_types.append(lane["type"])

            # move the inner lane edge to the outer lane edge
            inner_lane_edge = outer_lane_edge

        # move the geometry start index to the next lane section
        geometry_start_index = index-1

    return polygons, road_types


def construct_object_polygons(road):
    geometry = road["geometry"]
    objects = road["objects"]

    polygons = []
    road_types = []
    for object in objects:
        s = object["s"]

        # TODO: This works for almost all crosswalks but does not account for the orientation which
        #       may be the reason for the occasional misalignment.  Good enough for now.
        center_x = np.interp(s, geometry[:, 0], geometry[:, 1]) + object["t"] * np.cos(object["heading"] + np.pi / 2)
        center_y = np.interp(s, geometry[:, 0], geometry[:, 2]) + object["t"] * np.sin(object["heading"] + np.pi / 2)

        heading = np.interp(s, geometry[:, 0], geometry[:, 3]) + object["heading"]

        # rotate the object outline to match the road heading
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        corner_points = np.dot(rotation_matrix, np.array(object["outline"]).T).T

        # calculate the corner points of the crosswalk
        corner_points = corner_points + np.array([center_x, center_y])

        polygons.append(corner_points)
        road_types.append(object["type"])

    return polygons, road_types

def render( polygons, road_types, group_name, groups, map_origin, map_width, map_height, map_resolution):

    image_width = map_width
    image_height = map_height

    # create a blank image
    img = np.zeros((image_height, image_width, 3), np.uint8)

    for poly, road_type in zip(polygons, road_types):
        colour = groups.lookup_element(group_name, road_type)
        if colour is not None:
            points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
            cv2.fillPoly(img, [points], color=colour)

    return img


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
        else:
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
    roads = []
    mins = []
    maxs = []
    for road_data in roads_data:
        id = int(road_data.get("id"))

        name = road_data.get("name")
        length = float(road_data.get("length"))

        geometry, geo_mins, geo_maxs = parse_geometry(road_data)
        mins.append(geo_mins)
        maxs.append(geo_maxs)

        lanes = parse_lanes(road_data)
        objects = parse_objects(road_data)

        road = {"id": id, "name": name, "length": length, "geometry": geometry, "lanes": lanes, "objects": objects}
        roads.append(road)

    min_x, min_y = np.min(np.array(mins), axis=0)
    max_x, max_y = np.max(np.array(maxs), axis=0)

    print(f"Map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    map_origin = [min_x - MAP_MARGIN, min_y - MAP_MARGIN]

    map_width = int((max_x - min_x + 2 * MAP_MARGIN) / MAP_RESOLUTION)
    map_height = int((max_y - min_y + 2 * MAP_MARGIN) / MAP_RESOLUTION)

    # # draw the roads
    # centers_img = np.zeros((map_height, map_width, 3), np.uint8)
    # for road in roads:
    #     draw_road_polyline(centers_img, road["geometry"], map_origin, MAP_RESOLUTION)

    # consruct the road polygons
    polygons = []
    road_types = []
    for road in roads:

        left_polys, left_types = construct_polygon_array( road=road, side="left" )
        polygons.extend(left_polys)
        road_types.extend(left_types)

        right_polys, right_types = construct_polygon_array( road=road, side="right" )
        polygons.extend(right_polys)
        road_types.extend(right_types)

        object_polys, object_types = construct_object_polygons(road)
        polygons.extend(object_polys)
        road_types.extend(object_types)

    # render the map
    for group_name in groups.get_group_names():
        img = render(polygons, road_types, group_name, groups, map_origin, map_width, map_height, MAP_RESOLUTION)
        # write the map to a file

        # convert the image format to 3 x X x Y
        np_img = img.transpose(2, 0, 1)
        filename = os.path.join(args.output, f"{prefix}{args.town}_{group_name}_map.npy")
        np.save(filename, np_img)

        if not args.no_render:
            filename = os.path.join(args.output, f"{prefix}{args.town}_{group_name}_map.png")
            cv2.imwrite(filename, img)

    # write the map parameters to a file
    parameters = {
        "town": args.town,
        "prefix": args.prefix,
        "map_origin": map_origin,
        "map_width": map_width,
        "map_height": map_height,
        "map_resolution": MAP_RESOLUTION,
    }
    with open(os.path.join(args.output, f"{prefix}{args.town}_map_parameters.json"), "w") as f:
        json.dump(parameters, f)

    print("done")


if __name__ == "__main__":
    main()
