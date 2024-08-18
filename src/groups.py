import json

class GroupData:
    def __init__(self, filename):
        self.filename   = filename
        self.groups = self._parse_groups()

    def __str__(self):
        groups_str = ""
        for group in self.groups:
            groups_str += f"{group}: {self.groups[group]}\n"
        return f"GroupData: {self.filename}, {groups_str}"

    def __repr__(self):
        return self.__str__()

    def _parse_groups(self):
        with open(self.filename, 'r') as f:
            groups_data = json.load(f)

        try:
            groups_data = groups_data["groups"]
        except KeyError:
            print("No groups found in the file")
            return None

        groups = {}
        for group in groups_data:
            group_name = group["name"]
            try:
                default_color = group["default color"]
            except KeyError:
                default_color = (255, 255, 255)

            elements = {}
            try:
                elements_data = group["elements"]
            except KeyError:
                print(f"No elements found in the group: {group_name}")
                continue

            for element in elements_data:
                try:
                    element_name = element["element"]
                except KeyError:
                    print(f"Badly formed 'element' name found in the group: {group_name}")
                    continue

                try:
                    element_color = element["color"]
                except KeyError:
                    element_color = default_color

                elements[element_name] = element_color

            groups[group_name] = elements

        return groups

    def get_group_names(self):
        return self.groups.keys()

    def lookup_element( self, group_name, element ):
        '''
        Look up the color of an element in a group.  If the element is not found, return None
        '''
        if self.groups is None:
            return None

        try:
            color = self.groups[group_name][element]
        except KeyError:
            return None

        return color

if __name__ == "__main__":
    layers_file = 'groups.json'
    groups = GroupData(layers_file)

    print(groups)

    print( f"Driving: {groups._lookup_element_color('driving', 'driving')} ")
    print( f"Pedestrian: {groups._lookup_element_color('pedestrian', 'crosswalk')} ")


