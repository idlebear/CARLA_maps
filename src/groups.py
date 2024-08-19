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
                groups[group_name] = group["layers"]
            except KeyError:
                print(f"No layers found in the group: {group_name}")
                continue

        return groups

    def get_group_names(self):
        return [ k for k in self.groups.keys() ]

    def get_num_layers(self, group_name):
        '''
        Allocate an array of the same size as the number of layers in the group
        '''
        if self.groups is None:
            return None

        return len(self.groups[group_name])

    def get_layers(self, group_name):
        '''
        Return the layers in a group
        '''
        if self.groups is None:
            return None

        return self.groups[group_name]


if __name__ == "__main__":
    layers_file = 'groups.json'
    groups = GroupData(layers_file)

    print(groups)

    print( f"Driving: {groups._lookup_element_color('driving', 'driving')} ")
    print( f"Pedestrian: {groups._lookup_element_color('pedestrian', 'crosswalk')} ")


