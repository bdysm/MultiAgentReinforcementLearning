class Assets:

    def __init__(self):
        self.assets_list = []

    def append_asset(self, asset):
        self.assets_list.append(asset)


class SingleAsset:

    def __init__(self, name, data, asset_type=None):
        self.name = name
        self.type = asset_type
        self.data = data


class AssetData:

    def __init__(self, time_line, origin_data, relative_data, sswn_data):
        self.time_line = time_line
        self.origin_data = origin_data
        self.relative_data = relative_data
        self.sswn_data = sswn_data
