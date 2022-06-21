from datetime import datetime
import ezodf
from assets import SingleAsset, AssetData, Assets


class LoadData:

    def __init__(self, filepath):
        self.file = filepath
        self.assets = self.get_all_assets_data()

    def get_all_assets_data(self):
        doc = ezodf.opendoc(self.file)
        all_assets = Assets()
        print('Available {in input file} assets for portfolios:')
        for sheet in doc.sheets:
            print(sheet.name)
            column_count = sheet.ncols()
            asset_type = None
            if column_count > 4:
                cell = sheet['E1']
                asset_type = cell.value

            data = []
            time_line = []
            sswn_data = []
            relative_data = []

            for i, row in enumerate(sheet.rows()):

                for j, cell in enumerate(row):

                    if (j == 0) and cell.value is not None:
                        time_line.append(datetime.strptime(cell.value, '%Y-%m-%d'))
                    if (j == 1) and cell.value is not None:
                        data.append(cell.value)
                    if (j == 2) and cell.value is not None:
                        relative_data.append(cell.value)
                    if (j == 3) and cell.value is not None:
                        sswn_data.append(cell.value)

            asset_data = AssetData(time_line, data, relative_data, sswn_data)
            single_asset = SingleAsset(sheet.name, asset_data, asset_type=asset_type)
            all_assets.append_asset(single_asset)

        return all_assets
