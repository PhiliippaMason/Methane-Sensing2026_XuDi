import math
from concurrent.futures import ThreadPoolExecutor
import cdsapi
import pandas as pd

# from matplotlib import pyplot as plt


def download_wind(year, month, day, locations, filepath):
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": locations,
        },
        filepath,
    )


def process_data(data_row):
    """Process individual data rows and download the wind file."""
    year, month, day = data_row["path"].split("/")[-1].split("-")
    x1, y1, x2, y2 = (
        math.ceil(data_row["latitude"]),
        math.floor(data_row["longitude"]),
        math.floor(data_row["latitude"]),
        math.ceil(data_row["longitude"]),
    )
    locations = [x1, y1, x2, y2]
    subname = str(data_row["index"])
    filepath = f"./dataset/wind/{subname}.nc"

    # Download
    download_wind(year, month, day, locations, filepath)
    print(f"Downloaded: {locations}, {filepath}")


# Filter out plume data
data = pd.read_csv("data_details.csv")
plume_data = data[data["plume"] == 2]

# Create a thread pool and process the data
with ThreadPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(process_data, row) for _, row in plume_data.iterrows()]

for future in futures:
    future.result()
