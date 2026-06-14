import cdsapi

client = cdsapi.Client()


for month in range(1,13):

	client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable" : [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_precipitation",
],
        "year": ["2025"],
        "month": [f"{month}"],
         "day": [f"{d:02d}" for d in range(1, 31)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [50.25, -5.25, 50.00, -5.00],
        "data_format": "netcdf",
    },
    f"data/weather_{month}_2025.nc",
)
