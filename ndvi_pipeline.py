import os
from datetime import datetime
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, MimeType, bbox_to_dimensions

# Set Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = 'ENTER YOUR CLIENT ID'
config.sh_client_secret = 'ENTER YOUR CLIENT SECRET ID'

# Function to download Sentinel satellite images
def download_images(bbox, time_interval, output_folder, resolution=10, satellite='SENTINEL_2A', cloud_cover=10):
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for start_date, end_date in time_interval:
        bbox_obj = BBox(bbox, CRS.WGS84)
        dwn_size = bbox_to_dimensions(bbox_obj, resolution=resolution)
        print(f"Image shape at {resolution} m resolution: {dwn_size} pixels")
        
        request = SentinelHubRequest(
            evalscript='''//VERSION=3
                function setup() {
                    return {
                        input: ["B02", "B03", "B04", "B08"],
                        output: { bands: 4 }
                    };
                }

                function evaluatePixel(sample) {
                    return [ sample.B08, sample.B04, sample.B03, sample.B02 ];
                }
            ''',
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=("2022-01-01", "2024-03-06"),
                    other_args={
                        "dataFilter": {         
                        'maxCloudCoverage': cloud_cover,
                        'resolution': f'{resolution}m',
                        #'collection': data_collection_str,
                        'satellite': satellite}
                    }
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=dwn_size,
            config=config
        )
        
        # Get the data
        data = request.get_data()

        # Save the images
        for idx, image_data in enumerate(data):
            image_filename = f'{start_date}_{end_date}_{idx}.tif'
            image_filepath = os.path.join(output_folder, image_filename)

            # Write the image data to a GeoTIFF file using rasterio
            with rasterio.open(image_filepath, 'w', driver='GTiff', width=image_data.shape[1], height=image_data.shape[0],
                               count=image_data.shape[2], dtype=str(image_data.dtype), crs='EPSG:4326',
                               transform=Affine.translation(bbox_obj.min_x, bbox_obj.max_y) * Affine.scale(resolution, -resolution)) as dst:
                # Write the data from all bands
                for i in range(image_data.shape[2]):
                    dst.write(image_data[:, :, i], indexes=i+1)


# Function to calculate NDVI statistics
def calculate_ndvi_stats(image_path):
    with rasterio.open(image_path) as src:
        red = src.read(3)
        nir = src.read(4)

    ndvi = (nir - red) / (nir + red)
    ndvi_mean = np.nanmean(ndvi)
    ndvi_median = np.nanmedian(ndvi)
    ndvi_max = np.nanmax(ndvi)
    ndvi_min = np.nanmin(ndvi)

    return ndvi_mean, ndvi_median, ndvi_max, ndvi_min

# Function to plot NDVI over time
def plot_ndvi_over_time(ndvi_stats, dates):
    plt.plot(dates, ndvi_stats, marker='o')
    plt.xlabel('Date')
    plt.ylabel('NDVI')
    plt.title('Normalized Difference Vegetation Index (NDVI) Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # User inputs
    aoi_bbox = (-1.821087,5.567851,-1.709975,5.686242)
    start_date = '2022-01-01'  # Start date for image acquisition
    end_date = '2024-03-06'  # End date for image acquisition
    output_folder = 'output_images'  # Output folder to save downloaded images
    resolution = 10  # Resolution in meters
    #satellite = 'Sentinel-2A'  # Satellite type
    cloud_cover = 10  # Maximum cloud cover percentage

    # Download images
    download_images(aoi_bbox, [(start_date, end_date)], output_folder, resolution, cloud_cover)

    # Calculate NDVI statistics
    ndvi_stats = []
    dates = []
    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                ndvi_mean, _, _, _ = calculate_ndvi_stats(image_path)
                ndvi_stats.append(ndvi_mean)
                dates.append(datetime.strptime(file[:10], '%Y-%m-%d'))

    # Plot NDVI over time
    plot_ndvi_over_time(ndvi_stats, dates)
    print(dates)
    print(ndvi_stats)


if __name__ == "__main__":
    main()