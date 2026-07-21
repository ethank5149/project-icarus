import cdsapi
import os
from tqdm import tqdm
import requests

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cdsapirc')
with open(config_path, 'r') as f:
    lines = f.read().splitlines()

url = None
key = None
for line in lines:
    if line.startswith('url:'):
        url = line.split(':', 1)[1].strip()
    elif line.startswith('key:'):
        key = line.split(':', 1)[1].strip()

c = cdsapi.Client(url=url, key=key)
# Download 1997 months 9-12 manually
target_years = [
    '2015',
    # '2016', '2017', '2018', '2019', '2020',
    # '2021', '2022', '2023', '2024', '2025',
    # '2026'
]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]


for year in target_years:
    for month in months:
        output_filename = f"era5_global_native_025_{year}_{month}.grib"
        
        if os.path.exists(output_filename):
            continue
            
        print(f"\nProcessing Batch: {year}-{month}")
        
        try:
            # 2. Request the execution pointer from the server
            result = c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': ['temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity'],
                    'pressure_level': [
                        # Stratosphere / Upper Edge (Crucial for high-altitude aerodynamic tracking)
                        '1', '2', '3', '5', '7', '10', '20', '30', '50', '70', 
    
                        # Tropopause / Jet Stream Core (High-density sampling to capture maximum wind shear)
                        '100', '150', '200', '250', '300', '400', '500', 
    
                        # Planetary Boundary Layer (Dense sampling for silo departure and terminal re-entry)
                        '700', '850', '925', '1000'
                    ],
                    'year': [year],
                    'month': [month],
                    'day': days,
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                }
            )
            
            # 3. Handle the download manually to inject a real-time progress bar
            # This triggers the moment the server status changes from 'running' to 'completed'
            download_url = result.location
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            print(f"Server processing complete. Streaming {total_size / (1024**2):.2f} MB to disk...")
            
            with open(output_filename, 'wb') as file, tqdm(
                desc=output_filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))
                        
            print(f"Successfully finalized: {output_filename}")
            
        except Exception as e:
            print(f"Execution disrupted on {year}-{month}: {e}")
