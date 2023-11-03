# Server request handling
import requests
from bs4 import BeautifulSoup

# String manipulation
import re
from datetime import datetime

# Directory and time functionality
import os
import time


def run_flake_simulation(latitude, longitude, lakeDepth, extinctionCoefficient, lakeFetch,
                         surfaceTemperature=9.44, meanTemperature=9.44, bottomTemperature=9.44,
                         mixedLayerThickness=0, iceThickness=0):

    # Extinction Coefficient can only take certain values.
    assert extinctionCoefficient in [0.4, 1, 2, 4]

    # Define constants for URLs and headers
    ORIGIN_URL = "http://www.flake.igb-berlin.de"
    GET_URL = f"{ORIGIN_URL}/model/run"
    POST_URL = f"{ORIGIN_URL}/model/running"

    # Make a GET request to the website to initialize session and get CSRF token
    response = requests.get(GET_URL)
    print("====================")

    # Check if the GET request was successful
    if response.status_code == 200:
        cookie = response.cookies["_csrf"]  # Extract session cookie

        # Parse the HTML to extract CSRF token
        soup = BeautifulSoup(response.text, 'html.parser')
        csrf_token = soup.find('meta', {'name': 'csrf-token'})['content']

        print("GET request successful")
        print("CSRF TOKEN:", csrf_token)
        print("COOKIE:", cookie)
        print("====================")
    else:
        # Handle non-200 responses by saving the response to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"get_response_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Unexpected response to GET request. Response with status code: <{response.status_code}> has been saved to {filename}.")
        raise SystemError("GET request failed")

    # Prepare headers for the POST request
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0; For academic research purposes)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': ORIGIN_URL,
        'Connection': 'keep-alive',
        'Referer': GET_URL,
        'Cookie': f'_csrf={cookie}; cookieconsent_status=dismiss'
    }

    # Prepare data for the POST request
    data = {
        '_csrf': csrf_token,
        'RunForm[latitude]': f'{latitude}',
        'RunForm[longitude]': f'{longitude}',
        'RunForm[lakeDepth]': f'{lakeDepth}',
        'RunForm[extinctionCoefficient]': f'{extinctionCoefficient}',
        'RunForm[lakeFetch]': f'{lakeFetch}',
        'RunForm[perpetual]': '1',
        'RunForm[advancedConfig]': '0',
        'RunForm[surfaceTemperature]': f'{surfaceTemperature}',
        'RunForm[meanTemperature]': f'{meanTemperature}',
        'RunForm[bottomTemperature]': f'{bottomTemperature}',
        'RunForm[mixedLayerThickness]': f'{mixedLayerThickness}',
        'RunForm[iceThickness]': f'{iceThickness}'
    }

    # Make a POST request with CSRF token, cookie, and data
    response = requests.post(POST_URL, data=data, headers=headers)

    # Check if the POST request was successful
    if response.status_code == 200:
        print("POST request successful")
        print("====================")
    else:
        # Handle non-200 responses by saving the response to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"post_response_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Unexpected response to POST request. Response with status code: <{response.status_code}> has been saved to {filename}.")
        raise SystemError("POST request failed")


    # Parse the HTML text to extract the result ID
    post_response = response.text
    match = re.search(r'/model/check-meteo-file\?id=(\d+)', post_response)
    if match:
        result_id = match.group(1)
        print(f"Received valid ID: <{result_id}>")
        print("====================")
    else:
        print("No result ID found in response")
        raise SystemError("Failed to retrieve result ID")


    # Wait 45 seconds before querying server for CSV file.
    print("Waiting for simulation to finish...")
    time.sleep(45)

    # Download CSV file with retries
    max_retries = 8  # 8 retries * 15 seconds = 2 minutes
    retry_interval = 15  # 15 seconds
    for i in range(max_retries):
        csv_url = f"{ORIGIN_URL}/model/download-t-map-data?id={result_id}"
        csv_response = requests.get(csv_url)

        # Checks how the server responded to file query
        if csv_response.status_code == 200:
            # Server responded OK, the file is ready to download
            print("====================")
            print("File ready! Downloading...")

            # Create folder if it doesn't exist
            os.makedirs("Datasources", exist_ok=True)

            # Save CSV file
            csv_filename = f"Datasources/result_{result_id}.csv"
            with open(csv_filename, 'wb') as f:
                f.write(csv_response.content)
            print(f"CSV file has been successfully downloaded and saved as {csv_filename}")
            print("====================")
            return csv_filename

        # If the file is not ready, the server will respond with a 404 code.
        elif csv_response.status_code in (404, 500):
            if i < max_retries - 1:
                print("File not ready yet. Retrying in 15 seconds...")
                time.sleep(retry_interval)
            else:
                print("Timed out waiting for file to be ready.")
                raise SystemError("Server failed to provide results file")
        else:
            print(f"Unexpected response while querying file. Status code: <{csv_response.status_code}>")
            raise SystemError("Unexpected server response to file query")
