import requests
import pandas as pd
from datetime import datetime
import time
import random
import os

INPUT_FILE = "flight_legs_with_delays.csv"
OUTPUT_FILE = "flight_legs_with_weather_all-2.csv"
CACHE_FILE = "weather_cache_log.csv"

# --- Round down time to full hour ---
def floor_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

# --- Load persistent cache ---
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            cache = {
                (round(r["latitude"], 3), round(r["longitude"], 3),
                 str(r["date"]), int(r["hour"])): r.to_dict()
                for _, r in df.iterrows()
            }
            print(f"🔁 Loaded {len(cache)} cached weather records.")
            return cache
        except Exception as e:
            print(f" Failed to load cache: {e}")
    return {}

# --- Save cache safely ---
def save_cache(cache):
    if cache:
        pd.DataFrame(cache.values()).to_csv(CACHE_FILE, index=False)
        print(f"💾 Cache saved ({len(cache)} entries).")

# --- Fetch weather for given location and hour ---
def fetch_weather(lat, lon, date_str, hour, cache):
    key = (round(lat, 3), round(lon, 3), date_str, hour)
    if key in cache:
        return cache[key]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,precipitation,"
        f"snowfall,visibility,cloudcover,windspeed_10m,winddirection_10m,weathercode"
        f"&timezone=UTC"
    )

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            print(f" Attempt {attempt+1}/3 failed for {key}: {e}")
            time.sleep(2 ** attempt)
    else:
        print(f"❌ Failed all retries for {key}")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f" No hourly data for {key}")
        return None

    target_ts = f"{date_str}T{hour:02d}:00"
    if target_ts not in times:
        print(f" No match for {target_ts} at {key}")
        return None

    idx = times.index(target_ts)
    weather = {
        "date": date_str,
        "hour": hour,
        "latitude": lat,
        "longitude": lon,
        "temp": hourly.get("temperature_2m", [None]*len(times))[idx],
        "humidity": hourly.get("relative_humidity_2m", [None]*len(times))[idx],
        "windspeed": hourly.get("windspeed_10m", [None]*len(times))[idx],
        "winddirection": hourly.get("winddirection_10m", [None]*len(times))[idx],
        "pressure": hourly.get("pressure_msl", [None]*len(times))[idx],
        "precip": hourly.get("precipitation", [None]*len(times))[idx],
        "snow": hourly.get("snowfall", [None]*len(times))[idx],
        "visibility": hourly.get("visibility", [None]*len(times))[idx],
        "cloudcover": hourly.get("cloudcover", [None]*len(times))[idx],
        "conditions": hourly.get("weathercode", [None]*len(times))[idx],
    }

    cache[key] = weather
    time.sleep(random.uniform(0.2, 0.5))  # rate limiting
    return weather


if __name__ == "__main__":
    flights = pd.read_csv(INPUT_FILE)
    print(f" Loaded {len(flights)} flights from {INPUT_FILE}")

    # --- Fix any column name issues ---
    if "acutal_departure_time" in flights.columns:
        flights.rename(columns={"acutal_departure_time": "actual_departure_time"}, inplace=True)
    if "Origin-Latitute" in flights.columns:
        flights.rename(columns={"Origin-Latitute": "Origin-Latitude"}, inplace=True)

    # --- Resume progress if file exists ---
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        start_index = len(existing)
        print(f" Resuming from flight {start_index}/{len(flights)}")

    cache = load_cache()

    dep_weather_list = []
    arr_weather_list = []

    for idx, row in flights.iloc[start_index:].iterrows():
        global_index = start_index + idx + 1
        print(f"\n✈️ Processing flight {global_index}/{len(flights)}")

        # --- Departure weather ---
        dep_weather = None
        try:
            dep_time = pd.to_datetime(row["scheduled_departure_time"], errors="coerce")
            if pd.notna(dep_time):
                dep_floor = floor_hour(dep_time)
                dep_weather = fetch_weather(
                    row["Origin-Latitude"],
                    row["Origin-Longitude"],
                    dep_floor.strftime("%Y-%m-%d"),
                    dep_floor.hour,
                    cache,
                )
        except Exception as e:
            print(f"Departure fetch error: {e}")

        # --- Arrival weather ---
        arr_weather = None
        try:
            arr_time = pd.to_datetime(row["scheduled_arrival_time"], errors="coerce")
            if pd.notna(arr_time):
                arr_floor = floor_hour(arr_time)
                arr_weather = fetch_weather(
                    row["Destination-Latitude"],
                    row["Destination-Longitude"],
                    arr_floor.strftime("%Y-%m-%d"),
                    arr_floor.hour,
                    cache,
                )
        except Exception as e:
            print(f"Arrival fetch error: {e}")

        dep_weather_list.append(dep_weather)
        arr_weather_list.append(arr_weather)

        # --- Periodic saving ---
        if (global_index % 500 == 0) or (global_index == len(flights)):
            print(f"\n Saving progress at flight {global_index}...")
            temp_df = flights.iloc[:global_index].copy()
            temp_df["dep_weather"] = dep_weather_list + [None] * (global_index - len(dep_weather_list))
            temp_df["arr_weather"] = arr_weather_list + [None] * (global_index - len(arr_weather_list))

            dep_df = pd.json_normalize(temp_df["dep_weather"]).add_prefix("dep_")
            arr_df = pd.json_normalize(temp_df["arr_weather"]).add_prefix("arr_")
            partial = pd.concat(
                [temp_df.drop(columns=["dep_weather", "arr_weather"], errors="ignore"), dep_df, arr_df],
                axis=1,
            )

            partial.to_csv(OUTPUT_FILE, index=False)
            save_cache(cache)
            print(f" Progress saved up to flight {global_index}.")

    print("\n🎉 All flights processed successfully!")
    save_cache(cache)
