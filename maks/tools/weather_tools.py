"""Live weather via Open-Meteo — free, no API key required.

Kept as a plain function (not an MCP tool) because main.py's wake-up greeting
calls it directly and synchronously — that's an internal convenience call,
not an LLM tool call. The LLM-facing version lives in
maks/mcp_servers/api_connectors.py and just wraps this function.
"""

from __future__ import annotations

import requests

from maks.settings import settings

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> short human description.
_WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with rime",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}


def _resolve_location() -> tuple[float, float, str]:
    if settings.weather_lat is not None and settings.weather_lon is not None:
        return settings.weather_lat, settings.weather_lon, settings.weather_city

    resp = requests.get(_GEOCODE_URL, params={"name": settings.weather_city, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Could not find a location matching '{settings.weather_city}'")
    place = results[0]
    return place["latitude"], place["longitude"], place.get("name", settings.weather_city)


def get_current_weather(city: str | None = None) -> str:
    """Return a short natural-language weather summary for `city` (or the
    configured default city). Raises on network/lookup failure — callers
    should catch and fall back gracefully rather than inventing a number.
    """
    if city:
        resp = requests.get(_GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return f"I couldn't find weather data for {city}."
        lat, lon, place_name = results[0]["latitude"], results[0]["longitude"], results[0].get("name", city)
    else:
        lat, lon, place_name = _resolve_location()

    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json().get("current", {})

    temp = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    code = current.get("weather_code")
    wind = current.get("wind_speed_10m")
    condition = _WEATHER_CODES.get(code, "unclear conditions")

    return (
        f"In {place_name}, it's currently {temp}°C ({condition}), "
        f"feels like {feels_like}°C, wind at {wind} km/h."
    )
