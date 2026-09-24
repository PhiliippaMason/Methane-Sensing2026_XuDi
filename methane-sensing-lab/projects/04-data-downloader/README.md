# 04-data-downloader

A multi-source satellite and meteorological data downloading tool.

This module provides a command-line interface for downloading remote sensing and weather datasets from multiple sources, including Sentinel-2, Sentinel-2 metadata, EMIT, and ERA5. It is designed to support geospatial workflows that require consistent data acquisition over a defined area of interest and time range.

## Supported Data Sources

- **Sentinel-2**
- **Sentinel-2 Metadata**
- **EMIT**
- **ERA5**

## Features

- Download satellite and meteorological datasets from multiple sources
- Support area-of-interest based data filtering based on specifc location
- Support time-range based queries
- Support Query metadata including sunZenithAngle, viewZenithAngle, sunAzimuthAngle, viewAzimuthAngle
- Support cloud-cover filtering for optical satellite data
- Suitable for remote sensing, atmospheric analysis, and environmental monitoring workflows

## Project Structure

```text
04-data-downloader/
├── data_download.ipynb
├── README.md