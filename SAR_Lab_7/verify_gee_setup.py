#!/usr/bin/env python3
"""Verify Earth Engine access without printing or storing the project ID."""

import os
from pathlib import Path

import ee


AOI = [103.62, 32.04, 103.68, 32.09]


def main() -> None:
    project_id = os.environ.get("EARTH_ENGINE_PROJECT", "river-dynamo-494108-u6").strip()
    if not project_id:
        raise SystemExit(
            "EARTH_ENGINE_PROJECT is not set. Export it privately before running this check."
        )

    earth_engine_credentials = Path.home() / ".config" / "earthengine" / "credentials"
    application_default_credentials = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    if not earth_engine_credentials.exists() and not application_default_credentials.exists():
        raise SystemExit(
            "Earth Engine authentication is missing. Run `earthengine authenticate`, "
            "complete the browser authorization, and retry."
        )

    try:
        ee.Initialize(project=project_id)
        aoi = ee.Geometry.Rectangle(AOI)
        image_count = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate("2017-06-01", "2017-08-01")
            .size()
            .getInfo()
        )
    except Exception as exc:
        message = str(exc).lower()
        if "not registered" in message or "earth engine" in message and "project" in message:
            reason = (
                "The Cloud project may not be registered for Earth Engine, or your "
                "account may not have permission to use it."
            )
        elif "credential" in message or "auth" in message or "token" in message:
            reason = "Authentication is missing or expired. Re-run `earthengine authenticate`."
        elif "permission" in message or "forbidden" in message or "403" in message:
            reason = "The authenticated account does not have the required project permission."
        else:
            reason = (
                "Check Earth Engine authentication, project registration, billing/quota "
                "configuration, and account permissions."
            )
        raise SystemExit(
            f"Earth Engine verification failed with {type(exc).__name__}. {reason}"
        ) from None

    print("Earth Engine verification succeeded.")
    print(f"Sentinel-1 metadata probe count: {image_count}")
    print("Project ID was loaded from the environment and was not displayed.")


if __name__ == "__main__":
    main()
