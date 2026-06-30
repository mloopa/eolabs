#!/usr/bin/env python3
"""Build the final SAR Lab 7 PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


LAB_DIR = Path(__file__).resolve().parent
REPORT_DIR = LAB_DIR / "report"
FIGURES_DIR = LAB_DIR / "figures"
OUTPUT_DIR = LAB_DIR / "output"
PDF_FILE = REPORT_DIR / "SAR_Lab_7_report.pdf"
SOURCE_FILE = REPORT_DIR / "SAR_Lab_7_report.md"


def read_outputs() -> dict:
    coherence = pd.read_csv(
        OUTPUT_DIR / "event_coherence" / "20170607_20170725_coherence_stats.csv"
    ).set_index("sample")
    s1_stats = pd.read_csv(OUTPUT_DIR / "earth_engine" / "sentinel1_vv_stats.csv").set_index(
        "sample"
    )
    s1_selection = json.loads(
        (OUTPUT_DIR / "earth_engine" / "sentinel1_vv_selection.json").read_text()
    )
    s2_stats = pd.read_csv(OUTPUT_DIR / "earth_engine" / "sentinel2_stats.csv").set_index(
        "sample"
    )
    s2_selection = json.loads(
        (OUTPUT_DIR / "earth_engine" / "sentinel2_selection.json").read_text()
    )
    return {
        "coherence": coherence,
        "s1_stats": s1_stats,
        "s1_selection": s1_selection,
        "s2_stats": s2_stats,
        "s2_selection": s2_selection,
    }


def fnum(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def make_markdown(values: dict) -> str:
    coherence = values["coherence"]
    s1_stats = values["s1_stats"]
    s1_selection = values["s1_selection"]
    s2_stats = values["s2_stats"]
    s2_selection = values["s2_selection"]

    coh_land = coherence.loc["Landslide-centered sample", "mean_coherence"]
    coh_ref = coherence.loc["Stable reference sample", "mean_coherence"]
    vv_land = s1_stats.loc["Landslide-centered sample", "vv_change_db_mean"]
    vv_ref = s1_stats.loc["Stable reference sample", "vv_change_db_mean"]
    affected = s2_stats.loc["Classified affected area"]
    stable = s2_stats.loc["Stable reference sample"]

    return f"""# SAR Lab 7 Report: Xinmo Landslide Analysis

Event: Xinmo landslide, Sichuan Province, China

Event date: 2017-06-24

AOI: 103.62 to 103.68 E, 32.04 to 32.09 N

LiCSAR frame: 062D_05831_131313

## 1. Introduction and Study Objective

This report recreates the pre-failure deformation analysis requested in SAR Lab 7
using public COMET-LiCSAR interferograms and MintPy SBAS processing. It then
compares event-spanning LiCSAR coherence, Sentinel-1 GRD VV backscatter change,
and Sentinel-2 optical changes for mapping the 2017 Xinmo landslide.

The goal is not to exactly reproduce the supplied PS-based study. Instead, the
goal is to evaluate what can be recovered from publicly available pre-processed
LiCSAR interferograms and open Sentinel data.

## 2. Data and Study Area

The study area is centered near 103.6506 E, 32.0661 N. The working AOI is
103.62, 32.04, 103.68, 32.09. The SAR time-series analysis uses LiCSAR frame
062D_05831_131313. Event mapping uses one event-spanning LiCSAR coherence pair,
Sentinel-1 GRD VV images before and after the failure, and Sentinel-2 imagery
before and after the failure.

## 3. LiCSAR and MintPy SBAS Methodology

Pre-event LiCSAR unwrapped phase and coherence products were downloaded for
interferograms before 2017-06-24. A 72-day temporal-baseline selection produced
a disconnected network, so the maximum temporal baseline was expanded to 108
days. The final MintPy input stack contains 42 acquisitions and 127
interferograms from 2014-10-09 to 2017-06-07. The perpendicular baseline range
is approximately -178 m to +189 m.

The LiCSAR products were converted to MintPy HDF5 inputs, subset to the AOI, and
processed with `smallbaselineApp.py`. Network inversion used coherence weighting.

## 4. Reference-Point Selection

The automatic reference point was used for the first run. The final run used a
manual local reference point at Y/X = 28,31, lat/lon = 32.06233, 103.65083. This
point was selected because it is close to the landslide but outside the
interpreted affected area and has high temporal coherence in the MintPy result.

The manual reference reduces sensitivity to long-wavelength atmospheric,
seasonal, or topographic differences between a distant automatic reference and
the landslide area. The tradeoff is that all displacement is interpreted
relative to this selected pixel.

## 5. Pre-Failure Time-Series Results

The representative landslide pixel at Y/X = 29,29, lat/lon = 32.06133,
103.64883 shows a broad negative LOS trend of about -1.07 +/- 0.08 cm/year.
This supports pre-event motion at the landslide, but the available stack does
not show a clear late-stage acceleration.

The key limitation is acquisition timing. The MintPy stack ends on 2017-06-07,
which is 17 days before the 2017-06-24 failure. The assignment notes that the
last few points in the supplied PS-based paper's figure were not processed by
COMET-LiCSAR. Therefore, this SBAS recreation can support pre-event motion but
cannot independently confirm the final acceleration reported by the published
PS-based analysis.

## 6. Coherence Analysis

The selected event-spanning LiCSAR pair is 20170607_20170725, with a 48-day
temporal baseline. The mean coherence in the landslide-centered sample is
{fnum(coh_land)}. The mean coherence at the manual-reference sample is
{fnum(coh_ref)}. Both local samples are very low coherence, so this pair does
not cleanly separate the landslide from nearby low-coherence terrain.

Coherence loss is therefore weak supporting evidence in this workflow. It is
not sufficient as a standalone landslide boundary because the mountainous,
vegetated AOI has broad temporal and geometric decorrelation.

## 7. Sentinel-1 VV Analysis

The Sentinel-1 GRD comparison uses descending relative orbit
{s1_selection["relative_orbit"]}. The pre-event acquisition is
{s1_selection["pre_date"]}, and the post-event acquisition is
{s1_selection["post_date"]}. VV change is computed as post-event VV minus
pre-event VV in dB.

The landslide-centered sample has mean VV change of {fnum(vv_land, 2)} dB. The
stable comparison sample has mean VV change of {fnum(vv_ref, 2)} dB. The local
contrast is small, and the map is affected by speckle, moisture, and terrain
geometry. VV change is therefore weak for mapping the landslide in this case.

## 8. Sentinel-2 and Spectral-Index Analysis

Sentinel-2 surface-reflectance imagery did not provide a usable 2017 pre/post
pair for this AOI, so the workflow used `COPERNICUS/S2_HARMONIZED` Level-1C
top-of-atmosphere imagery. The pre-event image date is
{s2_selection["pre_date"]}; the post-event image date is
{s2_selection["post_date"]}. Both have AOI clear fraction 1.000 after the
QA60 cloud/cirrus mask.

The final affected-area mask uses a post-event optical-scar rule:
post brightness >= 0.115, post NDVI <= 0.0, and NDVI change <= -0.10. The
largest connected component inside a 1.8 km landslide-centered search radius is
retained. The resulting affected area is {fnum(s2_selection["affected_area_ha"], 2)}
ha, or {fnum(s2_selection["affected_area_km2"], 4)} km2.

Mean NDVI change in the affected area is {fnum(affected["ndvi_change"])}. The
stable reference sample has mean NDVI change of {fnum(stable["ndvi_change"])}.
This contrast supports vegetation loss or burial by fresh debris in the mapped
scar. BSI is less useful for this pair: the affected area has mean BSI change
of {fnum(affected["bsi_change"])}, so a simple positive-BSI-change threshold
would under-detect the scar.

## 9. Comparison and Discussion

SBAS is useful for pre-failure deformation, but not for mapping the event
extent. The inspected landslide pixel shows a negative pre-event LOS trend, yet
the available LiCSAR stack ends before the final 17 days.

Sentinel-2 true color plus NDVI decrease gives the clearest landslide extent.
The bright post-event scar is visually distinct, and the refined optical mask
captures the source and runout area. Coherence and Sentinel-1 VV change are
weaker in this workflow. The coherence pair has very low coherence in both the
landslide and manual-reference samples, while the Sentinel-1 VV change has only
small local contrast.

## 10. Conclusions

The main conclusions are:

- MintPy SBAS supports pre-event motion at the landslide, with an inspected
  landslide-pixel trend of about -1.07 +/- 0.08 cm/year.
- The manually selected reference point gives a local relative basis for the
  time series and is preferable to relying only on an automatic reference.
- The available LiCSAR stack does not include the final days before failure, so
  the late acceleration from the supplied PS-based study cannot be reproduced.
- Coherence loss is weak for this event-spanning pair because both local samples
  have very low coherence.
- Sentinel-1 VV change is also weak because local backscatter contrast is small.
- Sentinel-2 true color plus NDVI decrease provides the clearest spatial map,
  with a refined affected-area estimate of {fnum(s2_selection["affected_area_ha"], 2)}
  ha.
- NDVI change supports vegetation loss or burial; BSI change is not a reliable
  positive-threshold discriminator for this image pair.

## 11. References

- SAR Lab 7 assignment handout, `SAR_Lab_7.pdf`.
- Supplied Xinmo landslide PS-based study: DOI 10.1007/s10346-017-0915-7.
- COMET-LiCSAR public interferogram products, frame 062D_05831_131313.
- Yunjun, Z., Fattahi, H., and Amelung, F. MintPy: a Python package for
  InSAR time series analysis.
- ESA Copernicus Sentinel-1 GRD imagery.
- ESA Copernicus Sentinel-2 imagery.
- Google Earth Engine cloud-processing platform.
"""


def styles():
    sample = getSampleStyleSheet()
    sample["Title"].fontName = "Helvetica-Bold"
    sample["Title"].fontSize = 18
    sample["Title"].leading = 22
    sample["Title"].alignment = TA_CENTER
    sample["Heading1"].spaceBefore = 14
    sample["Heading1"].spaceAfter = 8
    sample["Heading1"].fontSize = 13
    sample["Heading1"].leading = 16
    sample["Heading2"].fontSize = 11
    sample["Heading2"].leading = 14
    sample["BodyText"].fontSize = 9
    sample["BodyText"].leading = 12
    sample.add(
        ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#333333"),
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    sample.add(
        ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontSize=8,
            leading=10,
        )
    )
    return sample


def paragraph(text: str, style) -> Paragraph:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, style)


def add_paragraphs(story: list, text: str, style) -> None:
    for part in text.strip().split("\n\n"):
        story.append(paragraph(" ".join(line.strip() for line in part.splitlines()), style))
        story.append(Spacer(1, 0.06 * inch))


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure(path: str, caption: str, styles_, max_height: float = 3.7 * inch) -> KeepTogether:
    img_path = FIGURES_DIR / path
    return KeepTogether(
        [
            scaled_image(img_path, 7.0 * inch, max_height),
            paragraph(caption, styles_["Caption"]),
        ]
    )


def table(data: list[list[str]], widths: list[float]) -> Table:
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def bullet_list(items: list[str], style) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, style), bulletColor=colors.black) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "SAR Lab 7 - Xinmo landslide report")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(values: dict, markdown_source: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles_ = styles()
    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    coherence = values["coherence"]
    s1_stats = values["s1_stats"]
    s2_stats = values["s2_stats"]
    s2_selection = values["s2_selection"]

    coh_land = coherence.loc["Landslide-centered sample", "mean_coherence"]
    coh_ref = coherence.loc["Stable reference sample", "mean_coherence"]
    vv_land = s1_stats.loc["Landslide-centered sample", "vv_change_db_mean"]
    vv_ref = s1_stats.loc["Stable reference sample", "vv_change_db_mean"]
    affected = s2_stats.loc["Classified affected area"]
    stable = s2_stats.loc["Stable reference sample"]

    story = []
    story.append(Paragraph("SAR Lab 7: Xinmo Landslide Analysis", styles_["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    add_paragraphs(
        story,
        "Event: Xinmo landslide, Sichuan Province, China. Event date: 2017-06-24. "
        "AOI: 103.62 to 103.68 E, 32.04 to 32.09 N. LiCSAR frame: 062D_05831_131313.",
        styles_["BodyText"],
    )

    sections = [
        (
            "1. Introduction and Study Objective",
            "This report recreates the SAR Lab 7 Xinmo landslide workflow using "
            "public COMET-LiCSAR interferograms and MintPy SBAS processing. It then "
            "compares LiCSAR coherence, Sentinel-1 GRD VV change, and Sentinel-2 "
            "optical changes for event mapping. The objective is to evaluate what "
            "can be recovered from public pre-processed interferograms and open "
            "Sentinel data, not to exactly reproduce the supplied PS-based study.",
        ),
        (
            "2. Data and Study Area",
            "The study area is centered near 103.6506 E, 32.0661 N in Sichuan "
            "Province, China. The working AOI is 103.62, 32.04, 103.68, 32.09. "
            "The SAR time-series analysis uses LiCSAR frame 062D_05831_131313. "
            "Event mapping uses an event-spanning LiCSAR coherence pair, "
            "Sentinel-1 GRD VV images, and Sentinel-2 optical imagery.",
        ),
    ]
    for heading, text in sections:
        story.append(Paragraph(heading, styles_["Heading1"]))
        add_paragraphs(story, text, styles_["BodyText"])

    story.append(
        figure(
            "study_area_dem.png",
            "Figure 1. Study area and topographic context for the Xinmo landslide AOI. "
            "Source: lab-generated DEM/context figure; coordinates are longitude and latitude.",
            styles_,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. LiCSAR and MintPy SBAS Methodology", styles_["Heading1"]))
    add_paragraphs(
        story,
        "Pre-event LiCSAR unwrapped phase and coherence products were downloaded for "
        "interferograms before 2017-06-24. A 72-day temporal-baseline selection "
        "produced a disconnected network, so the maximum temporal baseline was "
        "expanded to 108 days. The final MintPy input stack contains 42 "
        "acquisitions and 127 interferograms from 2014-10-09 to 2017-06-07. "
        "The perpendicular baseline range is approximately -178 m to +189 m. "
        "The LiCSAR products were converted to MintPy HDF5 inputs, subset to the "
        "AOI, and processed with coherence-weighted SBAS inversion.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "mintpy_network_overview.png",
            "Figure 2. MintPy interferogram network for the pre-event LiCSAR stack. "
            "The network is connected and spans 2014-10-09 to 2017-06-07. "
            "Source: LiCSAR interferograms processed with MintPy.",
            styles_,
        )
    )

    story.append(Paragraph("4. Reference-Point Selection", styles_["Heading1"]))
    add_paragraphs(
        story,
        "The final run used a manual local reference point at Y/X = 28,31, "
        "lat/lon = 32.06233, 103.65083. This point is close to the landslide but "
        "outside the interpreted affected area and has high temporal coherence "
        "in the MintPy result. The manual reference reduces sensitivity to "
        "long-wavelength atmospheric, seasonal, or topographic differences. The "
        "tradeoff is that all displacement is relative to this selected pixel.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "mintpy_reference_context_temporal_coherence.png",
            "Figure 3. Manual reference context and temporal coherence. The selected "
            "reference point is local to the landslide and coherent in the MintPy result. "
            "Source: LiCSAR/MintPy temporal coherence.",
            styles_,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Pre-Failure Time-Series Results", styles_["Heading1"]))
    add_paragraphs(
        story,
        "The representative landslide pixel at Y/X = 29,29, lat/lon = 32.06133, "
        "103.64883 shows a broad negative LOS trend of about -1.07 +/- 0.08 "
        "cm/year. This supports pre-event motion at the landslide, but the "
        "available stack does not show a clear late-stage acceleration. The key "
        "limitation is acquisition timing: the stack ends on 2017-06-07, which "
        "is 17 days before the 2017-06-24 failure.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "mintpy_velocity_manual_reference.png",
            "Figure 4. MintPy LOS velocity after manual rereferencing. Units are "
            "cm/year. Source: LiCSAR/MintPy SBAS velocity product.",
            styles_,
        )
    )
    story.append(
        figure(
            "mintpy_landslide_timeseries_manual_reference.png",
            "Figure 5. Representative landslide-pixel displacement time series after "
            "manual rereferencing. Units are cm of relative LOS displacement. "
            "Source: LiCSAR/MintPy time-series product.",
            styles_,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("6. Coherence Analysis", styles_["Heading1"]))
    add_paragraphs(
        story,
        f"The selected event-spanning pair is 20170607_20170725 with a 48-day "
        f"temporal baseline. The mean coherence in the landslide-centered sample "
        f"is {fnum(coh_land)}. The mean coherence at the manual-reference sample "
        f"is {fnum(coh_ref)}. Both local samples are very low coherence, so this "
        f"pair does not cleanly separate the landslide from nearby low-coherence "
        f"terrain.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "event_spanning_coherence.png",
            "Figure 6. Event-spanning LiCSAR coherence for pair 20170607_20170725. "
            "Color scale is coherence from 0 to 1. Source: LiCSAR .geo.cc.tif product.",
            styles_,
        )
    )

    story.append(Paragraph("7. Sentinel-1 VV Analysis", styles_["Heading1"]))
    add_paragraphs(
        story,
        f"The Sentinel-1 GRD comparison uses descending relative orbit 62. The "
        f"pre-event acquisition is 2017-06-19 and the post-event acquisition is "
        f"2017-07-13. VV change is post-event VV minus pre-event VV in dB. The "
        f"landslide-centered sample has mean VV change of {fnum(vv_land, 2)} dB, "
        f"while the stable comparison sample has mean VV change of {fnum(vv_ref, 2)} "
        f"dB. The local contrast is small, so VV change is weak for mapping the "
        f"landslide in this case.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "sentinel1_vv_change.png",
            "Figure 7. Sentinel-1 descending VV backscatter before and after the "
            "event, plus post-minus-pre VV change in dB. Source: Copernicus "
            "Sentinel-1 GRD via Google Earth Engine.",
            styles_,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("8. Sentinel-2 and Spectral-Index Analysis", styles_["Heading1"]))
    add_paragraphs(
        story,
        f"Sentinel-2 surface-reflectance imagery did not provide a usable 2017 "
        f"pre/post pair for this AOI, so the workflow used COPERNICUS/S2_HARMONIZED "
        f"Level-1C top-of-atmosphere imagery. The pre-event date is "
        f"{values['s2_selection']['pre_date']}; the post-event date is "
        f"{values['s2_selection']['post_date']}. The final affected-area mask uses "
        f"post brightness >= 0.115, post NDVI <= 0.0, and NDVI change <= -0.10. "
        f"The largest connected component inside a 1.8 km search radius is retained. "
        f"The resulting affected area is {fnum(s2_selection['affected_area_ha'], 2)} "
        f"ha, or {fnum(s2_selection['affected_area_km2'], 4)} km2.",
        styles_["BodyText"],
    )
    story.append(
        figure(
            "sentinel2_true_color_extent.png",
            "Figure 8. Sentinel-2 pre-event true color, post-event true color, "
            "and refined affected-area extent. Source: Copernicus Sentinel-2 "
            "Level-1C via Google Earth Engine.",
            styles_,
        )
    )
    story.append(
        figure(
            "sentinel2_ndvi_change.png",
            "Figure 9. Sentinel-2 NDVI before, after, and post-minus-pre change. "
            "NDVI is unitless. Source: Copernicus Sentinel-2 via Google Earth Engine.",
            styles_,
        )
    )
    story.append(
        figure(
            "sentinel2_bsi_change.png",
            "Figure 10. Sentinel-2 BSI before, after, and post-minus-pre change. "
            "BSI is unitless. Source: Copernicus Sentinel-2 via Google Earth Engine.",
            styles_,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("9. Comparison and Discussion", styles_["Heading1"]))
    comp_table = [
        ["Method", "Main signal", "Result", "Interpretation"],
        ["MintPy SBAS", "Pre-failure LOS motion", "-1.07 +/- 0.08 cm/year", "Useful for motion, not extent"],
        ["LiCSAR coherence", "Event disturbance", f"{fnum(coh_land)} vs {fnum(coh_ref)}", "Weak separator"],
        ["Sentinel-1 VV", "Backscatter change", f"{fnum(vv_land, 2)} dB vs {fnum(vv_ref, 2)} dB", "Weak contrast"],
        ["Sentinel-2 true color", "Optical scar", f"{fnum(s2_selection['affected_area_ha'], 2)} ha", "Clearest extent"],
        ["NDVI change", "Vegetation loss", f"{fnum(affected['ndvi_change'])} vs {fnum(stable['ndvi_change'])}", "Strong support"],
        ["BSI change", "Bare soil/debris", f"{fnum(affected['bsi_change'])} vs {fnum(stable['bsi_change'])}", "Not reliable alone"],
    ]
    story.append(table(comp_table, [1.2 * inch, 1.35 * inch, 1.25 * inch, 3.0 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    add_paragraphs(
        story,
        "Sentinel-2 true color plus NDVI decrease gives the clearest landslide "
        "extent. SBAS is useful for pre-failure deformation but cannot map the "
        "event extent. Coherence and Sentinel-1 VV are weaker in this workflow "
        "because their local contrasts are small or spatially ambiguous.",
        styles_["BodyText"],
    )

    story.append(Paragraph("10. Conclusions", styles_["Heading1"]))
    story.append(
        bullet_list(
            [
                "MintPy SBAS supports pre-event motion at the landslide, with an inspected trend of about -1.07 +/- 0.08 cm/year.",
                "The manual reference point gives a local relative basis for displacement interpretation.",
                "The available LiCSAR stack ends on 2017-06-07, so late acceleration before the 2017-06-24 failure cannot be reproduced.",
                "Coherence loss is weak for this pair because both local samples have very low coherence.",
                "Sentinel-1 VV change is also weak because local backscatter contrast is small.",
                f"Sentinel-2 true color plus NDVI decrease provides the clearest spatial map, with affected area {fnum(s2_selection['affected_area_ha'], 2)} ha.",
                "NDVI change supports vegetation loss or burial; BSI change is not a reliable positive-threshold discriminator for this pair.",
            ],
            styles_["BodyText"],
        )
    )

    story.append(Paragraph("11. References", styles_["Heading1"]))
    refs = [
        "SAR Lab 7 assignment handout, SAR_Lab_7.pdf.",
        "Supplied Xinmo landslide PS-based study: DOI 10.1007/s10346-017-0915-7.",
        "COMET-LiCSAR public interferogram products, frame 062D_05831_131313.",
        "Yunjun, Z., Fattahi, H., and Amelung, F. MintPy: a Python package for InSAR time series analysis.",
        "ESA Copernicus Sentinel-1 GRD imagery.",
        "ESA Copernicus Sentinel-2 imagery.",
        "Google Earth Engine cloud-processing platform.",
    ]
    story.append(bullet_list(refs, styles_["Small"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    SOURCE_FILE.write_text(markdown_source)


def main() -> None:
    values = read_outputs()
    markdown_source = make_markdown(values)
    build_pdf(values, markdown_source)
    print(f"Wrote {PDF_FILE.relative_to(LAB_DIR)}")
    print(f"Wrote {SOURCE_FILE.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
