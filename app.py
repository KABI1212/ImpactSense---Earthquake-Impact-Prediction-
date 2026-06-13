"""ImpactSense Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from impact_logic import (
    ENGINEERED_FEATURE_COLUMNS,
    build_driver_summary,
    build_prediction_headline,
    categorize_depth,
    categorize_fault_proximity,
    categorize_magnitude,
    classify_risk_level,
    encode_fault_category,
    validate_inputs,
)

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover - optional visualization dependency
    pdk = None


APP_TITLE = "ImpactSense - Earthquake Impact Predictor"
BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.json"
MODELS_DIR = BASE_DIR / "models"
PREPROCESSED_DATA_PATH = BASE_DIR / "data" / "earthquake_preprocessed.csv"
BACKGROUND_IMAGE_CANDIDATES = (
    BASE_DIR / "assets" / "earthquake_background.jpg",
    BASE_DIR / "assets" / "earthquake_background.jpeg",
    BASE_DIR / "assets" / "earthquake_background.png",
    BASE_DIR / "earthquake_background.jpg",
    BASE_DIR / "earthquake_background.jpeg",
    BASE_DIR / "earthquake_background.png",
    BASE_DIR / "background.jpg",
    BASE_DIR / "background.jpeg",
    BASE_DIR / "background.png",
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="I",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_background_image_path() -> Path | None:
    """Return the first matching local background image file."""
    return next((path for path in BACKGROUND_IMAGE_CANDIDATES if path.exists()), None)


@st.cache_data(show_spinner=False)
def encode_image_file(image_path: str) -> str:
    """Encode a local image as base64 for CSS embedding."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def get_background_style() -> tuple[str, str]:
    """Build the app background CSS and whether decorative orbs should be shown."""
    image_path = get_background_image_path()
    if image_path is None:
        return (
            """
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.72), transparent 24%),
                radial-gradient(circle at left center, rgba(191, 230, 213, 0.45), transparent 20%),
                linear-gradient(180deg, var(--sky-1) 0%, var(--sky-2) 46%, var(--sky-3) 100%);
            """,
            "block",
        )

    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded_image = encode_image_file(str(image_path))
    return (
        f"""
        background:
            linear-gradient(180deg, rgba(4, 10, 26, 0.52) 0%, rgba(8, 16, 38, 0.78) 100%),
            url("data:{mime_type};base64,{encoded_image}") center center / cover fixed no-repeat;
        """,
        "none",
    )


def inject_styles() -> None:
    """Apply the shared visual system."""
    app_background, orb_display = get_background_style()
    css = """
        <style>
        :root {
            --sky-1: #eef4ff;
            --sky-2: #dce8ff;
            --sky-3: #c9d9ff;
            --panel: rgba(255, 255, 255, 0.62);
            --panel-strong: rgba(255, 255, 255, 0.78);
            --border: rgba(133, 160, 214, 0.42);
            --shadow: rgba(95, 124, 178, 0.16);
            --text: #46658f;
            --text-strong: #314d77;
            --text-soft: #6f88ad;
            --accent: #7fa8e8;
            --accent-2: #9cc0f6;
            --mint: #bfe6d5;
            --peach: #ffd8c8;
            --gold: #f5df9f;
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"],
        [data-testid="stMarkdownContainer"], button, input, textarea, select, label, p, span, div {
            font-family: "Times New Roman", Times, serif !important;
        }

        header[data-testid="stHeader"],
        #MainMenu,
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        .stApp {
            __APP_BACKGROUND__
            color: var(--text);
        }

        .main .block-container {
            width: 100%;
            max-width: none;
            margin: 0 auto;
            padding-top: 1rem;
            padding-right: 2.5rem;
            padding-bottom: 3rem;
            padding-left: 2.5rem;
            position: relative;
            z-index: 2;
        }

        div[data-testid="stForm"] {
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 28px !important;
            backdrop-filter: blur(12px);
            box-shadow: 0 24px 60px var(--shadow);
            padding: 1.75rem !important;
        }

        div[data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.82) !important;
            border: 1px solid rgba(133, 160, 214, 0.28) !important;
            border-radius: 18px !important;
        }

        div[data-testid="stTabs"] button {
            color: var(--text-strong) !important;
        }

        .page-shell {
            position: relative;
            overflow: hidden;
        }

        .soft-orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(22px);
            pointer-events: none;
            z-index: 0;
            display: __ORB_DISPLAY__;
        }

        .orb-a {
            width: 280px;
            height: 280px;
            top: 80px;
            left: 40px;
            background: rgba(255, 255, 255, 0.42);
        }

        .orb-b {
            width: 340px;
            height: 340px;
            right: 40px;
            top: 160px;
            background: rgba(191, 230, 213, 0.34);
        }

        .orb-c {
            width: 280px;
            height: 280px;
            left: 42%;
            bottom: 70px;
            background: rgba(255, 216, 200, 0.26);
        }

        .hero-card,
        .glass-card,
        .info-card,
        .result-card,
        .run-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 28px;
            backdrop-filter: blur(12px);
            box-shadow: 0 24px 60px var(--shadow);
        }

        .hero-card {
            padding: 1.55rem 1.65rem;
            margin-bottom: 1.25rem;
            max-width: none;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
        }

        .login-hero-card {
            max-width: 1120px;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.66);
            color: var(--text-strong);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-card h1 {
            color: var(--text-strong);
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.15;
            letter-spacing: 0.02em;
        }

        .hero-card p {
            color: var(--text);
            margin: 0.75rem 0 0 0;
            font-size: 1.08rem;
            line-height: 1.75;
        }

        .login-shell {
            max-width: 1200px;
            margin: 0 auto;
        }

        .feature-card {
            padding: 1.4rem;
            min-height: 100%;
        }

        .feature-title {
            color: var(--text-strong);
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .feature-copy {
            color: var(--text);
            line-height: 1.8;
            font-size: 1.03rem;
        }

        .feature-list {
            display: grid;
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .feature-pill {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(141, 167, 210, 0.34);
            border-radius: 20px;
            padding: 0.9rem 1rem;
            color: var(--text-strong);
            font-size: 1rem;
        }

        .login-card {
            padding: 1.75rem;
        }

        .login-card h2 {
            color: var(--text-strong);
            font-size: 2.2rem;
            margin: 0 0 0.5rem 0;
            text-align: center;
            letter-spacing: 0.04em;
        }

        .login-card p {
            color: var(--text);
            text-align: center;
            margin: 0 0 1.2rem 0;
            line-height: 1.7;
        }

        .login-form-title {
            color: var(--text-strong);
            font-size: 2.3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.45rem;
            letter-spacing: 0.03em;
        }

        .login-form-copy {
            color: var(--text);
            text-align: center;
            margin-bottom: 1.35rem;
            line-height: 1.7;
            font-size: 1.02rem;
        }

        .helper-strip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-top: 0.8rem;
            margin-bottom: 1.2rem;
            color: var(--text-soft);
            font-size: 0.98rem;
        }

        .footer-note {
            margin-top: 1rem;
            text-align: center;
            color: var(--text-soft);
            font-size: 0.96rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="select"] > div {
            background: var(--panel-strong) !important;
            border: 1px solid rgba(136, 163, 207, 0.42) !important;
            border-radius: 18px !important;
            min-height: 3.2rem !important;
            color: var(--text-strong) !important;
        }

        div[data-baseweb="input"] input {
            color: var(--text-strong) !important;
            font-size: 1.08rem !important;
        }

        label {
            color: var(--text-strong) !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        button[kind="primary"] {
            background: linear-gradient(180deg, var(--accent-2) 0%, var(--accent) 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 18px !important;
            min-height: 3.25rem !important;
            font-size: 1.08rem !important;
            font-weight: 700 !important;
            box-shadow: 0 15px 28px rgba(127, 168, 232, 0.28) !important;
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease !important;
        }

        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 34px rgba(127, 168, 232, 0.32) !important;
            filter: brightness(1.02);
        }

        button[kind="primary"]:active {
            transform: translateY(0);
            box-shadow: 0 12px 22px rgba(127, 168, 232, 0.24) !important;
        }

        button[kind="secondary"] {
            background: linear-gradient(180deg, #f6f9ff 0%, #dfe9fa 100%) !important;
            color: var(--text-strong) !important;
            border: 1px solid rgba(137, 164, 210, 0.42) !important;
            border-radius: 18px !important;
            min-height: 3.15rem !important;
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 22px rgba(137, 164, 210, 0.15) !important;
        }

        div[data-testid="stTabs"] button {
            transition: transform 180ms ease, background 180ms ease, box-shadow 180ms ease, color 180ms ease !important;
        }

        div[data-testid="stTabs"] button:hover {
            background: rgba(255, 255, 255, 0.82) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stTabs"] button[aria-selected="true"]:hover {
            background: linear-gradient(180deg, rgba(248, 252, 255, 0.98) 0%, rgba(226, 237, 255, 0.98) 100%) !important;
        }

        .dashboard-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1.15rem;
            text-align: center;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
        }

        .dashboard-user {
            color: var(--text-soft);
            font-size: 1rem;
            margin-top: 0.3rem;
        }

        .section-label {
            color: var(--text-strong);
            font-size: 1.65rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        }

        .input-panel {
            padding: 1.45rem;
        }

        .result-card {
            padding: 1.45rem;
            min-height: 100%;
        }

        .result-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
            align-items: stretch;
        }

        .result-card-main,
        .result-card-side,
        .result-card-location {
            background: rgba(255, 255, 255, 0.44);
            border: 1px solid rgba(136, 163, 207, 0.24);
            border-radius: 24px;
            padding: 1rem;
            min-width: 0;
            min-height: 0;
            gap: 0.75rem;
        }

        .result-card-main {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .result-card-side {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .result-card-location {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .result-card-side .result-summary,
        .result-card-side .result-card-hint {
            margin-top: 0;
        }

        .result-card-side .result-summary {
            margin-top: 0;
        }

        .result-card-side {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(150px, 0.85fr);
            gap: 0.85rem;
            align-items: center;
        }

        .result-card-side .result-summary p {
            max-width: none;
        }

        .result-card-location h4 {
            color: var(--text-strong);
            margin: 0 0 0.45rem 0;
            font-size: 1.08rem;
        }

        .result-card-location p {
            color: var(--text);
            margin: 0;
            line-height: 1.55;
        }

        .result-card-location {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(130px, 0.92fr);
            gap: 0.75rem 0.9rem;
            align-items: center;
        }

        .result-card-location h4,
        .result-card-location p,
        .result-location-value {
            grid-column: 1;
        }

        .result-card-location .result-card-hint {
            grid-column: 2;
            grid-row: 1 / span 3;
            align-self: center;
            margin-top: 0;
        }

        .result-location-value {
            color: var(--text-strong);
            font-size: 1.26rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .result-card-hint {
            color: var(--text-soft);
            font-size: 0.86rem;
            line-height: 1.6;
            margin-top: 0.85rem;
        }

        .impact-meter {
            margin-top: 1.1rem;
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(136, 163, 207, 0.26);
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(145px, 0.9fr);
            grid-template-areas:
                "top top"
                "track note";
            gap: 0.65rem 0.85rem;
            align-items: center;
        }

        .impact-meter-top {
            grid-area: top;
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
        }

        .impact-meter-title {
            color: var(--text-strong);
            font-size: 1rem;
            font-weight: 700;
        }

        .impact-meter-value {
            color: var(--text-strong);
            font-size: 1.1rem;
            font-weight: 700;
        }

        .impact-meter-track {
            grid-area: track;
            position: relative;
            overflow: hidden;
            height: 14px;
            border-radius: 999px;
            background: rgba(184, 204, 234, 0.45);
            box-shadow: inset 0 1px 2px rgba(83, 112, 163, 0.12);
        }

        .impact-meter-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #80a9ea 0%, #6d99e0 50%, #8fd0ef 100%);
            box-shadow: 0 10px 18px rgba(110, 153, 224, 0.2);
            transition: width 240ms ease;
        }

        .impact-meter-fill.low {
            background: linear-gradient(90deg, #98d9c3 0%, #77c8a9 100%);
        }

        .impact-meter-fill.moderate {
            background: linear-gradient(90deg, #f4d58a 0%, #eabf60 100%);
        }

        .impact-meter-fill.high {
            background: linear-gradient(90deg, #ffb9a7 0%, #e78b78 100%);
        }

        .impact-meter-note {
            grid-area: note;
            color: var(--text-soft);
            font-size: 0.86rem;
            margin-top: 0;
            line-height: 1.45;
        }

        .result-summary {
            margin-top: 1rem;
            padding: 1rem 1.05rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(136, 163, 207, 0.26);
        }

        .result-summary h4 {
            color: var(--text-strong);
            margin: 0 0 0.5rem 0;
            font-size: 1.08rem;
        }

        .result-summary p {
            color: var(--text);
            margin: 0;
            line-height: 1.55;
        }

        .result-insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .result-insight-card {
            padding: 0.95rem 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.56);
            border: 1px solid rgba(136, 163, 207, 0.26);
        }

        .result-insight-card h5 {
            color: var(--text-soft);
            margin: 0 0 0.35rem 0;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-insight-card p {
            color: var(--text-strong);
            margin: 0;
            line-height: 1.6;
            font-size: 0.96rem;
        }

        .flash-banner {
            margin-bottom: 1rem;
        }

        .risk-chip {
            display: inline-block;
            padding: 0.4rem 0.78rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 0.9rem;
        }

        .chip-none {
            background: rgba(222, 235, 255, 0.76);
            color: #6487b8;
        }

        .chip-low {
            background: rgba(191, 230, 213, 0.6);
            color: #4f8468;
        }

        .chip-moderate {
            background: rgba(245, 223, 159, 0.55);
            color: #9c7f30;
        }

        .chip-high {
            background: rgba(255, 216, 200, 0.7);
            color: #b07158;
        }

        .result-title {
            color: var(--text-strong);
            font-size: 1.95rem;
            margin-bottom: 0.55rem;
        }

        .result-copy {
            color: var(--text);
            line-height: 1.78;
            font-size: 1.03rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.2rem;
        }

        .metric-box {
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(136, 163, 207, 0.26);
            border-radius: 22px;
            padding: 1rem;
        }

        .metric-box h3 {
            color: var(--text-soft);
            margin: 0;
            font-size: 0.92rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-box p {
            color: var(--text-strong);
            margin: 0.55rem 0 0 0;
            font-size: 1.22rem;
            font-weight: 700;
        }

        .snapshot-box {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(136, 163, 207, 0.26);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            margin-top: 1.2rem;
            color: var(--text-strong);
        }

        .world-map-header {
            margin-top: 0;
            margin-bottom: 0.85rem;
            padding: 1rem 1.05rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(236, 244, 255, 0.9) 100%);
            border: 1px solid rgba(136, 163, 207, 0.26);
            box-shadow: 0 18px 34px rgba(95, 124, 178, 0.12);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .world-map-header h4 {
            color: var(--text-strong);
            margin: 0;
            font-size: 1.12rem;
            flex: 0 0 auto;
        }

        .world-map-header p {
            color: var(--text);
            margin: 0;
            line-height: 1.5;
            text-align: right;
            max-width: 48ch;
        }

        .world-map-note {
            color: var(--text-soft);
            font-size: 0.92rem;
            margin-top: 0;
        }

        .world-panel-section {
            margin-top: 1.2rem;
        }

        .world-panel-title {
            color: var(--text-strong);
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0 0 0.8rem 0;
            text-align: center;
        }

        .world-panel-grid {
            display: grid;
            grid-template-columns: minmax(260px, 0.82fr) minmax(0, 1.18fr);
            gap: 1rem;
            align-items: stretch;
        }

        .world-map-strip {
            margin-top: 0.85rem;
            padding: 1rem 1.05rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(136, 163, 207, 0.26);
            box-shadow: 0 22px 54px rgba(95, 124, 178, 0.14);
            display: grid;
            grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
            gap: 1rem;
            align-items: center;
        }

        .world-map-strip-copy h4 {
            color: var(--text-strong);
            margin: 0 0 0.35rem 0;
            font-size: 1.15rem;
        }

        .world-map-strip-copy p {
            color: var(--text);
            margin: 0;
            line-height: 1.55;
        }

        .world-map-strip-pills {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.6rem;
        }

        .world-map-strip-pill {
            padding: 0.8rem 0.85rem;
            border-radius: 20px;
            background: rgba(127, 168, 232, 0.12);
            border: 1px solid rgba(127, 168, 232, 0.2);
            color: var(--text-strong);
            text-align: center;
        }

        .world-map-strip-pill span {
            display: block;
            color: var(--text-soft);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .world-map-strip-pill strong {
            display: block;
            font-size: 0.92rem;
            font-weight: 700;
        }

        .world-panel-copy {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(136, 163, 207, 0.26);
            border-radius: 28px;
            padding: 1.1rem;
            box-shadow: 0 22px 54px rgba(95, 124, 178, 0.14);
            min-width: 0;
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(150px, 0.85fr);
            gap: 0.9rem;
            align-items: start;
        }

        .world-panel-copy h4 {
            color: var(--text-strong);
            margin: 0 0 0.45rem 0;
            font-size: 1.2rem;
        }

        .world-panel-copy p {
            color: var(--text);
            margin: 0;
            line-height: 1.58;
        }

        .world-panel-pill-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0;
            align-content: start;
        }

        .world-panel-pill {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            width: 100%;
            padding: 0.5rem 0.6rem;
            border-radius: 999px;
            background: rgba(127, 168, 232, 0.12);
            border: 1px solid rgba(127, 168, 232, 0.2);
            color: var(--text-strong);
            font-size: 0.86rem;
            font-weight: 700;
            text-align: center;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .mini-card {
            padding: 1rem;
        }

        .mini-card h4 {
            color: var(--text-strong);
            margin-top: 0;
            margin-bottom: 0.35rem;
            font-size: 1.12rem;
        }

        .mini-card p {
            color: var(--text);
            line-height: 1.7;
            margin: 0;
        }

        .run-card {
            padding: 1.4rem;
        }

        .run-card h3 {
            color: var(--text-strong);
            margin-top: 0;
            font-size: 1.65rem;
        }

        .run-card p {
            color: var(--text);
            margin-bottom: 0.7rem;
        }

        @media (max-width: 1100px) {
            .main .block-container {
                width: 100%;
                padding-right: 1rem;
                padding-left: 1rem;
            }
        }

        @media (max-width: 760px) {
            .result-card-grid,
            .metric-grid,
            .mini-grid,
            .result-insight-grid {
                grid-template-columns: 1fr;
            }

            .impact-meter,
            .result-card-side,
            .result-card-location {
                display: flex;
                flex-direction: column;
                align-items: stretch;
            }

            .impact-meter {
                grid-template-columns: 1fr;
                grid-template-areas: none;
            }

            .result-card-location h4,
            .result-card-location p,
            .result-location-value,
            .result-card-location .result-card-hint {
                grid-column: auto;
                grid-row: auto;
            }
        }

        @media (max-width: 900px) {
            .hero-card,
            .result-card,
            .run-card,
            .input-panel {
                padding: 1rem;
            }

            .result-title {
                font-size: 1.55rem;
            }

            .world-panel-title {
                font-size: 1.25rem;
            }

            .world-panel-copy {
                padding: 0.95rem;
                grid-template-columns: 1fr;
            }

            .world-map-strip {
                grid-template-columns: 1fr;
            }

            .world-map-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .world-map-header p {
                text-align: left;
                max-width: none;
            }

            .world-panel-pill-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .world-map-strip-pills {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .result-insight-card,
            .impact-meter,
            .result-summary {
                padding: 0.85rem;
            }

            .metric-box p {
                font-size: 1.05rem;
            }
        }
        </style>
        """
    css = css.replace("__APP_BACKGROUND__", app_background)
    css = css.replace("__ORB_DISPLAY__", orb_display)
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize session state values."""
    defaults = {
        "authenticated": False,
        "current_page": "login",
        "username": "",
        "prediction": None,
        "flash_kind": "",
        "flash_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def load_prediction_resources() -> dict[str, Any]:
    """Load trained prediction artifacts once per Streamlit process."""
    feature_config = joblib.load(MODELS_DIR / "feature_config.pkl")
    feature_columns = feature_config.get("feature_columns", ENGINEERED_FEATURE_COLUMNS)
    if list(feature_columns) != ENGINEERED_FEATURE_COLUMNS:
        raise ValueError("Feature configuration does not match impact_logic.ENGINEERED_FEATURE_COLUMNS.")

    preprocessed = pd.read_csv(PREPROCESSED_DATA_PATH)
    cluster_risk_scores = (
        preprocessed.groupby("Geo_Cluster")["Risk_Score"].mean().astype(float).to_dict()
    )
    observed_ranges = {
        "magnitude": (
            float(preprocessed["Magnitude"].min()),
            float(preprocessed["Magnitude"].max()),
        ),
        "depth": (
            float(preprocessed["Depth"].min()),
            float(preprocessed["Depth"].max()),
        ),
        "latitude": (
            float(preprocessed["Latitude"].min()),
            float(preprocessed["Latitude"].max()),
        ),
        "longitude": (
            float(preprocessed["Longitude"].min()),
            float(preprocessed["Longitude"].max()),
        ),
        "fault_proximity": (
            float(preprocessed["Fault_Proximity"].min()),
            float(preprocessed["Fault_Proximity"].max()),
        ),
    }

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    scaler_feature_names = list(getattr(scaler, "feature_names_in_", ENGINEERED_FEATURE_COLUMNS))
    if scaler_feature_names != ENGINEERED_FEATURE_COLUMNS:
        raise ValueError("scaler.pkl was not fitted with ENGINEERED_FEATURE_COLUMNS.")

    return {
        "model": joblib.load(MODELS_DIR / "xgboost_model.pkl"),
        "risk_regressor": joblib.load(MODELS_DIR / "risk_regressor.pkl"),
        "scaler": scaler,
        "label_encoder": joblib.load(MODELS_DIR / "label_encoder.pkl"),
        "depth_encoder": joblib.load(MODELS_DIR / "depth_encoder.pkl"),
        "magnitude_encoder": joblib.load(MODELS_DIR / "magnitude_encoder.pkl"),
        "geo_kmeans": joblib.load(MODELS_DIR / "geo_kmeans.pkl"),
        "feature_config": feature_config,
        "cluster_risk_scores": cluster_risk_scores,
        "observed_ranges": observed_ranges,
        "fallback_location_risk_score": float(preprocessed["Risk_Score"].mean()),
    }


def build_engineered_feature_row(
    payload: dict[str, float],
    resources: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one inference row using the same engineered features as preprocessing.py."""
    resources = resources or load_prediction_resources()

    depth_category = categorize_depth(payload["depth"])
    magnitude_category = categorize_magnitude(payload["magnitude"])
    fault_category = categorize_fault_proximity(payload["fault_proximity"])

    geo_cluster = int(
        resources["geo_kmeans"].predict([[payload["latitude"], payload["longitude"]]])[0]
    )
    location_risk_score = float(
        resources["cluster_risk_scores"].get(
            geo_cluster,
            resources["fallback_location_risk_score"],
        )
    )

    row = {
        "Magnitude": float(payload["magnitude"]),
        "Depth": float(payload["depth"]),
        "Latitude": float(payload["latitude"]),
        "Longitude": float(payload["longitude"]),
        "Fault_Proximity": float(payload["fault_proximity"]),
        "Depth_Category_Encoded": int(resources["depth_encoder"].transform([depth_category])[0]),
        "Magnitude_Category_Encoded": int(
            resources["magnitude_encoder"].transform([magnitude_category])[0]
        ),
        "Mag_Depth_Ratio": float(payload["magnitude"]) / (float(payload["depth"]) + 1.0),
        "Mag_Depth_Product": float(payload["magnitude"]) * float(payload["depth"]),
        "Log_Depth": math.log1p(float(payload["depth"])),
        "Mag_Squared": float(payload["magnitude"]) ** 2,
        "Geo_Cluster": geo_cluster,
        "Location_Risk_Score": location_risk_score,
        "Fault_Category_Encoded": encode_fault_category(fault_category),
    }
    return pd.DataFrame([row], columns=ENGINEERED_FEATURE_COLUMNS)


def load_users() -> dict[str, dict[str, str]]:
    """Load locally stored user accounts."""
    if not USERS_FILE.exists():
        return {}

    try:
        raw_data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    return raw_data if isinstance(raw_data, dict) else {}


def save_users(users: dict[str, dict[str, str]]) -> None:
    """Persist user accounts to the local JSON store."""
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def hash_password(password: str) -> str:
    """Create a simple password hash for the demo auth system."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def normalize_lookup_value(value: str) -> str:
    """Normalize usernames and emails for comparison."""
    return value.strip().lower()


def validate_email(email: str) -> bool:
    """Perform a minimal email format check."""
    clean_email = email.strip()
    return "@" in clean_email and "." in clean_email.split("@", 1)[-1]


def validate_user(identity: str, password: str) -> tuple[bool, str, dict[str, str] | None]:
    """Validate login credentials against locally created accounts."""
    if not identity.strip() or not password.strip():
        return False, "Enter both your email or username and your password.", None

    lookup_value = normalize_lookup_value(identity)
    users = load_users()

    for user in users.values():
        matches_identity = lookup_value in {
            normalize_lookup_value(user["username"]),
            normalize_lookup_value(user["email"]),
        }
        if not matches_identity:
            continue

        if user["password_hash"] != hash_password(password):
            return False, "Incorrect password. Please try again.", None

        return True, "", user

    return False, "Account not found. Create an account first.", None


def create_user_account(
    first_name: str,
    last_name: str,
    email: str,
    username: str,
    password: str,
    confirm_password: str,
) -> tuple[bool, str, dict[str, str] | None]:
    """Create a new local user account."""
    clean_first_name = first_name.strip()
    clean_last_name = last_name.strip()
    clean_email = email.strip()
    clean_username = username.strip()

    if not all([clean_first_name, clean_last_name, clean_email, clean_username, password, confirm_password]):
        return False, "Fill in all account details before continuing.", None

    if not validate_email(clean_email):
        return False, "Enter a valid email address.", None

    if len(clean_username) < 3:
        return False, "Username must be at least 3 characters long.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    if password != confirm_password:
        return False, "Passwords do not match.", None

    users = load_users()
    username_key = normalize_lookup_value(clean_username)
    email_key = normalize_lookup_value(clean_email)

    if username_key in users:
        return False, "That username is already registered.", None

    if any(normalize_lookup_value(user["email"]) == email_key for user in users.values()):
        return False, "That email address is already registered.", None

    user_record = {
        "first_name": clean_first_name,
        "last_name": clean_last_name,
        "email": clean_email,
        "username": clean_username,
        "password_hash": hash_password(password),
    }
    users[username_key] = user_record
    save_users(users)
    return True, "", user_record


def logout() -> None:
    """Clear authentication state and return to login."""
    st.session_state["authenticated"] = False
    st.session_state["current_page"] = "login"
    st.session_state["username"] = ""
    st.session_state["prediction"] = None
    st.session_state["flash_kind"] = ""
    st.session_state["flash_message"] = ""


def set_flash_message(kind: str, message: str) -> None:
    """Store a one-time banner message for the next screen."""
    st.session_state["flash_kind"] = kind
    st.session_state["flash_message"] = message


def render_flash_message() -> None:
    """Render and clear any pending one-time banner message."""
    message = st.session_state.get("flash_message", "")
    if not message:
        return

    kind = st.session_state.get("flash_kind", "info")
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)

    st.session_state["flash_kind"] = ""
    st.session_state["flash_message"] = ""


def validate_login_form(identity: str, password: str) -> tuple[dict[str, str], dict[str, str] | None]:
    """Validate login fields and resolve the matching user if possible."""
    errors: dict[str, str] = {}
    clean_identity = identity.strip()
    clean_password = password.strip()

    if not clean_identity:
        errors["identity"] = "Enter your email or username."

    if not clean_password:
        errors["password"] = "Enter your password."

    if errors:
        return errors, None

    is_valid, message, user = validate_user(identity, password)
    if is_valid and user is not None:
        return {}, user

    if message == "Incorrect password. Please try again.":
        errors["password"] = message
    else:
        errors["identity"] = message

    return errors, None


def validate_signup_form(
    first_name: str,
    last_name: str,
    email: str,
    username: str,
    password: str,
    confirm_password: str,
) -> dict[str, str]:
    """Validate signup fields and return field-specific errors."""
    errors: dict[str, str] = {}
    clean_first_name = first_name.strip()
    clean_last_name = last_name.strip()
    clean_email = email.strip()
    clean_username = username.strip()
    clean_password = password.strip()
    clean_confirm_password = confirm_password.strip()
    users = load_users()

    if not clean_first_name:
        errors["first_name"] = "First name is required."

    if not clean_last_name:
        errors["last_name"] = "Last name is required."

    if not clean_email:
        errors["email"] = "Email address is required."
    elif not validate_email(clean_email):
        errors["email"] = "Enter a valid email address."

    if not clean_username:
        errors["username"] = "Username is required."
    elif len(clean_username) < 3:
        errors["username"] = "Username must be at least 3 characters long."

    if not clean_password:
        errors["password"] = "Password is required."
    elif len(clean_password) < 6:
        errors["password"] = "Password must be at least 6 characters long."

    if not clean_confirm_password:
        errors["confirm_password"] = "Confirm your password."
    elif clean_password and clean_password != clean_confirm_password:
        errors["confirm_password"] = "Passwords do not match."

    if clean_username and normalize_lookup_value(clean_username) in users:
        errors["username"] = "That username is already registered."

    if clean_email and any(
        normalize_lookup_value(user["email"]) == normalize_lookup_value(clean_email)
        for user in users.values()
    ):
        errors["email"] = "That email address is already registered."

    return errors


def predict_impact(
    magnitude: float,
    depth: float,
    latitude: float,
    longitude: float,
    fault_proximity: float,
) -> dict[str, Any]:
    """Run the trained classifier and risk regressor for one earthquake scenario."""
    payload = {
        "magnitude": float(magnitude),
        "depth": float(depth),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "fault_proximity": float(fault_proximity),
    }
    resources = load_prediction_resources()
    validation = validate_inputs(payload, resources["observed_ranges"])
    if validation["errors"]:
        raise ValueError(" ".join(validation["errors"]))

    feature_row = build_engineered_feature_row(payload, resources)
    scaled_features = resources["scaler"].transform(feature_row)

    model = resources["model"]
    predicted_class = int(model.predict(scaled_features)[0])
    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(scaled_features)[0]
        class_index = list(model.classes_).index(1) if 1 in model.classes_ else predicted_class
        high_impact_probability = float(class_probabilities[class_index])
    else:
        high_impact_probability = float(predicted_class)

    raw_risk_score = float(resources["risk_regressor"].predict(scaled_features)[0])
    impact_score = int(round(max(0.0, min(100.0, raw_risk_score))))
    level = classify_risk_level(high_impact_probability, impact_score)
    risk_level = f"{level} Risk"
    headline = build_prediction_headline(level, high_impact_probability, impact_score)
    reasons = build_driver_summary(payload, high_impact_probability, impact_score)
    if validation["warnings"]:
        reasons.append(validation["warnings"][0])

    chip_by_level = {
        "Low": "chip-low",
        "Elevated": "chip-moderate",
        "High": "chip-high",
        "Severe": "chip-high",
    }

    return {
        "risk_level": risk_level,
        "message": headline,
        "chip": chip_by_level[level],
        "magnitude": magnitude,
        "depth": depth,
        "latitude": latitude,
        "longitude": longitude,
        "fault_proximity": fault_proximity,
        "impact_score": impact_score,
        "impact_band": level,
        "summary": (
            f"XGBoost predicted class {predicted_class} with "
            f"{high_impact_probability * 100:.1f}% high-impact probability; "
            f"the trained risk regressor estimated {raw_risk_score:.1f}/100."
        ),
        "reasons": reasons,
        "high_impact_probability": high_impact_probability,
        "raw_risk_score": raw_risk_score,
    }


def render_background_orbs() -> None:
    """Render decorative background orbs."""
    st.markdown(
        """
        <div class="page-shell">
            <div class="soft-orb orb-a"></div>
            <div class="soft-orb orb-b"></div>
            <div class="soft-orb orb-c"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_login_styles() -> None:
    """Apply the professional auth-page styling."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.6rem;
        }

        .auth-shell {
            max-width: 680px;
            margin: 0 auto;
        }

        .auth-welcome {
            max-width: 680px;
            margin: 0 auto 1rem auto;
            text-align: center;
        }

        .auth-welcome-badge {
            color: #6d88af;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .auth-welcome h1 {
            color: #26466f;
            margin: 0;
            font-size: 2.8rem;
            line-height: 1.1;
        }

        .auth-welcome p {
            color: #5b7397;
            margin: 0.55rem 0 0 0;
            font-size: 1rem;
            line-height: 1.7;
        }

        div[data-testid="stTabs"] {
            max-width: 680px;
            margin: 0 auto;
            width: 100%;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.6rem;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(133, 160, 214, 0.32);
            border-radius: 20px;
            padding: 0.42rem;
            box-shadow: 0 18px 42px rgba(95, 124, 178, 0.12);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: transparent !important;
        }

        div[data-testid="stTabs"] button {
            background: transparent !important;
            border-radius: 16px !important;
            color: #6480a7 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            min-height: 3.1rem !important;
            flex: 1 1 0% !important;
            justify-content: center !important;
            transition: transform 180ms ease, background 180ms ease, box-shadow 180ms ease, color 180ms ease !important;
        }

        div[data-testid="stTabs"] button:hover {
            background: rgba(255, 255, 255, 0.86) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(180deg, rgba(248, 252, 255, 0.98) 0%, rgba(226, 237, 255, 0.92) 100%) !important;
            border: 1px solid rgba(121, 155, 213, 0.42) !important;
            color: #26466f !important;
            box-shadow: 0 10px 24px rgba(119, 150, 201, 0.16) !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"]:hover {
            background: linear-gradient(180deg, rgba(248, 252, 255, 0.98) 0%, rgba(226, 237, 255, 0.98) 100%) !important;
        }

        div[data-testid="stForm"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82) 0%, rgba(240, 246, 255, 0.74) 100%) !important;
            border: 1px solid rgba(133, 160, 214, 0.34) !important;
            border-radius: 30px !important;
            backdrop-filter: blur(18px);
            box-shadow: 0 28px 68px rgba(95, 124, 178, 0.18);
            padding: 2rem !important;
            margin-top: 1rem !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {
            background: rgba(255, 255, 255, 0.88) !important;
            border: 1px solid rgba(133, 160, 214, 0.38) !important;
            border-radius: 18px !important;
            min-height: 3.35rem !important;
            color: #26466f !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.46);
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            border-color: rgba(107, 146, 209, 0.8) !important;
            box-shadow: 0 0 0 4px rgba(108, 156, 236, 0.12) !important;
        }

        div[data-baseweb="input"] input {
            color: #26466f !important;
            font-size: 1.05rem !important;
        }

        div[data-baseweb="input"] input::placeholder {
            color: rgba(100, 126, 160, 0.64) !important;
        }

        div[data-baseweb="input"] svg {
            color: #6983a7 !important;
            fill: #6983a7 !important;
        }

        label {
            color: #2d4972 !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        .auth-form-title {
            color: #26466f;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .auth-form-copy {
            color: #5b7397;
            margin-bottom: 1.15rem;
            line-height: 1.7;
            font-size: 1rem;
        }

        .auth-inline-note {
            color: #6983a7;
            font-size: 0.95rem;
            margin-top: 0.42rem;
            line-height: 1.6;
        }

        .field-note {
            margin-top: 0.32rem;
            font-size: 0.9rem;
            line-height: 1.45;
            color: #6f88ad;
        }

        .field-note.error {
            color: #b25d54;
        }

        .field-note.hint {
            color: #6f88ad;
        }

        button[kind="primary"] {
            background: linear-gradient(180deg, #86aceb 0%, #6e99e0 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 18px !important;
            min-height: 3.35rem !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            box-shadow: 0 14px 30px rgba(110, 153, 224, 0.24) !important;
        }

        div[data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.82) !important;
            border: 1px solid rgba(133, 160, 214, 0.28) !important;
        }

        @media (max-width: 1100px) {
            div[data-testid="stTabs"] {
                max-width: 680px;
            }
        }

        @media (min-width: 1450px) {
            .auth-welcome,
            .auth-shell,
            div[data-testid="stTabs"] {
                max-width: 640px;
            }
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-right: 1rem;
                padding-left: 1rem;
            }

            .auth-welcome h1 {
                font-size: 2.3rem;
            }

            .auth-welcome p {
                font-size: 0.96rem;
            }

            .auth-form-title {
                font-size: 1.7rem;
            }

            div[data-testid="stForm"] {
                padding: 1.45rem !important;
            }

            .field-note {
                font-size: 0.88rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login() -> None:
    """Render the login page."""
    render_background_orbs()
    inject_login_styles()
    left_space, auth_col, right_space = st.columns([1, 1.15, 1], gap="large")

    with auth_col:
        st.markdown(
            """
            <div class="auth-welcome">
                <div class="auth-welcome-badge">ImpactSense Access</div>
                <h1>Welcome back</h1>
                <p>Sign in to continue to your earthquake dashboard, or create an account if you are new here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sign_in_tab, create_account_tab = st.tabs(["Sign In", "Create Account"])

        with sign_in_tab:
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="auth-form-title">Sign in</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="auth-form-copy">Use the username or email address from your account.</div>',
                    unsafe_allow_html=True,
                )
                identity = st.text_input(
                    "Email address or username",
                    placeholder="you@example.com or your username",
                    key="login_identity",
                )
                identity_note = st.empty()
                identity_note.markdown(
                    '<div class="field-note hint">Use the email address or username you created for this project.</div>',
                    unsafe_allow_html=True,
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )
                password_note = st.empty()
                password_note.markdown(
                    '<div class="field-note hint">Password is case-sensitive.</div>',
                    unsafe_allow_html=True,
                )
                sign_in_submitted = st.form_submit_button(
                    "Sign In",
                    width="stretch",
                    type="primary",
                )

            if sign_in_submitted:
                login_errors, user = validate_login_form(identity, login_password)
                if user is not None:
                    set_flash_message(
                        "success",
                        f"Signed in successfully. Welcome back, {user.get('first_name') or user['username']}.",
                    )
                    st.session_state["authenticated"] = True
                    st.session_state["current_page"] = "dashboard"
                    st.session_state["username"] = user["username"]
                    st.rerun()
                identity_kind = "error" if "identity" in login_errors else "hint"
                identity_message = login_errors.get(
                    "identity",
                    "Use the email address or username you created for this project.",
                )
                password_kind = "error" if "password" in login_errors else "hint"
                password_message = login_errors.get("password", "Password is case-sensitive.")

                identity_note.markdown(
                    f'<div class="field-note {identity_kind}">{identity_message}</div>',
                    unsafe_allow_html=True,
                )
                password_note.markdown(
                    f'<div class="field-note {password_kind}">{password_message}</div>',
                    unsafe_allow_html=True,
                )
                if login_errors:
                    st.error("Please fix the highlighted fields and try again.")

        with create_account_tab:
            with st.form("create_account_form", clear_on_submit=False):
                st.markdown('<div class="auth-form-title">Create account</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="auth-form-copy">Register with your first name, last name, email, and username.</div>',
                    unsafe_allow_html=True,
                )
                first_col, last_col = st.columns(2)
                with first_col:
                    first_name = st.text_input("First Name", placeholder="First name", key="signup_first_name")
                    first_name_note = st.empty()
                    first_name_note.markdown(
                        '<div class="field-note hint">This appears on your local dashboard profile.</div>',
                        unsafe_allow_html=True,
                    )
                with last_col:
                    last_name = st.text_input("Last Name", placeholder="Last name", key="signup_last_name")
                    last_name_note = st.empty()
                    last_name_note.markdown(
                        '<div class="field-note hint">Use the family or last name you want displayed.</div>',
                        unsafe_allow_html=True,
                    )
                email = st.text_input("Email Address", placeholder="you@example.com", key="signup_email")
                email_note = st.empty()
                email_note.markdown(
                    '<div class="field-note hint">We use this to match your local account.</div>',
                    unsafe_allow_html=True,
                )
                username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
                username_note = st.empty()
                username_note.markdown(
                    '<div class="field-note hint">Choose at least 3 characters.</div>',
                    unsafe_allow_html=True,
                )
                new_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a password",
                    key="signup_password",
                )
                password_note = st.empty()
                password_note.markdown(
                    '<div class="field-note hint">Use at least 6 characters.</div>',
                    unsafe_allow_html=True,
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password",
                    key="signup_confirm_password",
                )
                confirm_password_note = st.empty()
                confirm_password_note.markdown(
                    '<div class="field-note hint">Repeat the same password once more.</div>',
                    unsafe_allow_html=True,
                )
                create_account_submitted = st.form_submit_button(
                    "Create Account",
                    width="stretch",
                    type="primary",
                )

            if create_account_submitted:
                signup_errors = validate_signup_form(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    username=username,
                    password=new_password,
                    confirm_password=confirm_password,
                )

                first_name_kind = "error" if "first_name" in signup_errors else "hint"
                last_name_kind = "error" if "last_name" in signup_errors else "hint"
                email_kind = "error" if "email" in signup_errors else "hint"
                username_kind = "error" if "username" in signup_errors else "hint"
                password_kind = "error" if "password" in signup_errors else "hint"
                confirm_kind = "error" if "confirm_password" in signup_errors else "hint"

                first_name_note.markdown(
                    f'<div class="field-note {first_name_kind}">{signup_errors.get("first_name", "This appears on your local dashboard profile.")}</div>',
                    unsafe_allow_html=True,
                )
                last_name_note.markdown(
                    f'<div class="field-note {last_name_kind}">{signup_errors.get("last_name", "Use the family or last name you want displayed.")}</div>',
                    unsafe_allow_html=True,
                )
                email_note.markdown(
                    f'<div class="field-note {email_kind}">{signup_errors.get("email", "We use this to match your local account.")}</div>',
                    unsafe_allow_html=True,
                )
                username_note.markdown(
                    f'<div class="field-note {username_kind}">{signup_errors.get("username", "Choose at least 3 characters.")}</div>',
                    unsafe_allow_html=True,
                )
                password_note.markdown(
                    f'<div class="field-note {password_kind}">{signup_errors.get("password", "Use at least 6 characters.")}</div>',
                    unsafe_allow_html=True,
                )
                confirm_password_note.markdown(
                    f'<div class="field-note {confirm_kind}">{signup_errors.get("confirm_password", "Repeat the same password once more.")}</div>',
                    unsafe_allow_html=True,
                )

                if signup_errors:
                    st.error("Please fix the highlighted fields and try again.")
                else:
                    created, message, user = create_user_account(
                        first_name=first_name,
                        last_name=last_name,
                        email=email,
                        username=username,
                        password=new_password,
                        confirm_password=confirm_password,
                    )
                    if created and user is not None:
                        set_flash_message(
                            "success",
                            f"Account created successfully. Welcome, {user.get('first_name') or user['username']}.",
                        )
                        st.session_state["authenticated"] = True
                        st.session_state["current_page"] = "dashboard"
                        st.session_state["username"] = user["username"]
                        st.rerun()
                    st.error(message)


def render_dashboard_header() -> None:
    """Render the dashboard page header."""
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <div style="color:#314d77;font-size:2.2rem;font-weight:700;letter-spacing:0.03em;">
                    ImpactSense Dashboard
                </div>
                <div class="dashboard-user">Signed in as {st.session_state["username"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_guide() -> None:
    """Show the run instructions card."""
    st.markdown(
        """
        <div class="run-card">
            <h3>Run The Application</h3>
            <p>Start the app from the project folder using the command below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("streamlit run app.py")


def show_prediction_result() -> None:
    """Display the current prediction if one exists."""
    prediction = st.session_state.get("prediction")
    if not prediction:
        st.info("Enter earthquake details and click `Predict Impact` to view the result.")
        return

    impact_score = int(prediction.get("impact_score", 0))
    impact_band = prediction.get("impact_band", "Moderate")
    summary = prediction.get("summary", "")
    reasons = prediction.get("reasons", [])
    fill_class = {
        "Low": "low",
        "Elevated": "high",
        "High": "high",
        "Severe": "high",
    }.get(impact_band, "moderate")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="risk-chip {prediction["chip"]}">{prediction["risk_level"]}</div>
            <div class="result-title">Impact Assessment</div>
            <div class="result-copy">{prediction["message"]}</div>
            <div class="result-card-grid">
                <div class="result-card-main">
                    <div class="impact-meter">
                        <div class="impact-meter-top">
                            <div class="impact-meter-title">Impact score</div>
                            <div class="impact-meter-value">{impact_score}/100</div>
                        </div>
                        <div class="impact-meter-track">
                            <div class="impact-meter-fill {fill_class}" style="width: {impact_score}%;"></div>
                        </div>
                        <div class="impact-meter-note">{impact_band} impact based on the trained model.</div>
                    </div>
                </div>
                <div class="result-card-side">
                    <div class="result-summary">
                        <h4>Why this result</h4>
                        <p>{summary}</p>
                    </div>
                    <div class="result-card-hint">The rest of the dashboard stays in wide rows below.</div>
                </div>
                <div class="result-card-location">
                    <h4>Location Snapshot</h4>
                    <p>Exact coordinates on the globe.</p>
                    <div class="result-location-value">{prediction["latitude"]:.4f}, {prediction["longitude"]:.4f}</div>
                    <div class="result-card-hint">Marker follows the coordinates.</div>
                </div>
            </div>
            <div class="result-insight-grid">
                <div class="result-insight-card">
                    <h5>Magnitude</h5>
                    <p>{reasons[0] if len(reasons) > 0 else "Magnitude shaped the base risk band."}</p>
                </div>
                <div class="result-insight-card">
                    <h5>Depth</h5>
                    <p>{reasons[1] if len(reasons) > 1 else "Depth adjusted how strongly the event can affect the surface."}</p>
                </div>
                <div class="result-insight-card">
                    <h5>Score Tuning</h5>
                    <p>{reasons[2] if len(reasons) > 2 else "The score is softened so the result stays conservative."}</p>
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric-box">
                    <h3>Magnitude</h3>
                    <p>{prediction["magnitude"]:.1f}</p>
                </div>
                <div class="metric-box">
                    <h3>Depth</h3>
                    <p>{prediction["depth"]:.1f} km</p>
                </div>
                <div class="metric-box">
                    <h3>Coordinates</h3>
                    <p>{prediction["latitude"]:.4f}, {prediction["longitude"]:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Fault Proximity</h3>
                    <p>{prediction["fault_proximity"]:.1f} km</p>
                </div>
                <div class="metric-box">
                    <h3>High-Impact Probability</h3>
                    <p>{prediction["high_impact_probability"] * 100:.1f}%</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_world_location_map(latitude: float, longitude: float) -> None:
    """Render a flat map view centered on the selected coordinates."""
    st.markdown(
        f"""
        <div class="world-map-header">
            <h4>Flat World Map</h4>
            <p>Centered on the current coordinates and kept straight for a clear landscape view.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pdk is None:
        st.map([{"lat": latitude, "lon": longitude}], zoom=1)
        st.caption("The marker shows the selected location on Earth.")
        return

    marker_data = [
        {
            "lat": latitude,
            "lon": longitude,
            "label": f"{latitude:.4f}, {longitude:.4f}",
        }
    ]
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        marker_data,
        get_position=["lon", "lat"],
        get_fill_color=[255, 255, 255, 235],
        get_line_color=[255, 180, 140, 240],
        stroked=True,
        filled=True,
        get_radius=120000,
        radius_min_pixels=10,
        radius_max_pixels=18,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )
    halo_layer = pdk.Layer(
        "ScatterplotLayer",
        marker_data,
        get_position=["lon", "lat"],
        get_fill_color=[111, 174, 255, 70],
        stroked=False,
        filled=True,
        get_radius=260000,
        radius_min_pixels=16,
        radius_max_pixels=30,
        pickable=False,
    )
    deck = pdk.Deck(
        layers=[halo_layer, marker_layer],
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=1.25,
            pitch=0,
            bearing=0,
        ),
        views=[pdk.View(type="MapView", controller=True)],
        map_style=pdk.map_styles.CARTO_ROAD,
        height=300,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck, width="stretch")
    st.caption("The flat map updates as latitude and longitude change.")


def render_world_location_panel(latitude: float, longitude: float) -> None:
    """Render the horizontal Earth and coordinate lane."""
    st.markdown('<div class="world-panel-title">Real World Position</div>', unsafe_allow_html=True)
    render_world_location_map(latitude, longitude)
    st.markdown(
        f"""
        <div class="world-map-strip">
            <div class="world-map-strip-copy">
                <h4>Floating Coordinates</h4>
                <p>Live map anchored to the exact latitude and longitude you entered.</p>
            </div>
            <div class="world-map-strip-pills">
                <div class="world-map-strip-pill"><span>Latitude</span><strong>{latitude:.4f}</strong></div>
                <div class="world-map-strip-pill"><span>Longitude</span><strong>{longitude:.4f}</strong></div>
                <div class="world-map-strip-pill"><span>View</span><strong>Flat map</strong></div>
                <div class="world-map-strip-pill"><span>Status</span><strong>Live update</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main_app() -> None:
    """Render the main dashboard UI."""
    render_background_orbs()
    render_dashboard_header()
    render_flash_message()
    left_space, logout_col, right_space = st.columns([1.2, 0.4, 1.2])
    with logout_col:
        st.button("Logout", on_click=logout, width="stretch")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Impact Predictor</div>
            <h1>ImpactSense - Earthquake Impact Predictor</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Prediction Workspace</div>', unsafe_allow_html=True)
    workspace_left, workspace_center, workspace_right = st.columns([0.06, 0.88, 0.06], gap="large")

    with workspace_center:
        left_input, right_input = st.columns(2)
        with left_input:
            magnitude = st.number_input(
                "Magnitude",
                min_value=0.0,
                max_value=10.0,
                value=5.5,
                step=0.1,
            )
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=28.6139,
                step=0.0001,
                format="%.4f",
            )
        with right_input:
            depth = st.number_input(
                "Depth (km)",
                min_value=0.0,
                max_value=700.0,
                value=35.0,
                step=1.0,
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=77.2090,
                step=0.0001,
                format="%.4f",
            )
            fault_proximity = st.number_input(
                "Fault Proximity (km)",
                min_value=0.0,
                max_value=120.0,
                value=24.0,
                step=1.0,
            )
        submitted = st.button(
            "Predict Impact",
            width="stretch",
            type="primary",
        )

        if submitted:
            with st.spinner("Analyzing earthquake inputs..."):
                time.sleep(0.35)
                st.session_state["prediction"] = predict_impact(
                    magnitude=magnitude,
                    depth=depth,
                    latitude=latitude,
                    longitude=longitude,
                    fault_proximity=fault_proximity,
                )

        st.markdown(
            """
            <div class="mini-grid">
                <div class="info-card mini-card">
                    <h4>Magnitude Logic</h4>
                    <p>Magnitude is encoded by the trained category bands and interaction features.</p>
                </div>
                <div class="info-card mini-card">
                    <h4>Depth Effect</h4>
                    <p>Shallow events below 70 km are encoded as a stronger surface-impact signal.</p>
                </div>
                <div class="info-card mini-card">
                    <h4>Fault Proximity</h4>
                    <p>Near-fault locations are encoded with the same bands used during training.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
        show_prediction_result()

        st.markdown('<div style="height: 0.9rem;"></div>', unsafe_allow_html=True)
        render_world_location_panel(latitude, longitude)


def main() -> None:
    """Route between the login page and the dashboard."""
    inject_styles()
    initialize_session_state()

    if not st.session_state["authenticated"] or st.session_state["current_page"] == "login":
        login()
        return

    main_app()


if __name__ == "__main__":
    main()
