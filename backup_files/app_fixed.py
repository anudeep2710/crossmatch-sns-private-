"""
Streamlit web application for cross-platform user identification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Union, Tuple, Any

# Wrap PyTorch imports in try-except to avoid Streamlit file watcher errors
try:
    # Import project modules
    from src.models.cross_platform_identifier import CrossPlatformUserIdentifier
    from src.data.data_loader import DataLoader
    from src.utils.visualizer import Visualizer
except RuntimeError as e:
    if "__path__._path" in str(e):
        # This is the PyTorch/Streamlit file watcher error, we can ignore it
        # and try importing again
        from src.models.cross_platform_identifier import CrossPlatformUserIdentifier
        from src.data.data_loader import DataLoader
        from src.utils.visualizer import Visualizer

# Set page config
st.set_page_config(
    page_title="LinkedIn-Instagram User Identification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'identifier' not in st.session_state:
    st.session_state.identifier = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False
if 'matches' not in st.session_state:
    st.session_state.matches = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer(use_plotly=True)

def load_data():
    """Load data from selected sources."""
    st.header("Data Loading")

    data_source = st.radio(
        "Select data source",
        ["LinkedIn Scraping", "Instagram Scraping", "Local Files"]
    )

    if data_source == "LinkedIn Scraping":
        with st.form("linkedin_scraping_form"):
            st.subheader("Scrape LinkedIn Data")
            st.warning("LinkedIn scraping requires Selenium and a LinkedIn account. Use at your own risk and ensure compliance with LinkedIn's terms of service.")

            email = st.text_input("LinkedIn Email")
            password = st.text_input("LinkedIn Password", type="password")
            profile_urls = st.text_area("LinkedIn Profile URLs (one per line)")
            headless = st.checkbox("Run in headless mode", value=True)

            scrape_button = st.form_submit_button("Scrape LinkedIn")

            if scrape_button:
                if not email or not password or not profile_urls:
                    st.error("Please provide email, password, and at least one profile URL.")
                else:
                    try:
                        from src.data.linkedin_scraper import LinkedInScraper

                        # Parse profile URLs
                        profile_url_list = [url.strip() for url in profile_urls.split('\n') if url.strip()]

                        with st.spinner(f"Scraping {len(profile_url_list)} LinkedIn profiles..."):
                            # Initialize scraper
                            scraper = LinkedInScraper(headless=headless, output_dir="data/linkedin")

                            # Login
                            if scraper.login(email, password):
                                # Scrape profiles
                                scraper.scrape_profiles(profile_url_list)

                                # Close browser
                                scraper.close()

                                st.success(f"Successfully scraped {len(profile_url_list)} LinkedIn profiles. Data saved to 'data/linkedin/'.")

                                # Initialize identifier
                                if st.session_state.identifier is None:
                                    st.session_state.identifier = CrossPlatformUserIdentifier()

                                # Load data
                                try:
                                    # Check if Instagram data exists
                                    instagram_exists = os.path.exists("data/instagram/profiles.csv")
                                    instagram_path = "data/instagram" if instagram_exists else None
                                    
                                    # Load data with only LinkedIn if Instagram doesn't exist
                                    st.session_state.identifier.load_data(
                                        platform1_path="data/linkedin",
                                        platform2_path=instagram_path,
                                        ground_truth_path=None
                                    )
                                    
                                    if not instagram_exists:
                                        st.info("Only LinkedIn data was loaded. You'll need to scrape Instagram data as well for cross-platform analysis.")
                                except Exception as e:
                                    st.warning(f"Note: {str(e)}")
                                    st.info("Only LinkedIn data was loaded. You'll need to scrape Instagram data as well for cross-platform analysis.")

                                st.session_state.data_loaded = True

                                # Show sample data
                                st.subheader("Sample LinkedIn profiles")
                                st.dataframe(st.session_state.identifier.data["linkedin"]["profiles"].head())
                            else:
                                st.error("Failed to login to LinkedIn. Please check your credentials.")
                    except Exception as e:
                        st.error(f"Error scraping LinkedIn profiles: {str(e)}")
                        st.info("Make sure you have installed all required dependencies: pip install selenium webdriver-manager")
    
    elif data_source == "Instagram Scraping":
        with st.form("instagram_scraping_form"):
            st.subheader("Scrape Instagram Data")
            st.warning("Instagram scraping requires Selenium and an Instagram account. Use at your own risk and ensure compliance with Instagram's terms of service.")

            username = st.text_input("Instagram Username")
            password = st.text_input("Instagram Password", type="password")
            target_usernames = st.text_area("Target Instagram Usernames (one per line)")
            headless = st.checkbox("Run in headless mode", value=True)

            scrape_button = st.form_submit_button("Scrape Instagram")

            if scrape_button:
                if not username or not password or not target_usernames:
                    st.error("Please provide username, password, and at least one target username.")
                else:
                    try:
                        from src.data.instagram_scraper import InstagramScraper

                        # Parse target usernames
                        username_list = [uname.strip() for uname in target_usernames.split('\n') if uname.strip()]

                        with st.spinner(f"Scraping {len(username_list)} Instagram profiles..."):
                            # Initialize scraper
                            scraper = InstagramScraper(headless=headless, output_dir="data/instagram")

                            # Login
                            if scraper.login(username, password):
                                # Scrape profiles
                                scraper.scrape_profiles(username_list)

                                # Close browser
                                scraper.close()

                                st.success(f"Successfully scraped {len(username_list)} Instagram profiles. Data saved to 'data/instagram/'.")

                                # Initialize identifier
                                if st.session_state.identifier is None:
                                    st.session_state.identifier = CrossPlatformUserIdentifier()

                                # Load data
                                try:
                                    # Check if LinkedIn data exists
                                    linkedin_exists = os.path.exists("data/linkedin/profiles.csv")
                                    linkedin_path = "data/linkedin" if linkedin_exists else None
                                    
                                    # Load data with only Instagram if LinkedIn doesn't exist
                                    st.session_state.identifier.load_data(
                                        platform1_path="data/instagram",
                                        platform2_path=linkedin_path,
                                        ground_truth_path=None
                                    )
                                    
                                    if not linkedin_exists:
                                        st.info("Only Instagram data was loaded. You'll need to scrape LinkedIn data as well for cross-platform analysis.")
                                except Exception as e:
                                    st.warning(f"Note: {str(e)}")
                                    st.info("Only Instagram data was loaded. You'll need to scrape LinkedIn data as well for cross-platform analysis.")

                                st.session_state.data_loaded = True

                                # Show sample data
                                st.subheader("Sample Instagram profiles")
                                st.dataframe(st.session_state.identifier.data["instagram"]["profiles"].head())
                            else:
                                st.error("Failed to login to Instagram. Please check your credentials.")
                    except Exception as e:
                        st.error(f"Error scraping Instagram profiles: {str(e)}")
                        st.info("Make sure you have installed all required dependencies: pip install selenium webdriver-manager")
    
    elif data_source == "Local Files":
        with st.form("local_files_form"):
            st.subheader("Load Data from Local Files")

            platform1_path = st.text_input("Platform 1 directory path", "data/linkedin")
            platform2_path = st.text_input("Platform 2 directory path", "data/instagram")
            ground_truth_path = st.text_input("Ground truth file path", "data/ground_truth.csv")

            load_button = st.form_submit_button("Load Data")

            if load_button:
                # Check if paths exist
                if not os.path.exists(platform1_path):
                    st.error(f"Platform 1 directory not found: {platform1_path}")
                    return

                if not os.path.exists(platform2_path):
                    st.error(f"Platform 2 directory not found: {platform2_path}")
                    return

                with st.spinner("Loading data..."):
                    # Initialize identifier
                    if st.session_state.identifier is None:
                        st.session_state.identifier = CrossPlatformUserIdentifier()

                    # Load data
                    try:
                        st.session_state.identifier.load_data(
                            platform1_path=platform1_path,
                            platform2_path=platform2_path,
                            ground_truth_path=ground_truth_path if os.path.exists(ground_truth_path) else None
                        )

                        st.session_state.data_loaded = True
                        st.success("Data loaded successfully")

                        # Show sample data
                        platform_names = list(st.session_state.identifier.data.keys())
                        if len(platform_names) >= 2:
                            platform1_name = platform_names[0]
                            platform2_name = platform_names[1]

                            st.subheader(f"Sample profiles from {platform1_name}")
                            st.dataframe(st.session_state.identifier.data[platform1_name]['profiles'].head())

                            st.subheader(f"Sample profiles from {platform2_name}")
                            st.dataframe(st.session_state.identifier.data[platform2_name]['profiles'].head())

                            if hasattr(st.session_state.identifier.data_loader, 'ground_truth'):
                                st.subheader("Sample ground truth")
                                st.dataframe(st.session_state.identifier.data_loader.ground_truth.head())

                    except Exception as e:
                        st.error(f"Error loading data: {e}")
