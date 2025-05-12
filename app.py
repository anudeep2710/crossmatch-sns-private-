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
        ["LinkedIn Scraping", "Instagram Scraping", "Instagram Direct Login", "Manual Data Upload", "Local Files"]
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

            # Advanced options
            with st.expander("Advanced Options"):
                headless = st.checkbox("Run in headless mode", value=True,
                                      help="Run browser in background without showing the window")
                use_cookies = st.checkbox("Use cookies for login", value=True,
                                         help="Save and use cookies to avoid login challenges in future sessions")
                wait_for_manual_verification = st.checkbox("Wait for manual verification", value=True,
                                                         help="If Instagram requires verification, wait for you to manually complete it")

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

                            # Create cookies directory if it doesn't exist
                            cookies_dir = os.path.join("data/instagram", "cookies")
                            os.makedirs(cookies_dir, exist_ok=True)
                            cookies_path = os.path.join(cookies_dir, f"{username}_cookies.json")

                            # Login with cookies if enabled
                            login_success = scraper.login(username, password, use_cookies=use_cookies, cookies_path=cookies_path)

                            if login_success:
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
                                # Check if a screenshot was saved
                                screenshot_path = os.path.join("data/instagram", "login_error.png")
                                if os.path.exists(screenshot_path):
                                    st.error("Failed to login to Instagram. See screenshot below for details.")

                                    # Display the screenshot
                                    from PIL import Image
                                    image = Image.open(screenshot_path)
                                    st.image(image, caption="Login Error Screenshot")
                                else:
                                    st.error("Failed to login to Instagram. Please check your credentials.")

                                st.info("Instagram may be requiring additional verification. Try the following:")
                                st.markdown("""
                                1. Uncheck the 'Run in headless mode' option to see the browser
                                2. Enable 'Wait for manual verification' to give you time to complete any verification steps
                                3. Use 'Use cookies for login' to save your session for future use
                                """)
                    except Exception as e:
                        st.error(f"Error scraping Instagram profiles: {str(e)}")
                        st.info("Make sure you have installed all required dependencies: pip install selenium webdriver-manager")

    elif data_source == "Instagram Direct Login":
        st.subheader("Instagram Direct Login")
        st.info("This option allows you to use saved cookies from a previous login session.")

        # Check if any cookies exist
        cookies_dir = os.path.join("data/instagram", "cookies")
        os.makedirs(cookies_dir, exist_ok=True)

        cookie_files = [f for f in os.listdir(cookies_dir) if f.endswith('_cookies.json')]

        if not cookie_files:
            st.warning("No saved cookies found. Please use the 'Instagram Scraping' option first with 'Use cookies for login' enabled.")
        else:
            # Extract usernames from cookie filenames
            usernames = [f.replace('_cookies.json', '') for f in cookie_files]

            with st.form("instagram_direct_login_form"):
                st.subheader("Login with Saved Cookies")

                # Username selection
                selected_username = st.selectbox("Select Instagram Account", usernames)

                # Target usernames
                target_usernames = st.text_area("Target Instagram Usernames (one per line)")

                # Advanced options
                with st.expander("Advanced Options"):
                    headless = st.checkbox("Run in headless mode", value=True,
                                         help="Run browser in background without showing the window")

                login_button = st.form_submit_button("Login and Scrape")

                if login_button:
                    if not target_usernames:
                        st.error("Please provide at least one target username.")
                    else:
                        try:
                            from src.data.instagram_scraper import InstagramScraper

                            # Parse target usernames
                            username_list = [uname.strip() for uname in target_usernames.split('\n') if uname.strip()]

                            with st.spinner(f"Logging in as {selected_username} and scraping {len(username_list)} Instagram profiles..."):
                                # Initialize scraper
                                scraper = InstagramScraper(headless=headless, output_dir="data/instagram")

                                # Get cookies path
                                cookies_path = os.path.join(cookies_dir, f"{selected_username}_cookies.json")

                                # Login with cookies
                                if scraper.login(selected_username, "", use_cookies=True, cookies_path=cookies_path):
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
                                    st.error("Failed to login with saved cookies. The cookies may have expired.")
                                    st.info("Please use the 'Instagram Scraping' option to login with your username and password.")
                        except Exception as e:
                            st.error(f"Error logging in with cookies: {str(e)}")
                            st.info("Make sure you have installed all required dependencies: pip install selenium webdriver-manager")

    elif data_source == "Manual Data Upload":
        st.subheader("Manual Data Upload")
        st.info("""
        This option allows you to manually upload your Instagram and LinkedIn data files.

        You can use the template files as a starting point:
        1. Download the template files
        2. Edit them with your actual data
        3. Upload the edited files here
        """)

        # Platform selection
        platform = st.selectbox("Select Platform", ["Instagram", "LinkedIn"])

        # Skip template downloads and go directly to file upload
        if platform == "Instagram":
            st.info("Please use the file upload section below to upload your Instagram data files.")
        else:  # LinkedIn
            st.info("Please use the file upload section below to upload your LinkedIn data files.")

        # File upload section
        st.subheader(f"Upload {platform} Data Files")

        with st.form(f"{platform.lower()}_upload_form"):
            # File uploads
            profiles_file = st.file_uploader(f"{platform} Profiles CSV", type=["csv"])
            posts_file = st.file_uploader(f"{platform} Posts CSV", type=["csv"])
            network_file = st.file_uploader(f"{platform} Network Edgelist", type=["edgelist", "txt"])

            upload_button = st.form_submit_button("Upload Files")

            if upload_button:
                if not profiles_file or not posts_file or not network_file:
                    st.error("Please upload all three files: profiles, posts, and network.")
                else:
                    try:
                        # Create output directory
                        output_dir = f"data/{platform.lower()}"
                        os.makedirs(output_dir, exist_ok=True)

                        # Save uploaded files
                        with open(os.path.join(output_dir, "profiles.csv"), "wb") as f:
                            f.write(profiles_file.getvalue())

                        with open(os.path.join(output_dir, "posts.csv"), "wb") as f:
                            f.write(posts_file.getvalue())

                        with open(os.path.join(output_dir, "network.edgelist"), "wb") as f:
                            f.write(network_file.getvalue())

                        st.success(f"Successfully uploaded {platform} data files.")

                        # Initialize identifier
                        if st.session_state.identifier is None:
                            st.session_state.identifier = CrossPlatformUserIdentifier()

                        # Load data
                        try:
                            # Check if the other platform data exists
                            other_platform = "linkedin" if platform.lower() == "instagram" else "instagram"
                            other_platform_exists = os.path.exists(f"data/{other_platform}/profiles.csv")
                            other_platform_path = f"data/{other_platform}" if other_platform_exists else None

                            # Load data
                            st.session_state.identifier.load_data(
                                platform1_path=output_dir,
                                platform2_path=other_platform_path,
                                ground_truth_path="data/ground_truth.csv" if os.path.exists("data/ground_truth.csv") else None
                            )

                            if not other_platform_exists:
                                st.info(f"Only {platform} data was loaded. You'll need to upload {other_platform.capitalize()} data as well for cross-platform analysis.")

                            st.session_state.data_loaded = True

                            # Show sample data
                            st.subheader(f"Sample {platform} profiles")
                            st.dataframe(st.session_state.identifier.data[platform.lower()]['profiles'].head())

                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")

                    except Exception as e:
                        st.error(f"Error uploading files: {str(e)}")

        # Ground truth upload
        st.subheader("Upload Ground Truth (Optional)")

        with st.form("ground_truth_upload_form"):
            ground_truth_file = st.file_uploader("Ground Truth CSV", type=["csv"])

            # Show ground truth example
            if st.checkbox("Show Ground Truth Example"):
                ground_truth_data = {
                    "user_id_1": ["linkedin_user1", "linkedin_user2", "linkedin_user1"],
                    "user_id_2": ["instagram_user1", "instagram_user2", "instagram_user2"],
                    "is_same_user": [1, 1, 0]
                }

                st.dataframe(pd.DataFrame(ground_truth_data))
                st.info("Ground truth should have columns: user_id_1, user_id_2, is_same_user")

            upload_gt_button = st.form_submit_button("Upload Ground Truth")

            if upload_gt_button and ground_truth_file:
                try:
                    # Save uploaded file
                    with open("data/ground_truth.csv", "wb") as f:
                        f.write(ground_truth_file.getvalue())

                    st.success("Successfully uploaded ground truth file.")

                    # Show sample data
                    ground_truth_df = pd.read_csv("data/ground_truth.csv")
                    st.dataframe(ground_truth_df.head())

                except Exception as e:
                    st.error(f"Error uploading ground truth file: {str(e)}")

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
def run_analysis():
    """Run analysis on loaded data."""
    st.header("Analysis")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to perform cross-platform analysis.")
        return

    # Get platform names before creating the form
    platform_names = list(st.session_state.identifier.data.keys())
    if len(platform_names) < 2:
        st.error("Both LinkedIn and Instagram data are required for analysis.")
        st.info("Please go to the Data Loading tab and scrape data from both platforms.")
        return

    with st.form("analysis_form"):
        st.subheader("Configure Analysis")

        # Set platform names
        linkedin_exists = "linkedin" in platform_names
        instagram_exists = "instagram" in platform_names

        if linkedin_exists and instagram_exists:
            platform1_name = "linkedin"
            platform2_name = "instagram"
            st.success("Found data for both LinkedIn and Instagram.")
        else:
            missing_platforms = []
            if not linkedin_exists:
                missing_platforms.append("LinkedIn")
            if not instagram_exists:
                missing_platforms.append("Instagram")

            st.error(f"Missing data for {', '.join(missing_platforms)}.")
            st.info("Please go to the Data Loading tab and scrape data from the missing platform(s).")

            # Use available platforms for now
            platform1_name = platform_names[0]
            platform2_name = platform_names[1] if len(platform_names) > 1 else platform_names[0]

        # Feature extraction parameters
        st.subheader("Feature Extraction")

        network_method = st.selectbox("Network Embedding Method",
                                    ["node2vec"],
                                    index=0,
                                    help="node2vec is used to generate network embeddings from user connections.")

        semantic_model = st.selectbox("Semantic Model", [
            "sentence-transformers/all-MiniLM-L6-v2"
        ], index=0,
        help="This model is used to generate semantic embeddings from user posts and profile text.")

        # Matching parameters
        st.subheader("User Matching")

        matching_method = st.selectbox("Matching Method", ["cosine"], index=0,
                                     help="Cosine similarity is used to measure the similarity between user embeddings.")

        matching_threshold = st.slider("Matching Threshold", 0.01, 0.9, 0.05, 0.01,
                                     help="Users with similarity above this threshold are considered matches.")

        # Run analysis
        run_button = st.form_submit_button("Run Analysis")

        if run_button:
            with st.spinner("Running analysis..."):
                # Update configuration
                config = {
                    'network_method': network_method,
                    'semantic_model_name': semantic_model,
                    'matching_method': matching_method,
                    'matching_threshold': matching_threshold
                }

                # Update identifier configuration
                st.session_state.identifier.config.update(config)

                # Preprocess data
                st.session_state.identifier.preprocess()

                # Extract features
                st.session_state.identifier.extract_features()
                st.session_state.embeddings_generated = True

                # Match users
                matches = st.session_state.identifier.match_users(
                    platform1_name=platform1_name,
                    platform2_name=platform2_name,
                    embedding_type='fusion'
                )
                st.session_state.matches = matches

                # Evaluate if ground truth is available
                if hasattr(st.session_state.identifier.data_loader, 'ground_truth'):
                    metrics = st.session_state.identifier.evaluate()
                    st.session_state.metrics = metrics

                st.success("Analysis completed successfully")

def display_results():
    """Display LinkedIn-Instagram matching results."""
    st.header("LinkedIn-Instagram Matching Results")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to view matching results.")
        return

    if not st.session_state.embeddings_generated:
        st.warning("Please run analysis first on the Analysis tab.")
        st.info("This will generate the embeddings and matches needed to view results.")
        return

    # Get platform names
    platform_names = list(st.session_state.identifier.data.keys())

    # Check if both LinkedIn and Instagram data are available
    linkedin_exists = "linkedin" in platform_names
    instagram_exists = "instagram" in platform_names

    if not linkedin_exists or not instagram_exists:
        missing_platforms = []
        if not linkedin_exists:
            missing_platforms.append("LinkedIn")
        if not instagram_exists:
            missing_platforms.append("Instagram")

        st.error(f"Missing data for {', '.join(missing_platforms)}.")
        st.info("Please go to the Data Loading tab and scrape data from the missing platform(s).")
        return

    # Set platform names
    platform1_name = "linkedin"
    platform2_name = "instagram"

    st.subheader("Cross-Platform User Matches")
    st.write("These are the users identified as potentially being the same person across LinkedIn and Instagram:")
    st.write("Higher confidence scores indicate a stronger likelihood of being the same person.")

    # Display matches
    if st.session_state.matches is not None:
        st.subheader("User Matches")
        st.dataframe(st.session_state.matches)

        # Download matches as CSV
        csv = st.session_state.matches.to_csv(index=False)
        st.download_button(
            label="Download Matches as CSV",
            data=csv,
            file_name="matches.csv",
            mime="text/csv"
        )

    # Display evaluation metrics
    if st.session_state.metrics is not None:
        st.subheader("Evaluation Metrics")

        match_key = f"{platform1_name}_{platform2_name}_fusion"
        if match_key in st.session_state.metrics:
            metrics = st.session_state.metrics[match_key]

            # Create metrics display
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{metrics['precision']:.4f}")
            col2.metric("Recall", f"{metrics['recall']:.4f}")
            col3.metric("F1 Score", f"{metrics['f1']:.4f}")

            # Display confusion matrix
            if 'confusion_matrix' in st.session_state.identifier.evaluator.metrics:
                st.subheader("Confusion Matrix")
                cm = np.array(st.session_state.identifier.evaluator.metrics['confusion_matrix']['matrix'])

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    colorscale='Blues',
                    showscale=False
                ))

                fig.update_layout(
                    title=f"Confusion Matrix (Threshold = {metrics['best_threshold']:.2f})",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )

                st.plotly_chart(fig)
def compare_users():
    """Compare specific users between LinkedIn and Instagram."""
    st.header("LinkedIn-Instagram User Comparison")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to perform user comparison.")
        return

    if not st.session_state.embeddings_generated:
        st.warning("Please run analysis first on the Analysis tab.")
        st.info("This will generate the embeddings needed for user comparison.")
        return

    # Get platform names
    platform_names = list(st.session_state.identifier.data.keys())

    # Check if both LinkedIn and Instagram data are available
    linkedin_exists = "linkedin" in platform_names
    instagram_exists = "instagram" in platform_names

    if not linkedin_exists or not instagram_exists:
        missing_platforms = []
        if not linkedin_exists:
            missing_platforms.append("LinkedIn")
        if not instagram_exists:
            missing_platforms.append("Instagram")

        st.error(f"Missing data for {', '.join(missing_platforms)}.")
        st.info("Please go to the Data Loading tab and scrape data from the missing platform(s).")
        return

    # Set platform names
    platform1_name = "linkedin"
    platform2_name = "instagram"

    # Get user IDs
    user_ids1 = list(st.session_state.identifier.data[platform1_name]['profiles']['user_id'])
    user_ids2 = list(st.session_state.identifier.data[platform2_name]['profiles']['user_id'])

    # User selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LinkedIn User")
        user_id1 = st.selectbox("Select LinkedIn User", user_ids1)

        # Show LinkedIn profile info
        if user_id1:
            profile1 = st.session_state.identifier.data[platform1_name]['profiles'][
                st.session_state.identifier.data[platform1_name]['profiles']['user_id'] == user_id1
            ]
            if not profile1.empty:
                st.info(f"Name: {profile1.iloc[0].get('name', 'N/A')}")
                st.info(f"Headline: {profile1.iloc[0].get('headline', 'N/A')}")
                st.info(f"Location: {profile1.iloc[0].get('location', 'N/A')}")

    with col2:
        st.subheader("Instagram User")
        user_id2 = st.selectbox("Select Instagram User", user_ids2)

        # Show Instagram profile info
        if user_id2:
            profile2 = st.session_state.identifier.data[platform2_name]['profiles'][
                st.session_state.identifier.data[platform2_name]['profiles']['user_id'] == user_id2
            ]
            if not profile2.empty:
                st.info(f"Username: {profile2.iloc[0].get('username', 'N/A')}")
                st.info(f"Full Name: {profile2.iloc[0].get('full_name', 'N/A')}")
                st.info(f"Bio: {profile2.iloc[0].get('bio', 'N/A')}")

    # Compare button
    if st.button("Compare Users"):
        with st.spinner("Comparing users..."):
            # Get embeddings
            if platform1_name in st.session_state.identifier.embeddings and platform2_name in st.session_state.identifier.embeddings:
                embeddings1 = st.session_state.identifier.embeddings[platform1_name]['fusion']
                embeddings2 = st.session_state.identifier.embeddings[platform2_name]['fusion']

                # Check if user IDs exist in embeddings
                if user_id1 in embeddings1 and user_id2 in embeddings2:
                    # Get embeddings for selected users
                    embedding1 = embeddings1[user_id1]
                    embedding2 = embeddings2[user_id2]

                    # Check if embeddings have the same dimension
                    if embedding1.shape[0] != embedding2.shape[0]:
                        st.warning("Embeddings have different dimensions. Using PCA to normalize dimensions.")

                        # Use PCA to normalize dimensions
                        from sklearn.decomposition import PCA

                        # Find the minimum dimension
                        min_dim = min(embedding1.shape[0], embedding2.shape[0])

                        # Apply PCA
                        pca = PCA(n_components=min_dim)
                        if embedding1.shape[0] > min_dim:
                            embedding1 = pca.fit_transform(embedding1.reshape(1, -1))[0]
                        if embedding2.shape[0] > min_dim:
                            embedding2 = pca.fit_transform(embedding2.reshape(1, -1))[0]

                    # Calculate similarity
                    similarity = st.session_state.identifier.user_matcher.calculate_similarity(
                        embedding1.reshape(1, -1),
                        embedding2.reshape(1, -1)
                    )[0][0]

                    # Display similarity
                    st.subheader("Similarity Score")
                    st.info(f"Similarity: {similarity:.4f}")

                    # Interpret similarity
                    threshold = st.session_state.identifier.config.get('matching_threshold', 0.7)
                    if similarity >= threshold:
                        st.success("These users are likely the same person.")
                    else:
                        st.warning("These users are likely different people.")
                else:
                    if user_id1 not in embeddings1:
                        st.error(f"No embedding found for LinkedIn user: {user_id1}")
                    if user_id2 not in embeddings2:
                        st.error(f"No embedding found for Instagram user: {user_id2}")
            else:
                st.error("Embeddings not found. Please run analysis first.")

def main():
    """Main function for the Streamlit app."""
    st.title("LinkedIn-Instagram User Identification")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Loading", "Analysis", "Results", "User Comparison"])

    # Display selected page
    if page == "Home":
        st.header("Welcome to LinkedIn-Instagram User Identification")
        st.write("""
        This application helps identify and match users across LinkedIn and Instagram
        using machine learning techniques.

        ### Features
        - Data scraping from LinkedIn and Instagram
        - Preprocessing and normalization of user data
        - Network-based user embeddings (Node2Vec)
        - Semantic embeddings from user content (BERT)
        - Temporal embeddings from user activity patterns
        - Multi-modal embedding fusion
        - User matching across platforms
        - Visualization of matching results

        ### Getting Started
        1. Go to the **Data Loading** page to scrape LinkedIn and Instagram data
        2. Run the analysis on the **Analysis** page
        3. View the results on the **Results** page
        4. Compare specific users on the **User Comparison** page

        ### Important Notes
        - Web scraping may violate the Terms of Service of LinkedIn and Instagram
        - Use this application responsibly and for educational purposes only
        - Ensure you have proper authorization to access the profiles you scrape
        """)

    elif page == "Data Loading":
        load_data()

    elif page == "Analysis":
        run_analysis()

    elif page == "Results":
        display_results()

    elif page == "User Comparison":
        compare_users()

if __name__ == "__main__":
    main()
