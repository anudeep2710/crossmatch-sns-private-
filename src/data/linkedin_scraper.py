"""
Module for scraping LinkedIn data.
"""

import time
import pandas as pd
import logging
import re
import os
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinkedInScraper:
    """
    Class for scraping LinkedIn data.
    
    Attributes:
        driver (webdriver.Chrome): Selenium Chrome webdriver
        is_logged_in (bool): Whether the user is logged in
        output_dir (str): Directory to save scraped data
    """
    
    def __init__(self, headless: bool = False, output_dir: str = "data/linkedin"):
        """
        Initialize the LinkedIn scraper.
        
        Args:
            headless (bool): Whether to run the browser in headless mode
            output_dir (str): Directory to save scraped data
        """
        self.output_dir = output_dir
        self.is_logged_in = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--start-maximized")
        
        # Initialize Chrome driver
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome driver initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Chrome driver: {e}")
            raise
    
    def login(self, email: str, password: str) -> bool:
        """
        Log in to LinkedIn.
        
        Args:
            email (str): LinkedIn email
            password (str): LinkedIn password
            
        Returns:
            bool: Whether login was successful
        """
        logger.info("Logging in to LinkedIn")
        
        try:
            # Navigate to LinkedIn login page
            self.driver.get("https://www.linkedin.com/login")
            
            # Wait for login form to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            
            # Enter email and password
            self.driver.find_element(By.ID, "username").send_keys(email)
            self.driver.find_element(By.ID, "password").send_keys(password)
            
            # Click login button
            self.driver.find_element(By.XPATH, "//button[@type='submit']").click()
            
            # Wait for login to complete
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "global-nav"))
            )
            
            self.is_logged_in = True
            logger.info("Logged in to LinkedIn successfully")
            return True
        
        except TimeoutException:
            logger.error("Timeout while logging in to LinkedIn")
            return False
        
        except Exception as e:
            logger.error(f"Error logging in to LinkedIn: {e}")
            return False
    
    def scrape_profile(self, profile_url: str) -> Dict[str, Any]:
        """
        Scrape a LinkedIn profile.
        
        Args:
            profile_url (str): URL of the LinkedIn profile
            
        Returns:
            Dict[str, Any]: Dictionary containing profile data
        """
        if not self.is_logged_in:
            logger.error("Not logged in to LinkedIn")
            return {}
        
        logger.info(f"Scraping profile: {profile_url}")
        
        try:
            # Navigate to profile page
            self.driver.get(profile_url)
            
            # Wait for profile to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "pv-top-card"))
            )
            
            # Extract profile data
            profile_data = {}
            
            # Extract name
            try:
                name_element = self.driver.find_element(By.XPATH, "//h1[@class='text-heading-xlarge inline t-24 v-align-middle break-words']")
                profile_data["name"] = name_element.text.strip()
            except NoSuchElementException:
                profile_data["name"] = ""
            
            # Extract headline
            try:
                headline_element = self.driver.find_element(By.XPATH, "//div[@class='text-body-medium break-words']")
                profile_data["headline"] = headline_element.text.strip()
            except NoSuchElementException:
                profile_data["headline"] = ""
            
            # Extract location
            try:
                location_element = self.driver.find_element(By.XPATH, "//span[@class='text-body-small inline t-black--light break-words']")
                profile_data["location"] = location_element.text.strip()
            except NoSuchElementException:
                profile_data["location"] = ""
            
            # Extract about section
            try:
                # Click "Show more" button if it exists
                try:
                    show_more_button = self.driver.find_element(By.XPATH, "//button[contains(@class, 'inline-show-more-text__button')]")
                    show_more_button.click()
                    time.sleep(1)
                except NoSuchElementException:
                    pass
                
                about_element = self.driver.find_element(By.XPATH, "//div[@class='pv-shared-text-with-see-more full-width t-14 t-normal t-black display-flex align-items-center']")
                profile_data["about"] = about_element.text.strip()
            except NoSuchElementException:
                profile_data["about"] = ""
            
            # Extract profile URL and username
            profile_data["profile_url"] = profile_url
            username_match = re.search(r"linkedin.com/in/([^/]+)", profile_url)
            profile_data["username"] = username_match.group(1) if username_match else ""
            
            # Generate a user_id
            profile_data["user_id"] = f"linkedin_{profile_data['username']}"
            
            # Extract connection count
            try:
                connections_element = self.driver.find_element(By.XPATH, "//span[contains(@class, 'distance-badge') and contains(@class, 'separator')]")
                connections_text = connections_element.text.strip()
                connections_match = re.search(r"(\d+)", connections_text)
                profile_data["connections"] = int(connections_match.group(1)) if connections_match else 0
            except NoSuchElementException:
                profile_data["connections"] = 0
            
            logger.info(f"Scraped profile: {profile_data['name']}")
            return profile_data
        
        except TimeoutException:
            logger.error(f"Timeout while scraping profile: {profile_url}")
            return {}
        
        except Exception as e:
            logger.error(f"Error scraping profile: {e}")
            return {}
    
    def scrape_posts(self, profile_url: str, max_posts: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape posts from a LinkedIn profile.
        
        Args:
            profile_url (str): URL of the LinkedIn profile
            max_posts (int): Maximum number of posts to scrape
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing post data
        """
        if not self.is_logged_in:
            logger.error("Not logged in to LinkedIn")
            return []
        
        logger.info(f"Scraping posts from profile: {profile_url}")
        
        try:
            # Navigate to profile page
            activity_url = f"{profile_url}/recent-activity/shares/"
            self.driver.get(activity_url)
            
            # Wait for posts to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2"))
                )
            except TimeoutException:
                logger.warning(f"No posts found for profile: {profile_url}")
                return []
            
            # Extract posts
            posts = []
            post_elements = self.driver.find_elements(By.CLASS_NAME, "feed-shared-update-v2")
            
            # Extract username from profile URL
            username_match = re.search(r"linkedin.com/in/([^/]+)", profile_url)
            username = username_match.group(1) if username_match else ""
            user_id = f"linkedin_{username}"
            
            for i, post_element in enumerate(post_elements[:max_posts]):
                try:
                    # Extract post text
                    try:
                        text_element = post_element.find_element(By.CLASS_NAME, "feed-shared-update-v2__description")
                        post_text = text_element.text.strip()
                    except NoSuchElementException:
                        post_text = ""
                    
                    # Extract post timestamp
                    try:
                        timestamp_element = post_element.find_element(By.CLASS_NAME, "feed-shared-actor__sub-description")
                        timestamp_text = timestamp_element.text.strip()
                    except NoSuchElementException:
                        timestamp_text = ""
                    
                    # Extract post metrics
                    try:
                        metrics_element = post_element.find_element(By.CLASS_NAME, "social-details-social-counts")
                        metrics_text = metrics_element.text.strip()
                        
                        # Extract likes
                        likes_match = re.search(r"(\d+) reactions?", metrics_text)
                        likes = int(likes_match.group(1)) if likes_match else 0
                        
                        # Extract comments
                        comments_match = re.search(r"(\d+) comments?", metrics_text)
                        comments = int(comments_match.group(1)) if comments_match else 0
                    except NoSuchElementException:
                        likes = 0
                        comments = 0
                    
                    # Create post dictionary
                    post = {
                        "post_id": f"{user_id}_post_{i}",
                        "user_id": user_id,
                        "content": post_text,
                        "timestamp": timestamp_text,
                        "likes": likes,
                        "comments": comments
                    }
                    
                    posts.append(post)
                
                except Exception as e:
                    logger.error(f"Error extracting post: {e}")
            
            logger.info(f"Scraped {len(posts)} posts from profile: {profile_url}")
            return posts
        
        except Exception as e:
            logger.error(f"Error scraping posts: {e}")
            return []
    
    def scrape_connections(self, profile_url: str, max_connections: int = 50) -> nx.Graph:
        """
        Scrape connections from a LinkedIn profile.
        
        Args:
            profile_url (str): URL of the LinkedIn profile
            max_connections (int): Maximum number of connections to scrape
            
        Returns:
            nx.Graph: NetworkX graph of connections
        """
        if not self.is_logged_in:
            logger.error("Not logged in to LinkedIn")
            return nx.Graph()
        
        logger.info(f"Scraping connections from profile: {profile_url}")
        
        try:
            # Extract username from profile URL
            username_match = re.search(r"linkedin.com/in/([^/]+)", profile_url)
            username = username_match.group(1) if username_match else ""
            user_id = f"linkedin_{username}"
            
            # Navigate to connections page
            connections_url = f"{profile_url}/connections/"
            self.driver.get(connections_url)
            
            # Wait for connections to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "mn-connection-card"))
                )
            except TimeoutException:
                logger.warning(f"No connections found for profile: {profile_url}")
                return nx.Graph()
            
            # Create graph
            G = nx.Graph()
            G.add_node(user_id)
            
            # Extract connections
            connection_elements = self.driver.find_elements(By.CLASS_NAME, "mn-connection-card")
            
            for i, connection_element in enumerate(connection_elements[:max_connections]):
                try:
                    # Extract connection name
                    try:
                        name_element = connection_element.find_element(By.CLASS_NAME, "mn-connection-card__name")
                        connection_name = name_element.text.strip()
                    except NoSuchElementException:
                        connection_name = f"Connection {i}"
                    
                    # Extract connection profile URL
                    try:
                        link_element = connection_element.find_element(By.CLASS_NAME, "mn-connection-card__link")
                        connection_url = link_element.get_attribute("href")
                        
                        # Extract username from URL
                        connection_username_match = re.search(r"linkedin.com/in/([^/]+)", connection_url)
                        connection_username = connection_username_match.group(1) if connection_username_match else f"user_{i}"
                    except NoSuchElementException:
                        connection_username = f"user_{i}"
                    
                    # Create connection ID
                    connection_id = f"linkedin_{connection_username}"
                    
                    # Add connection to graph
                    G.add_node(connection_id)
                    G.add_edge(user_id, connection_id)
                
                except Exception as e:
                    logger.error(f"Error extracting connection: {e}")
            
            logger.info(f"Scraped {G.number_of_edges()} connections from profile: {profile_url}")
            return G
        
        except Exception as e:
            logger.error(f"Error scraping connections: {e}")
            return nx.Graph()
    
    def scrape_profiles(self, profile_urls: List[str]) -> None:
        """
        Scrape multiple LinkedIn profiles and save the data.
        
        Args:
            profile_urls (List[str]): List of LinkedIn profile URLs
        """
        if not self.is_logged_in:
            logger.error("Not logged in to LinkedIn")
            return
        
        logger.info(f"Scraping {len(profile_urls)} LinkedIn profiles")
        
        # Initialize data structures
        profiles = []
        all_posts = []
        network = nx.Graph()
        
        # Scrape each profile
        for profile_url in profile_urls:
            # Scrape profile
            profile_data = self.scrape_profile(profile_url)
            if profile_data:
                profiles.append(profile_data)
            
            # Scrape posts
            posts = self.scrape_posts(profile_url)
            all_posts.extend(posts)
            
            # Scrape connections
            connections = self.scrape_connections(profile_url)
            network = nx.compose(network, connections)
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        # Save data
        if profiles:
            profiles_df = pd.DataFrame(profiles)
            profiles_df.to_csv(os.path.join(self.output_dir, "profiles.csv"), index=False)
            logger.info(f"Saved {len(profiles)} profiles to {os.path.join(self.output_dir, 'profiles.csv')}")
        
        if all_posts:
            posts_df = pd.DataFrame(all_posts)
            posts_df.to_csv(os.path.join(self.output_dir, "posts.csv"), index=False)
            logger.info(f"Saved {len(all_posts)} posts to {os.path.join(self.output_dir, 'posts.csv')}")
        
        if network.number_of_nodes() > 0:
            nx.write_edgelist(network, os.path.join(self.output_dir, "network.edgelist"))
            logger.info(f"Saved network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges to {os.path.join(self.output_dir, 'network.edgelist')}")
    
    def close(self):
        """Close the browser."""
        if hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("Browser closed")
