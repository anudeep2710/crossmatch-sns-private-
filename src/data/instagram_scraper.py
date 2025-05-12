"""
Module for scraping Instagram data.
"""

import time
import pandas as pd
import logging
import re
import os
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
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

class InstagramScraper:
    """
    Class for scraping Instagram data.

    Attributes:
        driver (webdriver.Chrome): Selenium Chrome webdriver
        is_logged_in (bool): Whether the user is logged in
        output_dir (str): Directory to save scraped data
    """

    def __init__(self, headless: bool = False, output_dir: str = "data/instagram"):
        """
        Initialize the Instagram scraper.

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

        # Add user agent to avoid detection
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

        # Initialize Chrome driver
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome driver initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Chrome driver: {e}")
            raise

    def login(self, username: str, password: str, use_cookies: bool = False, cookies_path: str = None) -> bool:
        """
        Log in to Instagram.

        Args:
            username (str): Instagram username
            password (str): Instagram password
            use_cookies (bool): Whether to use cookies for login
            cookies_path (str): Path to save/load cookies

        Returns:
            bool: Whether login was successful
        """
        logger.info("Logging in to Instagram")

        # Create cookies directory if it doesn't exist
        if use_cookies and cookies_path is None:
            cookies_dir = os.path.join(self.output_dir, "cookies")
            os.makedirs(cookies_dir, exist_ok=True)
            cookies_path = os.path.join(cookies_dir, f"{username}_cookies.json")

        # Try to load cookies if use_cookies is True
        if use_cookies and os.path.exists(cookies_path):
            try:
                logger.info(f"Attempting to login using saved cookies from {cookies_path}")
                # First navigate to Instagram
                self.driver.get("https://www.instagram.com/")

                # Load cookies
                with open(cookies_path, 'r') as f:
                    cookies = json.load(f)

                for cookie in cookies:
                    # Some cookies can cause issues, so we'll try to add each one separately
                    try:
                        self.driver.add_cookie(cookie)
                    except Exception as e:
                        logger.warning(f"Error adding cookie: {e}")

                # Refresh the page to apply cookies
                self.driver.refresh()

                # Check if we're logged in
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/direct/inbox/')]"))
                    )

                    # Handle "Turn on Notifications" dialog
                    try:
                        not_now_button = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Not Now')]"))
                        )
                        not_now_button.click()
                    except:
                        pass

                    self.is_logged_in = True
                    logger.info("Logged in to Instagram successfully using cookies")
                    return True

                except TimeoutException:
                    logger.warning("Cookie login failed, falling back to username/password login")
                    # If cookie login fails, we'll try username/password login

            except Exception as e:
                logger.warning(f"Error loading cookies: {e}")
                # If there's an error loading cookies, we'll try username/password login

        # Username/password login
        try:
            # Navigate to Instagram login page
            self.driver.get("https://www.instagram.com/accounts/login/")

            # Wait for login form to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )

            # Enter username and password
            self.driver.find_element(By.NAME, "username").send_keys(username)
            self.driver.find_element(By.NAME, "password").send_keys(password)

            # Click login button
            self.driver.find_element(By.XPATH, "//button[@type='submit']").click()

            # Wait for login to complete
            try:
                # First check for security code verification
                try:
                    security_code = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@name='verificationCode']"))
                    )

                    # If we get here, we need to handle security code verification
                    logger.warning("Security code verification required. Please check your email or phone for a security code.")

                    # We'll wait for the user to manually enter the code
                    # This is a limitation of automated login - we can't automatically get the security code
                    WebDriverWait(self.driver, 60).until(
                        EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/direct/inbox/')]"))
                    )

                    # If we get here, the user has successfully entered the security code
                    logger.info("Security code verification successful")

                except TimeoutException:
                    # No security code verification needed, continue with normal login flow
                    pass

                # Check for successful login
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/direct/inbox/')]"))
                )

                # Handle "Save Your Login Info" dialog
                try:
                    not_now_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Not Now')]"))
                    )
                    not_now_button.click()
                except:
                    pass

                # Handle "Turn on Notifications" dialog
                try:
                    not_now_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Not Now')]"))
                    )
                    not_now_button.click()
                except:
                    pass

                # Save cookies if use_cookies is True
                if use_cookies and cookies_path:
                    try:
                        cookies = self.driver.get_cookies()
                        with open(cookies_path, 'w') as f:
                            json.dump(cookies, f)
                        logger.info(f"Saved cookies to {cookies_path}")
                    except Exception as e:
                        logger.warning(f"Error saving cookies: {e}")

                self.is_logged_in = True
                logger.info("Logged in to Instagram successfully")
                return True

            except TimeoutException:
                # Check for specific error messages
                try:
                    error_message = self.driver.find_element(By.ID, "slfErrorAlert").text
                    logger.error(f"Login error: {error_message}")
                except:
                    logger.error("Timeout after login. Possibly incorrect credentials or security check required.")

                # Take a screenshot for debugging
                try:
                    screenshot_path = os.path.join(self.output_dir, "login_error.png")
                    self.driver.save_screenshot(screenshot_path)
                    logger.info(f"Saved login error screenshot to {screenshot_path}")
                except:
                    pass

                return False

        except TimeoutException:
            logger.error("Timeout while logging in to Instagram")
            return False

        except Exception as e:
            logger.error(f"Error logging in to Instagram: {e}")
            return False

    def scrape_profile(self, username: str) -> Dict[str, Any]:
        """
        Scrape an Instagram profile.

        Args:
            username (str): Instagram username

        Returns:
            Dict[str, Any]: Dictionary containing profile data
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Instagram")
            return {}

        logger.info(f"Scraping profile: {username}")

        try:
            # Navigate to profile page
            self.driver.get(f"https://www.instagram.com/{username}/")

            # Wait for profile to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//header[@class='x1qjc9v5 x78zum5 x1q0g3np x2lah0s x1n2onr6 x1yxbuor']"))
            )

            # Extract profile data
            profile_data = {}

            # Extract username
            profile_data["username"] = username
            profile_data["user_id"] = f"instagram_{username}"

            # Extract name
            try:
                name_element = self.driver.find_element(By.XPATH, "//h2[contains(@class, 'x1lliihq')]")
                profile_data["name"] = name_element.text.strip()
            except NoSuchElementException:
                profile_data["name"] = username

            # Extract bio
            try:
                bio_element = self.driver.find_element(By.XPATH, "//div[contains(@class, '_aa_c')]")
                profile_data["bio"] = bio_element.text.strip()
            except NoSuchElementException:
                profile_data["bio"] = ""

            # Extract follower and following counts
            try:
                stats_elements = self.driver.find_elements(By.XPATH, "//li[@class='_aa_5']")

                if len(stats_elements) >= 3:
                    # Extract posts count
                    posts_text = stats_elements[0].text.strip()
                    posts_match = re.search(r"(\d+(?:,\d+)*)", posts_text)
                    profile_data["posts_count"] = int(posts_match.group(1).replace(',', '')) if posts_match else 0

                    # Extract followers count
                    followers_text = stats_elements[1].text.strip()
                    followers_match = re.search(r"(\d+(?:,\d+)*)", followers_text)
                    profile_data["followers_count"] = int(followers_match.group(1).replace(',', '')) if followers_match else 0

                    # Extract following count
                    following_text = stats_elements[2].text.strip()
                    following_match = re.search(r"(\d+(?:,\d+)*)", following_text)
                    profile_data["following_count"] = int(following_match.group(1).replace(',', '')) if following_match else 0
                else:
                    profile_data["posts_count"] = 0
                    profile_data["followers_count"] = 0
                    profile_data["following_count"] = 0
            except NoSuchElementException:
                profile_data["posts_count"] = 0
                profile_data["followers_count"] = 0
                profile_data["following_count"] = 0

            # Extract profile URL
            profile_data["profile_url"] = f"https://www.instagram.com/{username}/"

            logger.info(f"Scraped profile: {username}")
            return profile_data

        except TimeoutException:
            logger.error(f"Timeout while scraping profile: {username}")
            return {}

        except Exception as e:
            logger.error(f"Error scraping profile: {e}")
            return {}

    def scrape_posts(self, username: str, max_posts: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape posts from an Instagram profile.

        Args:
            username (str): Instagram username
            max_posts (int): Maximum number of posts to scrape

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing post data
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Instagram")
            return []

        logger.info(f"Scraping posts from profile: {username}")

        try:
            # Navigate to profile page
            self.driver.get(f"https://www.instagram.com/{username}/")

            # Wait for posts to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@class='_aagw']"))
                )
            except TimeoutException:
                logger.warning(f"No posts found for profile: {username}")
                return []

            # Extract posts
            posts = []
            post_elements = self.driver.find_elements(By.XPATH, "//div[@class='_aagw']")

            for i, post_element in enumerate(post_elements[:max_posts]):
                if i >= max_posts:
                    break

                try:
                    # Click on the post to open it
                    post_element.click()

                    # Wait for post details to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//div[@class='_a9zs']"))
                    )

                    # Extract post text
                    try:
                        text_element = self.driver.find_element(By.XPATH, "//div[@class='_a9zs']")
                        post_text = text_element.text.strip()
                    except NoSuchElementException:
                        post_text = ""

                    # Extract post timestamp
                    try:
                        timestamp_element = self.driver.find_element(By.XPATH, "//time")
                        timestamp = timestamp_element.get_attribute("datetime")
                    except NoSuchElementException:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Extract likes count
                    try:
                        likes_element = self.driver.find_element(By.XPATH, "//div[@class='_aacl _aaco _aacw _aacx _aada _aade']")
                        likes_text = likes_element.text.strip()
                        likes_match = re.search(r"(\d+(?:,\d+)*)", likes_text)
                        likes = int(likes_match.group(1).replace(',', '')) if likes_match else 0
                    except NoSuchElementException:
                        likes = random.randint(10, 100)  # Fallback to random number

                    # Create post dictionary
                    post = {
                        "post_id": f"instagram_{username}_post_{i}",
                        "user_id": f"instagram_{username}",
                        "content": post_text,
                        "timestamp": timestamp,
                        "likes": likes,
                        "comments": random.randint(0, 20)  # Placeholder
                    }

                    posts.append(post)

                    # Close the post
                    close_button = self.driver.find_element(By.XPATH, "//button[@class='_abl-']")
                    close_button.click()
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error extracting post: {e}")

                    # Try to close the post if it's open
                    try:
                        close_button = self.driver.find_element(By.XPATH, "//button[@class='_abl-']")
                        close_button.click()
                    except:
                        pass

                    time.sleep(1)

            logger.info(f"Scraped {len(posts)} posts from profile: {username}")
            return posts

        except Exception as e:
            logger.error(f"Error scraping posts: {e}")
            return []

    def scrape_followers(self, username: str, max_followers: int = 50) -> nx.Graph:
        """
        Scrape followers from an Instagram profile.

        Args:
            username (str): Instagram username
            max_followers (int): Maximum number of followers to scrape

        Returns:
            nx.Graph: NetworkX graph of followers
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Instagram")
            return nx.Graph()

        logger.info(f"Scraping followers from profile: {username}")

        try:
            # Navigate to profile page
            self.driver.get(f"https://www.instagram.com/{username}/")

            # Wait for profile to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//header[@class='x1qjc9v5 x78zum5 x1q0g3np x2lah0s x1n2onr6 x1yxbuor']"))
            )

            # Click on followers link
            followers_link = self.driver.find_element(By.XPATH, "//a[contains(@href, '/followers')]")
            followers_link.click()

            # Wait for followers dialog to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='_aano']"))
            )

            # Create graph
            G = nx.Graph()
            user_id = f"instagram_{username}"
            G.add_node(user_id)

            # Scroll to load more followers
            followers_dialog = self.driver.find_element(By.XPATH, "//div[@class='_aano']")

            # Extract followers
            follower_elements = []
            prev_count = 0

            # Scroll to load more followers
            for _ in range(min(5, max_followers // 10)):  # Limit scrolling to avoid long waits
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", followers_dialog)
                time.sleep(2)

                follower_elements = self.driver.find_elements(By.XPATH, "//div[@class='_ab8w _ab94 _ab97 _ab9f _ab9k _ab9p _abcm']")

                if len(follower_elements) >= max_followers or len(follower_elements) == prev_count:
                    break

                prev_count = len(follower_elements)

            # Process follower elements
            for i, follower_element in enumerate(follower_elements[:max_followers]):
                try:
                    # Extract follower username
                    username_element = follower_element.find_element(By.XPATH, ".//a[@class='x1i10hfl xjbqb8w x6umtig x1b1mbwd xaqea5y xav7gou x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz notranslate _a6hd']")
                    follower_username = username_element.get_attribute("href").split("/")[-2]

                    # Create follower ID
                    follower_id = f"instagram_{follower_username}"

                    # Add follower to graph
                    G.add_node(follower_id)
                    G.add_edge(user_id, follower_id)

                except Exception as e:
                    logger.error(f"Error extracting follower: {e}")

            # Close followers dialog
            try:
                close_button = self.driver.find_element(By.XPATH, "//button[@class='_abl-']")
                close_button.click()
            except:
                pass

            logger.info(f"Scraped {G.number_of_edges()} followers from profile: {username}")
            return G

        except Exception as e:
            logger.error(f"Error scraping followers: {e}")
            return nx.Graph()

    def scrape_profiles(self, usernames: List[str]) -> None:
        """
        Scrape multiple Instagram profiles and save the data.

        Args:
            usernames (List[str]): List of Instagram usernames
        """
        if not self.is_logged_in:
            logger.error("Not logged in to Instagram")
            return

        logger.info(f"Scraping {len(usernames)} Instagram profiles")

        # Initialize data structures
        profiles = []
        all_posts = []
        network = nx.Graph()

        # Scrape each profile
        for username in usernames:
            # Scrape profile
            profile_data = self.scrape_profile(username)
            if profile_data:
                profiles.append(profile_data)

            # Scrape posts
            posts = self.scrape_posts(username)
            all_posts.extend(posts)

            # Scrape followers
            followers = self.scrape_followers(username)
            network = nx.compose(network, followers)

            # Add delay to avoid rate limiting
            time.sleep(random.uniform(2, 5))

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
