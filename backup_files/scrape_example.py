"""
Example script for scraping LinkedIn and Instagram data.

DISCLAIMER: Use this script responsibly and in compliance with the terms of service
of the respective platforms. This script is provided for educational purposes only.
"""

import argparse
import logging
import os
import sys
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_linkedin(email: str, password: str, profile_urls: List[str], headless: bool = True) -> bool:
    """
    Scrape LinkedIn profiles.
    
    Args:
        email (str): LinkedIn email
        password (str): LinkedIn password
        profile_urls (List[str]): List of LinkedIn profile URLs
        headless (bool): Whether to run the browser in headless mode
        
    Returns:
        bool: Whether scraping was successful
    """
    try:
        from src.data.linkedin_scraper import LinkedInScraper
        
        logger.info(f"Scraping {len(profile_urls)} LinkedIn profiles")
        
        # Initialize scraper
        scraper = LinkedInScraper(headless=headless, output_dir="data/linkedin")
        
        # Login
        if not scraper.login(email, password):
            logger.error("Failed to login to LinkedIn")
            scraper.close()
            return False
        
        # Scrape profiles
        scraper.scrape_profiles(profile_urls)
        
        # Close browser
        scraper.close()
        
        logger.info(f"Successfully scraped {len(profile_urls)} LinkedIn profiles")
        return True
    
    except Exception as e:
        logger.error(f"Error scraping LinkedIn profiles: {e}")
        return False

def scrape_instagram(username: str, password: str, target_usernames: List[str], headless: bool = True) -> bool:
    """
    Scrape Instagram profiles.
    
    Args:
        username (str): Instagram username
        password (str): Instagram password
        target_usernames (List[str]): List of Instagram usernames to scrape
        headless (bool): Whether to run the browser in headless mode
        
    Returns:
        bool: Whether scraping was successful
    """
    try:
        from src.data.instagram_scraper import InstagramScraper
        
        logger.info(f"Scraping {len(target_usernames)} Instagram profiles")
        
        # Initialize scraper
        scraper = InstagramScraper(headless=headless, output_dir="data/instagram")
        
        # Login
        if not scraper.login(username, password):
            logger.error("Failed to login to Instagram")
            scraper.close()
            return False
        
        # Scrape profiles
        scraper.scrape_profiles(target_usernames)
        
        # Close browser
        scraper.close()
        
        logger.info(f"Successfully scraped {len(target_usernames)} Instagram profiles")
        return True
    
    except Exception as e:
        logger.error(f"Error scraping Instagram profiles: {e}")
        return False

def main():
    """Run the example script."""
    parser = argparse.ArgumentParser(description="Scrape LinkedIn and Instagram data")
    
    # Add arguments
    parser.add_argument("--platform", type=str, choices=["linkedin", "instagram"], required=True,
                        help="Platform to scrape (linkedin or instagram)")
    parser.add_argument("--username", type=str, help="Username/email for login")
    parser.add_argument("--password", type=str, help="Password for login")
    parser.add_argument("--targets", type=str, nargs="+", help="Target profiles to scrape")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check required arguments
    if not args.username or not args.password or not args.targets:
        logger.error("Username, password, and targets are required")
        parser.print_help()
        sys.exit(1)
    
    # Scrape data
    if args.platform == "linkedin":
        success = scrape_linkedin(args.username, args.password, args.targets, args.headless)
    else:  # instagram
        success = scrape_instagram(args.username, args.password, args.targets, args.headless)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("""
    DISCLAIMER: Use this script responsibly and in compliance with the terms of service
    of the respective platforms. This script is provided for educational purposes only.
    """)
    
    # Ask for confirmation
    confirmation = input("Do you understand and agree to use this script responsibly? (yes/no): ")
    if confirmation.lower() != "yes":
        print("Exiting without scraping.")
        sys.exit(0)
    
    main()
