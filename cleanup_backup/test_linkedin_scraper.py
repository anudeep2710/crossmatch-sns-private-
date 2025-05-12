"""
Test script for LinkedIn scraper.
"""

import os
import pandas as pd
from src.data.linkedin_scraper import LinkedInScraper

# Create output directory
os.makedirs("data/linkedin", exist_ok=True)

# Initialize scraper
scraper = LinkedInScraper(headless=True, output_dir="data/linkedin")

# Login
email = input("Enter your LinkedIn email: ")
password = input("Enter your LinkedIn password: ")

if scraper.login(email, password):
    print("Login successful!")
    
    # Scrape profiles
    profile_urls = input("Enter LinkedIn profile URLs (comma-separated): ").split(",")
    profile_urls = [url.strip() for url in profile_urls if url.strip()]
    
    if profile_urls:
        print(f"Scraping {len(profile_urls)} profiles...")
        scraper.scrape_profiles(profile_urls)
        print("Scraping completed!")
        
        # Check if profiles.csv was created
        if os.path.exists("data/linkedin/profiles.csv"):
            print("Profiles saved successfully!")
            profiles = pd.read_csv("data/linkedin/profiles.csv")
            print(f"Number of profiles: {len(profiles)}")
            print("Sample profiles:")
            print(profiles.head())
        else:
            print("Error: profiles.csv not found!")
    else:
        print("No profile URLs provided.")
else:
    print("Login failed!")

# Close browser
scraper.close()
