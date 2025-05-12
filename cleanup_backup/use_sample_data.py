"""
Script to use sample tech industry data for cross-platform user identification.

This script copies the sample data files to the correct locations for the application to use.
"""

import os
import shutil
import argparse

def use_sample_data():
    """Copy sample data files to the correct locations."""
    # Create directories if they don't exist
    os.makedirs("data/instagram", exist_ok=True)
    os.makedirs("data/linkedin", exist_ok=True)
    
    # Copy Instagram files
    shutil.copy("data/instagram/sample_profiles.csv", "data/instagram/profiles.csv")
    shutil.copy("data/instagram/sample_posts.csv", "data/instagram/posts.csv")
    shutil.copy("data/instagram/sample_network.edgelist", "data/instagram/network.edgelist")
    
    # Copy LinkedIn files
    shutil.copy("data/linkedin/sample_profiles.csv", "data/linkedin/profiles.csv")
    shutil.copy("data/linkedin/sample_posts.csv", "data/linkedin/posts.csv")
    shutil.copy("data/linkedin/sample_network.edgelist", "data/linkedin/network.edgelist")
    
    # Copy ground truth file
    shutil.copy("data/sample_ground_truth.csv", "data/ground_truth.csv")
    
    print("Sample data files have been copied to the correct locations.")
    print("You can now run the application and use the 'Local Files' option to load the data.")
    print("The sample data includes profiles for tech industry leaders like Sundar Pichai, Tim Cook, Mark Zuckerberg, etc.")

def main():
    parser = argparse.ArgumentParser(description="Use sample tech industry data for cross-platform user identification")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files without confirmation")
    
    args = parser.parse_args()
    
    # Check if files already exist
    existing_files = []
    for file_path in ["data/instagram/profiles.csv", "data/instagram/posts.csv", "data/instagram/network.edgelist",
                      "data/linkedin/profiles.csv", "data/linkedin/posts.csv", "data/linkedin/network.edgelist",
                      "data/ground_truth.csv"]:
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    if existing_files and not args.force:
        print("The following files already exist:")
        for file_path in existing_files:
            print(f"  - {file_path}")
        
        response = input("Do you want to overwrite these files? (y/n): ")
        if response.lower() != "y":
            print("Operation cancelled.")
            return
    
    use_sample_data()

if __name__ == "__main__":
    main()
