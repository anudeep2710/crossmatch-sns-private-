"""
Manual data preparation script for cross-platform user identification.

This script helps users manually prepare their Instagram and LinkedIn data
without needing to scrape the platforms directly.
"""

import os
import pandas as pd
import shutil
import argparse

def create_template_files(platform, output_dir):
    """Create template files for manual data entry."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define template data
    if platform == "instagram":
        # Profiles template
        profiles_data = {
            "user_id": ["instagram_user1", "instagram_user2", "instagram_user3"],
            "username": ["user1", "user2", "user3"],
            "name": ["User One", "User Two", "User Three"],
            "bio": ["This is user one's bio", "This is user two's bio", "This is user three's bio"],
            "posts_count": [10, 20, 30],
            "followers_count": [100, 200, 300],
            "following_count": [200, 300, 400],
            "profile_url": ["https://instagram.com/user1", "https://instagram.com/user2", "https://instagram.com/user3"]
        }
        
        # Posts template
        posts_data = {
            "post_id": ["instagram_user1_post_1", "instagram_user1_post_2", "instagram_user2_post_1", 
                        "instagram_user2_post_2", "instagram_user3_post_1", "instagram_user3_post_2"],
            "user_id": ["instagram_user1", "instagram_user1", "instagram_user2", 
                        "instagram_user2", "instagram_user3", "instagram_user3"],
            "content": ["This is a post by user one", "Another post by user one", 
                        "This is a post by user two", "Another post by user two",
                        "This is a post by user three", "Another post by user three"],
            "timestamp": ["2023-01-01 12:00:00", "2023-01-02 12:00:00", 
                          "2023-01-01 13:00:00", "2023-01-02 13:00:00",
                          "2023-01-01 14:00:00", "2023-01-02 14:00:00"],
            "likes": [50, 60, 70, 80, 90, 100],
            "comments": [5, 6, 7, 8, 9, 10]
        }
        
        # Network template
        network_data = """instagram_user1 instagram_user2
instagram_user1 instagram_user3
instagram_user2 instagram_user3"""
        
    elif platform == "linkedin":
        # Profiles template
        profiles_data = {
            "user_id": ["linkedin_user1", "linkedin_user2", "linkedin_user3"],
            "name": ["User One", "User Two", "User Three"],
            "headline": ["Software Engineer at Company A", "Data Scientist at Company B", "Product Manager at Company C"],
            "location": ["New York, NY", "San Francisco, CA", "Seattle, WA"],
            "summary": ["Experienced software engineer", "Data scientist with ML expertise", "Product manager with 5 years experience"],
            "experience": ["Company A, Company B", "Company C, Company D", "Company E, Company F"],
            "education": ["University X", "University Y", "University Z"],
            "skills": ["Python, Java", "Machine Learning, SQL", "Product Management, Agile"],
            "profile_url": ["https://linkedin.com/in/user1", "https://linkedin.com/in/user2", "https://linkedin.com/in/user3"]
        }
        
        # Posts template
        posts_data = {
            "post_id": ["linkedin_user1_post_1", "linkedin_user1_post_2", "linkedin_user2_post_1", 
                        "linkedin_user2_post_2", "linkedin_user3_post_1", "linkedin_user3_post_2"],
            "user_id": ["linkedin_user1", "linkedin_user1", "linkedin_user2", 
                        "linkedin_user2", "linkedin_user3", "linkedin_user3"],
            "content": ["This is a post by user one on LinkedIn", "Another post by user one on LinkedIn", 
                        "This is a post by user two on LinkedIn", "Another post by user two on LinkedIn",
                        "This is a post by user three on LinkedIn", "Another post by user three on LinkedIn"],
            "timestamp": ["2023-01-01 12:00:00", "2023-01-02 12:00:00", 
                          "2023-01-01 13:00:00", "2023-01-02 13:00:00",
                          "2023-01-01 14:00:00", "2023-01-02 14:00:00"],
            "likes": [50, 60, 70, 80, 90, 100],
            "comments": [5, 6, 7, 8, 9, 10]
        }
        
        # Network template
        network_data = """linkedin_user1 linkedin_user2
linkedin_user1 linkedin_user3
linkedin_user2 linkedin_user3"""
    
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    # Save templates
    profiles_df = pd.DataFrame(profiles_data)
    posts_df = pd.DataFrame(posts_data)
    
    profiles_path = os.path.join(output_dir, "profiles_template.csv")
    posts_path = os.path.join(output_dir, "posts_template.csv")
    network_path = os.path.join(output_dir, "network_template.edgelist")
    
    profiles_df.to_csv(profiles_path, index=False)
    posts_df.to_csv(posts_path, index=False)
    
    with open(network_path, "w") as f:
        f.write(network_data)
    
    print(f"Created template files for {platform} in {output_dir}:")
    print(f"  - {profiles_path}")
    print(f"  - {posts_path}")
    print(f"  - {network_path}")
    print("\nInstructions:")
    print("1. Edit these template files with your actual data")
    print("2. Save the edited files as 'profiles.csv', 'posts.csv', and 'network.edgelist' in the same directory")
    print("3. Run the cross-platform user identification app and use the 'Local Files' option to load your data")

def main():
    parser = argparse.ArgumentParser(description="Create template files for manual data preparation")
    parser.add_argument("platform", choices=["instagram", "linkedin"], help="Platform to create templates for")
    parser.add_argument("--output-dir", default="data", help="Output directory for template files")
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.platform)
    create_template_files(args.platform, output_dir)

if __name__ == "__main__":
    main()
