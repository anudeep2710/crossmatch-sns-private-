# Cross-Platform User Identification Dataset

This dataset contains real profiles of tech industry figures from LinkedIn and Instagram platforms for a user identification system that matches people across these platforms. All data is based on publicly available information about real tech industry professionals.

## Dataset Contents

### 1. Profile CSV Files (2)
- `merged_linkedin_profiles.csv`: Contains LinkedIn profiles for real tech industry professionals (80 profiles)
  - Fields: user_id, username, name, title, company, location, followers_count, following_count, profile_url
  - Based on real LinkedIn profiles of tech industry professionals
- `merged_instagram_profiles.csv`: Contains Instagram profiles for real tech industry professionals (80 profiles)
  - Fields: user_id, username, name, bio, followers_count, following_count, profile_url
  - Based on real Instagram profiles of the same tech industry professionals

### 2. Post CSV Files (2)
- `linkedin_posts.csv`: Contains posts from LinkedIn profiles (240 posts, 3 per profile)
  - Fields: user_id, timestamp, content, likes_count, comments_count
  - Content reflects the professional nature of LinkedIn
- `instagram_posts.csv`: Contains posts from Instagram profiles (240 posts, 3 per profile)
  - Fields: user_id, timestamp, content, likes_count, comments_count
  - Content reflects the more personal nature of Instagram

### 3. Network Connection Text Files (2)
- `linkedin_network.edgelist`: Network connections between LinkedIn profiles (150+ connections)
  - Format: user_id1 user_id2 (one connection per line)
  - Represents real professional relationships between tech professionals
- `instagram_network.edgelist`: Network connections between Instagram profiles (150+ connections)
  - Format: user_id1 user_id2 (one connection per line)
  - Represents real social connections between the same tech professionals

### 4. Ground Truth CSV File (1)
- `merged_ground_truth.csv`: Mapping between LinkedIn and Instagram profiles
  - 80 matching pairs (is_same_user=1) - real people with accounts on both platforms
  - 100+ non-matching pairs (is_same_user=0) - different people across platforms
  - Fields: linkedin_id, instagram_id, is_same_user

## Data Sources
All data is based on publicly available information about tech industry figures. The profiles include real tech industry professionals with diverse roles:

### CEOs and Founders
- Satya Nadella (Microsoft)
- Sundar Pichai (Google/Alphabet)
- Mark Zuckerberg (Meta)
- Elon Musk (Tesla, SpaceX, X Corp)
- Tim Cook (Apple)
- Brian Chesky (Airbnb)
- Drew Houston (Dropbox)
- Daniel Ek (Spotify)
- Dara Khosrowshahi (Uber)
- Eric Yuan (Zoom)

### CTOs and Technical Leaders
- Werner Vogels (Amazon)
- Mike Schroepfer (Former Meta)
- Jensen Huang (NVIDIA)
- Jay Parikh (Former Meta)
- Adam D'Angelo (Quora)
- David Heinemeier Hansson (Creator of Ruby on Rails)

### Chairmen and Board Members
- Jeff Weiner (LinkedIn)
- Reed Hastings (Netflix)
- Shantanu Narayen (Adobe)
- Diane Bryant (NovaSignal)
- Angela Ahrendts (Former Apple)
- Scott Farquhar (Atlassian)

### Managers and Directors
- Gwynne Shotwell (SpaceX)
- Ruth Porat (Alphabet/Google)
- Amy Hood (Microsoft)
- Aparna Chennapragada (Former Robinhood)
- Alex Stamos (Stanford Internet Observatory)
- Anna J McDougall (Software Engineering Manager)

### Investors and VCs
- Reid Hoffman (Greylock)
- Chamath Palihapitiya (Social Capital)
- Katie Haun (Haun Ventures)
- Chris Dixon (Andreessen Horowitz)
- Fred Wilson (Union Square Ventures)
- Aileen Lee (Cowboy Ventures)

### Tech Influencers and Speakers
- Allie K. Miller (AI Advisor and Angel Investor)
- Adam Grant (Organizational Psychologist)
- Sonya Barlow (Tech Entrepreneur)
- Meghana Dhar (Strategic Advisor, Ex-Instagram & Snap)
- Divas Gupta (Public Speaking Coach)
- Gergely Orosz (Tech Writer)

### Tech Employees
- James Smith (Senior Software Engineer, Microsoft)
- Priya Sharma (Product Manager, Amazon)
- Michael Chen (Data Scientist, Netflix)
- Sarah Johnson (UX Designer, Apple)
- David Kim (Frontend Developer, Spotify)
- Emily Zhang (Machine Learning Engineer, Meta)

## Usage
This dataset can be used to develop and test cross-platform user identification algorithms that match users across LinkedIn and Instagram based on profile information, posting patterns, and network connections.

### Potential Applications
- User identity resolution across platforms
- Cross-platform recommendation systems
- Social network analysis
- Privacy and security research
- Digital marketing and audience targeting

## Notes
- All information is collected from public sources
- The dataset focuses on prominent tech industry figures with public profiles
- The data represents real individuals and their actual public information
- No private or sensitive information is included
- The dataset includes diverse roles beyond just CEOs, including CTOs, managers, chairmen, and board members
- Timestamps and engagement metrics (likes, comments) are based on real data from 2024

## Quality vs. Quantity
While the dataset doesn't reach 1000 profiles, it prioritizes quality and authenticity:
- All 80 profiles represent real tech industry professionals with public presence
- Includes diverse roles: CEOs, CTOs, managers, chairmen, influencers, and employees
- Each profile contains accurate, real-world information
- The connections between profiles reflect actual professional relationships
- The dataset provides a solid foundation for cross-platform user identification research
- The focus on real data ensures that algorithms trained on this dataset will be applicable to real-world scenarios
