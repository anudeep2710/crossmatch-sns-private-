#!/usr/bin/env python3
"""
Create Hard Test Samples for Cross-Platform User Identification
Tests the model with challenging, realistic scenarios
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def create_challenging_test_samples():
    """Create very challenging test samples to stress-test the model."""
    
    print("🔥 CREATING CHALLENGING TEST SAMPLES")
    print("=" * 40)
    
    # Create challenging scenarios
    test_cases = []
    
    # CASE 1: Same person, completely different writing styles
    test_cases.append({
        'case_id': 1,
        'description': 'Same person, different writing styles (professional vs casual)',
        'linkedin_profile': {
            'user_id': 'ln_test_001',
            'name': 'Dr. Sarah Johnson',
            'bio': 'Senior Data Scientist at TechCorp. PhD in Machine Learning from Stanford. Published 15+ papers in top-tier conferences. Expertise in deep learning, NLP, and computer vision.',
            'location': 'San Francisco, CA',
            'field_of_interest': 'Data Science'
        },
        'instagram_profile': {
            'user_id': 'ig_test_001', 
            'name': 'sarah_j_adventures',
            'bio': '🌟 coffee addict ☕ hiking enthusiast 🏔️ weekend warrior 💪 living my best life in SF 🌉 #blessed #wanderlust',
            'location': 'SF Bay Area',
            'field_of_interest': 'Travel'
        },
        'linkedin_posts': [
            'Excited to present our latest research on transformer architectures at NeurIPS 2024. The results show significant improvements in few-shot learning scenarios.',
            'Just published a comprehensive survey on attention mechanisms in deep learning. Link in comments for those interested in the technical details.'
        ],
        'instagram_posts': [
            'omg this latte art is EVERYTHING ☕✨ barista skills on point today! #coffeelover #latteartislife',
            'weekend hike complete! 🥾 those mountain views hit different when you need a break from coding 💻➡️🏔️'
        ],
        'is_match': True
    })
    
    # CASE 2: Different people, very similar profiles
    test_cases.append({
        'case_id': 2,
        'description': 'Different people, nearly identical professional backgrounds',
        'linkedin_profile': {
            'user_id': 'ln_test_002',
            'name': 'Michael Chen',
            'bio': 'Software Engineer at Google. Stanford CS graduate. Passionate about machine learning and distributed systems.',
            'location': 'Mountain View, CA',
            'field_of_interest': 'Software Engineering'
        },
        'instagram_profile': {
            'user_id': 'ig_test_002',
            'name': 'mike_codes',
            'bio': 'SWE @Google | Stanford alum | ML enthusiast | Building the future one line of code at a time 💻',
            'location': 'Mountain View',
            'field_of_interest': 'Technology'
        },
        'linkedin_posts': [
            'Working on some exciting distributed systems challenges at Google. The scale of our infrastructure never ceases to amaze me.',
            'Great discussion at the Stanford AI meetup yesterday. The future of machine learning is looking bright!'
        ],
        'instagram_posts': [
            'Another day at the Googleplex! Working on some cool distributed systems stuff 🔧 #googlelife #engineering',
            'Stanford AI meetup was incredible! So many brilliant minds in one room 🧠 #artificialintelligence #networking'
        ],
        'is_match': False  # Different people with similar backgrounds
    })
    
    # CASE 3: Same person, different languages/cultures
    test_cases.append({
        'case_id': 3,
        'description': 'Same person, bilingual with cultural code-switching',
        'linkedin_profile': {
            'user_id': 'ln_test_003',
            'name': 'Maria Rodriguez',
            'bio': 'Marketing Director at Global Brands Inc. MBA from Wharton. Specializing in international market expansion and cross-cultural communication.',
            'location': 'New York, NY',
            'field_of_interest': 'Marketing'
        },
        'instagram_profile': {
            'user_id': 'ig_test_003',
            'name': 'maria_worldwide',
            'bio': '🌍 Mercadóloga internacional | Amante de la cultura | NYC vibes | Connecting worlds through brands ✨',
            'location': 'Nueva York',
            'field_of_interest': 'Marketing'
        },
        'linkedin_posts': [
            'Successful launch of our Q4 campaign in Latin American markets. Cultural sensitivity and local insights were key to achieving 150% of our target metrics.',
            'Attending the Global Marketing Summit next week. Looking forward to sharing insights on cross-cultural brand positioning.'
        ],
        'instagram_posts': [
            '¡Qué emoción! Nuestra campaña en Latinoamérica fue un éxito total 🎉 cuando entiendes la cultura, todo fluye mejor 💫',
            'NYC energy + Latin passion = magic in marketing ✨ grateful for this multicultural life 🌎 #marketinglife #latina'
        ],
        'is_match': True
    })
    
    # CASE 4: Same person, major life transition
    test_cases.append({
        'case_id': 4,
        'description': 'Same person, career transition (tech to art)',
        'linkedin_profile': {
            'user_id': 'ln_test_004',
            'name': 'Alex Thompson',
            'bio': 'Former Senior Software Engineer transitioning to Digital Art. 8 years at Microsoft. Now pursuing MFA in Digital Media Arts. Bridging technology and creativity.',
            'location': 'Seattle, WA',
            'field_of_interest': 'Digital Arts'
        },
        'instagram_profile': {
            'user_id': 'ig_test_004',
            'name': 'alex_creates',
            'bio': '🎨 Digital artist | Former tech person | Creating beauty through code and pixels | Seattle based | Art is the new algorithm ✨',
            'location': 'Seattle',
            'field_of_interest': 'Art'
        },
        'linkedin_posts': [
            'One year ago I left my comfortable software engineering role to pursue art. Best decision ever. My technical background gives me a unique perspective in digital media.',
            'Speaking at the Tech-to-Art career transition panel next month. Happy to share insights with others considering similar paths.'
        ],
        'instagram_posts': [
            'from debugging code to creating art 🎨➡️💻 never thought my programming skills would help with generative art but here we are! #techart #careertransition',
            'seattle rain = perfect studio weather ☔ working on a new piece that combines my love for algorithms and visual aesthetics 🌧️✨'
        ],
        'is_match': True
    })
    
    # CASE 5: Different people, shared interests and location
    test_cases.append({
        'case_id': 5,
        'description': 'Different people, same city, same hobbies, similar age',
        'linkedin_profile': {
            'user_id': 'ln_test_005',
            'name': 'Jennifer Kim',
            'bio': 'Product Manager at Startup Inc. Love hiking, photography, and craft coffee. Always looking for the next adventure.',
            'location': 'Austin, TX',
            'field_of_interest': 'Product Management'
        },
        'instagram_profile': {
            'user_id': 'ig_test_005',
            'name': 'jenny_explores',
            'bio': '📸 Austin photographer | ☕ Coffee enthusiast | 🥾 Weekend hiker | Capturing life one shot at a time',
            'location': 'Austin, Texas',
            'field_of_interest': 'Photography'
        },
        'linkedin_posts': [
            'Great product launch this quarter! Our user engagement metrics exceeded expectations by 40%. Team collaboration was key to this success.',
            'Attending the Austin Product Management meetup tonight. Always excited to learn from fellow PMs in the community.'
        ],
        'instagram_posts': [
            'austin sunrise from zilker park 🌅 nothing beats starting the day with nature and coffee ☕ #austinlife #photography',
            'product management meetup tonight! excited to connect with fellow austin PMs 💼 this city has such an amazing tech community 🤘'
        ],
        'is_match': False  # Different people, just similar interests
    })
    
    return test_cases

def create_network_data(test_cases):
    """Create challenging network scenarios."""
    
    network_scenarios = []
    
    # Scenario 1: Same person, different network structures
    network_scenarios.append({
        'case_id': 'net_001',
        'description': 'Same person, professional vs personal networks',
        'linkedin_network': [
            ('ln_test_001', 'ln_colleague_001'),
            ('ln_test_001', 'ln_colleague_002'),
            ('ln_test_001', 'ln_boss_001'),
            ('ln_test_001', 'ln_industry_expert_001')
        ],
        'instagram_network': [
            ('ig_test_001', 'ig_friend_001'),
            ('ig_test_001', 'ig_family_001'),
            ('ig_test_001', 'ig_hobby_friend_001'),
            ('ig_test_001', 'ig_neighbor_001')
        ],
        'is_match': True,
        'challenge': 'No overlapping connections between professional and personal networks'
    })
    
    # Scenario 2: Different people, some mutual connections
    network_scenarios.append({
        'case_id': 'net_002', 
        'description': 'Different people with mutual professional connections',
        'linkedin_network': [
            ('ln_test_002', 'ln_mutual_001'),
            ('ln_test_002', 'ln_mutual_002'),
            ('ln_test_002', 'ln_unique_002')
        ],
        'instagram_network': [
            ('ig_test_002', 'ig_mutual_001'),  # Same person as ln_mutual_001
            ('ig_test_002', 'ig_mutual_002'),  # Same person as ln_mutual_002
            ('ig_test_002', 'ig_unique_002')
        ],
        'is_match': False,
        'challenge': 'Mutual connections but different people (colleagues)'
    })
    
    return network_scenarios

def create_temporal_challenges():
    """Create challenging temporal patterns."""
    
    temporal_scenarios = []
    
    # Same person, different posting schedules
    base_time = datetime.now() - timedelta(days=30)
    
    # Professional posting pattern (weekdays, business hours)
    linkedin_times = []
    for i in range(20):
        day_offset = i * 1.5  # Every 1.5 days
        post_time = base_time + timedelta(days=day_offset)
        # Adjust to business hours (9 AM - 6 PM, weekdays)
        if post_time.weekday() < 5:  # Monday to Friday
            post_time = post_time.replace(hour=random.randint(9, 18))
            linkedin_times.append(post_time)
    
    # Personal posting pattern (evenings, weekends)
    instagram_times = []
    for i in range(25):
        day_offset = i * 1.2  # Every 1.2 days
        post_time = base_time + timedelta(days=day_offset)
        # Adjust to personal hours (evenings, weekends)
        if post_time.weekday() >= 5:  # Weekends
            post_time = post_time.replace(hour=random.randint(10, 22))
        else:  # Weekday evenings
            post_time = post_time.replace(hour=random.randint(18, 23))
        instagram_times.append(post_time)
    
    temporal_scenarios.append({
        'case_id': 'temp_001',
        'description': 'Same person, different posting schedules',
        'linkedin_times': linkedin_times,
        'instagram_times': instagram_times,
        'is_match': True,
        'challenge': 'Professional vs personal posting patterns'
    })
    
    return temporal_scenarios

def save_test_samples(test_cases):
    """Save test samples to CSV files."""
    
    os.makedirs('test_samples', exist_ok=True)
    
    # Create LinkedIn profiles
    linkedin_profiles = []
    instagram_profiles = []
    ground_truth = []
    linkedin_posts = []
    instagram_posts = []
    
    for case in test_cases:
        # Add profiles
        linkedin_profiles.append(case['linkedin_profile'])
        instagram_profiles.append(case['instagram_profile'])
        
        # Add ground truth
        ground_truth.append({
            'linkedin_id': case['linkedin_profile']['user_id'],
            'instagram_id': case['instagram_profile']['user_id'],
            'is_same_user': 1 if case['is_match'] else 0,
            'case_description': case['description']
        })
        
        # Add posts
        for i, post in enumerate(case['linkedin_posts']):
            linkedin_posts.append({
                'user_id': case['linkedin_profile']['user_id'],
                'content': post,
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
                'likes': random.randint(5, 100),
                'comments': random.randint(0, 20)
            })
        
        for i, post in enumerate(case['instagram_posts']):
            instagram_posts.append({
                'user_id': case['instagram_profile']['user_id'],
                'content': post,
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
                'likes': random.randint(10, 500),
                'comments': random.randint(0, 50)
            })
    
    # Save to CSV files
    pd.DataFrame(linkedin_profiles).to_csv('test_samples/linkedin_profiles.csv', index=False)
    pd.DataFrame(instagram_profiles).to_csv('test_samples/instagram_profiles.csv', index=False)
    pd.DataFrame(ground_truth).to_csv('test_samples/ground_truth.csv', index=False)
    pd.DataFrame(linkedin_posts).to_csv('test_samples/linkedin_posts.csv', index=False)
    pd.DataFrame(instagram_posts).to_csv('test_samples/instagram_posts.csv', index=False)
    
    print("✅ Test samples saved to test_samples/ folder")
    return len(test_cases)

def main():
    """Create and save challenging test samples."""
    
    print("🔥 CREATING HARD TEST SAMPLES FOR MODEL EVALUATION")
    print("=" * 55)
    print()
    
    # Create challenging test cases
    test_cases = create_challenging_test_samples()
    
    print(f"📊 Created {len(test_cases)} challenging test cases:")
    for case in test_cases:
        print(f"   • Case {case['case_id']}: {case['description']}")
        print(f"     Match: {'✅ Yes' if case['is_match'] else '❌ No'}")
    
    print()
    
    # Save test samples
    num_cases = save_test_samples(test_cases)
    
    print()
    print("🎯 CHALLENGE LEVELS:")
    print("==================")
    print("🔥 EXTREME: Same person, completely different writing styles")
    print("🔥 HARD: Different people, nearly identical backgrounds") 
    print("🔥 COMPLEX: Bilingual users with cultural code-switching")
    print("🔥 TRICKY: Major life/career transitions")
    print("🔥 DECEPTIVE: Different people, shared interests/location")
    print()
    print("📁 Files created in test_samples/ folder:")
    print("   • linkedin_profiles.csv")
    print("   • instagram_profiles.csv") 
    print("   • ground_truth.csv")
    print("   • linkedin_posts.csv")
    print("   • instagram_posts.csv")
    print()
    print("🚀 Ready to test the model's limits!")
    
    return True

if __name__ == "__main__":
    main()
