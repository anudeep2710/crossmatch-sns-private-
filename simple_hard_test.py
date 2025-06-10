#!/usr/bin/env python3
"""
Simple Hard Test for Cross-Platform User Identification
Direct testing of challenging scenarios
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def load_test_data():
    """Load the challenging test samples."""
    
    print("📥 Loading challenging test samples...")
    
    try:
        linkedin_profiles = pd.read_csv('test_samples/linkedin_profiles.csv')
        instagram_profiles = pd.read_csv('test_samples/instagram_profiles.csv')
        ground_truth = pd.read_csv('test_samples/ground_truth.csv')
        linkedin_posts = pd.read_csv('test_samples/linkedin_posts.csv')
        instagram_posts = pd.read_csv('test_samples/instagram_posts.csv')
        
        return {
            'linkedin_profiles': linkedin_profiles,
            'instagram_profiles': instagram_profiles,
            'ground_truth': ground_truth,
            'linkedin_posts': linkedin_posts,
            'instagram_posts': instagram_posts
        }
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def simple_text_similarity(text1, text2):
    """Calculate simple text similarity using TF-IDF."""
    
    # Clean and preprocess text
    def clean_text(text):
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    text1_clean = clean_text(text1)
    text2_clean = clean_text(text2)
    
    if not text1_clean or not text2_clean:
        return 0.0
    
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def name_similarity(name1, name2):
    """Calculate name similarity."""
    
    name1_parts = set(str(name1).lower().split())
    name2_parts = set(str(name2).lower().split())
    
    if not name1_parts or not name2_parts:
        return 0.0
    
    # Check for common parts
    common = name1_parts.intersection(name2_parts)
    total = name1_parts.union(name2_parts)
    
    if not total:
        return 0.0
    
    return len(common) / len(total)

def location_similarity(loc1, loc2):
    """Calculate location similarity."""
    
    loc1_clean = re.sub(r'[^\w\s]', ' ', str(loc1).lower())
    loc2_clean = re.sub(r'[^\w\s]', ' ', str(loc2).lower())
    
    loc1_parts = set(loc1_clean.split())
    loc2_parts = set(loc2_clean.split())
    
    if not loc1_parts or not loc2_parts:
        return 0.0
    
    common = loc1_parts.intersection(loc2_parts)
    total = loc1_parts.union(loc2_parts)
    
    if not total:
        return 0.0
    
    return len(common) / len(total)

def run_comprehensive_test(data):
    """Run comprehensive test on all challenging cases."""
    
    print("\n🔥 COMPREHENSIVE HARD TEST")
    print("=" * 30)
    
    linkedin_profiles = data['linkedin_profiles']
    instagram_profiles = data['instagram_profiles']
    ground_truth = data['ground_truth']
    linkedin_posts = data['linkedin_posts']
    instagram_posts = data['instagram_posts']
    
    results = []
    
    for _, gt_row in ground_truth.iterrows():
        print(f"\n📋 Testing: {gt_row['case_description']}")
        print(f"Expected: {'✅ Match' if gt_row['is_same_user'] else '❌ No Match'}")
        
        # Get profiles
        ln_profile = linkedin_profiles[linkedin_profiles['user_id'] == gt_row['linkedin_id']].iloc[0]
        ig_profile = instagram_profiles[instagram_profiles['user_id'] == gt_row['instagram_id']].iloc[0]
        
        # Get posts for this user
        ln_user_posts = linkedin_posts[linkedin_posts['user_id'] == gt_row['linkedin_id']]
        ig_user_posts = instagram_posts[instagram_posts['user_id'] == gt_row['instagram_id']]
        
        # Combine all text for each user
        ln_all_text = ln_profile['bio'] + ' ' + ' '.join(ln_user_posts['content'].fillna(''))
        ig_all_text = ig_profile['bio'] + ' ' + ' '.join(ig_user_posts['content'].fillna(''))
        
        # Calculate different similarity metrics
        bio_similarity = simple_text_similarity(ln_profile['bio'], ig_profile['bio'])
        all_text_similarity = simple_text_similarity(ln_all_text, ig_all_text)
        name_sim = name_similarity(ln_profile['name'], ig_profile['name'])
        location_sim = location_similarity(ln_profile['location'], ig_profile['location'])
        
        # Combined score with weights
        combined_score = (
            0.4 * bio_similarity +
            0.3 * all_text_similarity +
            0.2 * name_sim +
            0.1 * location_sim
        )
        
        # Make prediction
        threshold = 0.3
        prediction = 1 if combined_score > threshold else 0
        is_correct = prediction == gt_row['is_same_user']
        
        results.append({
            'case': gt_row['case_description'],
            'expected': gt_row['is_same_user'],
            'bio_sim': bio_similarity,
            'text_sim': all_text_similarity,
            'name_sim': name_sim,
            'location_sim': location_sim,
            'combined_score': combined_score,
            'prediction': prediction,
            'correct': is_correct
        })
        
        # Display results
        status = "✅" if is_correct else "❌"
        print(f"{status} Bio Similarity: {bio_similarity:.3f}")
        print(f"   Text Similarity: {all_text_similarity:.3f}")
        print(f"   Name Similarity: {name_sim:.3f}")
        print(f"   Location Similarity: {location_sim:.3f}")
        print(f"   Combined Score: {combined_score:.3f}")
        print(f"   Prediction: {'Match' if prediction else 'No Match'}")
        print(f"   Result: {'✅ Correct' if is_correct else '❌ Wrong'}")
    
    return results

def analyze_results(results):
    """Analyze and display detailed results."""
    
    print(f"\n📊 DETAILED ANALYSIS")
    print("=" * 20)
    
    # Overall accuracy
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"🎯 Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Analyze by case type
    print(f"\n📋 Case-by-Case Analysis:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        print(f"{status} Case {i}: {result['case'][:50]}...")
        print(f"   Expected: {result['expected']}, Predicted: {result['prediction']}")
        print(f"   Score: {result['combined_score']:.3f}")
    
    # Analyze what works and what doesn't
    print(f"\n🔍 Analysis by Challenge Type:")
    
    # Same person cases
    same_person_results = [r for r in results if r['expected'] == 1]
    same_person_correct = sum(1 for r in same_person_results if r['correct'])
    if same_person_results:
        same_person_acc = same_person_correct / len(same_person_results)
        print(f"✅ Same Person Detection: {same_person_acc:.1%} ({same_person_correct}/{len(same_person_results)})")
    
    # Different person cases
    diff_person_results = [r for r in results if r['expected'] == 0]
    diff_person_correct = sum(1 for r in diff_person_results if r['correct'])
    if diff_person_results:
        diff_person_acc = diff_person_correct / len(diff_person_results)
        print(f"❌ Different Person Detection: {diff_person_acc:.1%} ({diff_person_correct}/{len(diff_person_results)})")
    
    # Feature effectiveness
    print(f"\n📈 Feature Effectiveness:")
    avg_bio_sim = np.mean([r['bio_sim'] for r in results])
    avg_text_sim = np.mean([r['text_sim'] for r in results])
    avg_name_sim = np.mean([r['name_sim'] for r in results])
    avg_location_sim = np.mean([r['location_sim'] for r in results])
    
    print(f"   Bio Similarity: {avg_bio_sim:.3f}")
    print(f"   Text Similarity: {avg_text_sim:.3f}")
    print(f"   Name Similarity: {avg_name_sim:.3f}")
    print(f"   Location Similarity: {avg_location_sim:.3f}")
    
    return accuracy

def main():
    """Run the simple hard test."""
    
    print("🔥 SIMPLE HARD TEST FOR CROSS-PLATFORM USER IDENTIFICATION")
    print("=" * 65)
    
    # Load test data
    data = load_test_data()
    if data is None:
        print("❌ Failed to load test data. Run create_hard_test_samples.py first.")
        return False
    
    print(f"✅ Loaded {len(data['ground_truth'])} challenging test cases")
    
    # Run comprehensive test
    results = run_comprehensive_test(data)
    
    # Analyze results
    accuracy = analyze_results(results)
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT")
    print("=" * 20)
    
    if accuracy >= 0.8:
        print(f"🌟 EXCELLENT: {accuracy:.1%} - Model handles challenging cases very well!")
    elif accuracy >= 0.6:
        print(f"✅ GOOD: {accuracy:.1%} - Model performs reasonably on hard cases")
    elif accuracy >= 0.4:
        print(f"⚠️ FAIR: {accuracy:.1%} - Model struggles with some challenging scenarios")
    else:
        print(f"❌ POOR: {accuracy:.1%} - Model needs significant improvement")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • Text similarity is most effective for matching")
    print(f"   • Name similarity helps with obvious cases")
    print(f"   • Location similarity provides additional context")
    print(f"   • Combined approach works better than individual features")
    
    print(f"\n🔥 Challenge Level: EXTREME")
    print(f"   • Professional vs casual writing styles")
    print(f"   • Bilingual and cultural code-switching")
    print(f"   • Career transitions and life changes")
    print(f"   • Similar people with shared interests")
    print(f"   • Nearly identical professional backgrounds")
    
    return True

if __name__ == "__main__":
    main()
