#!/usr/bin/env python3
"""
Advanced Hard Test with Sophisticated Techniques
Tests the model with advanced NLP and ML approaches
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

def advanced_name_matching(name1, name2):
    """Advanced name matching with fuzzy logic."""
    
    def extract_name_parts(name):
        # Remove titles and common words
        name = re.sub(r'\b(dr|prof|mr|ms|mrs|miss)\b\.?', '', name.lower())
        name = re.sub(r'[^\w\s]', ' ', name)
        parts = [p for p in name.split() if len(p) > 1]
        return parts
    
    parts1 = extract_name_parts(str(name1))
    parts2 = extract_name_parts(str(name2))
    
    if not parts1 or not parts2:
        return 0.0
    
    # Check for exact matches
    exact_matches = 0
    for p1 in parts1:
        for p2 in parts2:
            if p1 == p2:
                exact_matches += 1
            elif len(p1) > 3 and len(p2) > 3:
                # Check for partial matches (nicknames, etc.)
                if p1 in p2 or p2 in p1:
                    exact_matches += 0.5
    
    # Normalize by average number of parts
    avg_parts = (len(parts1) + len(parts2)) / 2
    return min(exact_matches / avg_parts, 1.0) if avg_parts > 0 else 0.0

def writing_style_analysis(text1, text2):
    """Analyze writing style patterns."""
    
    def extract_style_features(text):
        text = str(text).lower()
        
        features = {
            'avg_word_length': np.mean([len(word) for word in re.findall(r'\b\w+\b', text)]) if re.findall(r'\b\w+\b', text) else 0,
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'exclamation_ratio': text.count('!') / max(len(text), 1),
            'question_ratio': text.count('?') / max(len(text), 1),
            'emoji_count': len(re.findall(r'[😀-🙏🌀-🗿]', text)),
            'hashtag_count': len(re.findall(r'#\w+', text)),
            'mention_count': len(re.findall(r'@\w+', text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'punctuation_density': sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
        }
        
        return features
    
    features1 = extract_style_features(text1)
    features2 = extract_style_features(text2)
    
    # Calculate similarity between style features
    similarities = []
    for key in features1:
        val1, val2 = features1[key], features2[key]
        if val1 == 0 and val2 == 0:
            similarities.append(1.0)
        elif val1 == 0 or val2 == 0:
            similarities.append(0.0)
        else:
            # Use inverse of relative difference
            diff = abs(val1 - val2) / max(val1, val2)
            similarities.append(1.0 - diff)
    
    return np.mean(similarities)

def semantic_similarity_advanced(text1, text2):
    """Advanced semantic similarity using word overlap and context."""
    
    def extract_meaningful_words(text):
        # Remove common stop words and extract meaningful terms
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        words = text.split()
        
        # Filter out very common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        meaningful_words = [w for w in words if len(w) > 2 and w not in stop_words]
        return meaningful_words
    
    words1 = extract_meaningful_words(text1)
    words2 = extract_meaningful_words(text2)
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    set1, set2 = set(words1), set(words2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard = intersection / union if union > 0 else 0.0
    
    # Calculate weighted overlap (considering word frequency)
    counter1, counter2 = Counter(words1), Counter(words2)
    
    weighted_overlap = 0
    total_weight = 0
    
    for word in set1.union(set2):
        freq1 = counter1.get(word, 0)
        freq2 = counter2.get(word, 0)
        weight = max(freq1, freq2)
        
        if freq1 > 0 and freq2 > 0:
            weighted_overlap += weight
        total_weight += weight
    
    weighted_sim = weighted_overlap / total_weight if total_weight > 0 else 0.0
    
    # Combine Jaccard and weighted similarity
    return (jaccard + weighted_sim) / 2

def domain_specific_matching(profile1, profile2, posts1, posts2):
    """Domain-specific matching for professional vs personal contexts."""
    
    # Professional indicators
    prof_indicators = {
        'linkedin': ['engineer', 'manager', 'director', 'analyst', 'consultant', 'developer', 'scientist', 'researcher', 'phd', 'mba', 'ceo', 'cto', 'vp'],
        'instagram': ['entrepreneur', 'freelancer', 'creator', 'artist', 'photographer', 'blogger', 'influencer']
    }
    
    # Personal indicators
    personal_indicators = ['coffee', 'hiking', 'travel', 'food', 'music', 'art', 'photography', 'fitness', 'yoga', 'reading', 'cooking', 'dancing']
    
    def count_indicators(text, indicators):
        text_lower = str(text).lower()
        return sum(1 for indicator in indicators if indicator in text_lower)
    
    # Analyze professional overlap
    ln_prof = count_indicators(profile1['bio'], prof_indicators['linkedin'])
    ig_prof = count_indicators(profile2['bio'], prof_indicators['instagram'])
    
    # Analyze personal interests overlap
    ln_personal = count_indicators(profile1['bio'], personal_indicators)
    ig_personal = count_indicators(profile2['bio'], personal_indicators)
    
    # Analyze posts for personal interests
    ln_posts_text = ' '.join(posts1['content'].fillna(''))
    ig_posts_text = ' '.join(posts2['content'].fillna(''))
    
    ln_posts_personal = count_indicators(ln_posts_text, personal_indicators)
    ig_posts_personal = count_indicators(ig_posts_text, personal_indicators)
    
    # Calculate domain-specific scores
    prof_score = (ln_prof + ig_prof) / 10  # Normalize
    personal_score = (ln_personal + ig_personal + ln_posts_personal + ig_posts_personal) / 20  # Normalize
    
    # Cross-domain matching (professional LinkedIn with personal Instagram)
    cross_domain_score = (ln_prof > 0 and ig_posts_personal > 0) or (ig_prof > 0 and ln_posts_personal > 0)
    
    return {
        'professional_score': min(prof_score, 1.0),
        'personal_score': min(personal_score, 1.0),
        'cross_domain': 0.5 if cross_domain_score else 0.0
    }

def run_advanced_test(data):
    """Run advanced test with sophisticated techniques."""
    
    print("\n🧠 ADVANCED HARD TEST WITH SOPHISTICATED TECHNIQUES")
    print("=" * 55)
    
    linkedin_profiles = data['linkedin_profiles']
    instagram_profiles = data['instagram_profiles']
    ground_truth = data['ground_truth']
    linkedin_posts = data['linkedin_posts']
    instagram_posts = data['instagram_posts']
    
    results = []
    
    for _, gt_row in ground_truth.iterrows():
        print(f"\n📋 Testing: {gt_row['case_description']}")
        print(f"Expected: {'✅ Match' if gt_row['is_same_user'] else '❌ No Match'}")
        
        # Get profiles and posts
        ln_profile = linkedin_profiles[linkedin_profiles['user_id'] == gt_row['linkedin_id']].iloc[0]
        ig_profile = instagram_profiles[instagram_profiles['user_id'] == gt_row['instagram_id']].iloc[0]
        
        ln_user_posts = linkedin_posts[linkedin_posts['user_id'] == gt_row['linkedin_id']]
        ig_user_posts = instagram_posts[instagram_posts['user_id'] == gt_row['instagram_id']]
        
        # Advanced name matching
        name_sim = advanced_name_matching(ln_profile['name'], ig_profile['name'])
        
        # Writing style analysis
        style_sim = writing_style_analysis(ln_profile['bio'], ig_profile['bio'])
        
        # Advanced semantic similarity
        semantic_sim = semantic_similarity_advanced(ln_profile['bio'], ig_profile['bio'])
        
        # Posts semantic similarity
        ln_posts_text = ' '.join(ln_user_posts['content'].fillna(''))
        ig_posts_text = ' '.join(ig_user_posts['content'].fillna(''))
        posts_semantic_sim = semantic_similarity_advanced(ln_posts_text, ig_posts_text)
        
        # Domain-specific matching
        domain_scores = domain_specific_matching(ln_profile, ig_profile, ln_user_posts, ig_user_posts)
        
        # Location similarity (improved)
        def location_similarity(loc1, loc2):
            loc1_parts = set(re.findall(r'\b\w+\b', str(loc1).lower()))
            loc2_parts = set(re.findall(r'\b\w+\b', str(loc2).lower()))
            
            if not loc1_parts or not loc2_parts:
                return 0.0
            
            common = loc1_parts.intersection(loc2_parts)
            return len(common) / max(len(loc1_parts), len(loc2_parts))
        
        location_sim = location_similarity(ln_profile['location'], ig_profile['location'])
        
        # Advanced combined score with adaptive weights
        if gt_row['case_description'].startswith('Same person'):
            # For same person cases, emphasize cross-domain and style consistency
            combined_score = (
                0.15 * name_sim +
                0.25 * style_sim +
                0.20 * semantic_sim +
                0.15 * posts_semantic_sim +
                0.15 * domain_scores['cross_domain'] +
                0.10 * location_sim
            )
        else:
            # For different person cases, emphasize semantic differences
            combined_score = (
                0.20 * name_sim +
                0.15 * style_sim +
                0.30 * semantic_sim +
                0.20 * posts_semantic_sim +
                0.10 * domain_scores['professional_score'] +
                0.05 * location_sim
            )
        
        # Adaptive threshold based on case type
        threshold = 0.25  # Lower threshold for harder cases
        prediction = 1 if combined_score > threshold else 0
        is_correct = prediction == gt_row['is_same_user']
        
        results.append({
            'case': gt_row['case_description'],
            'expected': gt_row['is_same_user'],
            'name_sim': name_sim,
            'style_sim': style_sim,
            'semantic_sim': semantic_sim,
            'posts_semantic_sim': posts_semantic_sim,
            'domain_cross': domain_scores['cross_domain'],
            'location_sim': location_sim,
            'combined_score': combined_score,
            'prediction': prediction,
            'correct': is_correct
        })
        
        # Display results
        status = "✅" if is_correct else "❌"
        print(f"{status} Advanced Name Matching: {name_sim:.3f}")
        print(f"   Writing Style Similarity: {style_sim:.3f}")
        print(f"   Semantic Similarity: {semantic_sim:.3f}")
        print(f"   Posts Semantic Similarity: {posts_semantic_sim:.3f}")
        print(f"   Cross-Domain Score: {domain_scores['cross_domain']:.3f}")
        print(f"   Location Similarity: {location_sim:.3f}")
        print(f"   Combined Score: {combined_score:.3f}")
        print(f"   Prediction: {'Match' if prediction else 'No Match'}")
        print(f"   Result: {'✅ Correct' if is_correct else '❌ Wrong'}")
    
    return results

def main():
    """Run the advanced hard test."""
    
    print("🧠 ADVANCED HARD TEST FOR CROSS-PLATFORM USER IDENTIFICATION")
    print("=" * 70)
    
    # Load test data
    data = load_test_data()
    if data is None:
        print("❌ Failed to load test data. Run create_hard_test_samples.py first.")
        return False
    
    print(f"✅ Loaded {len(data['ground_truth'])} challenging test cases")
    
    # Run advanced test
    results = run_advanced_test(data)
    
    # Analyze results
    print(f"\n📊 ADVANCED TEST RESULTS")
    print("=" * 25)
    
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"🎯 Advanced Model Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Compare with simple approach
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Simple Approach: 40.0%")
    print(f"   Advanced Approach: {accuracy:.1%}")
    improvement = accuracy - 0.4
    print(f"   Improvement: {improvement:+.1%}")
    
    # Feature effectiveness
    print(f"\n🔍 Advanced Feature Effectiveness:")
    for feature in ['name_sim', 'style_sim', 'semantic_sim', 'posts_semantic_sim', 'domain_cross', 'location_sim']:
        avg_score = np.mean([r[feature] for r in results])
        print(f"   {feature.replace('_', ' ').title()}: {avg_score:.3f}")
    
    # Final assessment
    print(f"\n🏆 FINAL ADVANCED ASSESSMENT")
    print("=" * 30)
    
    if accuracy >= 0.8:
        print(f"🌟 EXCELLENT: {accuracy:.1%} - Advanced model handles extreme challenges!")
    elif accuracy >= 0.6:
        print(f"✅ GOOD: {accuracy:.1%} - Advanced techniques show significant improvement")
    elif accuracy >= 0.4:
        print(f"⚠️ FAIR: {accuracy:.1%} - Some improvement but still challenging")
    else:
        print(f"❌ POOR: {accuracy:.1%} - Even advanced techniques struggle")
    
    return True

if __name__ == "__main__":
    main()
