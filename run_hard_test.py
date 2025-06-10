#!/usr/bin/env python3
"""
Run Hard Test on Cross-Platform User Identification Model
Tests the model with challenging scenarios in terminal
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import our system components
try:
    from models.cross_platform_identifier import CrossPlatformUserIdentifier
    from data.data_loader import DataLoader
    from models.evaluator import Evaluator
    from utils.visualizer import Visualizer
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def load_test_data():
    """Load the challenging test samples."""
    
    print("📥 Loading challenging test samples...")
    
    try:
        # Load test data
        linkedin_profiles = pd.read_csv('test_samples/linkedin_profiles.csv')
        instagram_profiles = pd.read_csv('test_samples/instagram_profiles.csv')
        ground_truth = pd.read_csv('test_samples/ground_truth.csv')
        linkedin_posts = pd.read_csv('test_samples/linkedin_posts.csv')
        instagram_posts = pd.read_csv('test_samples/instagram_posts.csv')
        
        print(f"✅ Loaded test data:")
        print(f"   • LinkedIn profiles: {len(linkedin_profiles)}")
        print(f"   • Instagram profiles: {len(instagram_profiles)}")
        print(f"   • Ground truth pairs: {len(ground_truth)}")
        print(f"   • LinkedIn posts: {len(linkedin_posts)}")
        print(f"   • Instagram posts: {len(instagram_posts)}")
        
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

def analyze_test_cases(data):
    """Analyze each test case individually."""
    
    print("\n🔍 ANALYZING INDIVIDUAL TEST CASES")
    print("=" * 40)
    
    ground_truth = data['ground_truth']
    linkedin_profiles = data['linkedin_profiles']
    instagram_profiles = data['instagram_profiles']
    
    for _, row in ground_truth.iterrows():
        print(f"\n📋 Case: {row['case_description']}")
        print(f"Expected Match: {'✅ Yes' if row['is_same_user'] else '❌ No'}")
        
        # Get profiles
        ln_profile = linkedin_profiles[linkedin_profiles['user_id'] == row['linkedin_id']].iloc[0]
        ig_profile = instagram_profiles[instagram_profiles['user_id'] == row['instagram_id']].iloc[0]
        
        print(f"LinkedIn: {ln_profile['name']} - {ln_profile['bio'][:50]}...")
        print(f"Instagram: {ig_profile['name']} - {ig_profile['bio'][:50]}...")

def run_semantic_analysis(data):
    """Test semantic analysis on challenging cases."""
    
    print("\n🧠 SEMANTIC ANALYSIS TEST")
    print("=" * 30)
    
    try:
        from features.semantic_embedder import SemanticEmbedder
        
        embedder = SemanticEmbedder()
        
        linkedin_profiles = data['linkedin_profiles']
        instagram_profiles = data['instagram_profiles']
        ground_truth = data['ground_truth']
        
        results = []
        
        for _, row in ground_truth.iterrows():
            ln_profile = linkedin_profiles[linkedin_profiles['user_id'] == row['linkedin_id']].iloc[0]
            ig_profile = instagram_profiles[instagram_profiles['user_id'] == row['instagram_id']].iloc[0]
            
            # Get semantic embeddings
            ln_embedding = embedder.get_text_embedding(ln_profile['bio'])
            ig_embedding = embedder.get_text_embedding(ig_profile['bio'])
            
            # Calculate similarity
            similarity = np.dot(ln_embedding, ig_embedding) / (
                np.linalg.norm(ln_embedding) * np.linalg.norm(ig_embedding)
            )
            
            results.append({
                'case': row['case_description'][:30] + "...",
                'expected': row['is_same_user'],
                'similarity': similarity,
                'prediction': 1 if similarity > 0.7 else 0
            })
            
            status = "✅" if (similarity > 0.7) == row['is_same_user'] else "❌"
            print(f"{status} {row['case_description'][:40]}... | Sim: {similarity:.3f}")
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['prediction'] == r['expected'])
        accuracy = correct / len(results)
        print(f"\n📊 Semantic Analysis Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
        
        return results
        
    except Exception as e:
        print(f"❌ Semantic analysis failed: {e}")
        return []

def run_profile_analysis(data):
    """Test profile-based analysis."""
    
    print("\n👤 PROFILE ANALYSIS TEST")
    print("=" * 25)
    
    try:
        linkedin_profiles = data['linkedin_profiles']
        instagram_profiles = data['instagram_profiles']
        ground_truth = data['ground_truth']
        
        results = []
        
        for _, row in ground_truth.iterrows():
            ln_profile = linkedin_profiles[linkedin_profiles['user_id'] == row['linkedin_id']].iloc[0]
            ig_profile = instagram_profiles[instagram_profiles['user_id'] == row['instagram_id']].iloc[0]
            
            # Simple profile matching
            name_sim = 1.0 if ln_profile['name'].lower() in ig_profile['name'].lower() or \
                             ig_profile['name'].lower() in ln_profile['name'].lower() else 0.0
            
            location_sim = 1.0 if ln_profile['location'].lower() in ig_profile['location'].lower() or \
                                 ig_profile['location'].lower() in ln_profile['location'].lower() else 0.0
            
            field_sim = 1.0 if ln_profile['field_of_interest'].lower() in ig_profile['field_of_interest'].lower() or \
                              ig_profile['field_of_interest'].lower() in ln_profile['field_of_interest'].lower() else 0.0
            
            # Combined score
            profile_score = (name_sim + location_sim + field_sim) / 3
            
            results.append({
                'case': row['case_description'][:30] + "...",
                'expected': row['is_same_user'],
                'score': profile_score,
                'prediction': 1 if profile_score > 0.3 else 0
            })
            
            status = "✅" if (profile_score > 0.3) == row['is_same_user'] else "❌"
            print(f"{status} {row['case_description'][:40]}... | Score: {profile_score:.3f}")
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['prediction'] == r['expected'])
        accuracy = correct / len(results)
        print(f"\n📊 Profile Analysis Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
        
        return results
        
    except Exception as e:
        print(f"❌ Profile analysis failed: {e}")
        return []

def run_ensemble_test(data):
    """Run the full ensemble system test."""
    
    print("\n🎯 ENSEMBLE SYSTEM TEST")
    print("=" * 25)
    
    try:
        # Initialize the system
        identifier = CrossPlatformUserIdentifier()
        
        # Prepare data for the system
        linkedin_data = {
            'profiles': data['linkedin_profiles'],
            'posts': data['linkedin_posts']
        }
        
        instagram_data = {
            'profiles': data['instagram_profiles'],
            'posts': data['instagram_posts']
        }
        
        print("🔄 Running ensemble matching...")
        
        # Load data into the identifier
        identifier.data = {
            'linkedin': linkedin_data,
            'instagram': instagram_data
        }

        # Preprocess and extract features
        identifier.preprocess()
        identifier.extract_features()

        # Run the matching
        matches = identifier.match_users(
            platform1_name='linkedin',
            platform2_name='instagram',
            embedding_type='fusion'
        )
        
        if matches is not None and not matches.empty:
            print(f"✅ Generated {len(matches)} matches")
            
            # Evaluate against ground truth
            evaluator = Evaluator()
            metrics = evaluator.evaluate(matches, data['ground_truth'])
            
            print(f"\n📊 ENSEMBLE RESULTS:")
            print(f"   • Precision: {metrics.get('precision', 0):.3f}")
            print(f"   • Recall: {metrics.get('recall', 0):.3f}")
            print(f"   • F1-Score: {metrics.get('f1', 0):.3f}")
            print(f"   • Best Threshold: {metrics.get('best_threshold', 0):.3f}")
            
            # Show individual matches
            print(f"\n🔍 INDIVIDUAL MATCH RESULTS:")
            for _, match in matches.iterrows():
                gt_row = data['ground_truth'][
                    (data['ground_truth']['linkedin_id'] == match['user_id1']) &
                    (data['ground_truth']['instagram_id'] == match['user_id2'])
                ]
                
                if not gt_row.empty:
                    expected = gt_row.iloc[0]['is_same_user']
                    predicted = 1 if match['confidence'] > metrics.get('best_threshold', 0.7) else 0
                    status = "✅" if predicted == expected else "❌"
                    
                    print(f"{status} {match['user_id1']} ↔ {match['user_id2']} | "
                          f"Conf: {match['confidence']:.3f} | Expected: {expected}")
            
            return metrics
        else:
            print("❌ No matches generated")
            return None
            
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run comprehensive hard tests."""
    
    print("🔥 HARD TEST SUITE FOR CROSS-PLATFORM USER IDENTIFICATION")
    print("=" * 60)
    print()
    
    # Load test data
    data = load_test_data()
    if data is None:
        print("❌ Failed to load test data. Run create_hard_test_samples.py first.")
        return False
    
    # Analyze test cases
    analyze_test_cases(data)
    
    # Run individual component tests
    print("\n" + "=" * 60)
    print("🧪 COMPONENT-WISE TESTING")
    print("=" * 60)
    
    semantic_results = run_semantic_analysis(data)
    profile_results = run_profile_analysis(data)
    
    # Run full ensemble test
    print("\n" + "=" * 60)
    print("🎯 FULL SYSTEM TESTING")
    print("=" * 60)
    
    ensemble_results = run_ensemble_test(data)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    if semantic_results:
        semantic_acc = sum(1 for r in semantic_results if r['prediction'] == r['expected']) / len(semantic_results)
        print(f"🧠 Semantic Analysis: {semantic_acc:.1%}")
    
    if profile_results:
        profile_acc = sum(1 for r in profile_results if r['prediction'] == r['expected']) / len(profile_results)
        print(f"👤 Profile Analysis: {profile_acc:.1%}")
    
    if ensemble_results:
        print(f"🎯 Ensemble System: {ensemble_results.get('f1', 0):.1%} F1-Score")
    
    print("\n🎉 Hard test suite completed!")
    return True

if __name__ == "__main__":
    main()
