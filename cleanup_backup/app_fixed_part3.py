def compare_users():
    """Compare specific users between LinkedIn and Instagram."""
    st.header("LinkedIn-Instagram User Comparison")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to perform user comparison.")
        return

    if not st.session_state.embeddings_generated:
        st.warning("Please run analysis first on the Analysis tab.")
        st.info("This will generate the embeddings needed for user comparison.")
        return

    # Get platform names
    platform_names = list(st.session_state.identifier.data.keys())
    
    # Check if both LinkedIn and Instagram data are available
    linkedin_exists = "linkedin" in platform_names
    instagram_exists = "instagram" in platform_names
    
    if not linkedin_exists or not instagram_exists:
        missing_platforms = []
        if not linkedin_exists:
            missing_platforms.append("LinkedIn")
        if not instagram_exists:
            missing_platforms.append("Instagram")
        
        st.error(f"Missing data for {', '.join(missing_platforms)}.")
        st.info("Please go to the Data Loading tab and scrape data from the missing platform(s).")
        return
    
    # Set platform names
    platform1_name = "linkedin"
    platform2_name = "instagram"

    # Get user IDs
    user_ids1 = list(st.session_state.identifier.data[platform1_name]['profiles']['user_id'])
    user_ids2 = list(st.session_state.identifier.data[platform2_name]['profiles']['user_id'])

    # User selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LinkedIn User")
        user_id1 = st.selectbox("Select LinkedIn User", user_ids1)
        
        # Show LinkedIn profile info
        if user_id1:
            profile1 = st.session_state.identifier.data[platform1_name]['profiles'][
                st.session_state.identifier.data[platform1_name]['profiles']['user_id'] == user_id1
            ]
            if not profile1.empty:
                st.info(f"Name: {profile1.iloc[0].get('name', 'N/A')}")
                st.info(f"Headline: {profile1.iloc[0].get('headline', 'N/A')}")
                st.info(f"Location: {profile1.iloc[0].get('location', 'N/A')}")

    with col2:
        st.subheader("Instagram User")
        user_id2 = st.selectbox("Select Instagram User", user_ids2)
        
        # Show Instagram profile info
        if user_id2:
            profile2 = st.session_state.identifier.data[platform2_name]['profiles'][
                st.session_state.identifier.data[platform2_name]['profiles']['user_id'] == user_id2
            ]
            if not profile2.empty:
                st.info(f"Username: {profile2.iloc[0].get('username', 'N/A')}")
                st.info(f"Full Name: {profile2.iloc[0].get('full_name', 'N/A')}")
                st.info(f"Bio: {profile2.iloc[0].get('bio', 'N/A')}")

    # Compare button
    if st.button("Compare Users"):
        with st.spinner("Comparing users..."):
            # Get embeddings
            if platform1_name in st.session_state.identifier.embeddings and platform2_name in st.session_state.identifier.embeddings:
                embeddings1 = st.session_state.identifier.embeddings[platform1_name]['fusion']
                embeddings2 = st.session_state.identifier.embeddings[platform2_name]['fusion']
                
                # Check if user IDs exist in embeddings
                if user_id1 in embeddings1 and user_id2 in embeddings2:
                    # Get embeddings for selected users
                    embedding1 = embeddings1[user_id1]
                    embedding2 = embeddings2[user_id2]
                    
                    # Check if embeddings have the same dimension
                    if embedding1.shape[0] != embedding2.shape[0]:
                        st.warning("Embeddings have different dimensions. Using PCA to normalize dimensions.")
                        
                        # Use PCA to normalize dimensions
                        from sklearn.decomposition import PCA
                        
                        # Find the minimum dimension
                        min_dim = min(embedding1.shape[0], embedding2.shape[0])
                        
                        # Apply PCA
                        pca = PCA(n_components=min_dim)
                        if embedding1.shape[0] > min_dim:
                            embedding1 = pca.fit_transform(embedding1.reshape(1, -1))[0]
                        if embedding2.shape[0] > min_dim:
                            embedding2 = pca.fit_transform(embedding2.reshape(1, -1))[0]
                    
                    # Calculate similarity
                    similarity = st.session_state.identifier.user_matcher.calculate_similarity(
                        embedding1.reshape(1, -1),
                        embedding2.reshape(1, -1)
                    )[0][0]
                    
                    # Display similarity
                    st.subheader("Similarity Score")
                    st.info(f"Similarity: {similarity:.4f}")
                    
                    # Interpret similarity
                    threshold = st.session_state.identifier.config.get('matching_threshold', 0.7)
                    if similarity >= threshold:
                        st.success("These users are likely the same person.")
                    else:
                        st.warning("These users are likely different people.")
                else:
                    if user_id1 not in embeddings1:
                        st.error(f"No embedding found for LinkedIn user: {user_id1}")
                    if user_id2 not in embeddings2:
                        st.error(f"No embedding found for Instagram user: {user_id2}")
            else:
                st.error("Embeddings not found. Please run analysis first.")

def main():
    """Main function for the Streamlit app."""
    st.title("LinkedIn-Instagram User Identification")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Loading", "Analysis", "Results", "User Comparison"])

    # Display selected page
    if page == "Home":
        st.header("Welcome to LinkedIn-Instagram User Identification")
        st.write("""
        This application helps identify and match users across LinkedIn and Instagram
        using machine learning techniques.

        ### Features
        - Data scraping from LinkedIn and Instagram
        - Preprocessing and normalization of user data
        - Network-based user embeddings (Node2Vec)
        - Semantic embeddings from user content (BERT)
        - Temporal embeddings from user activity patterns
        - Multi-modal embedding fusion
        - User matching across platforms
        - Visualization of matching results

        ### Getting Started
        1. Go to the **Data Loading** page to scrape LinkedIn and Instagram data
        2. Run the analysis on the **Analysis** page
        3. View the results on the **Results** page
        4. Compare specific users on the **User Comparison** page
        
        ### Important Notes
        - Web scraping may violate the Terms of Service of LinkedIn and Instagram
        - Use this application responsibly and for educational purposes only
        - Ensure you have proper authorization to access the profiles you scrape
        """)

    elif page == "Data Loading":
        load_data()

    elif page == "Analysis":
        run_analysis()

    elif page == "Results":
        display_results()

    elif page == "User Comparison":
        compare_users()

if __name__ == "__main__":
    main()
