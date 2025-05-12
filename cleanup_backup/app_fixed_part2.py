def run_analysis():
    """Run analysis on loaded data."""
    st.header("Analysis")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to perform cross-platform analysis.")
        return

    with st.form("analysis_form"):
        st.subheader("Configure Analysis")

        # Get platform names
        platform_names = list(st.session_state.identifier.data.keys())
        if len(platform_names) < 2:
            st.error("Both LinkedIn and Instagram data are required for analysis.")
            st.info("Please go to the Data Loading tab and scrape data from both platforms.")
            return

        # Set platform names
        linkedin_exists = "linkedin" in platform_names
        instagram_exists = "instagram" in platform_names
        
        if linkedin_exists and instagram_exists:
            platform1_name = "linkedin"
            platform2_name = "instagram"
            st.success("Found data for both LinkedIn and Instagram.")
        else:
            missing_platforms = []
            if not linkedin_exists:
                missing_platforms.append("LinkedIn")
            if not instagram_exists:
                missing_platforms.append("Instagram")
            
            st.error(f"Missing data for {', '.join(missing_platforms)}.")
            st.info("Please go to the Data Loading tab and scrape data from the missing platform(s).")
            
            # Use available platforms for now
            platform1_name = platform_names[0]
            platform2_name = platform_names[1] if len(platform_names) > 1 else platform_names[0]

        # Feature extraction parameters
        st.subheader("Feature Extraction")

        network_method = st.selectbox("Network Embedding Method",
                                    ["node2vec"],
                                    index=0,
                                    help="node2vec is used to generate network embeddings from user connections.")
        
        semantic_model = st.selectbox("Semantic Model", [
            "sentence-transformers/all-MiniLM-L6-v2"
        ], index=0, 
        help="This model is used to generate semantic embeddings from user posts and profile text.")

        # Matching parameters
        st.subheader("User Matching")

        matching_method = st.selectbox("Matching Method", ["cosine"], index=0,
                                     help="Cosine similarity is used to measure the similarity between user embeddings.")
        
        matching_threshold = st.slider("Matching Threshold", 0.1, 0.9, 0.7, 0.05,
                                     help="Users with similarity above this threshold are considered matches.")

        # Run analysis
        run_button = st.form_submit_button("Run Analysis")

        if run_button:
            with st.spinner("Running analysis..."):
                # Update configuration
                config = {
                    'network_method': network_method,
                    'semantic_model_name': semantic_model,
                    'matching_method': matching_method,
                    'matching_threshold': matching_threshold
                }

                # Update identifier configuration
                st.session_state.identifier.config.update(config)

                # Preprocess data
                st.session_state.identifier.preprocess()

                # Extract features
                st.session_state.identifier.extract_features()
                st.session_state.embeddings_generated = True

                # Match users
                matches = st.session_state.identifier.match_users(
                    platform1_name=platform1_name,
                    platform2_name=platform2_name,
                    embedding_type='fusion'
                )
                st.session_state.matches = matches

                # Evaluate if ground truth is available
                if hasattr(st.session_state.identifier.data_loader, 'ground_truth'):
                    metrics = st.session_state.identifier.evaluate()
                    st.session_state.metrics = metrics

                st.success("Analysis completed successfully")

def display_results():
    """Display LinkedIn-Instagram matching results."""
    st.header("LinkedIn-Instagram Matching Results")

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Loading tab.")
        st.info("You need to scrape both LinkedIn and Instagram data to view matching results.")
        return

    if not st.session_state.embeddings_generated:
        st.warning("Please run analysis first on the Analysis tab.")
        st.info("This will generate the embeddings and matches needed to view results.")
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
    
    st.subheader("Cross-Platform User Matches")
    st.write("These are the users identified as potentially being the same person across LinkedIn and Instagram:")
    st.write("Higher confidence scores indicate a stronger likelihood of being the same person.")

    # Display matches
    if st.session_state.matches is not None:
        st.subheader("User Matches")
        st.dataframe(st.session_state.matches)

        # Download matches as CSV
        csv = st.session_state.matches.to_csv(index=False)
        st.download_button(
            label="Download Matches as CSV",
            data=csv,
            file_name="matches.csv",
            mime="text/csv"
        )

    # Display evaluation metrics
    if st.session_state.metrics is not None:
        st.subheader("Evaluation Metrics")

        match_key = f"{platform1_name}_{platform2_name}_fusion"
        if match_key in st.session_state.metrics:
            metrics = st.session_state.metrics[match_key]

            # Create metrics display
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{metrics['precision']:.4f}")
            col2.metric("Recall", f"{metrics['recall']:.4f}")
            col3.metric("F1 Score", f"{metrics['f1']:.4f}")

            # Display confusion matrix
            if 'confusion_matrix' in st.session_state.identifier.evaluator.metrics:
                st.subheader("Confusion Matrix")
                cm = np.array(st.session_state.identifier.evaluator.metrics['confusion_matrix']['matrix'])

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    colorscale='Blues',
                    showscale=False
                ))

                fig.update_layout(
                    title=f"Confusion Matrix (Threshold = {metrics['best_threshold']:.2f})",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )

                st.plotly_chart(fig)
