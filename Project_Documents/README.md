# Cross-Platform User Identification

A Python project for identifying and matching users across different social media platforms using machine learning techniques.

## Project Structure

```
cross_platform_user_identification/
  ├── data/                  # Data storage
  │   ├── linkedin/          # LinkedIn data
  │   ├── instagram/         # Instagram data
  │   ├── processed/         # Processed data
  │   ├── raw/               # Raw data
  │   └── synthetic/         # Synthetic data for testing
  ├── src/                   # Source code
  │   ├── data/              # Data loading and processing modules
  │   ├── features/          # Feature extraction modules
  │   ├── models/            # Matching and identification models
  │   └── utils/             # Utility functions
  ├── output/                # Output files and visualizations
  ├── tests/                 # Unit tests
  ├── app.py                 # Streamlit web application
  ├── requirements.txt       # Project dependencies
  └── README.md              # Project documentation
```

## Features

- Data loading from multiple social media platforms
- Web scraping for LinkedIn and Instagram profiles (use responsibly and in compliance with terms of service)
- Preprocessing and normalization of user data
- Network-based user embeddings (Node2Vec, GCN)
- Semantic embeddings from user content (BERT)
- Temporal embeddings from user activity patterns
- Multi-modal embedding fusion
- User matching across platforms
- Evaluation metrics and visualization
- Web interface for analysis and visualization

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with the `stellargraph` package, it has been replaced with `karateclub` as an alternative for graph embeddings. The core functionality will work without it, but some advanced network embedding methods may not be available.

3. Check dependencies:
   ```
   python check_dependencies.py
   ```

4. Download NLTK resources:
   ```python
./run_app.sh

# Option 2: Running directly with Streamlit (recommended)
streamlit run app.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag is used to disable Streamlit's file watcher, which can cause issues with PyTorch.

## Web Scraping Disclaimer

This project includes functionality to scrape data from LinkedIn and Instagram. Please note:

1. Web scraping may violate the Terms of Service of these platforms
2. The scraping functionality is provided for educational purposes only
3. Use at your own risk and responsibility
4. Consider using official APIs when available
5. Respect rate limits and privacy of users
6. The developers of this project are not responsible for any misuse of this functionality

## Troubleshooting

### PyTorch Error Messages

If you see error messages related to PyTorch and `__path__._path`, these are warnings from Streamlit's file watcher and don't affect the functionality of the application. To eliminate these warnings, run the application with the file watcher disabled as shown above.

### Selenium WebDriver Issues

If you encounter issues with the LinkedIn or Instagram scrapers, make sure you have the latest version of Chrome and ChromeDriver installed. You can update ChromeDriver using:

```bash
pip install --upgrade webdriver-manager
```

### NoneType has no attribute 'rstrip' Error

If you encounter an error saying `'NoneType' object has no attribute 'rstrip'` when loading data, this is likely because you're trying to load data from a platform that doesn't exist yet. Make sure to scrape data from both LinkedIn and Instagram before running the analysis.

## License

MIT
