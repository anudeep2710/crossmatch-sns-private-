"""
Script to check if all required dependencies are installed.
"""

import importlib
import sys

def check_dependency(package_name, min_version=None):
    """
    Check if a package is installed and meets the minimum version requirement.
    
    Args:
        package_name (str): Name of the package to check
        min_version (str, optional): Minimum version required
        
    Returns:
        bool: True if package is installed and meets version requirement, False otherwise
    """
    try:
        package = importlib.import_module(package_name)
        print(f"✅ {package_name} is installed")
        
        if min_version and hasattr(package, '__version__'):
            version = package.__version__
            print(f"   Version: {version}")
            
            # Simple version comparison (not handling complex version strings)
            if version < min_version:
                print(f"   ⚠️ Warning: Version {version} is lower than required {min_version}")
                return False
        
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def main():
    """Check all dependencies."""
    print("Checking dependencies for Cross-Platform User Identification...\n")
    
    # Core dependencies
    core_dependencies = [
        ('numpy', '1.20.0'),
        ('pandas', '1.3.0'),
        ('networkx', '2.6.0'),
        ('scikit-learn', '1.0.0'),
        ('torch', '1.9.0'),
        ('transformers', '4.10.0'),
        ('sentence_transformers', '2.0.0'),
        ('matplotlib', '3.4.0'),
        ('seaborn', '0.11.0'),
        ('streamlit', '1.0.0'),
        ('nltk', '3.6.0'),
        ('yaml', None),
        ('tqdm', '4.62.0'),
        ('faker', '8.0.0'),
        ('node2vec', '0.4.0'),
        ('plotly', '5.0.0'),
        ('joblib', '1.1.0'),
        ('pytz', '2021.1')
    ]
    
    # Optional dependencies
    optional_dependencies = [
        ('karateclub', '1.0.0'),
        ('selenium', '4.0.0'),
        ('webdriver_manager', '3.5.0')
    ]
    
    # Check core dependencies
    print("Core dependencies:")
    core_status = [check_dependency(pkg, ver) for pkg, ver in core_dependencies]
    
    # Check optional dependencies
    print("\nOptional dependencies:")
    optional_status = [check_dependency(pkg, ver) for pkg, ver in optional_dependencies]
    
    # Summary
    print("\nSummary:")
    print(f"Core dependencies: {sum(core_status)}/{len(core_status)} installed")
    print(f"Optional dependencies: {sum(optional_status)}/{len(optional_status)} installed")
    
    if all(core_status):
        print("\n✅ All core dependencies are installed. You can run the application.")
    else:
        print("\n⚠️ Some core dependencies are missing. Please install them before running the application.")
        print("   Run: pip install -r requirements.txt")
    
    if not all(optional_status):
        print("\nℹ️ Some optional dependencies are missing. Some features may not work.")
        print("   For full functionality, install all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
