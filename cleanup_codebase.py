"""
Codebase Cleanup Script - Remove unnecessary files and organize the project
"""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodebaseCleanup:
    """Clean up unnecessary files from the codebase."""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.files_to_remove = []
        self.dirs_to_remove = []
        self.files_removed = 0
        self.space_saved = 0
        
    def identify_unnecessary_files(self):
        """Identify files and directories that can be safely removed."""
        
        # 1. Backup and duplicate files
        backup_patterns = [
            "*.bak", "*.backup", "*_backup*", "*_bak*",
            "*_old*", "*_copy*", "*_duplicate*"
        ]
        
        # 2. Cache and temporary files
        cache_patterns = [
            "__pycache__", "*.pyc", "*.pyo", "*.pyd",
            ".pytest_cache", ".coverage", "*.log",
            "cache/*", ".cache/*", "tmp/*", "temp/*"
        ]
        
        # 3. Development and debug files
        dev_patterns = [
            "debug_*", "test_*", "example_*", "sample_*",
            "*_test.py", "*_debug.py", "*_example.py"
        ]
        
        # 4. Redundant configuration files
        config_patterns = [
            "config_old*", "*_config_old*", "*.toml.bak"
        ]
        
        # 5. Template and sample data files (keep one set)
        template_patterns = [
            "template_*", "*_template*"
        ]
        
        # Scan for files to remove
        for pattern_list, description in [
            (backup_patterns, "Backup files"),
            (cache_patterns, "Cache files"),
            (dev_patterns, "Development files"),
            (config_patterns, "Old config files"),
        ]:
            self._find_files_by_patterns(pattern_list, description)
        
        # Handle specific directories
        self._identify_redundant_directories()
        
        # Handle duplicate files
        self._identify_duplicate_files()
        
    def _find_files_by_patterns(self, patterns, description):
        """Find files matching given patterns."""
        logger.info(f"Scanning for {description}...")
        
        for pattern in patterns:
            if "*" in pattern:
                # Use glob for pattern matching
                for file_path in self.project_root.rglob(pattern):
                    if file_path.is_file():
                        self.files_to_remove.append((file_path, description))
                    elif file_path.is_dir() and pattern.endswith("__pycache__"):
                        self.dirs_to_remove.append((file_path, description))
    
    def _identify_redundant_directories(self):
        """Identify redundant directories."""
        redundant_dirs = [
            "backup", "backup_files", "cleanup_backup",
            "__pycache__", ".pytest_cache"
        ]
        
        for dir_name in redundant_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.dirs_to_remove.append((dir_path, "Redundant directory"))
    
    def _identify_duplicate_files(self):
        """Identify duplicate files in the models directory."""
        models_dir = self.project_root / "src" / "models"
        if models_dir.exists():
            # Remove duplicate evaluator files (keep only evaluator.py)
            evaluator_files = [
                "evaluator_complete.py", "evaluator_complete2.py", 
                "evaluator_complete3.py", "evaluator_complete4.py",
                "evaluator_fixed.py", "evaluator_new.py", "evaluator.py.bak"
            ]
            
            for file_name in evaluator_files:
                file_path = models_dir / file_name
                if file_path.exists():
                    self.files_to_remove.append((file_path, "Duplicate evaluator file"))
    
    def show_cleanup_plan(self):
        """Show what will be removed."""
        logger.info("=== CLEANUP PLAN ===")
        
        if self.files_to_remove:
            logger.info(f"\nFiles to remove ({len(self.files_to_remove)}):")
            for file_path, reason in self.files_to_remove:
                size = file_path.stat().st_size if file_path.exists() else 0
                logger.info(f"  - {file_path.relative_to(self.project_root)} ({size} bytes) - {reason}")
        
        if self.dirs_to_remove:
            logger.info(f"\nDirectories to remove ({len(self.dirs_to_remove)}):")
            for dir_path, reason in self.dirs_to_remove:
                logger.info(f"  - {dir_path.relative_to(self.project_root)}/ - {reason}")
        
        # Calculate total space to be saved
        total_size = 0
        for file_path, _ in self.files_to_remove:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        for dir_path, _ in self.dirs_to_remove:
            if dir_path.exists():
                total_size += sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        
        logger.info(f"\nTotal space to be saved: {total_size / 1024:.2f} KB")
        
    def execute_cleanup(self, confirm=True):
        """Execute the cleanup plan."""
        if confirm:
            response = input("\nProceed with cleanup? (y/N): ")
            if response.lower() != 'y':
                logger.info("Cleanup cancelled.")
                return
        
        logger.info("Starting cleanup...")
        
        # Remove files
        for file_path, reason in self.files_to_remove:
            try:
                if file_path.exists():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    self.files_removed += 1
                    self.space_saved += size
                    logger.info(f"Removed: {file_path.relative_to(self.project_root)}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        
        # Remove directories
        for dir_path, reason in self.dirs_to_remove:
            try:
                if dir_path.exists():
                    # Calculate size before removal
                    dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    shutil.rmtree(dir_path)
                    self.space_saved += dir_size
                    logger.info(f"Removed directory: {dir_path.relative_to(self.project_root)}/")
            except Exception as e:
                logger.error(f"Error removing directory {dir_path}: {e}")
        
        logger.info(f"\nCleanup completed!")
        logger.info(f"Files removed: {self.files_removed}")
        logger.info(f"Space saved: {self.space_saved / 1024:.2f} KB")
    
    def create_gitignore(self):
        """Create or update .gitignore file."""
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Cache
cache/
.cache/
*.cache

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
*.temp

# Backup files
*.bak
*.backup
*_backup*
*_old*

# OS
.DS_Store
Thumbs.db

# Model files (large)
*.pkl
*.joblib
*.h5
*.pt
*.pth

# Data files (if large)
# data/
# *.csv
# *.json

# MLflow
mlruns/
mlflow.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Coverage
.coverage
htmlcov/

# Testing
.pytest_cache/
.tox/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        
        logger.info("Created/updated .gitignore file")
    
    def organize_project_structure(self):
        """Organize the project structure."""
        logger.info("Organizing project structure...")
        
        # Create essential directories if they don't exist
        essential_dirs = [
            "docs", "tests", "scripts", "notebooks", "models", "logs"
        ]
        
        for dir_name in essential_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {dir_name}/")
        
        # Move files to appropriate locations
        self._move_performance_files()
        self._move_documentation_files()
    
    def _move_performance_files(self):
        """Move performance-related files to scripts directory."""
        performance_files = [
            "boost_performance.py",
            "quick_performance_boost.py", 
            "performance_optimization_guide.py",
            "performance_improvements.py"
        ]
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        for file_name in performance_files:
            src_path = self.project_root / file_name
            if src_path.exists():
                dst_path = scripts_dir / file_name
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    logger.info(f"Moved {file_name} to scripts/")
    
    def _move_documentation_files(self):
        """Move documentation files to docs directory."""
        doc_files = [
            "architecture_diagram.md",
            "PERFORMANCE_OPTIMIZATION_GUIDE.md",
            "ENHANCED_README.md"
        ]
        
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        for file_name in doc_files:
            src_path = self.project_root / file_name
            if src_path.exists():
                dst_path = docs_dir / file_name
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    logger.info(f"Moved {file_name} to docs/")

def main():
    """Main cleanup function."""
    logger.info("🧹 Starting codebase cleanup...")
    
    cleanup = CodebaseCleanup()
    
    # Step 1: Identify unnecessary files
    cleanup.identify_unnecessary_files()
    
    # Step 2: Show cleanup plan
    cleanup.show_cleanup_plan()
    
    # Step 3: Execute cleanup
    cleanup.execute_cleanup(confirm=True)
    
    # Step 4: Create .gitignore
    cleanup.create_gitignore()
    
    # Step 5: Organize project structure
    cleanup.organize_project_structure()
    
    logger.info("🎉 Codebase cleanup completed!")
    
    # Show final project structure
    logger.info("\n📁 Final project structure:")
    for root, dirs, files in os.walk("."):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Show only first 5 files per directory
            if not file.startswith('.') and not file.endswith('.pyc'):
                logger.info(f"{subindent}{file}")
        if len(files) > 5:
            logger.info(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    main()
