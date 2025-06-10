#!/usr/bin/env python3
"""
Research Paper Compilation Script
Compiles the IEEE format research paper with proper LaTeX processing
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_latex_installation():
    """Check if LaTeX is properly installed."""
    print("🔍 Checking LaTeX installation...")
    
    commands = ['pdflatex --version', 'bibtex --version']
    for cmd in commands:
        success, output = run_command(cmd)
        if success:
            print(f"✅ {cmd.split()[0]} found")
        else:
            print(f"❌ {cmd.split()[0]} not found")
            return False
    return True

def compile_paper():
    """Compile the research paper."""
    print("\n📄 Compiling IEEE Research Paper...")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    paper_name = "enhanced_cross_platform_user_identification"
    
    # First compilation
    print("🔄 First LaTeX compilation...")
    success, output = run_command(f'pdflatex -output-directory=output {paper_name}.tex')
    if not success:
        print(f"❌ First compilation failed")
        print(output)
        return False
    
    # Run bibtex
    print("📚 Processing bibliography...")
    success, output = run_command(f'bibtex output/{paper_name}')
    if not success:
        print(f"⚠️  BibTeX processing failed (this might be normal)")
        print(output)
    
    # Second compilation
    print("🔄 Second LaTeX compilation...")
    success, output = run_command(f'pdflatex -output-directory=output {paper_name}.tex')
    if not success:
        print(f"❌ Second compilation failed")
        print(output)
        return False
    
    # Third compilation
    print("🔄 Final LaTeX compilation...")
    success, output = run_command(f'pdflatex -output-directory=output {paper_name}.tex')
    if not success:
        print(f"❌ Final compilation failed")
        print(output)
        return False
    
    print("✅ Research paper compiled successfully!")
    return True

def compile_figures():
    """Compile the figures document."""
    print("\n🎨 Compiling figures...")
    
    success, output = run_command('pdflatex -output-directory=output figures.tex')
    if not success:
        print(f"❌ Figures compilation failed")
        print(output)
        return False
    
    print("✅ Figures compiled successfully!")
    return True

def validate_output():
    """Validate the generated PDF files."""
    print("\n🔍 Validating output files...")
    
    paper_pdf = "output/enhanced_cross_platform_user_identification.pdf"
    figures_pdf = "output/figures.pdf"
    
    files_to_check = [
        (paper_pdf, "Research Paper"),
        (figures_pdf, "Figures")
    ]
    
    all_valid = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 1000:  # At least 1KB
                print(f"✅ {description}: {file_path} ({size:,} bytes)")
            else:
                print(f"❌ {description}: File too small ({size} bytes)")
                all_valid = False
        else:
            print(f"❌ {description}: File not found - {file_path}")
            all_valid = False
    
    return all_valid

def create_submission_package():
    """Create a submission package with all necessary files."""
    print("\n📦 Creating submission package...")
    
    # Create submission directory
    os.makedirs('submission', exist_ok=True)
    
    # Files to include in submission
    files_to_copy = [
        ('enhanced_cross_platform_user_identification.tex', 'Main paper source'),
        ('references.bib', 'Bibliography'),
        ('figures.tex', 'Figures source'),
        ('output/enhanced_cross_platform_user_identification.pdf', 'Main paper PDF'),
        ('output/figures.pdf', 'Figures PDF'),
        ('Makefile', 'Compilation instructions')
    ]
    
    for source_file, description in files_to_copy:
        if os.path.exists(source_file):
            dest_file = f"submission/{os.path.basename(source_file)}"
            success, _ = run_command(f'cp "{source_file}" "{dest_file}"')
            if success:
                print(f"✅ Copied {description}")
            else:
                print(f"❌ Failed to copy {description}")
        else:
            print(f"⚠️  {description} not found: {source_file}")
    
    # Create archive
    success, _ = run_command('tar -czf research_paper_submission.tar.gz submission/')
    if success:
        print("✅ Submission package created: research_paper_submission.tar.gz")
        return True
    else:
        print("❌ Failed to create submission package")
        return False

def main():
    """Main compilation workflow."""
    print("🚀 IEEE Research Paper Compilation")
    print("=" * 50)
    
    # Check LaTeX installation
    if not check_latex_installation():
        print("\n❌ LaTeX installation incomplete.")
        print("💡 Please install LaTeX (TeX Live, MiKTeX, or MacTeX)")
        sys.exit(1)
    
    # Compile paper
    if not compile_paper():
        print("\n❌ Paper compilation failed")
        sys.exit(1)
    
    # Compile figures
    if not compile_figures():
        print("\n❌ Figures compilation failed")
        sys.exit(1)
    
    # Validate output
    if not validate_output():
        print("\n❌ Output validation failed")
        sys.exit(1)
    
    # Create submission package
    if not create_submission_package():
        print("\n❌ Submission package creation failed")
        sys.exit(1)
    
    # Success summary
    print(f"\n🎉 Research Paper Compilation Complete!")
    print("=" * 50)
    print("📄 Generated files:")
    print("   • output/enhanced_cross_platform_user_identification.pdf")
    print("   • output/figures.pdf")
    print("   • research_paper_submission.tar.gz")
    print("\n📋 Next steps:")
    print("   1. Review the generated PDF")
    print("   2. Check figures and formatting")
    print("   3. Submit to IEEE conference/journal")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Compilation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
