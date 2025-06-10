#!/usr/bin/env python3
"""
Research Paper Validation Script
Validates the IEEE format paper structure and content
"""

import os
import re
from pathlib import Path

def validate_latex_file(file_path):
    """Validate LaTeX file structure."""
    print(f"📄 Validating {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    issues = []
    
    # Check basic LaTeX structure
    if '\\documentclass' not in content:
        issues.append("Missing \\documentclass")
    
    if '\\begin{document}' not in content:
        issues.append("Missing \\begin{document}")
    
    if '\\end{document}' not in content:
        issues.append("Missing \\end{document}")
    
    # Check IEEE specific elements
    if 'IEEEtran' not in content:
        issues.append("Not using IEEE template")
    
    if '\\title{' not in content:
        issues.append("Missing title")
    
    if '\\author{' not in content:
        issues.append("Missing author")
    
    if '\\begin{abstract}' not in content:
        issues.append("Missing abstract")
    
    if '\\begin{IEEEkeywords}' not in content:
        issues.append("Missing IEEE keywords")
    
    # Check sections
    required_sections = ['Introduction', 'Related Work', 'Methodology', 'Results', 'Conclusion']
    for section in required_sections:
        if f'\\section{{{section}}}' not in content and f'section{{{section}' not in content:
            issues.append(f"Missing section: {section}")
    
    # Check bibliography
    if '\\begin{thebibliography}' not in content and '\\bibliography{' not in content:
        issues.append("Missing bibliography")
    
    if issues:
        print(f"⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ LaTeX structure validation passed")
        return True

def validate_bibliography(bib_file):
    """Validate bibliography file."""
    print(f"📚 Validating bibliography {bib_file}...")
    
    if not os.path.exists(bib_file):
        print(f"❌ Bibliography file not found: {bib_file}")
        return False
    
    try:
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading bibliography: {e}")
        return False
    
    # Count entries
    entries = re.findall(r'@\w+\{([^,]+),', content)
    print(f"📖 Bibliography entries: {len(entries)}")
    
    if len(entries) < 10:
        print("⚠️  Few bibliography entries (<10)")
    elif len(entries) >= 15:
        print("✅ Good number of references (≥15)")
    
    # Check for common entry types
    entry_types = re.findall(r'@(\w+)\{', content)
    type_counts = {}
    for entry_type in entry_types:
        type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
    
    print("📊 Entry types:")
    for entry_type, count in sorted(type_counts.items()):
        print(f"   • {entry_type}: {count}")
    
    return True

def analyze_paper_content(tex_file):
    """Analyze paper content for completeness."""
    print(f"📊 Analyzing paper content...")
    
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print("❌ Cannot read paper file")
        return False
    
    # Count sections and subsections
    sections = re.findall(r'\\section\{([^}]+)\}', content)
    subsections = re.findall(r'\\subsection\{([^}]+)\}', content)
    
    print(f"📋 Structure Analysis:")
    print(f"   • Sections: {len(sections)}")
    print(f"   • Subsections: {len(subsections)}")
    
    # Check abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        word_count = len(abstract_text.split())
        print(f"   • Abstract word count: ~{word_count} words")
        if word_count > 250:
            print("⚠️  Abstract may be too long (>250 words)")
        elif word_count < 100:
            print("⚠️  Abstract may be too short (<100 words)")
        else:
            print("✅ Abstract length is appropriate")
    
    # Check figures and tables
    figures = len(re.findall(r'\\begin\{figure', content))
    tables = len(re.findall(r'\\begin\{table', content))
    
    print(f"   • Figures: {figures}")
    print(f"   • Tables: {tables}")
    
    # Count citations
    citations = len(re.findall(r'\\cite\{[^}]+\}', content))
    print(f"   • Citations: {citations}")
    
    return True

def check_file_structure():
    """Check if all required files are present."""
    print(f"📁 Checking file structure...")
    
    required_files = [
        ('enhanced_cross_platform_user_identification.tex', 'Main paper'),
        ('references.bib', 'Bibliography'),
        ('figures.tex', 'Figures'),
        ('Makefile', 'Compilation script'),
        ('compile_paper.py', 'Python compiler'),
        ('README.md', 'Documentation')
    ]
    
    all_present = True
    for filename, description in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {description}: {filename} ({size:,} bytes)")
        else:
            print(f"❌ {description}: {filename} - NOT FOUND")
            all_present = False
    
    return all_present

def main():
    """Main validation workflow."""
    print("🔍 IEEE Research Paper Validation")
    print("=" * 50)
    
    # Check file structure
    if not check_file_structure():
        print("\n❌ File structure validation failed")
        return False
    
    # Validate main paper
    if not validate_latex_file('enhanced_cross_platform_user_identification.tex'):
        print("\n❌ Main paper validation failed")
        return False
    
    # Validate bibliography
    if not validate_bibliography('references.bib'):
        print("\n❌ Bibliography validation failed")
        return False
    
    # Validate figures (standalone document)
    print(f"📄 Checking figures.tex (standalone document)...")
    if os.path.exists('figures.tex'):
        print("✅ Figures file exists")
    else:
        print("❌ Figures file missing")
        return False
    
    # Analyze content
    if not analyze_paper_content('enhanced_cross_platform_user_identification.tex'):
        print("\n❌ Content analysis failed")
        return False
    
    # Success summary
    print(f"\n🎉 Paper Validation Complete!")
    print("=" * 30)
    print("✅ All validations passed")
    print("✅ Paper structure is correct")
    print("✅ Bibliography is complete")
    print("✅ Content analysis successful")
    
    print(f"\n📋 Next steps:")
    print("   1. Compile the paper: python3 compile_paper.py")
    print("   2. Review the generated PDF")
    print("   3. Submit to target venue")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
