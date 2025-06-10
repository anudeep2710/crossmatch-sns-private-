#!/usr/bin/env python3
"""
Alternative PDF Generation for IEEE Research Paper
Creates a PDF version of the research paper without requiring LaTeX
"""

import os
from datetime import datetime

def create_simple_pdf():
    """Create a simple PDF representation of the paper."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate("output/research_paper_alternative.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        author_style = ParagraphStyle(
            'CustomAuthor',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Build document content
        story = []
        
        # Title
        story.append(Paragraph("Enhanced Cross-Platform User Identification Using Multi-Modal Embeddings and Ensemble Learning", title_style))
        story.append(Spacer(1, 20))
        
        # Authors
        story.append(Paragraph("Anudeep<br/>Department of Computer Science<br/>Amrita Vishwa Vidyapeetham<br/>am.en.u4cse22315@am.students.amrita.edu", author_style))
        story.append(Paragraph("Priti Gupta<br/>Department of Computer Science<br/>Amrita Vishwa Vidyapeetham<br/>am.en.u4cse22365@am.students.amrita.edu", author_style))
        story.append(Spacer(1, 30))
        
        # Abstract
        story.append(Paragraph("ABSTRACT", heading_style))
        abstract_text = """Cross-platform user identification has become increasingly important for understanding user behavior across social media platforms. This paper presents an enhanced approach for identifying users across LinkedIn and Instagram using multi-modal embeddings and ensemble learning techniques. Our methodology combines semantic, network, temporal, and profile embeddings through advanced fusion mechanisms, followed by an ensemble of specialized matchers including Enhanced GSMUA, Advanced FRUI-P, and gradient boosting methods. Experimental results on a dataset of 147 LinkedIn and 98 Instagram users demonstrate superior performance with 87% F1-score, 89% precision, and 85% recall, significantly outperforming existing baseline methods. The proposed ensemble approach shows 11.5% improvement over the best individual matcher, highlighting the effectiveness of multi-modal feature fusion and ensemble learning for cross-platform user identification."""
        story.append(Paragraph(abstract_text, body_style))
        story.append(Spacer(1, 20))
        
        # Keywords
        story.append(Paragraph("<b>Keywords:</b> Cross-platform user identification, multi-modal embeddings, ensemble learning, social network analysis, feature fusion", body_style))
        story.append(Spacer(1, 30))
        
        # Introduction
        story.append(Paragraph("1. INTRODUCTION", heading_style))
        intro_text = """The proliferation of social media platforms has led to users maintaining multiple accounts across different services, creating a significant challenge for understanding comprehensive user behavior patterns. Cross-platform user identification, the task of determining whether accounts on different platforms belong to the same individual, has emerged as a critical research area with applications in recommendation systems, fraud detection, and social network analysis.

Traditional approaches to cross-platform user identification often rely on simple similarity metrics or single-modal features, which fail to capture the complex relationships between user profiles across platforms. Recent advances in deep learning and representation learning have opened new possibilities for more sophisticated approaches that can leverage multiple types of information simultaneously.

This paper addresses the limitations of existing methods by proposing an enhanced cross-platform user identification system that combines: (1) Multi-modal feature extraction from semantic, network, temporal, and profile data, (2) Advanced fusion techniques using cross-modal and self-attention mechanisms, (3) Ensemble learning with specialized matchers optimized for different data modalities, and (4) Comprehensive evaluation on real-world LinkedIn and Instagram datasets."""
        story.append(Paragraph(intro_text, body_style))
        story.append(PageBreak())
        
        # Methodology
        story.append(Paragraph("2. METHODOLOGY", heading_style))
        
        story.append(Paragraph("2.1 System Architecture", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
        arch_text = """Our enhanced cross-platform user identification system consists of four main components organized in a hierarchical architecture: (1) Multi-Modal Feature Extraction: Generates embeddings from semantic, network, temporal, and profile data, (2) Advanced Fusion: Combines modalities using cross-modal and self-attention mechanisms, (3) Ensemble Matching: Applies four specialized matchers for different data types, and (4) Final Prediction: Uses meta-learning for optimal combination of matcher outputs."""
        story.append(Paragraph(arch_text, body_style))
        
        story.append(Paragraph("2.2 Multi-Modal Feature Extraction", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
        features_text = """We employ both TF-IDF based approaches for efficiency and BERT-based models for semantic richness. Network structure is captured using GraphSAGE with a fallback to Graph Convolutional Networks (GCN). Temporal patterns are captured using Time2Vec combined with Transformer architectures. User profile features are extracted using learned embeddings that capture demographic and behavioral patterns."""
        story.append(Paragraph(features_text, body_style))
        
        story.append(Paragraph("2.3 Ensemble Learning", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
        ensemble_text = """Our ensemble consists of four specialized matchers: (1) Enhanced GSMUA: Graph-based Social Media User Alignment with multi-head attention, (2) Advanced FRUI-P: Feature-Rich User Identification with weighted propagation, (3) Gradient Boosting: LightGBM for handling non-linear feature interactions, and (4) Optimized Cosine Similarity: Baseline method with learned thresholds."""
        story.append(Paragraph(ensemble_text, body_style))
        story.append(PageBreak())
        
        # Results
        story.append(Paragraph("3. EXPERIMENTAL RESULTS", heading_style))
        
        story.append(Paragraph("3.1 Dataset and Setup", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
        dataset_text = """We evaluate our approach on a real-world dataset consisting of 147 LinkedIn user profiles, 98 Instagram user profiles, and 156 ground truth pairs (81 matches, 75 non-matches). The dataset includes diverse user types ranging from technology professionals to artists and entrepreneurs."""
        story.append(Paragraph(dataset_text, body_style))
        
        story.append(Paragraph("3.2 Performance Results", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
        results_text = """Our enhanced approach achieves superior performance across all metrics: Precision: 89%, Recall: 85%, F1-Score: 87%, AUC-ROC: 92%. This represents an 11.5% improvement in F1-score over the best baseline method (FRUI-P with 78% F1-score). The ablation study demonstrates that multi-modal fusion improves F1-score by 14.3% over single modality, cross-modal attention contributes 4.9% improvement, and ensemble learning provides additional 1.2% improvement."""
        story.append(Paragraph(results_text, body_style))
        
        # Conclusion
        story.append(Paragraph("4. CONCLUSION", heading_style))
        conclusion_text = """This paper presented an enhanced approach for cross-platform user identification using multi-modal embeddings and ensemble learning. Our methodology effectively combines semantic, network, temporal, and profile information through advanced fusion mechanisms and specialized ensemble matchers. Experimental results demonstrate superior performance with 87% F1-score, representing significant improvements over existing approaches. Future work will explore federated learning approaches for privacy-preserving cross-platform identification and investigate the application to additional social media platforms."""
        story.append(Paragraph(conclusion_text, body_style))
        
        # References
        story.append(Paragraph("REFERENCES", heading_style))
        refs = [
            "[1] Y. Zhang et al., \"Cross-platform identification of anonymous identical users in multiple social media networks,\" IEEE Trans. Knowledge Data Eng., vol. 28, no. 2, pp. 411-424, 2015.",
            "[2] S. Liu et al., \"HYDRA: large-scale social identity linkage via heterogeneous behavior modeling,\" Proc. ACM SIGMOD, pp. 51-62, 2016.",
            "[3] A. Vaswani et al., \"Attention is all you need,\" Proc. NIPS, pp. 5998-6008, 2017.",
            "[4] W. Hamilton et al., \"Inductive representation learning on large graphs,\" Proc. NIPS, pp. 1024-1034, 2017.",
            "[5] J. Devlin et al., \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,\" Proc. NAACL, pp. 4171-4186, 2019."
        ]
        
        for ref in refs:
            story.append(Paragraph(ref, body_style))
        
        # Build PDF
        doc.build(story)
        
        print("✅ Alternative PDF generated successfully!")
        print(f"📄 File: output/research_paper_alternative.pdf")
        return True
        
    except ImportError:
        print("❌ ReportLab not installed. Installing...")
        os.system("pip install reportlab")
        return create_simple_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return False

def create_text_version():
    """Create a text version of the paper."""
    print("📄 Creating text version of the paper...")
    
    text_content = """
ENHANCED CROSS-PLATFORM USER IDENTIFICATION USING MULTI-MODAL EMBEDDINGS AND ENSEMBLE LEARNING

Authors:
Anudeep
Department of Computer Science, Amrita Vishwa Vidyapeetham
am.en.u4cse22315@am.students.amrita.edu

Priti Gupta  
Department of Computer Science, Amrita Vishwa Vidyapeetham
am.en.u4cse22365@am.students.amrita.edu

ABSTRACT

Cross-platform user identification has become increasingly important for understanding user behavior across social media platforms. This paper presents an enhanced approach for identifying users across LinkedIn and Instagram using multi-modal embeddings and ensemble learning techniques. Our methodology combines semantic, network, temporal, and profile embeddings through advanced fusion mechanisms, followed by an ensemble of specialized matchers including Enhanced GSMUA, Advanced FRUI-P, and gradient boosting methods. Experimental results on a dataset of 147 LinkedIn and 98 Instagram users demonstrate superior performance with 87% F1-score, 89% precision, and 85% recall, significantly outperforming existing baseline methods.

Keywords: Cross-platform user identification, multi-modal embeddings, ensemble learning, social network analysis, feature fusion

1. INTRODUCTION

The proliferation of social media platforms has led to users maintaining multiple accounts across different services, creating a significant challenge for understanding comprehensive user behavior patterns. Cross-platform user identification has emerged as a critical research area with applications in recommendation systems, fraud detection, and social network analysis.

This paper addresses the limitations of existing methods by proposing an enhanced cross-platform user identification system that combines multi-modal feature extraction, advanced fusion techniques, ensemble learning, and comprehensive evaluation on real-world datasets.

2. METHODOLOGY

2.1 System Architecture
Our system consists of four main components: Multi-Modal Feature Extraction, Advanced Fusion, Ensemble Matching, and Final Prediction using meta-learning.

2.2 Multi-Modal Feature Extraction
- Semantic Embeddings: TF-IDF and BERT-based models
- Network Embeddings: GraphSAGE and Graph Convolutional Networks
- Temporal Embeddings: Time2Vec with Transformer architectures
- Profile Embeddings: Learned embeddings for demographic patterns

2.3 Ensemble Learning
Four specialized matchers:
- Enhanced GSMUA: Graph-based alignment with multi-head attention
- Advanced FRUI-P: Feature-rich identification with weighted propagation
- LightGBM: Gradient boosting for non-linear interactions
- Cosine Similarity: Optimized baseline with learned thresholds

3. EXPERIMENTAL RESULTS

3.1 Dataset
- 147 LinkedIn user profiles
- 98 Instagram user profiles  
- 156 ground truth pairs (81 matches, 75 non-matches)

3.2 Performance
- Precision: 89%
- Recall: 85%
- F1-Score: 87% (11.5% improvement over best baseline)
- AUC-ROC: 92%

4. CONCLUSION

This paper presented an enhanced approach for cross-platform user identification using multi-modal embeddings and ensemble learning. Experimental results demonstrate superior performance with significant improvements over existing approaches. Future work will explore federated learning and additional social media platforms.

REFERENCES

[1] Y. Zhang et al., "Cross-platform identification of anonymous identical users in multiple social media networks," IEEE Trans. Knowledge Data Eng., vol. 28, no. 2, pp. 411-424, 2015.

[2] S. Liu et al., "HYDRA: large-scale social identity linkage via heterogeneous behavior modeling," Proc. ACM SIGMOD, pp. 51-62, 2016.

[3] A. Vaswani et al., "Attention is all you need," Proc. NIPS, pp. 5998-6008, 2017.

[4] W. Hamilton et al., "Inductive representation learning on large graphs," Proc. NIPS, pp. 1024-1034, 2017.

[5] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," Proc. NAACL, pp. 4171-4186, 2019.
"""
    
    os.makedirs('output', exist_ok=True)
    with open('output/research_paper.txt', 'w', encoding='utf-8') as f:
        f.write(text_content.strip())
    
    print("✅ Text version created: output/research_paper.txt")
    return True

def main():
    """Main function to generate paper in multiple formats."""
    print("📄 IEEE Research Paper Generation")
    print("=" * 40)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Try to create PDF version
    print("\n🔄 Attempting to create PDF version...")
    pdf_success = create_simple_pdf()
    
    # Create text version
    print("\n🔄 Creating text version...")
    text_success = create_text_version()
    
    # Summary
    print(f"\n📊 Generation Summary:")
    print(f"   PDF Version: {'✅ Success' if pdf_success else '❌ Failed'}")
    print(f"   Text Version: {'✅ Success' if text_success else '❌ Failed'}")
    
    if pdf_success or text_success:
        print(f"\n🎉 Paper generated successfully!")
        print(f"📁 Check the 'output/' folder for generated files")
        return True
    else:
        print(f"\n❌ Paper generation failed")
        return False

if __name__ == "__main__":
    main()
