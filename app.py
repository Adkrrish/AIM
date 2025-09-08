"""
Instagram Competitor Analysis Tool
Enhanced with Apify API integration for reliable scraping
"""

import streamlit as st
import pandas as pd
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import zipfile
from PIL import Image

# Import our utilities
from utils import InstagramAnalyzer, call_groq_model, validate_post_data
from utils import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT
from utils import SUGGESTION_SYSTEM_PROMPT, SUGGESTION_USER_PROMPT  
from utils import CAMPAIGN_PROMPT_SYSTEM, CAMPAIGN_PROMPT_USER

# Page configuration
st.set_page_config(
    page_title="Instagram Competitor Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (same as before)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f1f1f;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}
.color-box {
    width: 30px;
    height: 30px;
    border-radius: 4px;
    display: inline-block;
    margin: 2px;
    border: 1px solid #ddd;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.debug-info {
    background: #e8f4fd;
    border: 1px solid #bee5eb;
    padding: 0.5rem;
    border-radius: 4px;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.apify-notice {
    background: #e7f3ff;
    border: 1px solid #b3d9ff;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'strategies' not in st.session_state:
    st.session_state.strategies = {}
if 'campaign_prompts' not in st.session_state:
    st.session_state.campaign_prompts = {}
if 'failed_posts' not in st.session_state:
    st.session_state.failed_posts = []
if 'manual_uploads' not in st.session_state:
    st.session_state.manual_uploads = {}

def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from secrets or environment"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY")

def get_apify_token() -> Optional[str]:
    """Get Apify token from secrets or environment"""
    try:
        return st.secrets["APIFY_TOKEN"]
    except:
        return os.getenv("APIFY_TOKEN")

def create_example_excel() -> BytesIO:
    """Generate example Excel file for competitor data"""
    data = {
        'competitor_name': ['Competitor A', 'Competitor A', 'Competitor A', 
                           'Competitor B', 'Competitor B', 'Competitor B',
                           'Competitor C', 'Competitor C', 'Competitor C'],
        'instagram_id': ['@competitora', '@competitora', '@competitora',
                        '@competitorb', '@competitorb', '@competitorb', 
                        '@competitorc', '@competitorc', '@competitorc'],
        'post_url': [
            'https://www.instagram.com/p/EXAMPLE1/',
            'https://www.instagram.com/p/EXAMPLE2/', 
            'https://www.instagram.com/p/EXAMPLE3/',
            'https://www.instagram.com/p/EXAMPLE4/',
            'https://www.instagram.com/p/EXAMPLE5/',
            'https://www.instagram.com/p/EXAMPLE6/',
            'https://www.instagram.com/p/EXAMPLE7/',
            'https://www.instagram.com/p/EXAMPLE8/',
            'https://www.instagram.com/p/EXAMPLE9/'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Competitors', index=False)
    output.seek(0)
    
    return output

def render_color_palette(colors: List[str]) -> str:
    """Render color palette as HTML"""
    html = "<div style='display: flex; gap: 5px; margin: 10px 0;'>"
    for color in colors[:5]:  # Show max 5 colors
        html += f"<div class='color-box' style='background-color: {color};' title='{color}'></div>"
    html += "</div>"
    return html

def analyze_single_post(analyzer: InstagramAnalyzer, competitor_name: str, 
                       instagram_id: str, post_url: str, groq_api_key: str) -> Dict[str, Any]:
    """Analyze a single Instagram post using Apify or fallback methods"""
    
    # Create status container for this post
    status_container = st.empty()
    
    with status_container.container():
        st.markdown(f'<div class="debug-info">🔍 Analyzing: {post_url}</div>', unsafe_allow_html=True)
    
    # Fetch post data using the enhanced analyzer
    post_data = analyzer.fetch_instagram_post(post_url)
    
    if post_data.get("errors"):
        with status_container.container():
            st.markdown(f'<div class="debug-info">❌ Fetch failed: {post_data["errors"]}</div>', unsafe_allow_html=True)
        
        return {
            "competitor_name": competitor_name,
            "instagram_id": instagram_id, 
            "post_url": post_url,
            "errors": post_data["errors"],
            "analysis": {},
            "needs_manual_upload": True
        }
    
    # Initialize variables with defaults
    visual_emotions = {"visual_emotions": ["neutral"], "brightness": 128, "score": 50}
    colors = ["#000000"]
    visual_description = "No image available"
    image_analysis_success = False
    
    # Try to download and analyze image
    if post_data.get("image_url"):
        with status_container.container():
            st.markdown(f'<div class="debug-info">🖼️ Found image URL: {post_data["image_url"][:50]}...</div>', unsafe_allow_html=True)
        
        image_bytes = analyzer.download_image(post_data["image_url"])
        if image_bytes:
            image = analyzer.image_to_pil(image_bytes)
            if image:
                try:
                    colors = analyzer.extract_colors(image)
                    visual_emotions = analyzer.analyze_visual_emotions(image)
                    visual_description = f"Image with dominant colors: {', '.join(colors[:3])}"
                    image_analysis_success = True
                    
                    with status_container.container():
                        st.markdown(f'<div class="debug-info">✅ Image analysis successful - Colors: {", ".join(colors[:3])}</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    with status_container.container():
                        st.markdown(f'<div class="debug-info">⚠️ Image processing failed: {str(e)}</div>', unsafe_allow_html=True)
            else:
                with status_container.container():
                    st.markdown(f'<div class="debug-info">⚠️ Failed to convert image to PIL format</div>', unsafe_allow_html=True)
        else:
            with status_container.container():
                st.markdown(f'<div class="debug-info">⚠️ Failed to download image</div>', unsafe_allow_html=True)
    else:
        with status_container.container():
            st.markdown(f'<div class="debug-info">⚠️ No image URL found in post data</div>', unsafe_allow_html=True)
    
    # Analyze caption using rules
    caption_analysis = analyzer.analyze_caption_rules(post_data.get("caption") or "")
    readability = analyzer.compute_readability(post_data.get("caption") or "")
    
    # Show caption analysis results
    if post_data.get("caption"):
        with status_container.container():
            st.markdown(f'<div class="debug-info">📝 Caption found: {len(post_data["caption"])} characters, {len(caption_analysis.get("hashtags", []))} hashtags</div>', unsafe_allow_html=True)
    
    # Generate image hash
    image_hash = hashlib.md5(post_data.get("image_url", "").encode()).hexdigest()
    
    # Use LLM for advanced analysis
    llm_analysis = {}
    if groq_api_key and post_data.get("caption"):
        with status_container.container():
            st.markdown(f'<div class="debug-info">🤖 Running LLM analysis...</div>', unsafe_allow_html=True)
            
        prompt = ANALYSIS_USER_PROMPT.format(
            caption=post_data["caption"][:500],
            visual_description=visual_description,
            colors=", ".join(colors[:3])
        )
        
        llm_response = call_groq_model(
            prompt=prompt,
            system=ANALYSIS_SYSTEM_PROMPT,
            groq_api_key=groq_api_key
        )
        
        if llm_response.get("content") and not llm_response.get("error"):
            try:
                llm_analysis = json.loads(llm_response["content"])
                with status_container.container():
                    st.markdown(f'<div class="debug-info">✅ LLM analysis completed</div>', unsafe_allow_html=True)
            except json.JSONDecodeError as e:
                with status_container.container():
                    st.markdown(f'<div class="debug-info">⚠️ LLM response parsing failed: {str(e)}</div>', unsafe_allow_html=True)
        else:
            with status_container.container():
                st.markdown(f'<div class="debug-info">⚠️ LLM analysis failed: {llm_response.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    
    # Combine rule-based and LLM analysis
    analysis = {
        "color_palette": {
            "dominant_hex": colors,
            "tone": llm_analysis.get("color_palette", {}).get("tone", "neutral"),
            "style": llm_analysis.get("color_palette", {}).get("style", "unknown"),
            "raw_score": llm_analysis.get("color_palette", {}).get("raw_score", 50),
            "evidence": f"Extracted {len(colors)} dominant colors from image" if image_analysis_success else "Image analysis failed - using defaults"
        },
        "tone_of_voice": {
            "label": llm_analysis.get("tone_of_voice", {}).get("label", "neutral"),
            "polarity": llm_analysis.get("tone_of_voice", {}).get("polarity", "neutral"),
            "intensity": llm_analysis.get("tone_of_voice", {}).get("intensity", 5),
            "evidence": llm_analysis.get("tone_of_voice", {}).get("evidence", []),
            "raw_score": llm_analysis.get("tone_of_voice", {}).get("raw_score", 50)
        },
        "cta": {
            "presence": "strong" if caption_analysis.get("cta_detected") else "none",
            "text": caption_analysis.get("cta_text"),
            "strength": llm_analysis.get("cta", {}).get("strength", "none"),
            "score": 80 if caption_analysis.get("cta_detected") else 20
        },
        "hashtags_keywords": {
            "hashtags": caption_analysis.get("hashtags", []),
            "top_keywords": caption_analysis.get("top_keywords", []),
            "recommendation": llm_analysis.get("hashtags_keywords", {}).get("recommendation", 
                                            "Add more relevant hashtags")
        },
        "readability": {
            "word_count": readability.get("word_count", 0),
            "skimmable": readability.get("skimmable", False),
            "score": readability.get("score", 0),
            "evidence": readability.get("evidence", "No analysis available")
        },
        "emotional_imagery": {
            "visual_emotions": llm_analysis.get("emotional_imagery", {}).get("visual_emotions", 
                                             visual_emotions.get("visual_emotions", ["neutral"])),
            "text_emotion": llm_analysis.get("emotional_imagery", {}).get("text_emotion", "neutral"),
            "alignment_score": llm_analysis.get("emotional_imagery", {}).get("alignment_score", 50),
            "score": llm_analysis.get("emotional_imagery", {}).get("score", 50)
        }
    }
    
    result = {
        "competitor_name": competitor_name,
        "instagram_id": instagram_id,
        "post_url": post_url,
        "timestamp": post_data.get("timestamp"),
        "caption_text": post_data.get("caption"),
        "image_url": post_data.get("image_url"),
        "image_hash": image_hash,
        "analysis": analysis,
        "errors": None if image_analysis_success else "Image scraping/processing failed",
        "needs_manual_upload": not image_analysis_success,
        "scraped_with_apify": hasattr(post_data, 'apify_data')
    }
    
    # Clear status for final result
    with status_container.container():
        if image_analysis_success:
            st.markdown(f'<div class="debug-info">✅ Analysis completed successfully</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="debug-info">⚠️ Analysis completed with image processing issues</div>', unsafe_allow_html=True)
    
    return result

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📱 Instagram Competitor Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze competitor Instagram posts using Apify API and AI-powered insights</p>', 
                unsafe_allow_html=True)
    
    # Apify notice
    st.markdown("""
    <div class="apify-notice">
    <strong>🚀 Enhanced with Apify Integration:</strong> This tool now uses Apify's professional Instagram scrapers for reliable data extraction.
    Apify provides stable, anti-blocking technology that significantly improves scraping success rates.
    <br><br>
    <strong>💰 Pricing:</strong> Apify charges approximately $2.70 per 1,000 Instagram posts scraped. 
    New users get $5 in free credits monthly, allowing you to test the integration before upgrading.
    </div>
    """, unsafe_allow_html=True)
    
    # Legal notice
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Legal Notice:</strong> This tool is for educational and research purposes. 
    Please ensure compliance with Instagram's Terms of Service and applicable data protection laws. 
    The tool respects rate limits and uses ethical scraping practices through Apify's infrastructure.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("API Configuration")
    
    # API Key setup
    groq_api_key = get_groq_api_key()
    apify_token = get_apify_token()
    
    if not groq_api_key:
        st.sidebar.error("🔑 GROQ API Key not found!")
        st.sidebar.info("Set GROQ_API_KEY in Streamlit secrets or environment variables")
        return
    else:
        st.sidebar.success("✅ Groq API Key configured")
    
    if not apify_token:
        st.sidebar.warning("⚠️ Apify Token not found!")
        st.sidebar.info("Set APIFY_TOKEN in Streamlit secrets for enhanced scraping")
        st.sidebar.markdown("Without Apify token, the tool will use fallback scraping methods with lower success rates.")
    else:
        st.sidebar.success("✅ Apify Token configured")
    
    # File uploads
    st.sidebar.subheader("📁 Upload Files")
    
    # Company metadata upload
    company_file = st.sidebar.file_uploader(
        "Company Metadata (JSON/CSV/Excel)", 
        type=['json', 'csv', 'xlsx'],
        help="Upload your company information and brand guidelines"
    )
    
    # Competitor data upload
    competitor_file = st.sidebar.file_uploader(
        "Competitor Data (Excel)", 
        type=['xlsx'],
        help="Upload Excel file with competitor Instagram post URLs"
    )
    
    # Download example template
    if st.sidebar.button("📥 Download Example Template"):
        example_file = create_example_excel()
        st.sidebar.download_button(
            label="
