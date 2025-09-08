"""
Instagram Competitor Analysis Tool
A Streamlit app for analyzing competitor Instagram posts and generating campaign strategies
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

# Custom CSS for better styling
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
.metric-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
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
.fallback-section {
    background: #f8f9fa;
    border: 2px dashed #dee2e6;
    padding: 1.5rem;
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
    """Analyze a single Instagram post with robust error handling"""
    
    # Create status container for this post
    status_container = st.empty()
    
    with status_container.container():
        st.markdown(f'<div class="debug-info">🔍 Analyzing: {post_url}</div>', unsafe_allow_html=True)
    
    # Fetch post data
    post_data = analyzer.fetch_instagram_post(post_url)
    
    if post_data["errors"]:
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
    
    # Initialize ALL variables with default values at the start
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
            caption=post_data["caption"][:500],  # Truncate for API limits
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
    
    # Combine rule-based and LLM analysis with safe defaults
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
        "needs_manual_upload": not image_analysis_success
    }
    
    # Clear status for final result
    with status_container.container():
        if image_analysis_success:
            st.markdown(f'<div class="debug-info">✅ Analysis completed successfully</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="debug-info">⚠️ Analysis completed with image processing issues</div>', unsafe_allow_html=True)
    
    return result

def process_manual_upload(post_data: Dict, uploaded_image: Any, manual_caption: str, groq_api_key: str) -> Dict[str, Any]:
    """Process manually uploaded image and caption"""
    
    try:
        # Initialize analyzer
        analyzer = InstagramAnalyzer(groq_api_key)
        
        # Process uploaded image
        image = Image.open(uploaded_image)
        colors = analyzer.extract_colors(image)
        visual_emotions = analyzer.analyze_visual_emotions(image)
        visual_description = f"Manually uploaded image with dominant colors: {', '.join(colors[:3])}"
        
        # Use manual caption or original caption
        caption_to_analyze = manual_caption if manual_caption.strip() else post_data.get("caption_text", "")
        
        # Analyze caption using rules
        caption_analysis = analyzer.analyze_caption_rules(caption_to_analyze)
        readability = analyzer.compute_readability(caption_to_analyze)
        
        # Use LLM for advanced analysis
        llm_analysis = {}
        if groq_api_key and caption_to_analyze:
            prompt = ANALYSIS_USER_PROMPT.format(
                caption=caption_to_analyze[:500],
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
                except json.JSONDecodeError:
                    pass
        
        # Update analysis with manual data
        analysis = {
            "color_palette": {
                "dominant_hex": colors,
                "tone": llm_analysis.get("color_palette", {}).get("tone", "neutral"),
                "style": llm_analysis.get("color_palette", {}).get("style", "unknown"),
                "raw_score": llm_analysis.get("color_palette", {}).get("raw_score", 75),
                "evidence": f"Manually uploaded image - extracted {len(colors)} dominant colors"
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
                "evidence": readability.get("evidence", "Manual caption analysis")
            },
            "emotional_imagery": {
                "visual_emotions": llm_analysis.get("emotional_imagery", {}).get("visual_emotions", 
                                                 visual_emotions.get("visual_emotions", ["neutral"])),
                "text_emotion": llm_analysis.get("emotional_imagery", {}).get("text_emotion", "neutral"),
                "alignment_score": llm_analysis.get("emotional_imagery", {}).get("alignment_score", 75),
                "score": llm_analysis.get("emotional_imagery", {}).get("score", 75)
            }
        }
        
        # Update post data
        updated_post = post_data.copy()
        updated_post.update({
            "caption_text": caption_to_analyze,
            "analysis": analysis,
            "errors": None,
            "needs_manual_upload": False,
            "manually_processed": True
        })
        
        return updated_post
        
    except Exception as e:
        st.error(f"Error processing manual upload: {str(e)}")
        return post_data

def display_manual_upload_interface():
    """Display manual upload interface for failed posts"""
    
    if not st.session_state.analysis_results:
        return
    
    # Collect failed posts
    failed_posts = []
    for competitor, posts in st.session_state.analysis_results.items():
        for post in posts:
            if post.get('needs_manual_upload') or post.get('errors'):
                failed_posts.append(post)
    
    st.session_state.failed_posts = failed_posts
    
    if not failed_posts:
        return
    
    st.markdown(f"""
    <div class="warning-box">
    <strong>⚠️ Manual Upload Required:</strong> {len(failed_posts)} posts failed automatic analysis. 
    Please upload images and captions manually below to complete the analysis.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📁 Manual Upload Interface", expanded=True):
        st.markdown('<div class="fallback-section">', unsafe_allow_html=True)
        
        groq_api_key = get_groq_api_key()
        
        for idx, post in enumerate(failed_posts):
            st.markdown(f"### Post {idx + 1}: {post['competitor_name']}")
            st.markdown(f"**URL:** {post['post_url']}")
            st.markdown(f"**Error:** {post.get('errors', 'Image processing failed')}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🖼️ Upload Image")
                uploaded_image = st.file_uploader(
                    f"Upload image for post {idx+1}",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    key=f"manual_upload_image_{idx}",
                    help="Upload the Instagram post image manually"
                )
                
                if uploaded_image:
                    st.image(uploaded_image, width=200, caption="Uploaded Image Preview")
            
            with col2:
                st.markdown("#### ✍️ Caption")
                manual_caption = st.text_area(
                    f"Enter caption for post {idx+1}",
                    value=post.get('caption_text', ''),
                    height=150,
                    key=f"manual_caption_{idx}",
                    help="Enter the Instagram post caption manually"
                )
                
                st.markdown("#### 📊 Status")
                if post.get('manually_processed'):
                    st.success("✅ Manually processed")
                else:
                    st.info("⏳ Waiting for manual input")
            
            # Process button
            col3, col4 = st.columns([1, 3])
            with col3:
                if st.button(f"🔄 Process Post {idx+1}", key=f"process_manual_{idx}"):
                    if uploaded_image:
                        with st.spinner(f"Processing post {idx+1}..."):
                            # Process the manual upload
                            updated_post = process_manual_upload(
                                post, uploaded_image, manual_caption, groq_api_key
                            )
                            
                            # Update in session state
                            for competitor, posts in st.session_state.analysis_results.items():
                                for i, p in enumerate(posts):
                                    if p['post_url'] == post['post_url']:
                                        st.session_state.analysis_results[competitor][i] = updated_post
                                        break
                            
                            st.success(f"✅ Post {idx+1} processed successfully!")
                            st.rerun()
                    else:
                        st.error("Please upload an image first!")
            
            with col4:
                if post.get('manually_processed'):
                    st.info("This post has been manually processed and updated in the results.")
            
            st.divider()
        
        # Bulk process button
        st.markdown("### 🚀 Bulk Actions")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Refresh Analysis Results", type="primary"):
                st.rerun()
        
        with col2:
            processed_count = sum(1 for post in failed_posts if post.get('manually_processed'))
            st.metric("Manually Processed", f"{processed_count}/{len(failed_posts)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📱 Instagram Competitor Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze competitor Instagram posts and generate strategic campaign recommendations</p>', 
                unsafe_allow_html=True)
    
    # Legal notice
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Legal Notice:</strong> This tool is for educational and research purposes. 
    Please ensure compliance with Instagram's Terms of Service and applicable data protection laws. 
    Respect rate limits and consider using official Instagram APIs for production use.
    <br><br>
    <strong>Note:</strong> Instagram actively blocks automated scraping. Some posts may fail to analyze due to anti-bot measures.
    You can manually upload images and captions for failed posts using the fallback interface.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key setup
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        st.sidebar.error("🔑 GROQ API Key not found!")
        st.sidebar.info("Set GROQ_API_KEY in Streamlit secrets or environment variables")
        return
    else:
        st.sidebar.success("🔑 API Key configured")
    
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
            label="💾 Download competitors_template.xlsx",
            data=example_file,
            file_name="competitors_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Analysis", "📁 Manual Upload", "🏆 Competitors", "💡 Strategies", "🎨 Campaigns", "💾 Export"])
    
    with tab1:
        st.header("Instagram Post Analysis")
        
        if competitor_file is not None:
            # Load competitor data
            try:
                df = pd.read_excel(competitor_file)
                required_columns = ['competitor_name', 'instagram_id', 'post_url']
                
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ Missing required columns: {required_columns}")
                    return
                
                # Validate data structure (3 competitors × 3 posts)
                competitor_counts = df['competitor_name'].value_counts()
                if len(competitor_counts) != 3:
                    st.warning(f"⚠️ Expected 3 competitors, found {len(competitor_counts)}")
                
                for competitor, count in competitor_counts.items():
                    if count != 3:
                        st.warning(f"⚠️ Competitor '{competitor}' has {count} posts (expected 3)")
                
                st.success(f"✅ Loaded {len(df)} posts from {len(competitor_counts)} competitors")
                
                # Show data preview
                with st.expander("📋 Data Preview", expanded=False):
                    st.dataframe(df, use_container_width=True)
                
                # Analysis controls
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Start Analysis", type="primary"):
                        analyze_posts(df, groq_api_key)
                
                with col2:
                    if st.button("🔄 Clear Results"):
                        st.session_state.analysis_results = {}
                        st.session_state.failed_posts = []
                        st.rerun()
                
                # Show analysis results
                if st.session_state.analysis_results:
                    display_analysis_results()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.info("👆 Please upload competitor data file to begin analysis")
            
            # Show instructions
            st.markdown("""
            ### 📋 Instructions
            
            1. **Upload competitor data**: Excel file with columns:
               - `competitor_name`: Name of the competitor
               - `instagram_id`: Instagram handle (@username)
               - `post_url`: Full Instagram post URL
            
            2. **Data structure**: 3 competitors × 3 posts each (9 total rows)
            
            3. **Optional**: Upload company metadata (brand guidelines, colors, etc.)
            
            4. **Click "Start Analysis"** to fetch and analyze posts
            
            5. **Use Manual Upload tab** for posts that fail automatic scraping
            
            6. **Review results** in the other tabs
            
            ### ⚠️ Important Notes
            
            - Instagram actively blocks automated scraping
            - Some posts may fail to analyze due to anti-bot measures
            - Image extraction may fail for recent posts or protected accounts
            - Use the "Manual Upload" tab to complete analysis for failed posts
            - The tool will attempt to analyze text content even if images fail
            """)
    
    with tab2:
        st.header("📁 Manual Upload Interface")
        st.markdown("Upload images and captions manually for posts that failed automatic scraping.")
        display_manual_upload_interface()
    
    with tab3:
        display_competitor_overview()
    
    with tab4:
        display_strategies(groq_api_key)
    
    with tab5:
        display_campaigns(groq_api_key)
    
    with tab6:
        display_export_options()

def analyze_posts(df: pd.DataFrame, groq_api_key: str):
    """Analyze all posts with progress tracking"""
    
    analyzer = InstagramAnalyzer(groq_api_key)
    results = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_posts = len(df)
    
    # Create container for detailed status updates
    status_container = st.container()
    
    for idx, row in df.iterrows():
        # Update progress
        progress = (idx + 1) / total_posts
        progress_bar.progress(progress)
        status_text.text(f"Analyzing post {idx + 1}/{total_posts}: {row['competitor_name']}")
        
        # Analyze post
        try:
            with status_container:
                result = analyze_single_post(
                    analyzer=analyzer,
                    competitor_name=row['competitor_name'],
                    instagram_id=row['instagram_id'],
                    post_url=row['post_url'],
                    groq_api_key=groq_api_key
                )
            
            # Store result
            competitor = row['competitor_name']
            if competitor not in results:
                results[competitor] = []
            results[competitor].append(result)
            
        except Exception as e:
            st.error(f"❌ Error analyzing {row['post_url']}: {str(e)}")
    
    # Store results in session state
    st.session_state.analysis_results = results
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Summary
    successful_posts = sum(1 for posts in results.values() 
                          for post in posts if not post.get('errors') and not post.get('needs_manual_upload'))
    failed_posts = sum(1 for posts in results.values() 
                      for post in posts if post.get('errors') or post.get('needs_manual_upload'))
    
    st.success(f"✅ Analysis complete! Processed {total_posts} posts from {len(results)} competitors")
    st.info(f"📊 Success rate: {successful_posts}/{total_posts} posts ({(successful_posts/total_posts)*100:.1f}%)")
    
    # Show failed posts if any
    if failed_posts > 0:
        st.warning(f"⚠️ {failed_posts} posts require manual upload (likely due to Instagram's anti-scraping measures)")
        st.info("👉 Use the 'Manual Upload' tab to complete analysis for failed posts")

def display_analysis_results():
    """Display analysis results in organized format"""
    
    if not st.session_state.analysis_results:
        st.info("No analysis results available")
        return
    
    st.subheader("📈 Analysis Results")
    
    # Summary metrics
    total_posts = sum(len(posts) for posts in st.session_state.analysis_results.values())
    successful_posts = sum(1 for posts in st.session_state.analysis_results.values() 
                          for post in posts if not post.get('errors') and not post.get('needs_manual_upload'))
    manual_posts = sum(1 for posts in st.session_state.analysis_results.values() 
                      for post in posts if post.get('manually_processed'))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", total_posts)
    col2.metric("Auto Success", successful_posts) 
    col3.metric("Manual Success", manual_posts)
    col4.metric("Complete Rate", f"{((successful_posts + manual_posts)/total_posts)*100:.1f}%" if total_posts > 0 else "0%")
    
    # Results by competitor
    for competitor, posts in st.session_state.analysis_results.items():
        with st.expander(f"🏢 {competitor} ({len(posts)} posts)", expanded=False):
            
            for idx, post in enumerate(posts, 1):
                st.markdown(f"**Post {idx}:** {post['post_url']}")
                
                # Status indicators
                if post.get('manually_processed'):
                    st.success("✅ Manually processed")
                elif post.get('errors') or post.get('needs_manual_upload'):
                    st.error(f"❌ Error: {post.get('errors', 'Needs manual upload')}")
                    continue
                else:
                    st.success("✅ Auto-processed")
                
                # Create columns for organized display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Caption preview
                    if post.get('caption_text'):
                        caption_preview = post['caption_text'][:100] + "..." if len(post['caption_text']) > 100 else post['caption_text']
                        st.markdown(f"**Caption:** {caption_preview}")
                    
                    # Color palette
                    if post['analysis'].get('color_palette', {}).get('dominant_hex'):
                        st.markdown("**Colors:**")
                        colors_html = render_color_palette(post['analysis']['color_palette']['dominant_hex'])
                        st.markdown(colors_html, unsafe_allow_html=True)
                
                with col2:
                    # Key metrics
                    analysis = post['analysis']
                    
                    if analysis.get('tone_of_voice'):
                        tone = analysis['tone_of_voice']
                        st.markdown(f"**Tone:** {tone.get('label', 'Unknown')} ({tone.get('intensity', 0)}/10)")
                    
                    if analysis.get('cta'):
                        cta = analysis['cta']
                        st.markdown(f"**CTA:** {cta.get('presence', 'None')} - Score: {cta.get('score', 0)}")
                    
                    if analysis.get('readability'):
                        read = analysis['readability']
                        st.markdown(f"**Readability:** {read.get('score', 0)}/100 ({read.get('word_count', 0)} words)")
                
                # Image thumbnail
                if post.get('image_url'):
                    try:
                        st.image(post['image_url'], width=200, caption=f"Post {idx} Image")
                    except:
                        st.text("🖼️ Image preview unavailable")
                
                st.divider()

# (Continue with all other functions from the previous complete app.py...)
# The remaining functions (display_competitor_overview, display_strategies, etc.) remain the same as in the previous version

def display_competitor_overview():
    """Display competitor comparison overview"""
    
    st.header("🏆 Competitor Overview")
    
    if not st.session_state.analysis_results:
        st.info("📊 Run analysis first to see competitor comparison")
        return
    
    # Aggregate competitor data
    competitor_stats = {}
    
    for competitor, posts in st.session_state.analysis_results.items():
        successful_posts = [p for p in posts if not p.get('errors') and not p.get('needs_manual_upload')]
        
        if not successful_posts:
            continue
        
        # Calculate averages
        stats = {
            'posts_analyzed': len(successful_posts),
            'avg_tone_intensity': 0,
            'avg_cta_score': 0,
            'avg_readability': 0,
            'avg_emotional_score': 0,
            'common_colors': [],
            'total_hashtags': 0,
            'dominant_tone': 'neutral'
        }
        
        # Aggregate metrics
        tone_intensities = []
        cta_scores = []
        readability_scores = []
        emotional_scores = []
        all_colors = []
        all_hashtags = []
        tone_labels = []
        
        for post in successful_posts:
            analysis = post['analysis']
            
            if analysis.get('tone_of_voice'):
                tone_intensities.append(analysis['tone_of_voice'].get('intensity', 0))
                tone_labels.append(analysis['tone_of_voice'].get('label', 'neutral'))
            
            if analysis.get('cta'):
                cta_scores.append(analysis['cta'].get('score', 0))
            
            if analysis.get('readability'):
                readability_scores.append(analysis['readability'].get('score', 0))
            
            if analysis.get('emotional_imagery'):
                emotional_scores.append(analysis['emotional_imagery'].get('score', 0))
            
            if analysis.get('color_palette', {}).get('dominant_hex'):
                all_colors.extend(analysis['color_palette']['dominant_hex'][:3])
            
            if analysis.get('hashtags_keywords', {}).get('hashtags'):
                all_hashtags.extend(analysis['hashtags_keywords']['hashtags'])
        
        # Calculate averages
        if tone_intensities:
            stats['avg_tone_intensity'] = sum(tone_intensities) / len(tone_intensities)
        if cta_scores:
            stats['avg_cta_score'] = sum(cta_scores) / len(cta_scores)
        if readability_scores:
            stats['avg_readability'] = sum(readability_scores) / len(readability_scores)
        if emotional_scores:
            stats['avg_emotional_score'] = sum(emotional_scores) / len(emotional_scores)
        
        # Most common elements
        if tone_labels:
            from collections import Counter
            stats['dominant_tone'] = Counter(tone_labels).most_common(1)[0][0]
        
        if all_colors:
            from collections import Counter
            stats['common_colors'] = [color for color, count in Counter(all_colors).most_common(5)]
        
        stats['total_hashtags'] = len(set(all_hashtags))
        
        competitor_stats[competitor] = stats
    
    if not competitor_stats:
        st.warning("⚠️ No successful analyses found for comparison")
        return
    
    # Display comparison table
    st.subheader("📊 Competitor Comparison")
    
    comparison_data = []
    for competitor, stats in competitor_stats.items():
        comparison_data.append({
            'Competitor': competitor,
            'Posts Analyzed': stats['posts_analyzed'],
            'Avg Tone Intensity': f"{stats['avg_tone_intensity']:.1f}/10",
            'Dominant Tone': stats['dominant_tone'].title(),
            'Avg CTA Score': f"{stats['avg_cta_score']:.0f}/100",
            'Avg Readability': f"{stats['avg_readability']:.0f}/100", 
            'Avg Emotional Score': f"{stats['avg_emotional_score']:.0f}/100",
            'Unique Hashtags': stats['total_hashtags']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visual comparison charts would go here
    st.subheader("🎨 Color Palette Comparison")
    
    cols = st.columns(len(competitor_stats))
    for idx, (competitor, stats) in enumerate(competitor_stats.items()):
        with cols[idx]:
            st.markdown(f"**{competitor}**")
            if stats['common_colors']:
                colors_html = render_color_palette(stats['common_colors'])
                st.markdown(colors_html, unsafe_allow_html=True)
            else:
                st.text("No colors extracted")

def display_strategies(groq_api_key: str):
    """Display and generate strategic recommendations"""
    
    st.header("💡 Strategic Recommendations")
    
    if not st.session_state.analysis_results:
        st.info("📊 Complete analysis first to generate strategies")
        return
    
    # Generate strategies button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Generate Strategies", type="primary"):
            generate_strategies(groq_api_key)
    
    with col2:
        if st.button("🔄 Regenerate"):
            st.session_state.strategies = {}
            generate_strategies(groq_api_key)
    
    # Display strategies
    if st.session_state.strategies:
        display_strategy_results()

def generate_strategies(groq_api_key: str):
    """Generate strategic recommendations using LLM"""
    
    if not groq_api_key:
        st.error("🔑 GROQ API key required for strategy generation")
        return
    
    # Prepare competitor data summary
    competitor_summary = {}
    
    for competitor, posts in st.session_state.analysis_results.items():
        successful_posts = [p for p in posts if not p.get('errors') and not p.get('needs_manual_upload')]
        
        if successful_posts:
            # Create summary for this competitor
            summary = {
                'name': competitor,
                'post_count': len(successful_posts),
                'sample_analysis': successful_posts[0]['analysis']  # Use first post as representative
            }
            competitor_summary[competitor] = summary
    
    # Format data for LLM
    competitor_data = json.dumps(competitor_summary, indent=2)
    
    # Call LLM for strategy generation
    with st.spinner("🤖 Generating strategic recommendations..."):
        prompt = SUGGESTION_USER_PROMPT.format(competitor_data=competitor_data)
        
        response = call_groq_model(
            prompt=prompt,
            system=SUGGESTION_SYSTEM_PROMPT,
            max_tokens=2048,
            groq_api_key=groq_api_key
        )
        
        if response.get("error"):
            st.error(f"❌ Strategy generation failed: {response['error']}")
            return
        
        try:
            strategies = json.loads(response["content"])
            st.session_state.strategies = strategies
            st.success("✅ Strategies generated successfully!")
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse strategy response: {str(e)}")
            with st.expander("🔍 Raw Response"):
                st.text(response["content"])

def display_strategy_results():
    """Display generated strategies"""
    
    strategies = st.session_state.strategies.get('strategies', {})
    
    if not strategies:
        st.warning("⚠️ No strategies available")
        return
    
    # Create tabs for each strategy
    strategy_tabs = st.tabs([f"Strategy {key.split('_')[-1].upper()}" for key in strategies.keys()])
    
    for idx, (strategy_key, strategy) in enumerate(strategies.items()):
        with strategy_tabs[idx]:
            
            # Strategy header
            st.subheader(f"🎯 {strategy.get('name', f'Strategy {strategy_key.split("_")[-1].upper()}')}")
            st.markdown(strategy.get('description', 'No description available'))
            
            # Confidence score
            confidence = strategy.get('confidence_score', 0)
            st.metric("Confidence Score", f"{confidence}/100")
            
            # Recommendations
            st.markdown("### 📋 Recommendations")
            
            recommendations = strategy.get('recommendations', [])
            
            for rec in recommendations:
                with st.expander(f"🎨 {rec.get('parameter', 'Unknown').replace('_', ' ').title()}", expanded=False):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Action:** {rec.get('action', 'No action specified')}")
                        st.markdown(f"**Rationale:** {rec.get('rationale', 'No rationale provided')}")
                    
                    with col2:
                        st.markdown(f"**KPI:** {rec.get('kpi', 'No KPI specified')}")

def display_campaigns(groq_api_key: str):
    """Display and generate campaign prompts"""
    
    st.header("🎨 Campaign Prompts")
    
    if not st.session_state.strategies:
        st.info("💡 Generate strategies first to create campaign prompts")
        return
    
    # Strategy selection
    strategies = st.session_state.strategies.get('strategies', {})
    strategy_names = [f"{key.split('_')[-1].upper()}: {strategy.get('name', 'Unnamed')}" 
                     for key, strategy in strategies.items()]
    
    selected_strategy = st.selectbox("🎯 Select Strategy", strategy_names)
    
    if not selected_strategy:
        return
    
    strategy_key = f"strategy_{selected_strategy.split(':')[0].lower()}"
    strategy = strategies.get(strategy_key, {})
    
    # Generate campaign prompts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎨 Generate Campaign Prompts", type="primary"):
            generate_campaign_prompts(strategy, strategy_key, groq_api_key)
    
    with col2:
        if st.button("🔄 Regenerate Prompts"):
            generate_campaign_prompts(strategy, strategy_key, groq_api_key)
    
    # Display campaign prompts
    if st.session_state.campaign_prompts.get(strategy_key):
        display_campaign_results(strategy_key)

def generate_campaign_prompts(strategy: Dict, strategy_key: str, groq_api_key: str):
    """Generate campaign prompts for selected strategy"""
    
    if not groq_api_key:
        st.error("🔑 GROQ API key required for campaign generation")
        return
    
    # Extract key information from strategy
    strategy_name = strategy.get('name', 'Unnamed Strategy')
    recommendations = strategy.get('recommendations', [])
    
    # Compile key actions
    key_actions = []
    for rec in recommendations:
        key_actions.append(f"{rec.get('parameter', '')}: {rec.get('action', '')}")
    
    # Get brand colors from analysis (use most common colors across competitors)
    all_colors = []
    for posts in st.session_state.analysis_results.values():
        for post in posts:
            if not post.get('errors') and post['analysis'].get('color_palette', {}).get('dominant_hex'):
                all_colors.extend(post['analysis']['color_palette']['dominant_hex'][:2])
    
    # Use most common colors or defaults
    from collections import Counter
    if all_colors:
        common_colors = [color for color, count in Counter(all_colors).most_common(3)]
    else:
        common_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Default vibrant palette
    
    brand_context = "Modern, engaging brand focused on authentic community building"
    
    # Generate prompts using LLM
    with st.spinner("🎨 Generating campaign prompts..."):
        prompt = CAMPAIGN_PROMPT_USER.format(
            strategy_name=strategy_name,
            key_actions="; ".join(key_actions[:3]),  # Limit for API
            colors=", ".join(common_colors),
            brand_context=brand_context
        )
        
        response = call_groq_model(
            prompt=prompt,
            system=CAMPAIGN_PROMPT_SYSTEM,
            max_tokens=2048,
            groq_api_key=groq_api_key
        )
        
        if response.get("error"):
            st.error(f"❌ Campaign generation failed: {response['error']}")
            return
        
        try:
            campaigns = json.loads(response["content"])
            
            # Store in session state
            if 'campaign_prompts' not in st.session_state:
                st.session_state.campaign_prompts = {}
            
            st.session_state.campaign_prompts[strategy_key] = campaigns
            st.success("✅ Campaign prompts generated!")
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse campaign response: {str(e)}")
            with st.expander("🔍 Raw Response"):
                st.text(response["content"])

def display_campaign_results(strategy_key: str):
    """Display generated campaign prompts"""
    
    campaigns = st.session_state.campaign_prompts.get(strategy_key, {})
    campaign_sets = campaigns.get('campaign_sets', [])
    
    if not campaign_sets:
        st.warning("⚠️ No campaign sets available")
        return
    
    st.subheader("🎨 Generated Campaign Sets")
    
    for idx, campaign_set in enumerate(campaign_sets, 1):
        with st.expander(f"📱 {campaign_set.get('name', f'Campaign Set {idx}')}", expanded=idx==1):
            
            # Image prompt
            st.markdown("#### 🖼️ Image Generation Prompt")
            st.code(campaign_set.get('image_prompt', 'No image prompt available'), language='text')
            
            # Video prompt  
            st.markdown("#### 🎬 Video Generation Prompt")
            st.code(campaign_set.get('video_prompt', 'No video prompt available'), language='text')
            
            # Hashtags
            st.markdown("#### #️⃣ Hashtags")
            hashtags = campaign_set.get('hashtags', [])
            if hashtags:
                hashtag_text = " ".join(hashtags)
                st.code(hashtag_text, language='text')
            
            # Caption starter
            st.markdown("#### ✍️ Caption Starter")
            st.code(campaign_set.get('caption_starter', 'No caption starter available'), language='text')
            
            # Model-ready prompt
            st.markdown("#### 🤖 Model-Ready Prompt")
            st.code(campaign_set.get('model_ready_prompt', 'No model prompt available'), language='text')

def display_export_options():
    """Display export and download options"""
    
    st.header("💾 Export Results")
    
    if not st.session_state.analysis_results:
        st.info("📊 Complete analysis to export results")
        return
    
    # Export options
    st.subheader("📁 Available Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        if st.button("📄 Export Analysis JSON", type="primary"):
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_results': st.session_state.analysis_results,
                'strategies': st.session_state.strategies,
                'campaign_prompts': st.session_state.campaign_prompts
            }
            
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="💾 Download analysis_results.json",
                data=json_str,
                file_name=f"instagram_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        # CSV export
        if st.button("📊 Export Analysis CSV"):
            csv_data = create_csv_export()
            
            st.download_button(
                label="💾 Download analysis_results.csv",
                data=csv_data,
                file_name=f"instagram_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Summary statistics
    st.subheader("📈 Export Summary")
    
    total_posts = sum(len(posts) for posts in st.session_state.analysis_results.values())
    total_competitors = len(st.session_state.analysis_results)
    total_strategies = len(st.session_state.strategies.get('strategies', {}))
    total_campaigns = sum(len(campaigns.get('campaign_sets', [])) 
                         for campaigns in st.session_state.campaign_prompts.values())
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts Analyzed", total_posts)
    col2.metric("Competitors", total_competitors)
    col3.metric("Strategies", total_strategies)
    col4.metric("Campaign Sets", total_campaigns)

def create_csv_export() -> str:
    """Create CSV export of analysis results"""
    
    rows = []
    
    for competitor, posts in st.session_state.analysis_results.items():
        for post in posts:
            if post.get('errors') and not post.get('manually_processed'):
                continue
            
            analysis = post['analysis']
            
            row = {
                'competitor_name': post['competitor_name'],
                'instagram_id': post['instagram_id'], 
                'post_url': post['post_url'],
                'timestamp': post.get('timestamp', ''),
                'caption_length': len(post.get('caption_text', '')),
                'image_hash': post['image_hash'],
                'processing_method': 'manual' if post.get('manually_processed') else 'automatic',
                
                # Analysis metrics
                'tone_label': analysis.get('tone_of_voice', {}).get('label', ''),
                'tone_intensity': analysis.get('tone_of_voice', {}).get('intensity', 0),
                'cta_presence': analysis.get('cta', {}).get('presence', ''),
                'cta_score': analysis.get('cta', {}).get('score', 0),
                'readability_score': analysis.get('readability', {}).get('score', 0),
                'word_count': analysis.get('readability', {}).get('word_count', 0),
                'hashtag_count': len(analysis.get('hashtags_keywords', {}).get('hashtags', [])),
                'emotional_score': analysis.get('emotional_imagery', {}).get('score', 0),
                'color_count': len(analysis.get('color_palette', {}).get('dominant_hex', [])),
                'dominant_colors': ','.join(analysis.get('color_palette', {}).get('dominant_hex', [])[:3])
            }
            
            rows.append(row)
    
    if not rows:
        return "No data available for export"
    
    # Convert to DataFrame and then CSV
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
