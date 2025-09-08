"""
Instagram Competitor Analysis Tool
Enhanced for caption scraping + manual image upload workflow
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
from PIL import Image

# Import our utilities
from utils import InstagramCaptionScraper, CombinedAnalyzer, call_groq_model

# Page configuration
st.set_page_config(
    page_title="Instagram Caption & Image Analysis",
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
.caption-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.analysis-box {
    background: #e8f4fd;
    border: 1px solid #bee5eb;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.recommendation-box {
    background: #f0f8ff;
    border-left: 4px solid #007bff;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scraped_captions' not in st.session_state:
    st.session_state.scraped_captions = {}
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}
if 'combined_analysis' not in st.session_state:
    st.session_state.combined_analysis = {}
if 'competitor_insights' not in st.session_state:
    st.session_state.competitor_insights = {}

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

def scrape_captions_from_urls(df: pd.DataFrame, apify_token: str) -> Dict[str, List[Dict]]:
    """Scrape captions from Instagram URLs"""
    
    scraper = InstagramCaptionScraper(apify_token)
    results = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_posts = len(df)
    
    for idx, row in df.iterrows():
        # Update progress
        progress = (idx + 1) / total_posts
        progress_bar.progress(progress)
        status_text.text(f"Scraping caption {idx + 1}/{total_posts}: {row['competitor_name']}")
        
        try:
            # Scrape caption
            caption_data = scraper.scrape_post_caption(row['post_url'])
            
            # Store result
            competitor = row['competitor_name']
            if competitor not in results:
                results[competitor] = []
            
            caption_data.update({
                'competitor_name': competitor,
                'instagram_id': row['instagram_id'],
                'post_id': f"{competitor}_{idx}"
            })
            
            results[competitor].append(caption_data)
            
        except Exception as e:
            st.error(f"❌ Error scraping {row['post_url']}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def display_caption_and_upload_interface():
    """Display captions and image upload interface"""
    
    if not st.session_state.scraped_captions:
        st.info("📊 Please scrape captions first using the Analysis tab")
        return
    
    st.header("📝 Caption Review & Image Upload")
    
    groq_api_key = get_groq_api_key()
    analyzer = CombinedAnalyzer(groq_api_key) if groq_api_key else None
    
    for competitor, posts in st.session_state.scraped_captions.items():
        with st.expander(f"🏢 {competitor} ({len(posts)} posts)", expanded=True):
            
            for idx, post in enumerate(posts):
                post_id = post.get('post_id', f"{competitor}_{idx}")
                
                st.markdown(f"### Post {idx + 1}")
                st.markdown(f"**URL:** {post['url']}")
                
                # Display caption
                if post.get('caption'):
                    st.markdown('<div class="caption-box">', unsafe_allow_html=True)
                    st.markdown(f"**Caption:** {post['caption']}")
                    
                    if post.get('hashtags'):
                        st.markdown(f"**Hashtags:** {', '.join(post['hashtags'])}")
                    
                    if post.get('timestamp'):
                        st.markdown(f"**Posted:** {post['timestamp']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No caption found for this post")
                
                # Image upload
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    uploaded_file = st.file_uploader(
                        f"Upload image for {competitor} Post {idx + 1}",
                        type=['jpg', 'jpeg', 'png', 'webp'],
                        key=f"upload_{post_id}",
                        help="Upload the corresponding Instagram post image"
                    )
                    
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                        st.image(image, width=250, caption=f"Uploaded for Post {idx + 1}")
                        
                        # Store uploaded image
                        st.session_state.uploaded_images[post_id] = image
                
                with col2:
                    # Analysis controls
                    if st.button(f"🔍 Analyze Post {idx + 1}", key=f"analyze_{post_id}"):
                        if post_id in st.session_state.uploaded_images and analyzer:
                            
                            with st.spinner("Analyzing caption and image..."):
                                # Analyze caption
                                caption_analysis = analyzer.analyze_caption_theme(post.get('caption', ''))
                                
                                # Analyze image
                                image_analysis = analyzer.analyze_image_properties(
                                    st.session_state.uploaded_images[post_id]
                                )
                                
                                # Combine analysis
                                combined_result = analyzer.combine_analysis(
                                    caption_analysis, 
                                    image_analysis, 
                                    post.get('caption', ''),
                                    competitor
                                )
                                
                                # Store results
                                st.session_state.combined_analysis[post_id] = combined_result
                                
                                st.success("✅ Analysis completed!")
                                st.rerun()
                        else:
                            st.error("Please upload an image and ensure API keys are configured")
                    
                    # Display existing analysis
                    if post_id in st.session_state.combined_analysis:
                        st.success("✅ Analysis completed - view results in Analysis Results tab")
                
                st.divider()

def display_analysis_results():
    """Display combined analysis results"""
    
    if not st.session_state.combined_analysis:
        st.info("📊 Complete caption and image analysis first")
        return
    
    st.header("📈 Combined Analysis Results")
    
    for post_id, analysis in st.session_state.combined_analysis.items():
        competitor = analysis['competitor_name']
        
        with st.expander(f"📊 {competitor} - Analysis Results", expanded=True):
            
            # Overall scores
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Score", f"{analysis['overall_score']}/100")
            col2.metric("Alignment Score", f"{analysis['alignment_score']}/100")
            col3.metric("Brand Consistency", analysis['brand_consistency'].title())
            
            # Caption Analysis
            st.markdown("#### 📝 Caption Analysis")
            caption_analysis = analysis['caption_analysis']
            
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Themes:** {', '.join(caption_analysis.get('themes', []))}")
                st.markdown(f"**Sentiment:** {caption_analysis.get('sentiment', 'Unknown').title()}")
                st.markdown(f"**Tone:** {caption_analysis.get('tone', 'Unknown').title()}")
            
            with col2:
                st.markdown(f"**Content Type:** {caption_analysis.get('content_type', 'Unknown').title()}")
                st.markdown(f"**Target Audience:** {caption_analysis.get('target_audience', 'Unknown').title()}")
                st.markdown(f"**Engagement Potential:** {caption_analysis.get('engagement_potential', 'Unknown').title()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image Analysis
            st.markdown("#### 🖼️ Image Analysis")
            image_analysis = analysis['image_analysis']
            
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Visual Style:** {image_analysis.get('visual_style', 'Unknown').replace('_', ' ').title()}")
                st.markdown(f"**Color Temperature:** {image_analysis.get('color_temperature', 'Unknown').title()}")
                st.markdown(f"**Brightness:** {image_analysis.get('brightness_level', 0)}/255")
            
            with col2:
                st.markdown(f"**Emotional Tone:** {image_analysis.get('emotional_tone', 'Unknown').replace('_', ' ').title()}")
                
                # Color palette
                colors = image_analysis.get('dominant_colors', [])
                if colors:
                    color_html = '<div style="display: flex; gap: 5px; margin: 5px 0;">'
                    for color in colors[:5]:
                        color_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #ddd; border-radius: 3px;" title="{color}"></div>'
                    color_html += '</div>'
                    st.markdown("**Dominant Colors:**")
                    st.markdown(color_html, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Strategy Insights
            if analysis.get('strategy_insights'):
                st.markdown("#### 💡 Strategy Insights")
                for insight in analysis['strategy_insights']:
                    st.markdown(f"<div class='analysis-box'>• {insight}</div>", unsafe_allow_html=True)
            
            # Recommendations
            if analysis.get('recommendations'):
                st.markdown("#### 🎯 Recommendations")
                for rec in analysis['recommendations']:
                    st.markdown(f"<div class='recommendation-box'>💡 {rec}</div>", unsafe_allow_html=True)

def generate_competitor_insights():
    """Generate overall competitor insights"""
    
    if not st.session_state.combined_analysis:
        st.info("📊 Complete individual post analysis first")
        return
    
    st.header("🏆 Competitor Strategic Insights")
    
    groq_api_key = get_groq_api_key()
    
    if st.button("🚀 Generate Competitor Insights", type="primary"):
        
        if not groq_api_key:
            st.error("🔑 Groq API key required for strategic insights")
            return
        
        with st.spinner("Generating competitor insights..."):
            
            # Aggregate data by competitor
            competitor_data = {}
            for post_id, analysis in st.session_state.combined_analysis.items():
                competitor = analysis['competitor_name']
                
                if competitor not in competitor_data:
                    competitor_data[competitor] = []
                
                competitor_data[competitor].append({
                    'themes': analysis['caption_analysis'].get('themes', []),
                    'sentiment': analysis['caption_analysis'].get('sentiment', 'neutral'),
                    'visual_style': analysis['image_analysis'].get('visual_style', 'unknown'),
                    'overall_score': analysis['overall_score'],
                    'alignment_score': analysis['alignment_score']
                })
            
            # Generate insights for each competitor
            insights = {}
            
            for competitor, posts in competitor_data.items():
                
                # Calculate aggregates
                avg_score = sum(p['overall_score'] for p in posts) / len(posts)
                avg_alignment = sum(p['alignment_score'] for p in posts) / len(posts)
                
                # Most common themes
                all_themes = []
                for post in posts:
                    all_themes.extend(post['themes'])
                
                from collections import Counter
                common_themes = [theme for theme, count in Counter(all_themes).most_common(3)]
                
                # Sentiment distribution
                sentiments = [p['sentiment'] for p in posts]
                dominant_sentiment = Counter(sentiments).most_common(1)[0][0]
                
                # Generate strategic insight using LLM
                insight_prompt = f"""
                Analyze this competitor's Instagram content strategy:
                
                Competitor: {competitor}
                Posts Analyzed: {len(posts)}
                Average Overall Score: {avg_score:.1f}/100
                Average Alignment Score: {avg_alignment:.1f}/100
                Common Themes: {', '.join(common_themes)}
                Dominant Sentiment: {dominant_sentiment}
                
                Provide strategic insights in JSON format:
                {{
                    "content_strategy": "description of their approach",
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"],
                    "differentiation_opportunities": ["opportunity1", "opportunity2"],
                    "recommended_counter_strategy": "strategic recommendation"
                }}
                """
                
                response = call_groq_model(
                    prompt=insight_prompt,
                    system="You are a strategic social media consultant analyzing competitor content strategies.",
                    groq_api_key=groq_api_key
                )
                
                try:
                    if response.get("content"):
                        competitor_insight = json.loads(response["content"])
                        competitor_insight.update({
                            'avg_score': avg_score,
                            'avg_alignment': avg_alignment,
                            'common_themes': common_themes,
                            'dominant_sentiment': dominant_sentiment,
                            'posts_count': len(posts)
                        })
                        insights[competitor] = competitor_insight
                except:
                    # Fallback insight
                    insights[competitor] = {
                        'content_strategy': f"Mixed content approach with focus on {', '.join(common_themes[:2])}",
                        'strengths': ["Consistent posting", "Diverse content themes"],
                        'weaknesses': ["Alignment could be improved"],
                        'avg_score': avg_score,
                        'avg_alignment': avg_alignment,
                        'common_themes': common_themes
                    }
            
            st.session_state.competitor_insights = insights
            st.success("✅ Competitor insights generated!")
    
    # Display insights
    if st.session_state.competitor_insights:
        
        for competitor, insight in st.session_state.competitor_insights.items():
            
            with st.expander(f"🎯 {competitor} Strategic Analysis", expanded=True):
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Score", f"{insight.get('avg_score', 0):.1f}/100")
                col2.metric("Alignment Score", f"{insight.get('avg_alignment', 0):.1f}/100")
                col3.metric("Posts Analyzed", insight.get('posts_count', 0))
                
                # Content Strategy
                st.markdown("#### 📋 Content Strategy")
                st.markdown(f"**Approach:** {insight.get('content_strategy', 'Unknown')}")
                
                if insight.get('common_themes'):
                    st.markdown(f"**Core Themes:** {', '.join(insight['common_themes'])}")
                
                # Strengths & Weaknesses
                col1, col2 = st.columns(2)
                
                with col1:
                    if insight.get('strengths'):
                        st.markdown("#### ✅ Strengths")
                        for strength in insight['strengths']:
                            st.markdown(f"• {strength}")
                
                with col2:
                    if insight.get('weaknesses'):
                        st.markdown("#### ⚠️ Weaknesses")
                        for weakness in insight['weaknesses']:
                            st.markdown(f"• {weakness}")
                
                # Opportunities & Recommendations
                if insight.get('differentiation_opportunities'):
                    st.markdown("#### 🎯 Differentiation Opportunities")
                    for opp in insight['differentiation_opportunities']:
                        st.markdown(f"<div class='recommendation-box'>🚀 {opp}</div>", unsafe_allow_html=True)
                
                if insight.get('recommended_counter_strategy'):
                    st.markdown("#### 🎪 Recommended Counter-Strategy")
                    st.markdown(f"<div class='success-box'>{insight['recommended_counter_strategy']}</div>", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📱 Instagram Caption & Image Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced competitor analysis combining AI-powered caption scraping with manual image upload</p>', 
                unsafe_allow_html=True)
    
    # Enhanced notice
    st.markdown("""
    <div class="success-box">
    <strong>🚀 New Workflow:</strong> This tool now focuses on accurate caption scraping via Apify, 
    combined with manual image uploads for comprehensive analysis. This approach ensures higher data quality 
    and enables deeper thematic mapping between text and visual content.
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
        st.sidebar.info("Set APIFY_TOKEN in Streamlit secrets for enhanced caption scraping")
        st.sidebar.markdown("Without Apify token, the tool will use fallback scraping methods.")
    else:
        st.sidebar.success("✅ Apify Token configured")
    
    # File uploads
    st.sidebar.subheader("📁 Upload Files")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Caption Scraping", "📁 Caption & Image Analysis", "📈 Analysis Results", "🏆 Competitor Insights"])
    
    with tab1:
        st.header("📝 Instagram Caption Scraping")
        
        if competitor_file is not None:
            # Load competitor data
            try:
                df = pd.read_excel(competitor_file)
                required_columns = ['competitor_name', 'instagram_id', 'post_url']
                
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ Missing required columns: {required_columns}")
                    return
                
                st.success(f"✅ Loaded {len(df)} posts from {len(df['competitor_name'].unique())} competitors")
                
                # Show data preview
                with st.expander("📋 Data Preview", expanded=False):
                    st.dataframe(df, use_container_width=True)
                
                # Scraping controls
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Scrape Captions", type="primary"):
                        scraped_data = scrape_captions_from_urls(df, apify_token)
                        st.session_state.scraped_captions = scraped_data
                        
                        # Show results
                        total_scraped = sum(len(posts) for posts in scraped_data.values())
                        successful_captions = sum(1 for posts in scraped_data.values() 
                                                for post in posts if post.get('caption'))
                        
                        st.success(f"✅ Scraping complete! {successful_captions}/{total_scraped} captions extracted")
                        
                        if successful_captions < total_scraped:
                            st.info("💡 Some posts may not have captions or failed to scrape. Proceed to next tab to upload images.")
                
                with col2:
                    if st.button("🔄 Clear Results"):
                        st.session_state.scraped_captions = {}
                        st.session_state.uploaded_images = {}
                        st.session_state.combined_analysis = {}
                        st.rerun()
                
                # Show scraping results
                if st.session_state.scraped_captions:
                    st.markdown("### 📊 Scraping Results")
                    
                    for competitor, posts in st.session_state.scraped_captions.items():
                        successful_posts = [p for p in posts if p.get('caption')]
                        st.markdown(f"**{competitor}:** {len(successful_posts)}/{len(posts)} captions scraped")
                    
                    st.info("👉 Proceed to 'Caption & Image Analysis' tab to upload images and run analysis")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.info("👆 Please upload competitor data file to begin caption scraping")
            
            # Show instructions
            st.markdown("""
            ### 📋 New Workflow Instructions
            
            1. **Upload competitor data**: Excel file with competitor Instagram post URLs
            2. **Scrape captions**: Extract text content using Apify's reliable API
            3. **Upload images**: Manually upload corresponding post images
            4. **Combined analysis**: AI analyzes caption themes and image properties
            5. **Strategic insights**: Generate competitor intelligence and recommendations
            
            ### 🎯 Benefits of This Approach
            
            - **Higher accuracy**: Reliable caption extraction via Apify
            - **Quality control**: Manual image upload ensures correct image-caption pairing
            - **Deeper insights**: Advanced thematic mapping between text and visuals
            - **Strategic value**: Comprehensive competitor intelligence
            """)
    
    with tab2:
        display_caption_and_upload_interface()
    
    with tab3:
        display_analysis_results()
    
    with tab4:
        generate_competitor_insights()

if __name__ == "__main__":
    main()
