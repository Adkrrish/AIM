"""
Instagram Competitor Analysis Tool
Focused workflow: Scrape captions → Upload images → Generate strategic recommendations → Create campaign prompts
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import BytesIO
from PIL import Image

from utils import InstagramCaptionScraper, CombinedAnalyzer, generate_strategic_recommendations, generate_campaign_prompts

# Page configuration
st.set_page_config(
    page_title="Instagram Competitor Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scraped_captions' not in st.session_state:
    st.session_state.scraped_captions = {}
if 'competitor_analysis' not in st.session_state:
    st.session_state.competitor_analysis = {}  
if 'strategic_recommendations' not in st.session_state:
    st.session_state.strategic_recommendations = {}
if 'campaign_prompts' not in st.session_state:
    st.session_state.campaign_prompts = {}

def get_groq_api_key() -> Optional[str]:
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY")

def get_apify_token() -> Optional[str]:
    try:
        return st.secrets["APIFY_TOKEN"]
    except:
        return os.getenv("APIFY_TOKEN")

def main():
    st.title("📱 Instagram Competitor Analysis")
    st.markdown("**Objective:** Analyze competitor strategies → Generate strategic recommendations → Create campaign prompts")
    
    # API Configuration
    st.sidebar.header("API Configuration")
    groq_api_key = get_groq_api_key()
    apify_token = get_apify_token()
    
    if not groq_api_key:
        st.sidebar.error("🔑 GROQ API Key required")
        return
    else:
        st.sidebar.success("✅ Groq API Key configured")
    
    if not apify_token:
        st.sidebar.warning("⚠️ Apify Token recommended for reliable scraping")
    else:
        st.sidebar.success("✅ Apify Token configured")
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Scrape Captions", 
        "2️⃣ Analyze Competitors", 
        "3️⃣ Strategic Recommendations", 
        "4️⃣ Campaign Prompts"
    ])
    
    with tab1:
        scrape_captions_tab(apify_token)
    
    with tab2:
        analyze_competitors_tab(groq_api_key)
    
    with tab3:
        strategic_recommendations_tab(groq_api_key)
    
    with tab4:
        campaign_prompts_tab(groq_api_key)

def scrape_captions_tab(apify_token: str):
    st.header("1️⃣ Scrape Instagram Captions")
    
    # File upload
    competitor_file = st.file_uploader("Upload competitor data (Excel)", type=['xlsx'])
    
    if competitor_file:
        df = pd.read_excel(competitor_file)
        required_columns = ['competitor_name', 'instagram_id', 'post_url']
        
        if all(col in df.columns for col in required_columns):
            st.success(f"✅ Loaded {len(df)} posts from {len(df['competitor_name'].unique())} competitors")
            
            if st.button("🚀 Scrape Captions", type="primary"):
                scraper = InstagramCaptionScraper(apify_token)
                results = {}
                
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    progress_bar.progress((idx + 1) / len(df))
                    
                    caption_data = scraper.scrape_post_caption(row['post_url'])
                    caption_data.update({
                        'competitor_name': row['competitor_name'],
                        'instagram_id': row['instagram_id'],
                        'post_id': f"{row['competitor_name']}_{idx}"
                    })
                    
                    competitor = row['competitor_name']
                    if competitor not in results:
                        results[competitor] = []
                    results[competitor].append(caption_data)
                
                st.session_state.scraped_captions = results
                progress_bar.empty()
                
                st.success("✅ Caption scraping completed!")
                st.info("👉 Proceed to 'Analyze Competitors' tab")
        else:
            st.error(f"❌ Missing columns: {required_columns}")
    else:
        st.info("📤 Upload Excel file with competitor Instagram post URLs")

def analyze_competitors_tab(groq_api_key: str):
    st.header("2️⃣ Competitor Analysis")
    
    if not st.session_state.scraped_captions:
        st.warning("⚠️ Please scrape captions first")
        return
    
    analyzer = CombinedAnalyzer(groq_api_key)
    
    # Display posts and image upload interface
    for competitor, posts in st.session_state.scraped_captions.items():
        with st.expander(f"🏢 {competitor}", expanded=True):
            
            for idx, post in enumerate(posts):
                st.markdown(f"### Post {idx + 1}")
                st.markdown(f"**URL:** {post['url']}")
                
                if post.get('caption'):
                    st.markdown(f"**Caption:** {post['caption'][:150]}...")
                    if post.get('hashtags'):
                        st.markdown(f"**Hashtags:** {', '.join(post['hashtags'][:5])}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    uploaded_file = st.file_uploader(
                        f"Upload image for {competitor} Post {idx + 1}",
                        type=['jpg', 'jpeg', 'png'],
                        key=f"upload_{post.get('post_id', f'{competitor}_{idx}')}"
                    )
                    
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                        st.image(image, width=200)
                
                with col2:
                    post_id = post.get('post_id', f"{competitor}_{idx}")
                    
                    if st.button(f"🔍 Analyze Post {idx + 1}", key=f"analyze_{post_id}"):
                        if uploaded_file:
                            with st.spinner("Analyzing..."):
                                image = Image.open(uploaded_file)
                                
                                analysis = analyzer.analyze_competitor_post(
                                    post.get('caption', ''),
                                    image,
                                    competitor,
                                    post['url']
                                )
                                
                                # Store analysis
                                if competitor not in st.session_state.competitor_analysis:
                                    st.session_state.competitor_analysis[competitor] = []
                                st.session_state.competitor_analysis[competitor].append(analysis)
                                
                                st.success("✅ Analysis completed!")
                        else:
                            st.error("Please upload an image first")
                    
                    # Show existing analysis
                    if competitor in st.session_state.competitor_analysis:
                        existing_analyses = [a for a in st.session_state.competitor_analysis[competitor] 
                                           if a['post_url'] == post['url']]
                        if existing_analyses:
                            analysis = existing_analyses[0]
                            st.success(f"✅ Strategic Score: {analysis['strategic_score']}/100")
                
                st.divider()
    
    # Generate competitor insights
    if st.session_state.competitor_analysis:
        if st.button("📊 Generate Competitor Intelligence", type="primary"):
            with st.spinner("Generating insights..."):
                insights = analyzer.generate_competitor_insights(st.session_state.competitor_analysis)
                
                st.subheader("🏆 Competitor Intelligence")
                for competitor, insight in insights.items():
                    with st.expander(f"📈 {competitor} Analysis"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Strategic Score", f"{insight['avg_strategic_score']}/100")
                        col2.metric("Posts Analyzed", insight['post_count'])
                        col3.metric("Consistency", insight['content_consistency'].title())
                        
                        st.markdown(f"**Common Themes:** {', '.join(insight['common_themes'])}")
                        st.markdown(f"**CTA Strategy:** {insight['dominant_cta_strategy'].title()}")
                
                st.info("👉 Proceed to 'Strategic Recommendations' tab")

def strategic_recommendations_tab(groq_api_key: str):
    st.header("3️⃣ Strategic Recommendations")
    
    if not st.session_state.competitor_analysis:
        st.warning("⚠️ Complete competitor analysis first")
        return
    
    if st.button("🎯 Generate Strategic Recommendations", type="primary"):
        with st.spinner("Generating strategic recommendations..."):
            
            # Prepare competitor insights
            analyzer = CombinedAnalyzer(groq_api_key)
            competitor_insights = analyzer.generate_competitor_insights(st.session_state.competitor_analysis)
            
            # Generate recommendations
            recommendations = generate_strategic_recommendations(competitor_insights, groq_api_key)
            
            if not recommendations.get("error"):
                st.session_state.strategic_recommendations = recommendations
                st.success("✅ Strategic recommendations generated!")
            else:
                st.error(f"❌ {recommendations['error']}")
    
    # Display recommendations
    if st.session_state.strategic_recommendations:
        st.subheader("🎯 Strategic Approaches")
        
        strategies = st.session_state.strategic_recommendations
        
        for strategy_name, strategy in strategies.items():
            with st.expander(f"📋 {strategy_name.replace('_', ' ').title()}", expanded=True):
                st.markdown(f"**Approach:** {strategy.get('approach', 'Not specified')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Content Pillars:**")
                    for pillar in strategy.get('content_pillars', []):
                        st.markdown(f"• {pillar}")
                
                with col2:
                    st.markdown(f"**Visual Direction:** {strategy.get('visual_direction', 'Not specified')}")
                    st.markdown(f"**Tone Strategy:** {strategy.get('tone_strategy', 'Not specified')}")
        
        st.info("👉 Proceed to 'Campaign Prompts' tab to generate content creation prompts")

def campaign_prompts_tab(groq_api_key: str):
    st.header("4️⃣ Campaign Prompts Generation")
    
    if not st.session_state.strategic_recommendations:
        st.warning("⚠️ Generate strategic recommendations first")
        return
    
    # Strategy selection
    strategies = st.session_state.strategic_recommendations
    strategy_options = list(strategies.keys())
    
    selected_strategy = st.selectbox(
        "Select strategy for campaign prompts:",
        strategy_options,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if st.button("🎨 Generate Campaign Prompts", type="primary"):
        with st.spinner("Generating campaign prompts..."):
            
            strategy_data = strategies[selected_strategy]
            campaign_prompts = generate_campaign_prompts(strategy_data, groq_api_key)
            
            if not campaign_prompts.get("error"):
                st.session_state.campaign_prompts[selected_strategy] = campaign_prompts
                st.success("✅ Campaign prompts generated!")
            else:
                st.error(f"❌ {campaign_prompts['error']}")
    
    # Display campaign prompts
    if selected_strategy in st.session_state.campaign_prompts:
        campaign_data = st.session_state.campaign_prompts[selected_strategy]
        campaign_sets = campaign_data.get('campaign_sets', [])
        
        st.subheader("🎬 Campaign Content Prompts")
        
        for idx, campaign_set in enumerate(campaign_sets, 1):
            with st.expander(f"🎯 {campaign_set.get('name', f'Campaign Set {idx}')}", expanded=True):
                
                st.markdown(f"**Theme:** {campaign_set.get('theme', 'Not specified')}")
                st.markdown(f"**Target Emotion:** {campaign_set.get('target_emotion', 'Not specified')}")
                
                # Image prompt
                st.markdown("#### 🖼️ Image Generation Prompt")
                st.code(campaign_set.get('image_prompt', 'No prompt available'), language='text')
                
                # Video prompt
                st.markdown("#### 🎬 Video Creation Prompt")
                st.code(campaign_set.get('video_prompt', 'No prompt available'), language='text')
                
                # Caption template
                st.markdown("#### ✍️ Caption Template")
                st.code(campaign_set.get('caption_template', 'No template available'), language='text')
                
                # Hashtags
                st.markdown("#### #️⃣ Recommended Hashtags")
                hashtags = campaign_set.get('hashtags', [])
                if hashtags:
                    st.code(' '.join(hashtags), language='text')
                
                # Copy buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📋 Copy Image Prompt", key=f"copy_img_{idx}"):
                        st.write("Copied to clipboard!")
                with col2:
                    if st.button(f"📋 Copy Video Prompt", key=f"copy_vid_{idx}"):
                        st.write("Copied to clipboard!")
                with col3:
                    if st.button(f"📋 Copy All Content", key=f"copy_all_{idx}"):
                        st.write("Copied to clipboard!")

if __name__ == "__main__":
    main()
