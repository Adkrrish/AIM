"""
Instagram Competitor Analysis Tool
Complete workflow: Company data → Caption scraping → 6-parameter analysis → 3 suggestion sets → Product-specific prompts
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import BytesIO
from PIL import Image

# CORRECTED IMPORTS - Using CombinedAnalyzer instead of SixParameterAnalyzer
from utils import (
    InstagramCaptionScraper, 
    CombinedAnalyzer,  # Fixed: was SixParameterAnalyzer
    generate_three_suggestion_sets,
    generate_product_specific_prompts
)

# Page configuration
st.set_page_config(
    page_title="Instagram Competitor Analysis - Product Focused",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'scraped_captions' not in st.session_state:
    st.session_state.scraped_captions = {}
if 'post_analyses' not in st.session_state:
    st.session_state.post_analyses = []
if 'suggestion_sets' not in st.session_state:
    st.session_state.suggestion_sets = {}
if 'product_prompts' not in st.session_state:
    st.session_state.product_prompts = {}

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
    st.markdown("**Complete Product-Focused Workflow:** Company Data → Caption Scraping → 6-Parameter Analysis → Strategic Suggestions → Product-Specific Prompts")
    
    # API Configuration Sidebar
    st.sidebar.header("🔧 API Configuration")
    groq_api_key = get_groq_api_key()
    apify_token = get_apify_token()
    
    if not groq_api_key:
        st.sidebar.error("🔑 GROQ API Key required")
        st.sidebar.stop()
    else:
        st.sidebar.success("✅ Groq API Key configured")
    
    if not apify_token:
        st.sidebar.warning("⚠️ Apify Token recommended")
    else:
        st.sidebar.success("✅ Apify Token configured")
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Company Data",
        "2️⃣ Scrape Captions", 
        "3️⃣ 6-Parameter Analysis",
        "4️⃣ Analysis Results",
        "5️⃣ Strategic Suggestions", 
        "6️⃣ Product Prompts"
    ])
    
    with tab1:
        company_data_tab()
    
    with tab2:
        scrape_captions_tab(apify_token)
    
    with tab3:
        six_parameter_analysis_tab(groq_api_key)
    
    with tab4:
        analysis_results_tab()
    
    with tab5:
        strategic_suggestions_tab(groq_api_key)
    
    with tab6:
        product_prompts_tab(groq_api_key)

def company_data_tab():
    """Tab 1: Upload and manage company data"""
    st.header("1️⃣ Company & Product Portfolio Data")
    st.markdown("Upload your company information and product portfolio for targeted analysis and recommendations.")
    
    # File upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Company Data")
        company_file = st.file_uploader(
            "Upload company metadata",
            type=['json', 'xlsx', 'csv'],
            help="JSON file with company info and product portfolio"
        )
        
        if company_file:
            if company_file.type == "application/json":
                company_data = json.load(company_file)
            elif company_file.name.endswith('.xlsx'):
                df = pd.read_excel(company_file)
                company_data = df.to_dict('records')
            else:
                df = pd.read_csv(company_file)
                company_data = df.to_dict('records')
            
            st.session_state.company_data = company_data
            st.success(f"✅ Loaded data for {len(company_data)} products")
    
    with col2:
        st.subheader("📝 Manual Entry")
        if st.button("➕ Add Product Manually"):
            if 'manual_products' not in st.session_state:
                st.session_state.manual_products = []
            
            with st.form("add_product_form"):
                product_name = st.text_input("Product Name")
                product_description = st.text_area("Product Description")
                product_category = st.selectbox("Category", 
                    ["Fashion", "Food", "Technology", "Beauty", "Lifestyle", "Fitness", "Travel", "Other"])
                product_price_range = st.selectbox("Price Range", 
                    ["Budget", "Mid-range", "Premium", "Luxury"])
                target_audience = st.text_input("Target Audience")
                
                if st.form_submit_button("Add Product"):
                    product_data = {
                        "name": product_name,
                        "description": product_description,
                        "category": product_category,
                        "price_range": product_price_range,
                        "target_audience": target_audience
                    }
                    st.session_state.manual_products.append(product_data)
                    st.success(f"Added {product_name}")
        
        if 'manual_products' in st.session_state and st.session_state.manual_products:
            st.session_state.company_data = st.session_state.manual_products
    
    # Display current company data
    if st.session_state.company_data:
        st.subheader("📊 Current Product Portfolio")
        
        for idx, product in enumerate(st.session_state.company_data):
            with st.expander(f"📦 {product.get('name', f'Product {idx+1}')}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Description:** {product.get('description', 'N/A')}")
                    st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                with col2:
                    st.markdown(f"**Price Range:** {product.get('price_range', 'N/A')}")
                    st.markdown(f"**Target Audience:** {product.get('target_audience', 'N/A')}")
        
        st.info("✅ Company data loaded. Proceed to 'Scrape Captions' tab.")
    else:
        st.info("📤 Please upload company data or add products manually to continue.")

def scrape_captions_tab(apify_token: str):
    """Tab 2: Scrape Instagram captions"""
    st.header("2️⃣ Instagram Caption Scraping")
    st.markdown("Upload competitor post URLs and scrape captions using Apify.")
    
    if not st.session_state.company_data:
        st.warning("⚠️ Please upload company data first in Tab 1")
        return
    
    # File upload for competitor data
    competitor_file = st.file_uploader(
        "Upload competitor posts Excel file",
        type=['xlsx'],
        help="Excel file with columns: competitor_name, instagram_id, post_url"
    )
    
    if competitor_file:
        df = pd.read_excel(competitor_file)
        required_columns = ['competitor_name', 'instagram_id', 'post_url']
        
        if all(col in df.columns for col in required_columns):
            st.success(f"✅ Loaded {len(df)} posts from {len(df['competitor_name'].unique())} competitors")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df)
            
            # Scraping controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🚀 Scrape All Captions", type="primary"):
                    scrape_all_captions(df, apify_token)
            
            with col2:
                if st.button("🔄 Clear Results"):
                    st.session_state.scraped_captions = {}
                    st.rerun()
        else:
            st.error(f"❌ Missing required columns: {required_columns}")
    
    # Display scraping results
    if st.session_state.scraped_captions:
        st.subheader("📊 Scraping Results")
        
        total_posts = sum(len(posts) for posts in st.session_state.scraped_captions.values())
        successful_posts = sum(1 for posts in st.session_state.scraped_captions.values() 
                              for post in posts if post.get('caption'))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts", total_posts)
        col2.metric("Successful Scrapes", successful_posts)
        col3.metric("Success Rate", f"{(successful_posts/total_posts)*100:.1f}%")
        
        st.info("✅ Caption scraping completed. Proceed to '6-Parameter Analysis' tab.")

def scrape_all_captions(df: pd.DataFrame, apify_token: str):
    """Scrape all captions with progress tracking"""
    scraper = InstagramCaptionScraper(apify_token)
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        progress_bar.progress((idx + 1) / len(df))
        status_text.text(f"Scraping {idx + 1}/{len(df)}: {row['competitor_name']}")
        
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
    status_text.empty()

def six_parameter_analysis_tab(groq_api_key: str):
    """Tab 3: 6-parameter analysis with image uploads"""
    st.header("3️⃣ 6-Parameter Analysis")
    st.markdown("""
    **Analyze competitor posts based on 6 key parameters:**
    1. Color Palette & Visual Style
    2. Tone of Voice in Captions  
    3. CTA Presence & Strength
    4. Hashtag & Keyword Strategy
    5. Readability & Clarity
    6. Emotional Appeal & Imagery
    """)
    
    if not st.session_state.scraped_captions:
        st.warning("⚠️ Please scrape captions first in Tab 2")
        return
    
    # FIXED: Using CombinedAnalyzer instead of SixParameterAnalyzer
    analyzer = CombinedAnalyzer(groq_api_key)
    
    # Display posts for analysis
    for competitor, posts in st.session_state.scraped_captions.items():
        with st.expander(f"🏢 {competitor} ({len(posts)} posts)", expanded=True):
            
            for idx, post in enumerate(posts):
                st.markdown(f"### Post {idx + 1}")
                st.markdown(f"**URL:** {post['url']}")
                
                if post.get('caption'):
                    st.markdown(f"**Caption:** {post['caption'][:200]}...")
                    if post.get('hashtags'):
                        st.markdown(f"**Hashtags:** {', '.join(post['hashtags'][:5])}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Image upload
                    uploaded_file = st.file_uploader(
                        f"Upload image for {competitor} Post {idx + 1}",
                        type=['jpg', 'jpeg', 'png'],
                        key=f"img_{post.get('post_id', f'{competitor}_{idx}')}"
                    )
                    
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                        st.image(image, width=200)
                
                with col2:
                    post_id = post.get('post_id', f"{competitor}_{idx}")
                    
                    if st.button(f"🔍 Analyze 6 Parameters", key=f"analyze_{post_id}"):
                        if uploaded_file:
                            with st.spinner("Running 6-parameter analysis..."):
                                image = Image.open(uploaded_file)
                                
                                analysis = analyzer.analyze_post(
                                    post.get('caption', ''),
                                    image,
                                    competitor,
                                    post['url']
                                )
                                
                                st.session_state.post_analyses.append(analysis)
                                st.success(f"✅ Analysis completed! Overall Score: {analysis['overall_score']}/100")
                                
                                # Show quick results preview
                                with st.expander("📊 Analysis Preview", expanded=False):
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Color/Visual", f"{analysis['parameter_1_color_visual']['score']}/100")
                                    col2.metric("Tone of Voice", f"{analysis['parameter_2_tone_voice']['score']}/100") 
                                    col3.metric("CTA Strength", f"{analysis['parameter_3_cta']['score']}/100")
                        else:
                            st.error("Please upload an image first")
                
                st.divider()
    
    # Analysis summary
    if st.session_state.post_analyses:
        st.subheader("📈 Analysis Summary")
        total_analyses = len(st.session_state.post_analyses)
        avg_score = sum(analysis['overall_score'] for analysis in st.session_state.post_analyses) / total_analyses
        
        col1, col2 = st.columns(2)
        col1.metric("Posts Analyzed", total_analyses)
        col2.metric("Average Score", f"{avg_score:.1f}/100")
        
        st.info("✅ 6-parameter analysis completed. View detailed results in Tab 4, then proceed to Strategic Suggestions.")

def analysis_results_tab():
    """Tab 4: Display detailed analysis results"""
    st.header("4️⃣ Detailed Analysis Results")
    
    if not st.session_state.post_analyses:
        st.info("📊 Complete 6-parameter analysis first in Tab 3")
        return
    
    # Results overview
    st.subheader("📊 Analysis Overview")
    
    # Calculate parameter averages
    param_scores = {
        "Color & Visual": [],
        "Tone of Voice": [],
        "CTA Strength": [],
        "Hashtag Strategy": [],
        "Readability": [],
        "Emotional Appeal": []
    }
    
    for analysis in st.session_state.post_analyses:
        param_scores["Color & Visual"].append(analysis["parameter_1_color_visual"]["score"])
        param_scores["Tone of Voice"].append(analysis["parameter_2_tone_voice"]["score"])
        param_scores["CTA Strength"].append(analysis["parameter_3_cta"]["score"])
        param_scores["Hashtag Strategy"].append(analysis["parameter_4_hashtag_keywords"]["score"])
        param_scores["Readability"].append(analysis["parameter_5_readability"]["score"])
        param_scores["Emotional Appeal"].append(analysis["parameter_6_emotional_appeal"]["score"])
    
    # Display parameter averages
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Color & Visual", f"{sum(param_scores['Color & Visual'])/len(param_scores['Color & Visual']):.1f}/100")
        st.metric("Tone of Voice", f"{sum(param_scores['Tone of Voice'])/len(param_scores['Tone of Voice']):.1f}/100")
    
    with col2:
        st.metric("CTA Strength", f"{sum(param_scores['CTA Strength'])/len(param_scores['CTA Strength']):.1f}/100")
        st.metric("Hashtag Strategy", f"{sum(param_scores['Hashtag Strategy'])/len(param_scores['Hashtag Strategy']):.1f}/100")
    
    with col3:
        st.metric("Readability", f"{sum(param_scores['Readability'])/len(param_scores['Readability']):.1f}/100")
        st.metric("Emotional Appeal", f"{sum(param_scores['Emotional Appeal'])/len(param_scores['Emotional Appeal']):.1f}/100")
    
    # Detailed results per post
    st.subheader("📋 Post-by-Post Analysis")
    
    for idx, analysis in enumerate(st.session_state.post_analyses):
        with st.expander(f"📊 {analysis['competitor_name']} - Post Analysis {idx + 1}", expanded=False):
            
            # Overall score
            st.markdown(f"**Overall Score: {analysis['overall_score']}/100**")
            st.markdown(f"**Post URL:** {analysis['post_url']}")
            
            # Parameter breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Parameter Scores")
                st.markdown(f"• **Color & Visual:** {analysis['parameter_1_color_visual']['score']}/100")
                st.markdown(f"• **Tone of Voice:** {analysis['parameter_2_tone_voice']['score']}/100")
                st.markdown(f"• **CTA Strength:** {analysis['parameter_3_cta']['score']}/100")
                st.markdown(f"• **Hashtag Strategy:** {analysis['parameter_4_hashtag_keywords']['score']}/100")
                st.markdown(f"• **Readability:** {analysis['parameter_5_readability']['score']}/100")
                st.markdown(f"• **Emotional Appeal:** {analysis['parameter_6_emotional_appeal']['score']}/100")
            
            with col2:
                st.markdown("#### Key Insights")
