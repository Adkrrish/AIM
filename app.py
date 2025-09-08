"""
Simplified Instagram Competitor Analysis Tool
Lightweight single-page application
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
from utils import SimpleCaptionScraper, SimpleAnalyzer, generate_simple_prompts

# Page config
st.set_page_config(
    page_title="Instagram Analysis - Simplified", 
    page_icon="📱",
    layout="wide"
)

# Get API keys
def get_api_keys():
    try:
        groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") 
        apify_token = st.secrets.get("APIFY_TOKEN") or os.getenv("APIFY_TOKEN")
        return groq_key, apify_token
    except:
        return os.getenv("GROQ_API_KEY"), os.getenv("APIFY_TOKEN")

def main():
    # Header
    st.title("📱 Instagram Competitor Analysis")
    st.markdown("**Simplified Workflow:** Upload Data → Scrape Captions → Analyze → Generate Prompts")
    
    # Initialize session state
    for key in ['company_data', 'scraped_data', 'analysis_results', 'prompts']:
        if key not in st.session_state:
            st.session_state[key] = []
    
    # API status
    groq_key, apify_token = get_api_keys()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Groq API" if groq_key else "❌ Groq API")
    with col2:
        st.success("✅ Apify API" if apify_token else "❌ Apify API")
    
    # Section 1: Company Data
    st.header("1️⃣ Company Products")
    
    company_file = st.file_uploader(
        "Upload product portfolio", 
        type=['json', 'csv', 'xlsx'],
        help="File with product names, descriptions, categories"
    )
    
    if company_file:
        try:
            if company_file.name.endswith('.json'):
                st.session_state.company_data = json.load(company_file)
            elif company_file.name.endswith('.csv'):
                df = pd.read_csv(company_file)
                st.session_state.company_data = df.to_dict('records')
            else:
                df = pd.read_excel(company_file)
                st.session_state.company_data = df.to_dict('records')
            
            st.success(f"✅ Loaded {len(st.session_state.company_data)} products")
            
            # Show first few products
            for i, product in enumerate(st.session_state.company_data[:3]):
                st.write(f"**{product.get('name', f'Product {i+1}')}:** {product.get('category', 'No category')}")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Manual product entry
    with st.expander("➕ Add Product Manually"):
        with st.form("add_product"):
            name = st.text_input("Product Name")
            category = st.selectbox("Category", ["Fashion", "Food", "Tech", "Beauty", "Lifestyle", "Other"])
            description = st.text_area("Description", height=100)
            
            if st.form_submit_button("Add Product"):
                if name:
                    new_product = {"name": name, "category": category, "description": description}
                    st.session_state.company_data.append(new_product)
                    st.success(f"Added {name}")
                    st.rerun()
    
    # Section 2: Competitor Data & Scraping
    st.header("2️⃣ Competitor Posts")
    
    competitor_file = st.file_uploader(
        "Upload competitor URLs",
        type=['xlsx'],
        help="Excel with columns: competitor_name, post_url"
    )
    
    if competitor_file:
        try:
            df = pd.read_excel(competitor_file)
            required_cols = ['competitor_name', 'post_url']
            
            if all(col in df.columns for col in required_cols):
                st.success(f"✅ Loaded {len(df)} competitor posts")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 Scrape Captions", type="primary"):
                    scraper = SimpleCaptionScraper(apify_token)
                    results = []
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for idx, row in df.iterrows():
                        progress.progress((idx + 1) / len(df))
                        status.text(f"Scraping {idx + 1}/{len(df)}: {row['competitor_name']}")
                        
                        result = scraper.scrape_caption(row['post_url'])
                        result['competitor_name'] = row['competitor_name']
                        results.append(result)
                    
                    st.session_state.scraped_data = results
                    progress.empty()
                    status.empty()
                    
                    successful = sum(1 for r in results if r['success'])
                    st.success(f"✅ Scraped {successful}/{len(results)} captions successfully")
                    
            else:
                st.error(f"Missing columns: {required_cols}")
                
        except Exception as e:
            st.error(f"Error loading competitor file: {e}")
    
    # Section 3: Results & Analysis
    if st.session_state.scraped_data:
        st.header("3️⃣ Scraped Results & Analysis")
        
        # Show scraped data
        with st.expander("📄 View Scraped Captions", expanded=False):
            for i, item in enumerate(st.session_state.scraped_data[:5]):
                if item['success']:
                    st.write(f"**{item['competitor_name']}:** {item['caption'][:150]}...")
                    st.write(f"Hashtags: {', '.join(item['hashtags'][:5])}")
                    st.divider()
        
        # Image upload and analysis
        st.subheader("📸 Upload Images for Analysis")
        
        # Select posts for analysis
        successful_posts = [item for item in st.session_state.scraped_data if item['success']]
        
        if successful_posts:
            selected_indices = st.multiselect(
                "Select posts to analyze (max 5)",
                range(min(5, len(successful_posts))),
                format_func=lambda x: f"{successful_posts[x]['competitor_name']} - {successful_posts[x]['caption'][:50]}...",
                max_selections=5
            )
            
            if selected_indices:
                analyzer = SimpleAnalyzer(groq_key)
                analyses = []
                
                for idx in selected_indices:
                    post = successful_posts[idx]
                    
                    st.write(f"**{post['competitor_name']}**")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        uploaded_img = st.file_uploader(
                            f"Image for {post['competitor_name']}",
                            type=['jpg', 'png'],
                            key=f"img_{idx}"
                        )
                        
                        if uploaded_img:
                            image = Image.open(uploaded_img)
                            st.image(image, width=200)
                    
                    with col2:
                        st.write(f"Caption: {post['caption'][:200]}...")
                        
                        if st.button(f"Analyze Post", key=f"analyze_{idx}"):
                            with st.spinner("Analyzing..."):
                                analysis = analyzer.analyze_post(
                                    post['caption'],
                                    Image.open(uploaded_img) if uploaded_img else None
                                )
                                
                                analyses.append(analysis)
                                
                                # Show results
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Score", f"{analysis['score']}/100")
                                col_b.metric("Words", analysis['word_count'])
                                col_c.metric("Hashtags", analysis['hashtag_count'])
                                
                                st.write(f"**Sentiment:** {analysis['sentiment']}")
                                st.write(f"**Themes:** {', '.join(analysis['themes']) or 'None detected'}")
                                st.write(f"**Has CTA:** {'Yes' if analysis['has_cta'] else 'No'}")
                                
                                # Color palette
                                if analysis['colors']:
                                    color_html = " ".join([
                                        f'<span style="background-color:{c}; padding:8px; margin:2px; display:inline-block; color:white; font-size:10px;">{c}</span>'
                                        for c in analysis['colors']
                                    ])
                                    st.markdown(f"**Colors:** {color_html}", unsafe_allow_html=True)
                    
                    st.divider()
                
                # Store analyses
                if analyses:
                    st.session_state.analysis_results = analyses
    
    # Section 4: Generate Prompts
    if st.session_state.analysis_results and st.session_state.company_data:
        st.header("4️⃣ Generate Content Prompts")
        
        if st.button("🎨 Generate Strategy & Prompts", type="primary"):
            with st.spinner("Generating strategic recommendations..."):
                prompts = generate_simple_prompts(
                    st.session_state.analysis_results,
                    st.session_state.company_data,
                    groq_key
                )
                
                st.session_state.prompts = prompts
                
                if not prompts.get('error'):
                    # Strategy overview
                    st.success("✅ Strategy Generated!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Strategy", prompts['strategy'].replace('_', ' ').title())
                        st.metric("Competitor Avg Score", f"{prompts['competitor_avg_score']}/100")
                    
                    with col2:
                        st.metric("Top Themes", ', '.join(prompts['top_competitor_themes']) or 'None')
                        st.metric("CTA Usage", prompts['cta_usage'])
                    
                    st.info(f"**Approach:** {prompts['approach']}")
                    
                    # Product-specific prompts
                    st.subheader("🎯 Product-Specific Prompts")
                    
                    for product_name, product_prompts in prompts['product_prompts'].items():
                        with st.expander(f"📦 {product_name} Content Prompts", expanded=True):
                            
                            st.markdown("#### 🖼️ Image Prompt")
                            st.code(product_prompts['image_prompt'])
                            
                            st.markdown("#### 🎬 Video Prompt")  
                            st.code(product_prompts['video_prompt'])
                            
                            st.markdown("#### ✍️ Caption Template")
                            st.code(product_prompts['caption_template'])
                            
                            st.markdown("#### #️⃣ Hashtags")
                            st.code(" ".join(product_prompts['hashtags']))
                            
                            st.markdown(f"**Strategy Applied:** {product_prompts['strategy_used'].replace('_', ' ').title()}")
                else:
                    st.error(f"Error: {prompts['error']}")
    
    # Progress indicator
    st.sidebar.header("📋 Progress")
    progress_items = [
        ("Company Data", bool(st.session_state.company_data)),
        ("Scraped Captions", bool(st.session_state.scraped_data)),
        ("Analysis Results", bool(st.session_state.analysis_results)),
        ("Generated Prompts", bool(st.session_state.prompts))
    ]
    
    for item, completed in progress_items:
        st.sidebar.write(f"{'✅' if completed else '⏳'} {item}")
    
    # Footer
    st.markdown("---")
    st.info("🚀 **Simplified Version:** Streamlined for better performance while maintaining core functionality.")

if __name__ == "__main__":
    main()
