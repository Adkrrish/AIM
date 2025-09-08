"""
Simplified Instagram Competitor Analysis Utilities
Lightweight version focusing on core functionality
"""

import os
import re
import json
import requests
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import nltk

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    ApifyClient = None

try:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    SentimentIntensityAnalyzer = None

class SimpleCaptionScraper:
    """Simplified caption scraper"""
    
    def __init__(self, apify_token: str = None):
        self.client = None
        if apify_token and APIFY_AVAILABLE:
            try:
                self.client = ApifyClient(apify_token)
            except:
                pass
    
    def scrape_caption(self, url: str) -> Dict:
        """Scrape caption from Instagram URL"""
        if self.client:
            try:
                run_input = {"postUrls": [url], "resultsLimit": 1}
                run = self.client.actor("apify/instagram-post-scraper").call(run_input=run_input)
                dataset = self.client.dataset(run["defaultDatasetId"])
                
                for item in dataset.iterate_items():
                    caption = item.get('caption') or item.get('text') or ""
                    hashtags = re.findall(r'#\w+', caption)
                    return {
                        "url": url,
                        "caption": caption,
                        "hashtags": hashtags,
                        "success": True
                    }
            except Exception as e:
                print(f"Apify error: {e}")
        
        # Fallback
        return self._fallback_scrape(url)
    
    def _fallback_scrape(self, url: str) -> Dict:
        """Basic fallback scraping"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible)'}
            response = requests.get(url, headers=headers, timeout=10)
            
            # Simple caption extraction
            patterns = [r'"caption":"([^"]+)"', r'<meta property="og:description" content="([^"]+)"']
            caption = ""
            
            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    caption = match.group(1)[:500]
                    break
            
            hashtags = re.findall(r'#\w+', caption)
            
            return {
                "url": url,
                "caption": caption,
                "hashtags": hashtags,
                "success": bool(caption)
            }
        except:
            return {"url": url, "caption": "", "hashtags": [], "success": False}

class SimpleAnalyzer:
    """Simplified post analyzer focusing on core parameters"""
    
    def __init__(self, groq_key: str = None):
        self.groq_key = groq_key
        self.sia = None
        if SentimentIntensityAnalyzer:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except:
                pass
    
    def analyze_post(self, caption: str, image: Image.Image = None) -> Dict:
        """Quick analysis of post"""
        
        # Caption analysis
        word_count = len(caption.split()) if caption else 0
        hashtag_count = len(re.findall(r'#\w+', caption))
        
        # CTA detection
        cta_words = ['shop', 'buy', 'click', 'visit', 'order', 'get', 'discover']
        has_cta = any(word in caption.lower() for word in cta_words)
        
        # Sentiment
        sentiment = "neutral"
        if self.sia and caption:
            scores = self.sia.polarity_scores(caption)
            if scores['pos'] > 0.3:
                sentiment = "positive"
            elif scores['neg'] > 0.3:
                sentiment = "negative"
        
        # Color analysis (if image provided)
        colors = ["#000000"]
        brightness = 128
        
        if image:
            try:
                img_resized = image.resize((100, 100))
                img_array = np.array(img_resized)
                brightness = int(np.mean(img_array))
                
                # Simple color extraction
                data = img_array.reshape((-1, 3))
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(data)
                
                colors = []
                for color in kmeans.cluster_centers_:
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(color[0]), int(color[1]), int(color[2])
                    )
                    colors.append(hex_color)
            except:
                pass
        
        # Overall score (simplified)
        score = 50
        if word_count > 20:
            score += 10
        if hashtag_count >= 3:
            score += 15
        if has_cta:
            score += 15
        if sentiment == "positive":
            score += 10
        
        return {
            "word_count": word_count,
            "hashtag_count": hashtag_count,
            "has_cta": has_cta,
            "sentiment": sentiment,
            "colors": colors[:3],
            "brightness": brightness,
            "score": min(100, score),
            "themes": self._detect_themes(caption)
        }
    
    def _detect_themes(self, caption: str) -> List[str]:
        """Simple theme detection"""
        if not caption:
            return []
        
        theme_words = {
            "lifestyle": ["life", "daily", "home", "family"],
            "fashion": ["style", "outfit", "wear", "fashion"],
            "food": ["food", "recipe", "delicious", "taste"],
            "fitness": ["workout", "gym", "health", "fit"],
            "travel": ["travel", "vacation", "explore", "trip"],
            "beauty": ["beauty", "makeup", "skin", "glow"]
        }
        
        detected = []
        caption_lower = caption.lower()
        
        for theme, words in theme_words.items():
            if any(word in caption_lower for word in words):
                detected.append(theme)
        
        return detected[:2]  # Max 2 themes

def generate_simple_prompts(analysis_results: List[Dict], products: List[Dict], groq_key: str = None) -> Dict:
    """Generate simplified content prompts"""
    
    if not products:
        return {"error": "No products provided"}
    
    # Analyze competitor patterns
    avg_score = sum(r.get('score', 0) for r in analysis_results) / max(len(analysis_results), 1)
    common_themes = []
    has_cta_ratio = sum(1 for r in analysis_results if r.get('has_cta', False)) / max(len(analysis_results), 1)
    
    for result in analysis_results:
        common_themes.extend(result.get('themes', []))
    
    theme_counts = Counter(common_themes)
    top_themes = [theme for theme, count in theme_counts.most_common(2)]
    
    # Generate strategy
    if avg_score > 70:
        strategy = "competitive_excellence"
        approach = "Match high-performing competitors with premium positioning"
    elif has_cta_ratio > 0.6:
        strategy = "conversion_focused"  
        approach = "Strong call-to-action emphasis for immediate action"
    else:
        strategy = "engagement_driven"
        approach = "Focus on storytelling and community engagement"
    
    # Generate product-specific prompts
    product_prompts = {}
    
    for product in products[:3]:  # Limit to 3 products for performance
        product_name = product.get('name', 'Product')
        category = product.get('category', 'General')
        
        # Simple prompt generation based on strategy
        if strategy == "competitive_excellence":
            image_prompt = f"Premium lifestyle shot of {product_name}, high-quality photography, luxury aesthetic, clean background"
            video_prompt = f"Professional product showcase of {product_name}, highlighting premium features and quality"
            caption_template = f"Introducing {product_name} - where quality meets innovation. ✨ #premium #{category.lower()}"
            
        elif strategy == "conversion_focused":
            image_prompt = f"Eye-catching product photo of {product_name} with clear benefits visible, call-to-action ready"
            video_prompt = f"Quick demo of {product_name} showing key benefits and ease of use"
            caption_template = f"Get your {product_name} today! Limited time offer. Shop now 👆 #sale #{category.lower()}"
            
        else:  # engagement_driven
            image_prompt = f"Lifestyle image showing {product_name} in everyday use, authentic and relatable"
            video_prompt = f"Behind-the-scenes or user-generated content style video featuring {product_name}"
            caption_template = f"How do you use your {product_name}? Share your story! 💬 #{category.lower()} #community"
        
        product_prompts[product_name] = {
            "image_prompt": image_prompt,
            "video_prompt": video_prompt,
            "caption_template": caption_template,
            "hashtags": [f"#{category.lower()}", f"#{product_name.lower().replace(' ', '')}", "#quality", "#innovation"],
            "strategy_used": strategy
        }
    
    return {
        "strategy": strategy,
        "approach": approach,
        "competitor_avg_score": round(avg_score, 1),
        "top_competitor_themes": top_themes,
        "cta_usage": f"{int(has_cta_ratio * 100)}%",
        "product_prompts": product_prompts
    }

def call_groq_simple(prompt: str, groq_key: str) -> str:
    """Simplified Groq API call"""
    if not groq_key or not Groq:
        return "API not available"
    
    try:
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except:
        return "API call failed"
