"""
Instagram Competitor Analysis Utilities
Focused on competitor intelligence and strategic recommendations
"""

import os
import re
import json
import hashlib
import requests
import numpy as np
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
import nltk
from groq import Groq

# Optional Apify client import
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    print("⚠️ Apify client not available. Install with: pip install apify-client")
    ApifyClient = None
    APIFY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class InstagramCaptionScraper:
    """Scraper focused on extracting Instagram captions via Apify"""
    
    def __init__(self, apify_token: str = None):
        self.apify_token = apify_token
        self.client = None
        
        if apify_token and APIFY_AVAILABLE:
            try:
                self.client = ApifyClient(apify_token)
                print("✅ Apify caption scraper initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Apify client: {e}")
    
    def scrape_post_caption(self, post_url: str) -> Dict[str, Any]:
        """Scrape caption from Instagram post URL"""
        if not self.client:
            return self._fallback_caption_scraping(post_url)
        
        try:
            print(f"🔍 Scraping caption from {post_url}")
            
            run_input = {
                "postUrls": [post_url],
                "resultsLimit": 1,
                "proxy": {"useApifyProxy": True}
            }
            
            run = self.client.actor("apify/instagram-post-scraper").call(run_input=run_input)
            dataset = self.client.dataset(run["defaultDatasetId"])
            
            for item in dataset.iterate_items():
                return self._parse_apify_result(item, post_url)
                
            return {
                "url": post_url,
                "caption": None,
                "hashtags": [],
                "timestamp": None,
                "errors": "No data returned from Apify"
            }
            
        except Exception as e:
            print(f"❌ Apify error: {str(e)}")
            return self._fallback_caption_scraping(post_url)
    
    def _parse_apify_result(self, item: Dict, original_url: str) -> Dict[str, Any]:
        """Parse Apify response for caption data"""
        try:
            caption = item.get('caption') or item.get('text') or item.get('description')
            hashtags = re.findall(r'#\w+', caption) if caption else []
            
            timestamp = None
            if item.get('timestamp'):
                try:
                    timestamp = datetime.fromtimestamp(item['timestamp']).isoformat()
                except:
                    timestamp = str(item['timestamp'])
            
            return {
                "url": original_url,
                "caption": caption,
                "hashtags": hashtags,
                "timestamp": timestamp,
                "errors": None
            }
            
        except Exception as e:
            return {
                "url": original_url,
                "caption": None,
                "hashtags": [],
                "timestamp": None,
                "errors": f"Parse error: {str(e)}"
            }
    
    def _fallback_caption_scraping(self, url: str) -> Dict[str, Any]:
        """Fallback scraping using requests"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            caption = self._extract_caption_from_html(response.text)
            hashtags = re.findall(r'#\w+', caption) if caption else []
            
            return {
                "url": url,
                "caption": caption,
                "hashtags": hashtags,
                "timestamp": None,
                "errors": None
            }
            
        except Exception as e:
            return {
                "url": url,
                "caption": None,
                "hashtags": [],
                "timestamp": None,
                "errors": f"Fallback failed: {str(e)}"
            }
    
    def _extract_caption_from_html(self, html: str) -> Optional[str]:
        """Extract caption from HTML"""
        patterns = [
            r'"caption":"([^"]*)"',
            r'<meta property="og:description" content="([^"]*)"',
            r'"edge_media_to_caption":\s*{"edges":\s*\[{"node":\s*{"text":\s*"([^"]*)"'
        ]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, html)
                if match:
                    caption = match.group(1)
                    if caption and len(caption.strip()) > 5:
                        try:
                            caption = caption.encode('utf-8').decode('unicode_escape')
                        except:
                            pass
                        return caption[:1000]
            except:
                continue
        return None

class CombinedAnalyzer:
    """Analyzer for competitor intelligence and strategic insights"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
    
    def analyze_competitor_post(self, caption: str, image: Image.Image, competitor_name: str, post_url: str) -> Dict[str, Any]:
        """Analyze single competitor post (caption + image)"""
        
        # Analyze caption
        caption_analysis = self._analyze_caption(caption)
        
        # Analyze image
        image_analysis = self._analyze_image(image)
        
        # Combined insights
        combined_insights = self._generate_combined_insights(caption_analysis, image_analysis)
        
        return {
            "competitor_name": competitor_name,
            "post_url": post_url,
            "caption_analysis": caption_analysis,
            "image_analysis": image_analysis,
            "combined_insights": combined_insights,
            "strategic_score": self._calculate_strategic_score(caption_analysis, image_analysis)
        }
    
    def _analyze_caption(self, caption: str) -> Dict[str, Any]:
        """Analyze caption for competitive intelligence"""
        if not caption:
            return {
                "themes": [],
                "sentiment": "neutral",
                "tone": "neutral",
                "cta_strength": "none",
                "hashtag_strategy": "basic"
            }
        
        # Theme detection
        themes = self._detect_themes(caption)
        
        # Sentiment analysis
        sentiment = "neutral"
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(caption)
            if scores['pos'] > 0.3:
                sentiment = "positive"
            elif scores['neg'] > 0.3:
                sentiment = "negative"
        
        # CTA detection
        cta_strength = self._detect_cta_strength(caption)
        
        # Hashtag strategy
        hashtags = re.findall(r'#\w+', caption)
        hashtag_strategy = self._analyze_hashtag_strategy(hashtags)
        
        return {
            "themes": themes,
            "sentiment": sentiment,
            "tone": self._detect_tone(caption),
            "cta_strength": cta_strength,
            "hashtag_strategy": hashtag_strategy,
            "hashtag_count": len(hashtags),
            "word_count": len(caption.split())
        }
    
    def _analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image for visual strategy insights"""
        try:
            # Color analysis
            colors = self._extract_colors(image)
            
            # Visual properties
            img_array = np.array(image)
            brightness = np.mean(img_array)
            
            # Color temperature
            mean_rgb = np.mean(img_array, axis=(0, 1))
            r, g, b = mean_rgb
            color_temp = "warm" if r > b else "cool"
            
            return {
                "dominant_colors": colors,
                "brightness_level": int(brightness),
                "color_temperature": color_temp,
                "visual_style": self._classify_visual_style(brightness),
                "color_strategy": self._analyze_color_strategy(colors)
            }
            
        except Exception as e:
            return {
                "dominant_colors": ["#000000"],
                "brightness_level": 128,
                "color_temperature": "neutral",
                "visual_style": "unknown",
                "color_strategy": "undefined"
            }
    
    def _extract_colors(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors"""
        try:
            image = image.resize((150, 150))
            data = np.array(image).reshape((-1, 3))
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(data)
            
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                colors.append(hex_color)
            
            return colors
        except:
            return ['#000000']
    
    def _detect_themes(self, caption: str) -> List[str]:
        """Detect content themes"""
        theme_keywords = {
            "lifestyle": ["life", "daily", "routine", "home"],
            "fashion": ["outfit", "style", "fashion", "wear"],
            "food": ["food", "recipe", "eat", "delicious"],
            "travel": ["travel", "trip", "vacation", "explore"],
            "fitness": ["workout", "fitness", "gym", "health"],
            "business": ["business", "work", "professional", "success"],
            "beauty": ["beauty", "makeup", "skincare", "glow"],
            "technology": ["tech", "digital", "app", "innovation"]
        }
        
        detected_themes = []
        caption_lower = caption.lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes[:3]  # Top 3 themes
    
    def _detect_tone(self, caption: str) -> str:
        """Detect communication tone"""
        caption_lower = caption.lower()
        
        if any(word in caption_lower for word in ["amazing", "incredible", "love", "excited"]):
            return "enthusiastic"
        elif any(word in caption_lower for word in ["professional", "expert", "industry"]):
            return "professional"
        elif any(word in caption_lower for word in ["hey", "guys", "lol", "fun"]):
            return "casual"
        else:
            return "neutral"
    
    def _detect_cta_strength(self, caption: str) -> str:
        """Detect call-to-action strength"""
        strong_ctas = ["click", "buy", "shop", "order", "subscribe", "follow", "sign up"]
        medium_ctas = ["check out", "see more", "learn", "discover"]
        
        caption_lower = caption.lower()
        
        if any(cta in caption_lower for cta in strong_ctas):
            return "strong"
        elif any(cta in caption_lower for cta in medium_ctas):
            return "medium"
        else:
            return "weak"
    
    def _analyze_hashtag_strategy(self, hashtags: List[str]) -> str:
        """Analyze hashtag strategy"""
        count = len(hashtags)
        
        if count >= 10:
            return "aggressive"
        elif count >= 5:
            return "moderate"
        elif count >= 2:
            return "minimal"
        else:
            return "none"
    
    def _classify_visual_style(self, brightness: float) -> str:
        """Classify visual style"""
        if brightness > 200:
            return "bright_modern"
        elif brightness < 80:
            return "dark_dramatic"
        else:
            return "balanced"
    
    def _analyze_color_strategy(self, colors: List[str]) -> str:
        """Analyze color strategy"""
        if len(colors) <= 2:
            return "minimalist"
        elif len(colors) >= 4:
            return "vibrant"
        else:
            return "balanced"
    
    def _generate_combined_insights(self, caption_analysis: Dict, image_analysis: Dict) -> List[str]:
        """Generate strategic insights from combined analysis"""
        insights = []
        
        # Theme-visual alignment
        themes = caption_analysis.get("themes", [])
        visual_style = image_analysis.get("visual_style", "")
        
        if "lifestyle" in themes and "bright" in visual_style:
            insights.append("Strong lifestyle branding with optimistic visual approach")
        
        # CTA-visual coherence
        cta_strength = caption_analysis.get("cta_strength", "weak")
        color_strategy = image_analysis.get("color_strategy", "balanced")
        
        if cta_strength == "strong" and color_strategy == "vibrant":
            insights.append("Aggressive conversion strategy with attention-grabbing visuals")
        
        # Engagement optimization
        if caption_analysis.get("hashtag_count", 0) > 5 and "bright" in visual_style:
            insights.append("High-engagement approach combining reach and visual appeal")
        
        return insights[:3]
    
    def _calculate_strategic_score(self, caption_analysis: Dict, image_analysis: Dict) -> int:
        """Calculate strategic effectiveness score"""
        score = 50  # Base score
        
        # Caption factors
        if caption_analysis.get("cta_strength") == "strong":
            score += 15
        if len(caption_analysis.get("themes", [])) >= 2:
            score += 10
        if caption_analysis.get("hashtag_count", 0) >= 5:
            score += 10
        
        # Visual factors
        if image_analysis.get("color_strategy") == "vibrant":
            score += 10
        if image_analysis.get("brightness_level", 0) > 150:
            score += 5
        
        return min(100, score)
    
    def generate_competitor_insights(self, competitor_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate overall competitor intelligence"""
        
        insights = {}
        
        for competitor, posts in competitor_data.items():
            if not posts:
                continue
            
            # Aggregate analysis
            total_score = sum(post.get("strategic_score", 0) for post in posts)
            avg_score = total_score / len(posts) if posts else 0
            
            # Common themes
            all_themes = []
            for post in posts:
                all_themes.extend(post.get("caption_analysis", {}).get("themes", []))
            
            from collections import Counter
            common_themes = [theme for theme, count in Counter(all_themes).most_common(3)]
            
            # Dominant strategies
            cta_strategies = [post.get("caption_analysis", {}).get("cta_strength", "weak") for post in posts]
            dominant_cta = Counter(cta_strategies).most_common(1)[0][0] if cta_strategies else "weak"
            
            insights[competitor] = {
                "avg_strategic_score": round(avg_score, 1),
                "post_count": len(posts),
                "common_themes": common_themes,
                "dominant_cta_strategy": dominant_cta,
                "content_consistency": "high" if len(set(all_themes)) <= 3 else "low"
            }
        
        return insights

def call_groq_model(prompt: str, system: str = None, max_tokens: int = 2048, groq_api_key: str = None) -> Dict[str, Any]:
    """Call Groq API for strategic analysis"""
    if not groq_api_key:
        return {"error": "GROQ API key not provided", "content": None}
    
    try:
        client = Groq(api_key=groq_api_key)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.1,
            max_completion_tokens=max_tokens,
            top_p=1,
            reasoning_effort="medium",
            stream=False
        )
        
        return {"content": completion.choices[0].message.content, "error": None}
        
    except Exception as e:
        return {"error": str(e), "content": None}

def generate_strategic_recommendations(competitor_insights: Dict, groq_api_key: str) -> Dict[str, Any]:
    """Generate strategic recommendations based on competitor analysis"""
    
    if not groq_api_key:
        return {"error": "API key required"}
    
    # Prepare competitor data for LLM
    competitor_summary = json.dumps(competitor_insights, indent=2)
    
    strategy_prompt = f"""Based on this competitor analysis, generate strategic recommendations:

{competitor_summary}

Provide 3 strategic approaches in JSON format:
{{
    "differentiation_strategy": {{
        "approach": "description",
        "content_pillars": ["pillar1", "pillar2", "pillar3"],
        "visual_direction": "description",
        "tone_strategy": "description"
    }},
    "competitive_advantage": {{
        "approach": "description", 
        "content_pillars": ["pillar1", "pillar2", "pillar3"],
        "visual_direction": "description",
        "tone_strategy": "description"
    }},
    "market_gap_strategy": {{
        "approach": "description",
        "content_pillars": ["pillar1", "pillar2", "pillar3"], 
        "visual_direction": "description",
        "tone_strategy": "description"
    }}
}}"""
    
    response = call_groq_model(
        prompt=strategy_prompt,
        system="You are a strategic social media consultant specializing in competitive differentiation.",
        groq_api_key=groq_api_key
    )
    
    try:
        if response.get("content"):
            return json.loads(response["content"])
    except:
        pass
    
    return {"error": "Could not generate recommendations"}

def generate_campaign_prompts(strategy: Dict, groq_api_key: str) -> Dict[str, Any]:
    """Generate image/video creation prompts based on strategy"""
    
    campaign_prompt = f"""Generate 5 campaign sets for this strategy:

Strategy: {json.dumps(strategy, indent=2)}

Return JSON with 5 campaign sets:
{{
    "campaign_sets": [
        {{
            "name": "Campaign Set 1",
            "theme": "specific theme",
            "image_prompt": "Detailed prompt for image generation (Midjourney/DALL-E style)",
            "video_prompt": "Detailed prompt for video creation", 
            "caption_template": "Template with placeholders for caption",
            "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"],
            "target_emotion": "emotion to evoke"
        }}
    ]
}}"""
    
    response = call_groq_model(
        prompt=campaign_prompt,
        system="You are a creative campaign director generating specific, actionable prompts for content creation.",
        groq_api_key=groq_api_key
    )
    
    try:
        if response.get("content"):
            return json.loads(response["content"])
    except:
        pass
    
    return {"error": "Could not generate campaign prompts"}
