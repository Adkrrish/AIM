"""
Instagram Competitor Analysis Utilities
Enhanced for caption scraping + manual image upload approach
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
import time
import nltk
import jsonschema
from groq import Groq

# Optional Apify client import
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    print("⚠️ Apify client not available. Install with: pip install apify-client")
    ApifyClient = None
    APIFY_AVAILABLE = False

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class InstagramCaptionScraper:
    """Focused scraper for Instagram captions only via Apify"""
    
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
        """
        Scrape only the caption from an Instagram post
        
        Args:
            post_url: Instagram post URL
            
        Returns:
            Dictionary containing caption data
        """
        if not self.client:
            return self._fallback_caption_scraping(post_url)
        
        try:
            print(f"🔍 Scraping caption from {post_url} with Apify...")
            
            # Use Apify Instagram Post Scraper for caption extraction
            run_input = {
                "postUrls": [post_url],
                "resultsLimit": 1,
                "onlyPostDetails": True,  # Focus on post details only
                "proxy": {
                    "useApifyProxy": True
                }
            }
            
            # Run the Apify actor
            run = self.client.actor("apify/instagram-post-scraper").call(run_input=run_input)
            
            # Get results from the dataset
            dataset_id = run["defaultDatasetId"]
            dataset = self.client.dataset(dataset_id)
            
            # Fetch the first (and only) result
            for item in dataset.iterate_items():
                return self._parse_apify_caption_result(item, post_url)
                
            # If no results found
            return {
                "url": post_url,
                "caption": None,
                "timestamp": None,
                "hashtags": [],
                "mentions": [],
                "errors": "No caption data returned from Apify scraper"
            }
            
        except Exception as e:
            print(f"❌ Apify caption scraping error: {str(e)}")
            return self._fallback_caption_scraping(post_url)
    
    def scrape_multiple_captions(self, post_urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape captions from multiple posts"""
        results = []
        
        for url in post_urls:
            result = self.scrape_post_caption(url)
            results.append(result)
        
        return results
    
    def _parse_apify_caption_result(self, item: Dict, original_url: str) -> Dict[str, Any]:
        """Parse Apify result focusing on caption data"""
        try:
            # Extract caption
            caption = item.get('caption') or item.get('text') or item.get('description')
            
            # Extract hashtags from caption
            hashtags = re.findall(r'#\w+', caption) if caption else []
            
            # Extract mentions from caption
            mentions = re.findall(r'@\w+', caption) if caption else []
            
            # Extract timestamp
            timestamp = None
            timestamp_field = item.get('timestamp') or item.get('takenAt') or item.get('createdTime')
            if timestamp_field:
                try:
                    if isinstance(timestamp_field, str):
                        timestamp = timestamp_field
                    elif isinstance(timestamp_field, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp_field).isoformat()
                except:
                    timestamp = str(timestamp_field)
            
            return {
                "url": original_url,
                "caption": caption,
                "timestamp": timestamp,
                "hashtags": hashtags,
                "mentions": mentions,
                "errors": None,
                "apify_data": item
            }
            
        except Exception as e:
            print(f"❌ Error parsing Apify caption result: {str(e)}")
            return {
                "url": original_url,
                "caption": None,
                "timestamp": None,
                "hashtags": [],
                "mentions": [],
                "errors": f"Parsing error: {str(e)}"
            }
    
    def _fallback_caption_scraping(self, url: str) -> Dict[str, Any]:
        """Fallback method for caption scraping using requests"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract caption using regex patterns
            caption = self._extract_caption_from_html(html_content)
            
            # Extract hashtags and mentions
            hashtags = re.findall(r'#\w+', caption) if caption else []
            mentions = re.findall(r'@\w+', caption) if caption else []
            
            return {
                "url": url,
                "caption": caption,
                "timestamp": None,
                "hashtags": hashtags,
                "mentions": mentions,
                "errors": None
            }
            
        except Exception as e:
            return {
                "url": url,
                "caption": None,
                "timestamp": None,
                "hashtags": [],
                "mentions": [],
                "errors": f"Fallback scraping failed: {str(e)}"
            }
    
    def _extract_caption_from_html(self, html: str) -> Optional[str]:
        """Extract caption from HTML using regex"""
        patterns = [
            r'"caption":"([^"]*)"',
            r'<meta property="og:description" content="([^"]*)"',
            r'"accessibility_caption":"([^"]*)"',
            r'"edge_media_to_caption":\s*{"edges":\s*\[{"node":\s*{"text":\s*"([^"]*)"',
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
    """Analyzer that combines caption and image analysis"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
    
    def analyze_caption_theme(self, caption: str) -> Dict[str, Any]:
        """Analyze caption for themes using LLM"""
        if not caption or not self.groq_api_key:
            return {
                "themes": ["general"],
                "sentiment": "neutral",
                "tone": "neutral",
                "content_type": "unknown",
                "brand_alignment": "unknown"
            }
        
        # Use LLM for thematic analysis
        theme_prompt = f"""Analyze this Instagram caption for themes and content strategy:

Caption: "{caption}"

Provide analysis in JSON format:
{{
    "themes": ["theme1", "theme2"],
    "sentiment": "positive|negative|neutral",
    "tone": "professional|casual|playful|inspiring|promotional",
    "content_type": "educational|promotional|lifestyle|behind_scenes|user_generated",
    "brand_alignment": "strong|medium|weak",
    "target_audience": "millennials|gen_z|professionals|general",
    "engagement_potential": "high|medium|low",
    "content_pillars": ["pillar1", "pillar2"]
}}"""
        
        try:
            response = call_groq_model(
                prompt=theme_prompt,
                system="You are an expert social media strategist analyzing Instagram content for themes and brand alignment.",
                groq_api_key=self.groq_api_key
            )
            
            if response.get("content"):
                result = json.loads(response["content"])
                return result
        except:
            pass
        
        # Fallback to rule-based analysis
        return self._fallback_caption_analysis(caption)
    
    def _fallback_caption_analysis(self, caption: str) -> Dict[str, Any]:
        """Fallback caption analysis using rules"""
        themes = []
        
        # Theme detection keywords
        theme_keywords = {
            "lifestyle": ["life", "daily", "routine", "home", "family"],
            "fashion": ["outfit", "style", "fashion", "wear", "look"],
            "food": ["food", "recipe", "eat", "delicious", "taste", "cook"],
            "travel": ["travel", "trip", "vacation", "explore", "adventure"],
            "fitness": ["workout", "fitness", "gym", "health", "exercise"],
            "nature": ["nature", "outdoor", "sunset", "landscape", "beach"],
            "business": ["business", "work", "professional", "career", "success"]
        }
        
        caption_lower = caption.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                themes.append(theme)
        
        if not themes:
            themes = ["general"]
        
        # Sentiment analysis
        sentiment = "neutral"
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(caption)
            if scores['pos'] > 0.5:
                sentiment = "positive"
            elif scores['neg'] > 0.5:
                sentiment = "negative"
        
        return {
            "themes": themes[:3],  # Top 3 themes
            "sentiment": sentiment,
            "tone": "casual",
            "content_type": "lifestyle",
            "brand_alignment": "medium",
            "target_audience": "general",
            "engagement_potential": "medium",
            "content_pillars": themes[:2]
        }
    
    def analyze_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze uploaded image for visual properties"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Color analysis
            colors = self._extract_dominant_colors(image)
            
            # Brightness analysis
            mean_rgb = np.mean(img_array, axis=(0, 1))
            brightness = np.mean(mean_rgb)
            
            # Color temperature analysis
            r, g, b = mean_rgb
            color_temperature = "warm" if r > b else "cool"
            
            # Visual style analysis
            visual_style = self._analyze_visual_style(img_array, brightness)
            
            # Emotional tone from colors
            emotional_tone = self._derive_emotional_tone(colors, brightness)
            
            return {
                "dominant_colors": colors,
                "brightness_level": int(brightness),
                "color_temperature": color_temperature,
                "visual_style": visual_style,
                "emotional_tone": emotional_tone,
                "composition_type": "unknown",  # Would need more sophisticated analysis
                "brand_consistency": "medium"
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                "dominant_colors": ["#000000"],
                "brightness_level": 128,
                "color_temperature": "neutral",
                "visual_style": "unknown",
                "emotional_tone": "neutral",
                "composition_type": "unknown",
                "brand_consistency": "unknown"
            }
    
    def _extract_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors using K-means clustering"""
        try:
            # Resize for faster processing
            image = image.resize((150, 150))
            data = np.array(image).reshape((-1, 3))
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(data)
            
            # Convert to hex colors
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                colors.append(hex_color)
            
            return colors
            
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return ['#000000']
    
    def _analyze_visual_style(self, img_array: np.ndarray, brightness: float) -> str:
        """Analyze visual style based on image properties"""
        if brightness > 200:
            return "bright_airy"
        elif brightness < 50:
            return "dark_moody"
        elif brightness > 150:
            return "clean_modern"
        else:
            return "balanced_natural"
    
    def _derive_emotional_tone(self, colors: List[str], brightness: float) -> str:
        """Derive emotional tone from colors and brightness"""
        # Simple color psychology mapping
        warm_colors = 0
        cool_colors = 0
        
        for color in colors[:3]:  # Check top 3 colors
            try:
                # Convert hex to RGB
                hex_color = color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                
                if r > b:  # More red than blue = warmer
                    warm_colors += 1
                else:
                    cool_colors += 1
            except:
                continue
        
        if brightness > 180:
            if warm_colors > cool_colors:
                return "energetic_positive"
            else:
                return "calm_serene"
        elif brightness < 80:
            if warm_colors > cool_colors:
                return "intimate_cozy"
            else:
                return "mysterious_dramatic"
        else:
            return "balanced_natural"
    
    def combine_analysis(self, caption_analysis: Dict, image_analysis: Dict, 
                        caption: str, competitor_name: str) -> Dict[str, Any]:
        """Combine caption and image analysis for comprehensive insights"""
        
        # Alignment analysis
        alignment_score = self._calculate_alignment_score(caption_analysis, image_analysis)
        
        # Brand consistency analysis
        brand_consistency = self._analyze_brand_consistency(caption_analysis, image_analysis)
        
        # Content strategy insights
        strategy_insights = self._generate_strategy_insights(caption_analysis, image_analysis)
        
        # Recommendations
        recommendations = self._generate_recommendations(caption_analysis, image_analysis, alignment_score)
        
        return {
            "competitor_name": competitor_name,
            "caption_analysis": caption_analysis,
            "image_analysis": image_analysis,
            "alignment_score": alignment_score,
            "brand_consistency": brand_consistency,
            "strategy_insights": strategy_insights,
            "recommendations": recommendations,
            "overall_score": self._calculate_overall_score(caption_analysis, image_analysis, alignment_score)
        }
    
    def _calculate_alignment_score(self, caption_analysis: Dict, image_analysis: Dict) -> int:
        """Calculate alignment between caption and image"""
        score = 50  # Base score
        
        # Sentiment and emotional tone alignment
        caption_sentiment = caption_analysis.get("sentiment", "neutral")
        image_tone = image_analysis.get("emotional_tone", "neutral")
        
        if caption_sentiment == "positive" and "positive" in image_tone:
            score += 20
        elif caption_sentiment == "negative" and "dark" in image_tone:
            score += 15
        
        # Theme and visual style alignment
        themes = caption_analysis.get("themes", [])
        visual_style = image_analysis.get("visual_style", "")
        
        if "lifestyle" in themes and "clean" in visual_style:
            score += 15
        elif "nature" in themes and "natural" in visual_style:
            score += 15
        
        return min(100, max(0, score))
    
    def _analyze_brand_consistency(self, caption_analysis: Dict, image_analysis: Dict) -> str:
        """Analyze overall brand consistency"""
        caption_tone = caption_analysis.get("tone", "neutral")
        image_style = image_analysis.get("visual_style", "unknown")
        
        if caption_tone == "professional" and "clean" in image_style:
            return "high"
        elif caption_tone == "casual" and "natural" in image_style:
            return "high"
        else:
            return "medium"
    
    def _generate_strategy_insights(self, caption_analysis: Dict, image_analysis: Dict) -> List[str]:
        """Generate strategic insights"""
        insights = []
        
        # Content type insights
        content_type = caption_analysis.get("content_type", "unknown")
        if content_type == "educational":
            insights.append("Educational content strategy - focus on value-driven messaging")
        elif content_type == "lifestyle":
            insights.append("Lifestyle content strategy - emphasize authenticity and relatability")
        
        # Visual insights
        visual_style = image_analysis.get("visual_style", "unknown")
        if "bright" in visual_style:
            insights.append("Bright, optimistic visual approach appeals to positive engagement")
        elif "dark" in visual_style:
            insights.append("Moody aesthetic creates emotional depth and sophistication")
        
        # Engagement potential
        engagement = caption_analysis.get("engagement_potential", "medium")
        if engagement == "high":
            insights.append("High engagement potential - strong call-to-action and relatable content")
        
        return insights[:3]  # Return top 3 insights
    
    def _generate_recommendations(self, caption_analysis: Dict, image_analysis: Dict, alignment_score: int) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if alignment_score < 70:
            recommendations.append("Improve alignment between caption tone and visual mood")
        
        # Caption recommendations
        if caption_analysis.get("engagement_potential") == "low":
            recommendations.append("Add more engaging CTAs and interactive elements to captions")
        
        # Visual recommendations
        brightness = image_analysis.get("brightness_level", 128)
        if brightness < 100:
            recommendations.append("Consider brighter, more vibrant visuals for better engagement")
        
        # Theme consistency
        themes = caption_analysis.get("themes", [])
        if len(themes) > 2:
            recommendations.append("Focus on 1-2 core themes per post for clearer messaging")
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def _calculate_overall_score(self, caption_analysis: Dict, image_analysis: Dict, alignment_score: int) -> int:
        """Calculate overall content quality score"""
        # Weighted scoring
        caption_score = 70 if caption_analysis.get("engagement_potential") == "high" else 50
        image_score = 70 if image_analysis.get("emotional_tone") != "neutral" else 50
        
        # Calculate weighted average
        overall_score = int(
            (caption_score * 0.4) + 
            (image_score * 0.3) + 
            (alignment_score * 0.3)
        )
        
        return min(100, max(0, overall_score))

def call_groq_model(prompt: str, system: str = None, max_tokens: int = 2048, groq_api_key: str = None) -> Dict[str, Any]:
    """Call Groq API for LLM analysis"""
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
            stream=False,
            stop=None
        )
        
        content = completion.choices[0].message.content
        return {"content": content, "error": None}
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"error": str(e), "content": None}

# Keep existing prompt templates for compatibility
ANALYSIS_SYSTEM_PROMPT = """You are an expert social media analyst specializing in Instagram content analysis."""

ANALYSIS_USER_PROMPT = """Analyze this Instagram post data and provide strategic insights."""

SUGGESTION_SYSTEM_PROMPT = """You are a strategic social media consultant creating competitive analysis reports."""

SUGGESTION_USER_PROMPT = """Based on competitor analysis data, generate strategic recommendations."""

CAMPAIGN_PROMPT_SYSTEM = """You are a creative campaign prompt generator for social media marketing."""

CAMPAIGN_PROMPT_USER = """Generate campaign prompts based on strategic recommendations."""
