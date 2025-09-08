"""
Instagram Competitor Analysis Utilities
Focused on 6-parameter analysis and product-specific recommendations
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
from collections import Counter

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
    """Scraper for Instagram captions via Apify"""
    
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

class SixParameterAnalyzer:
    """Analyzer based on the 6 core parameters for competitor analysis"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
    
    def analyze_post(self, caption: str, image: Image.Image, competitor_name: str, post_url: str) -> Dict[str, Any]:
        """Analyze single post based on 6 parameters"""
        
        # Parameter 1: Color Palette & Visual Style
        color_visual_analysis = self._analyze_color_palette_visual_style(image)
        
        # Parameter 2: Tone of Voice in Captions
        tone_analysis = self._analyze_tone_of_voice(caption)
        
        # Parameter 3: CTA Presence & Strength
        cta_analysis = self._analyze_cta_presence_strength(caption)
        
        # Parameter 4: Hashtag & Keyword Strategy
        hashtag_keyword_analysis = self._analyze_hashtag_keyword_strategy(caption)
        
        # Parameter 5: Readability & Clarity
        readability_analysis = self._analyze_readability_clarity(caption)
        
        # Parameter 6: Emotional Appeal & Imagery Use
        emotional_analysis = self._analyze_emotional_appeal_imagery(caption, image)
        
        return {
            "competitor_name": competitor_name,
            "post_url": post_url,
            "parameter_1_color_visual": color_visual_analysis,
            "parameter_2_tone_voice": tone_analysis,
            "parameter_3_cta": cta_analysis,
            "parameter_4_hashtag_keywords": hashtag_keyword_analysis,
            "parameter_5_readability": readability_analysis,
            "parameter_6_emotional_appeal": emotional_analysis,
            "overall_score": self._calculate_overall_score([
                color_visual_analysis, tone_analysis, cta_analysis,
                hashtag_keyword_analysis, readability_analysis, emotional_analysis
            ])
        }
    
    def _analyze_color_palette_visual_style(self, image: Image.Image) -> Dict[str, Any]:
        """Parameter 1: Color Palette & Visual Style Analysis"""
        try:
            # Extract dominant colors
            colors = self._extract_colors(image)
            
            # Analyze brightness and saturation
            img_array = np.array(image)
            brightness = np.mean(img_array)
            
            # Color temperature
            mean_rgb = np.mean(img_array, axis=(0, 1))
            r, g, b = mean_rgb
            
            # Determine tone
            tone = "bright" if brightness > 180 else "muted" if brightness < 100 else "balanced"
            
            # Determine luxury vs playful
            saturation = np.std(img_array)
            luxury_playful = "luxury" if saturation < 50 and brightness > 150 else "playful"
            
            # Style classification (simplified)
            style = "photo"  # Would need more sophisticated analysis for illustration/meme
            
            return {
                "dominant_colors": colors,
                "tone": tone,
                "luxury_playful": luxury_playful,
                "style": style,
                "brightness_score": int(brightness),
                "score": self._score_color_visual(colors, tone, luxury_playful)
            }
            
        except Exception as e:
            return {
                "dominant_colors": ["#000000"],
                "tone": "unknown",
                "luxury_playful": "unknown",
                "style": "unknown",
                "brightness_score": 0,
                "score": 0
            }
    
    def _analyze_tone_of_voice(self, caption: str) -> Dict[str, Any]:
        """Parameter 2: Tone of Voice Analysis"""
        if not caption:
            return {
                "primary_tone": "neutral",
                "secondary_tones": [],
                "intensity": 0,
                "score": 0
            }
        
        caption_lower = caption.lower()
        
        # Detect different tones
        tone_indicators = {
            "humor": ["lol", "haha", "funny", "joke", "hilarious", "😂", "🤣"],
            "inspiration": ["inspire", "motivate", "dream", "achieve", "believe", "✨", "💪"],
            "authority": ["expert", "professional", "proven", "research", "studies"],
            "casualness": ["hey", "guys", "tbh", "omg", "totally", "super"],
            "urgency": ["now", "hurry", "limited", "deadline", "quick", "⏰", "🔥"]
        }
        
        detected_tones = []
        for tone, indicators in tone_indicators.items():
            if any(indicator in caption_lower for indicator in indicators):
                detected_tones.append(tone)
        
        # Determine primary tone
        primary_tone = detected_tones[0] if detected_tones else "neutral"
        
        # Calculate intensity
        total_indicators = sum(1 for indicators in tone_indicators.values() 
                              for indicator in indicators if indicator in caption_lower)
        intensity = min(10, total_indicators)
        
        return {
            "primary_tone": primary_tone,
            "secondary_tones": detected_tones[1:3],
            "intensity": intensity,
            "score": self._score_tone(primary_tone, intensity)
        }
    
    def _analyze_cta_presence_strength(self, caption: str) -> Dict[str, Any]:
        """Parameter 3: CTA Presence & Strength Analysis"""
        if not caption:
            return {
                "cta_present": False,
                "cta_strength": "none",
                "cta_text": None,
                "score": 0
            }
        
        caption_lower = caption.lower()
        
        # Strong CTAs
        strong_ctas = [
            "shop now", "buy now", "order now", "click link", "swipe up",
            "don't miss out", "limited time", "get yours", "subscribe now"
        ]
        
        # Medium CTAs
        medium_ctas = [
            "check out", "see more", "learn more", "discover", "explore",
            "find out", "read more", "watch", "follow"
        ]
        
        # Weak/Implicit CTAs
        weak_ctas = [
            "link in bio", "thoughts?", "what do you think", "let me know",
            "comment below", "tag someone"
        ]
        
        cta_found = None
        strength = "none"
        
        for cta in strong_ctas:
            if cta in caption_lower:
                cta_found = cta
                strength = "strong"
                break
        
        if not cta_found:
            for cta in medium_ctas:
                if cta in caption_lower:
                    cta_found = cta
                    strength = "medium"
                    break
        
        if not cta_found:
            for cta in weak_ctas:
                if cta in caption_lower:
                    cta_found = cta
                    strength = "weak"
                    break
        
        return {
            "cta_present": cta_found is not None,
            "cta_strength": strength,
            "cta_text": cta_found,
            "score": self._score_cta(strength)
        }
    
    def _analyze_hashtag_keyword_strategy(self, caption: str) -> Dict[str, Any]:
        """Parameter 4: Hashtag & Keyword Strategy Analysis"""
        if not caption:
            return {
                "hashtag_count": 0,
                "hashtag_strategy": "none",
                "core_keywords": [],
                "content_themes": [],
                "score": 0
            }
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', caption)
        hashtag_count = len(hashtags)
        
        # Determine hashtag strategy
        if hashtag_count >= 15:
            hashtag_strategy = "aggressive"
        elif hashtag_count >= 8:
            hashtag_strategy = "moderate"
        elif hashtag_count >= 3:
            hashtag_strategy = "selective"
        elif hashtag_count >= 1:
            hashtag_strategy = "minimal"
        else:
            hashtag_strategy = "none"
        
        # Extract keywords (excluding hashtags and common words)
        words = re.findall(r'\b\w+\b', caption.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get most common keywords
        keyword_counts = Counter(keywords)
        core_keywords = [word for word, count in keyword_counts.most_common(5)]
        
        # Detect content themes
        themes = self._detect_content_themes(caption, hashtags)
        
        return {
            "hashtag_count": hashtag_count,
            "hashtag_strategy": hashtag_strategy,
            "core_keywords": core_keywords,
            "content_themes": themes,
            "hashtags": hashtags[:10],  # Limit for display
            "score": self._score_hashtag_keywords(hashtag_count, len(core_keywords), len(themes))
        }
    
    def _analyze_readability_clarity(self, caption: str) -> Dict[str, Any]:
        """Parameter 5: Readability & Clarity Analysis"""
        if not caption:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "readability_type": "none",
                "clarity_score": 0,
                "score": 0
            }
        
        words = caption.split()
        word_count = len(words)
        
        sentences = [s.strip() for s in caption.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Determine readability type
        if word_count <= 50:
            readability_type = "short_skimmable"
        elif word_count <= 150:
            readability_type = "medium_balanced"
        else:
            readability_type = "long_storytelling"
        
        # Calculate clarity score
        clarity_factors = 0
        if avg_sentence_length <= 15:  # Short sentences are clearer
            clarity_factors += 20
        if word_count <= 100:  # Concise is clearer
            clarity_factors += 20
        if sentence_count >= 2:  # Some structure
            clarity_factors += 10
        
        # Check for clear product benefits
        benefit_words = ["benefit", "feature", "advantage", "helps", "improves", "saves"]
        if any(word in caption.lower() for word in benefit_words):
            clarity_factors += 30
        
        clarity_score = min(100, clarity_factors)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "readability_type": readability_type,
            "clarity_score": clarity_score,
            "score": self._score_readability(word_count, clarity_score)
        }
    
    def _analyze_emotional_appeal_imagery(self, caption: str, image: Image.Image) -> Dict[str, Any]:
        """Parameter 6: Emotional Appeal & Imagery Analysis"""
        
        # Text sentiment analysis
        text_emotions = self._analyze_text_emotions(caption)
        
        # Image emotion analysis
        image_emotions = self._analyze_image_emotions(image)
        
        # Calculate alignment
        alignment_score = self._calculate_emotion_alignment(text_emotions, image_emotions)
        
        return {
            "text_emotions": text_emotions,
            "image_emotions": image_emotions,
            "alignment_score": alignment_score,
            "overall_emotional_impact": self._calculate_emotional_impact(text_emotions, image_emotions),
            "score": self._score_emotional_appeal(text_emotions, image_emotions, alignment_score)
        }
    
    def _extract_colors(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors using K-means clustering"""
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
    
    def _detect_content_themes(self, caption: str, hashtags: List[str]) -> List[str]:
        """Detect content themes from caption and hashtags"""
        all_text = (caption + ' ' + ' '.join(hashtags)).lower()
        
        theme_keywords = {
            "lifestyle": ["life", "daily", "routine", "home", "family", "weekend"],
            "fashion": ["outfit", "style", "fashion", "wear", "look", "trend"],
            "food": ["food", "recipe", "eat", "delicious", "taste", "cook", "meal"],
            "travel": ["travel", "trip", "vacation", "explore", "adventure", "destination"],
            "fitness": ["workout", "fitness", "gym", "health", "exercise", "training"],
            "beauty": ["beauty", "makeup", "skincare", "glow", "skin", "cosmetic"],
            "technology": ["tech", "digital", "app", "innovation", "gadget"],
            "business": ["business", "entrepreneur", "startup", "professional", "work"]
        }
        
        detected_themes = []
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes[:3]  # Top 3 themes
    
    def _analyze_text_emotions(self, caption: str) -> Dict[str, Any]:
        """Analyze emotions in text"""
        if not caption:
            return {"sentiment": "neutral", "emotions": [], "intensity": 0}
        
        # Sentiment analysis
        sentiment = "neutral"
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(caption)
            if scores['pos'] > 0.3:
                sentiment = "positive"
            elif scores['neg'] > 0.3:
                sentiment = "negative"
        
        # Emotion detection
        emotion_keywords = {
            "excitement": ["amazing", "incredible", "wow", "fantastic", "awesome"],
            "trust": ["trusted", "reliable", "proven", "guarantee", "secure"],
            "exclusivity": ["exclusive", "limited", "special", "vip", "premium"],
            "happiness": ["happy", "joy", "smile", "fun", "celebrate"],
            "urgency": ["hurry", "now", "quick", "limited", "deadline"]
        }
        
        detected_emotions = []
        caption_lower = caption.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        # Calculate intensity
        intensity = len(detected_emotions) * 2 + (len(re.findall(r'[!]{1,3}', caption)) * 1)
        intensity = min(10, intensity)
        
        return {
            "sentiment": sentiment,
            "emotions": detected_emotions,
            "intensity": intensity
        }
    
    def _analyze_image_emotions(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze emotions conveyed by image"""
        try:
            img_array = np.array(image)
            
            # Brightness indicates mood
            brightness = np.mean(img_array)
            
            # Color analysis for emotions
            colors = self._extract_colors(image, 3)
            
            # Simple emotion mapping based on brightness and colors
            if brightness > 180:
                mood = "bright_positive"
            elif brightness < 80:
                mood = "dark_dramatic"
            else:
                mood = "balanced"
            
            # Color psychology (simplified)
            warm_colors = 0
            cool_colors = 0
            for color in colors:
                try:
                    hex_color = color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    if r > b:
                        warm_colors += 1
                    else:
                        cool_colors += 1
                except:
                    continue
            
            color_emotion = "warm_inviting" if warm_colors > cool_colors else "cool_calming"
            
            return {
                "mood": mood,
                "color_emotion": color_emotion,
                "brightness_level": int(brightness),
                "emotional_indicators": [mood, color_emotion]
            }
            
        except:
            return {
                "mood": "unknown",
                "color_emotion": "unknown",
                "brightness_level": 128,
                "emotional_indicators": []
            }
    
    def _calculate_emotion_alignment(self, text_emotions: Dict, image_emotions: Dict) -> int:
        """Calculate alignment between text and image emotions"""
        text_sentiment = text_emotions.get("sentiment", "neutral")
        image_mood = image_emotions.get("mood", "unknown")
        
        alignment_score = 50  # Base score
        
        # Positive alignment
        if text_sentiment == "positive" and "positive" in image_mood:
            alignment_score += 30
        elif text_sentiment == "negative" and "dark" in image_mood:
            alignment_score += 20
        
        # Intensity alignment
        text_intensity = text_emotions.get("intensity", 0)
        image_brightness = image_emotions.get("brightness_level", 128)
        
        if text_intensity > 5 and image_brightness > 150:
            alignment_score += 20
        
        return min(100, alignment_score)
    
    def _calculate_emotional_impact(self, text_emotions: Dict, image_emotions: Dict) -> str:
        """Calculate overall emotional impact"""
        text_intensity = text_emotions.get("intensity", 0)
        image_indicators = len(image_emotions.get("emotional_indicators", []))
        
        total_impact = text_intensity + (image_indicators * 2)
        
        if total_impact >= 8:
            return "high"
        elif total_impact >= 4:
            return "medium"
        else:
            return "low"
    
    # Scoring functions for each parameter
    def _score_color_visual(self, colors: List[str], tone: str, luxury_playful: str) -> int:
        score = 50
        if len(colors) >= 3:
            score += 20
        if tone in ["bright", "luxury"]:
            score += 15
        return min(100, score)
    
    def _score_tone(self, primary_tone: str, intensity: int) -> int:
        score = 40
        if primary_tone != "neutral":
            score += 30
        score += min(30, intensity * 3)
        return min(100, score)
    
    def _score_cta(self, strength: str) -> int:
        scores = {"strong": 90, "medium": 70, "weak": 40, "none": 10}
        return scores.get(strength, 10)
    
    def _score_hashtag_keywords(self, hashtag_count: int, keyword_count: int, theme_count: int) -> int:
        score = 20
        if hashtag_count >= 5:
            score += 25
        if keyword_count >= 3:
            score += 25
        if theme_count >= 2:
            score += 30
        return min(100, score)
    
    def _score_readability(self, word_count: int, clarity_score: int) -> int:
        readability_score = clarity_score * 0.7
        if 50 <= word_count <= 150:  # Optimal length
            readability_score += 30
        return min(100, int(readability_score))
    
    def _score_emotional_appeal(self, text_emotions: Dict, image_emotions: Dict, alignment_score: int) -> int:
        text_intensity = text_emotions.get("intensity", 0)
        emotion_count = len(text_emotions.get("emotions", []))
        
        score = alignment_score * 0.4
        score += text_intensity * 3
        score += emotion_count * 5
        
        return min(100, int(score))
    
    def _calculate_overall_score(self, parameter_analyses: List[Dict]) -> int:
        """Calculate overall score from all 6 parameters"""
        scores = [analysis.get("score", 0) for analysis in parameter_analyses]
        return int(sum(scores) / len(scores)) if scores else 0

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

def generate_three_suggestion_sets(competitor_analyses: List[Dict], groq_api_key: str) -> Dict[str, Any]:
    """Generate 3 suggestion sets based on 6-parameter analysis"""
    
    if not groq_api_key:
        return {"error": "API key required"}
    
    # Aggregate parameter insights
    parameter_summary = _aggregate_parameter_insights(competitor_analyses)
    
    strategy_prompt = f"""Based on competitor analysis across 6 key parameters, generate 3 distinct strategic approaches:

Parameter Analysis Summary:
{json.dumps(parameter_summary, indent=2)}

Generate 3 suggestion sets focusing on different parameter combinations:

{{
    "suggestion_set_1": {{
        "name": "Strategy Name",
        "focus_parameters": ["parameter_1", "parameter_2"],
        "approach_description": "Clear description of this strategic approach",
        "color_visual_strategy": "specific visual direction",
        "tone_voice_strategy": "specific tone recommendation",
        "cta_strategy": "specific CTA approach",
        "hashtag_strategy": "specific hashtag approach",
        "readability_strategy": "specific readability approach", 
        "emotional_strategy": "specific emotional approach",
        "expected_impact": "high|medium|low"
    }},
    "suggestion_set_2": {{
        "name": "Strategy Name",
        "focus_parameters": ["parameter_3", "parameter_4"],
        "approach_description": "Clear description of this strategic approach",
        "color_visual_strategy": "specific visual direction",
        "tone_voice_strategy": "specific tone recommendation", 
        "cta_strategy": "specific CTA approach",
        "hashtag_strategy": "specific hashtag approach",
        "readability_strategy": "specific readability approach",
        "emotional_strategy": "specific emotional approach", 
        "expected_impact": "high|medium|low"
    }},
    "suggestion_set_3": {{
        "name": "Strategy Name", 
        "focus_parameters": ["parameter_5", "parameter_6"],
        "approach_description": "Clear description of this strategic approach",
        "color_visual_strategy": "specific visual direction",
        "tone_voice_strategy": "specific tone recommendation",
        "cta_strategy": "specific CTA approach", 
        "hashtag_strategy": "specific hashtag approach",
        "readability_strategy": "specific readability approach",
        "emotional_strategy": "specific emotional approach",
        "expected_impact": "high|medium|low"
    }}
}}"""
    
    response = call_groq_model(
        prompt=strategy_prompt,
        system="You are a strategic social media consultant specializing in parameter-based competitive differentiation for Instagram marketing.",
        groq_api_key=groq_api_key
    )
    
    try:
        if response.get("content"):
            return json.loads(response["content"])
    except:
        pass
    
    return {"error": "Could not generate suggestion sets"}

def _aggregate_parameter_insights(competitor_analyses: List[Dict]) -> Dict[str, Any]:
    """Aggregate insights from competitor analyses across all 6 parameters"""
    
    parameter_aggregation = {
        "color_visual_trends": {},
        "tone_voice_trends": {},
        "cta_trends": {},
        "hashtag_trends": {},
        "readability_trends": {},
        "emotional_trends": {}
    }
    
    for analysis in competitor_analyses:
        # Aggregate each parameter
        if "parameter_1_color_visual" in analysis:
            param = analysis["parameter_1_color_visual"]
            parameter_aggregation["color_visual_trends"]["tone"] = parameter_aggregation["color_visual_trends"].get("tone", {})
            tone = param.get("tone", "unknown")
            parameter_aggregation["color_visual_trends"]["tone"][tone] = parameter_aggregation["color_visual_trends"]["tone"].get(tone, 0) + 1
        
        # Similar aggregation for other parameters...
    
    return parameter_aggregation

def generate_product_specific_prompts(suggestion_set: Dict, company_products: List[Dict], groq_api_key: str) -> Dict[str, Any]:
    """Generate product-specific image and video prompts based on suggestion set"""
    
    product_prompts = {}
    
    for product in company_products:
        product_name = product.get("name", "Unknown Product")
        product_description = product.get("description", "")
        product_category = product.get("category", "")
        
        prompt_generation_request = f"""Generate specific image and video creation prompts for this product based on the strategic approach:

Product: {product_name}
Description: {product_description}
Category: {product_category}

Strategic Approach:
- Visual Strategy: {suggestion_set.get('color_visual_strategy', '')}
- Tone Strategy: {suggestion_set.get('tone_voice_strategy', '')}
- CTA Strategy: {suggestion_set.get('cta_strategy', '')}
- Emotional Strategy: {suggestion_set.get('emotional_strategy', '')}

Generate prompts in this JSON format:
{{
    "image_prompts": [
        "Detailed image generation prompt 1 for {product_name}",
        "Detailed image generation prompt 2 for {product_name}",
        "Detailed image generation prompt 3 for {product_name}"
    ],
    "video_prompts": [
        "Detailed video creation prompt 1 for {product_name}",
        "Detailed video creation prompt 2 for {product_name}",
        "Detailed video creation prompt 3 for {product_name}"
    ],
    "caption_templates": [
        "Caption template 1 with placeholders for {product_name}",
        "Caption template 2 with placeholders for {product_name}"
    ],
    "hashtag_suggestions": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"]
}}"""
        
        response = call_groq_model(
            prompt=prompt_generation_request,
            system="You are a creative content strategist generating specific, actionable prompts for product marketing content creation.",
            groq_api_key=groq_api_key
        )
        
        try:
            if response.get("content"):
                product_prompts[product_name] = json.loads(response["content"])
        except:
            product_prompts[product_name] = {"error": "Could not generate prompts"}
    
    return product_prompts
