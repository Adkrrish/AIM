"""
Instagram Competitor Analysis Utilities
Core functions for data processing and analysis
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

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class InstagramAnalyzer:
    """Main class for Instagram post analysis"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
    
    def fetch_instagram_post(self, url: str) -> Dict[str, Any]:
        """
        Fetch Instagram post data from URL
        
        Args:
            url: Instagram post URL
            
        Returns:
            Dictionary containing post data and metadata
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract caption using regex patterns
            caption = self._extract_caption(html_content)
            
            # Extract image URL
            image_url = self._extract_image_url(html_content)
            
            # Extract timestamp
            timestamp = self._extract_timestamp(html_content)
            
            return {
                "url": url,
                "caption": caption,
                "image_url": image_url,
                "timestamp": timestamp,
                "errors": None
            }
            
        except Exception as e:
            return {
                "url": url,
                "caption": None,
                "image_url": None,
                "timestamp": None,
                "errors": str(e)
            }
    
    def _extract_caption(self, html: str) -> Optional[str]:
        """Extract caption from Instagram HTML"""
        patterns = [
            r'"caption":"([^"]*)"',
            r'<meta property="og:description" content="([^"]*)"',
            r'"accessibility_caption":"([^"]*)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                caption = match.group(1)
                # Decode unicode escapes
                try:
                    caption = caption.encode().decode('unicode_escape')
                except:
                    pass
                return caption[:1000]  # Limit length
        
        return None
    
    def _extract_image_url(self, html: str) -> Optional[str]:
        """Extract image URL from Instagram HTML"""
        patterns = [
            r'"display_url":"([^"]*)"',
            r'<meta property="og:image" content="([^"]*)"',
            r'"src":"([^"]*\.jpg[^"]*)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                url = match.group(1)
                # Decode unicode escapes
                try:
                    url = url.encode().decode('unicode_escape')
                except:
                    pass
                if url.startswith('http'):
                    return url
        
        return None
    
    def _extract_timestamp(self, html: str) -> Optional[str]:
        """Extract timestamp from Instagram HTML"""
        patterns = [
            r'"taken_at_timestamp":(\d+)',
            r'"date":"([^"]*)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                if pattern.endswith('(\\d+)'):  # Unix timestamp
                    timestamp = int(match.group(1))
                    return datetime.fromtimestamp(timestamp).isoformat()
                else:
                    return match.group(1)
        
        return None
    
    def download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def image_to_pil(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Convert image bytes to PIL Image"""
        try:
            from io import BytesIO
            return Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error converting to PIL: {e}")
            return None
    
    def extract_colors(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors using K-means clustering"""
        try:
            # Resize image for faster processing
            image = image.resize((150, 150))
            
            # Convert to numpy array
            data = np.array(image)
            data = data.reshape((-1, 3))
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(data)
            
            # Convert colors to hex
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                colors.append(hex_color)
            
            return colors
            
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return ['#000000']  # Fallback to black
    
    def analyze_caption_rules(self, caption: str) -> Dict[str, Any]:
        """Analyze caption using rule-based methods"""
        if not caption:
            return {
                "hashtags": [],
                "top_keywords": [],
                "word_count": 0,
                "cta_detected": False,
                "cta_text": None,
                "readability_score": 0
            }
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', caption)
        
        # Extract keywords (simple approach)
        words = re.findall(r'\b\w+\b', caption.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(keywords)
        top_keywords = [word for word, count in word_counts.most_common(5)]
        
        # Detect CTA
        cta_patterns = [
            r'\b(click|swipe|tap|visit|shop|buy|order|book|subscribe|follow|like|share|comment)\b',
            r'\b(link in bio|dm us|contact us|call now|order now)\b'
        ]
        
        cta_detected = False
        cta_text = None
        for pattern in cta_patterns:
            match = re.search(pattern, caption.lower())
            if match:
                cta_detected = True
                cta_text = match.group(0)
                break
        
        # Simple readability score (based on sentence and word length)
        sentences = caption.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        readability_score = max(0, 100 - (avg_sentence_length * 2))  # Simple formula
        
        return {
            "hashtags": hashtags,
            "top_keywords": top_keywords,
            "word_count": len(words),
            "cta_detected": cta_detected,
            "cta_text": cta_text,
            "readability_score": int(readability_score)
        }
    
    def compute_readability(self, caption: str) -> Dict[str, Any]:
        """Compute detailed readability metrics"""
        if not caption:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "skimmable": False,
                "score": 0,
                "evidence": "No caption provided"
            }
        
        sentences = [s.strip() for s in caption.split('.') if s.strip()]
        words = caption.split()
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Determine if skimmable (short sentences, good structure)
        skimmable = avg_sentence_length <= 15 and word_count <= 100
        
        # Score based on readability factors
        score = 100
        if avg_sentence_length > 20:
            score -= 30
        if word_count > 150:
            score -= 20
        if sentence_count < 2:
            score -= 10
        
        score = max(0, score)
        
        evidence = f"Average sentence length: {avg_sentence_length:.1f} words"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "skimmable": skimmable,
            "score": score,
            "evidence": evidence
        }
    
    def analyze_visual_emotions(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze visual emotions using simple heuristics - FIXED VERSION"""
        try:
            # Convert to numpy for basic analysis
            img_array = np.array(image)
            
            # Calculate basic color statistics
            mean_rgb = np.mean(img_array, axis=(0, 1))
            brightness = np.mean(mean_rgb)
            
            # Simple emotion classification based on color analysis
            emotions = []
            
            if brightness > 180:
                emotions.append("bright")
                emotions.append("energetic")
            elif brightness < 80:
                emotions.append("moody")
                emotions.append("dramatic")
            else:
                emotions.append("balanced")
            
            # Color-based emotions
            r, g, b = mean_rgb
            if r > g and r > b:
                emotions.append("warm")
            elif b > r and b > g:
                emotions.append("cool")
            elif g > r and g > b:
                emotions.append("natural")
            
            # Ensure emotions is never empty
            if not emotions:
                emotions.append("neutral")
            
            return {
                "visual_emotions": emotions[:3],  # Limit to 3, guaranteed to have at least one
                "brightness": int(brightness),
                "score": min(100, int(brightness * 0.5 + 25))  # Simple scoring
            }
            
        except Exception as e:
            print(f"Error in visual emotion analysis: {e}")
            # Always return a valid structure with default values
            return {
                "visual_emotions": ["neutral"],
                "brightness": 128,
                "score": 50
            }

def call_groq_model(prompt: str, system: str = None, max_tokens: int = 2048, groq_api_key: str = None) -> Dict[str, Any]:
    """
    Call Groq API for LLM analysis using the official Groq client
    
    Args:
        prompt: User prompt
        system: System message (optional)
        max_tokens: Maximum tokens to generate
        groq_api_key: API key for Groq
        
    Returns:
        Dictionary containing model response
    """
    if not groq_api_key:
        return {"error": "GROQ API key not provided", "content": None}
    
    try:
        # Initialize Groq client with API key
        client = Groq(api_key=groq_api_key)
        
        # Prepare messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Create completion
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.1,  # Lower temperature for more consistent results
            max_completion_tokens=max_tokens,
            top_p=1,
            reasoning_effort="medium",
            stream=False,  # Set to False for simpler handling
            stop=None
        )
        
        # Extract content from response
        content = completion.choices[0].message.content
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens if hasattr(completion, 'usage') and completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if hasattr(completion, 'usage') and completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if hasattr(completion, 'usage') and completion.usage else 0
            },
            "error": None
        }
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"error": str(e), "content": None}

# JSON Schemas for validation
POST_SCHEMA = {
    "type": "object",
    "properties": {
        "competitor_name": {"type": "string"},
        "instagram_id": {"type": "string"},
        "post_url": {"type": "string"},
        "timestamp": {"type": ["string", "null"]},
        "caption_text": {"type": ["string", "null"]},
        "image_url": {"type": ["string", "null"]},
        "image_hash": {"type": "string"},
        "analysis": {
            "type": "object",
            "properties": {
                "color_palette": {
                    "type": "object",
                    "properties": {
                        "dominant_hex": {"type": "array", "items": {"type": "string"}},
                        "tone": {"type": "string"},
                        "style": {"type": "string"},
                        "raw_score": {"type": "integer"},
                        "evidence": {"type": "string"}
                    }
                },
                "tone_of_voice": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "polarity": {"type": "string"},
                        "intensity": {"type": "integer"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "raw_score": {"type": "integer"}
                    }
                },
                "cta": {
                    "type": "object",
                    "properties": {
                        "presence": {"type": "string"},
                        "text": {"type": ["string", "null"]},
                        "strength": {"type": "string"},
                        "score": {"type": "integer"}
                    }
                },
                "hashtags_keywords": {
                    "type": "object",
                    "properties": {
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "top_keywords": {"type": "array", "items": {"type": "string"}},
                        "recommendation": {"type": "string"}
                    }
                },
                "readability": {
                    "type": "object",
                    "properties": {
                        "word_count": {"type": "integer"},
                        "skimmable": {"type": "boolean"},
                        "score": {"type": "integer"},
                        "evidence": {"type": "string"}
                    }
                },
                "emotional_imagery": {
                    "type": "object",
                    "properties": {
                        "visual_emotions": {"type": "array", "items": {"type": "string"}},
                        "text_emotion": {"type": "string"},
                        "alignment_score": {"type": "integer"},
                        "score": {"type": "integer"}
                    }
                }
            }
        },
        "errors": {"type": ["string", "null"]}
    }
}

def validate_post_data(post_data: Dict) -> bool:
    """Validate post data against schema"""
    try:
        jsonschema.validate(post_data, POST_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e}")
        return False

# LLM Prompt Templates
ANALYSIS_SYSTEM_PROMPT = """You are an expert social media analyst specializing in Instagram content analysis. 
Analyze the provided Instagram post data and return a structured JSON response following the exact schema provided.
Focus on actionable insights for competitive analysis and campaign planning.
Be precise and provide specific evidence for your assessments."""

ANALYSIS_USER_PROMPT = """Analyze this Instagram post data:

Caption: "{caption}"
Visual Description: {visual_description}
Colors: {colors}

Return a JSON object with this exact structure:
{{
    "tone_of_voice": {{
        "label": "professional|casual|playful|inspiring|authoritative",
        "polarity": "positive|negative|neutral", 
        "intensity": 1-10,
        "evidence": ["specific phrase 1", "specific phrase 2"],
        "raw_score": 1-100
    }},
    "cta": {{
        "presence": "strong|weak|none",
        "text": "exact CTA text or null",
        "strength": "direct|subtle|implied|none",
        "score": 1-100
    }},
    "hashtags_keywords": {{
        "hashtags": ["#tag1", "#tag2"],
        "top_keywords": ["keyword1", "keyword2"],
        "recommendation": "specific recommendation text"
    }},
    "emotional_imagery": {{
        "visual_emotions": ["emotion1", "emotion2", "emotion3"],
        "text_emotion": "dominant emotion from caption",
        "alignment_score": 1-100,
        "score": 1-100
    }},
    "color_palette": {{
        "tone": "warm|cool|neutral|vibrant|muted",
        "style": "minimalist|bold|elegant|playful|corporate",
        "raw_score": 1-100,
        "evidence": "specific color analysis"
    }}
}}"""

SUGGESTION_SYSTEM_PROMPT = """You are a strategic social media consultant creating competitive analysis reports.
Generate three distinct strategy recommendations (A, B, C) based on competitor analysis data.
Each strategy should have 6 actionable recommendations covering: color palette, tone of voice, CTA approach, hashtag strategy, readability optimization, and emotional appeal.
Include 3 campaign prompt sets per strategy."""

SUGGESTION_USER_PROMPT = """Based on this competitor analysis data:

{competitor_data}

Generate 3 strategic recommendations as JSON:
{{
    "strategies": {{
        "strategy_a": {{
            "name": "Strategy Name",
            "description": "Brief strategy description",
            "recommendations": [
                {{"parameter": "color_palette", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "tone_of_voice", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "cta", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "hashtags", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "readability", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "emotional_appeal", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}}
            ],
            "confidence_score": 1-100
        }},
        "strategy_b": {{
            "name": "Strategy Name",
            "description": "Brief strategy description",
            "recommendations": [
                {{"parameter": "color_palette", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "tone_of_voice", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "cta", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "hashtags", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "readability", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "emotional_appeal", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}}
            ],
            "confidence_score": 1-100
        }},
        "strategy_c": {{
            "name": "Strategy Name", 
            "description": "Brief strategy description",
            "recommendations": [
                {{"parameter": "color_palette", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "tone_of_voice", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "cta", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "hashtags", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "readability", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}},
                {{"parameter": "emotional_appeal", "action": "specific action", "rationale": "why this works", "kpi": "measurable outcome"}}
            ],
            "confidence_score": 1-100
        }}
    }}
}}"""

CAMPAIGN_PROMPT_SYSTEM = """You are a creative campaign prompt generator for social media marketing.
Generate specific, actionable prompts for image generation, video creation, hashtags, captions, and model-ready prompts.
Focus on brand alignment and competitor differentiation."""

CAMPAIGN_PROMPT_USER = """Generate 3 campaign prompt sets for this strategy:

Strategy: {strategy_name}
Key Actions: {key_actions}
Brand Colors: {colors}
Brand Context: {brand_context}

Return JSON with 3 campaign sets:
{{
    "campaign_sets": [
        {{
            "name": "Campaign Set 1",
            "image_prompt": "Detailed image generation prompt",
            "video_prompt": "Detailed short video prompt", 
            "hashtags": ["#tag1", "#tag2", "#tag3"],
            "caption_starter": "Engaging caption opening...",
            "model_ready_prompt": "Concise prompt for AI tools"
        }},
        {{
            "name": "Campaign Set 2",
            "image_prompt": "Detailed image generation prompt",
            "video_prompt": "Detailed short video prompt",
            "hashtags": ["#tag1", "#tag2", "#tag3"],
            "caption_starter": "Engaging caption opening...",
            "model_ready_prompt": "Concise prompt for AI tools"
        }},
        {{
            "name": "Campaign Set 3",
            "image_prompt": "Detailed image generation prompt",
            "video_prompt": "Detailed short video prompt",
            "hashtags": ["#tag1", "#tag2", "#tag3"],
            "caption_starter": "Engaging caption opening...",
            "model_ready_prompt": "Concise prompt for AI tools"
        }}
    ]
}}"""
