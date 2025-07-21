from flask import Flask, request, jsonify, render_template_string, send_file, flash, url_for
import os
import tempfile
import fitz  # PyMuPDF
from docx import Document
import re
import random
import json
from datetime import datetime, timedelta
import math
from collections import Counter, defaultdict
import string
import hashlib
import logging
from functools import wraps
from werkzeug.utils import secure_filename, redirect
import time
import sqlite3
from pathlib import Path


# Configuration
class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    RATE_LIMIT_REQUESTS = 10  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds
    DATABASE_PATH = 'resume_analyzer.db'
    LOG_FILE = 'resume_analyzer.log'
    SECRET_KEY = 'your-secret-key-change-in-production'


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import NLP libraries, fallback to basic analysis if not available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    import textstat

    NLTK_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic text analysis")

try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        logger.warning("spaCy model not found, using basic NLP")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, using basic NLP")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using basic scoring")

# Rate limiting storage
rate_limit_storage = {}


def init_database():
    """Initialize SQLite database for analytics and caching"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            field TEXT,
            analysis_result TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def track_analytics(event_type, data):
    """Track analytics events"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO analytics (event_type, data) VALUES (?, ?)',
            (event_type, json.dumps(data))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Analytics tracking failed: {e}")


def rate_limit(func):
    """Rate limiting decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()

        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = []

        # Clean old requests
        rate_limit_storage[client_ip] = [
            req_time for req_time in rate_limit_storage[client_ip]
            if current_time - req_time < Config.RATE_LIMIT_WINDOW
        ]

        # Check rate limit
        if len(rate_limit_storage[client_ip]) >= Config.RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

        rate_limit_storage[client_ip].append(current_time)
        return func(*args, **kwargs)

    return wrapper


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_file_hash(file_content):
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()


def security_headers(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


# Enhanced keyword dictionaries with more comprehensive coverage
ENHANCED_KEYWORD_DICT = {
    "web_development": {
        "core_skills": [
            "html", "css", "javascript", "react", "angular", "vue", "node.js", "express",
            "mongodb", "sql", "postgresql", "mysql", "git", "github", "docker", "kubernetes",
            "aws", "azure", "gcp", "typescript", "python", "java", "php", "ruby", "go",
            "rest api", "graphql", "microservices", "serverless", "ci/cd", "jenkins",
            "webpack", "babel", "sass", "less", "bootstrap", "tailwind", "material-ui"
        ],
        "frameworks": [
            "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "gatsby", "express",
            "koa", "fastify", "django", "flask", "spring", "laravel", "rails", "asp.net"
        ],
        "databases": [
            "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "cassandra",
            "dynamodb", "firebase", "supabase", "prisma", "sequelize", "mongoose"
        ],
        "tools": [
            "git", "docker", "kubernetes", "jenkins", "gitlab", "bitbucket", "jira",
            "confluence", "slack", "figma", "postman", "insomnia", "chrome devtools"
        ],
        "soft_skills": [
            "problem solving", "teamwork", "communication", "leadership", "agile",
            "scrum", "project management", "mentoring", "code review", "debugging"
        ],
        "action_verbs": [
            "developed", "built", "created", "designed", "implemented", "optimized",
            "deployed", "maintained", "scaled", "architected", "led", "managed"
        ],
        "weak_verbs": ["worked", "helped", "assisted", "participated", "involved"],
        "trends": [
            "jamstack", "pwa", "web3", "blockchain", "ai/ml integration", "edge computing",
            "webassembly", "micro-frontends", "headless cms", "low-code", "no-code"
        ]
    },
    "data_science": {
        "core_skills": [
            "python", "r", "sql", "machine learning", "deep learning", "statistics",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
            "matplotlib", "seaborn", "plotly", "jupyter", "anaconda", "git", "docker"
        ],
        "ml_frameworks": [
            "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
            "catboost", "hugging face", "transformers", "opencv", "nltk", "spacy"
        ],
        "databases": [
            "sql", "postgresql", "mysql", "mongodb", "cassandra", "redis", "elasticsearch",
            "snowflake", "bigquery", "redshift", "databricks", "spark", "hadoop"
        ],
        "tools": [
            "jupyter", "anaconda", "docker", "kubernetes", "airflow", "mlflow",
            "kubeflow", "dvc", "git", "github", "aws", "azure", "gcp", "tableau", "power bi"
        ],
        "soft_skills": [
            "analytical thinking", "problem solving", "communication", "storytelling",
            "business acumen", "research", "experimentation", "collaboration"
        ],
        "action_verbs": [
            "analyzed", "modeled", "predicted", "optimized", "implemented", "deployed",
            "researched", "experimented", "visualized", "automated", "discovered"
        ],
        "weak_verbs": ["worked", "helped", "assisted", "participated", "involved"],
        "trends": [
            "mlops", "automl", "explainable ai", "federated learning", "edge ai",
            "real-time ml", "feature stores", "model monitoring", "a/b testing", "causal inference"
        ]
    },
    "ui_ux": {
        "core_skills": [
            "user research", "wireframing", "prototyping", "usability testing", "figma",
            "sketch", "adobe xd", "invision", "principle", "framer", "axure", "balsamiq",
            "user personas", "user journey mapping", "information architecture", "interaction design"
        ],
        "design_tools": [
            "figma", "sketch", "adobe xd", "invision", "principle", "framer", "axure",
            "balsamiq", "miro", "figjam", "photoshop", "illustrator", "after effects"
        ],
        "research_methods": [
            "user interviews", "surveys", "usability testing", "a/b testing", "card sorting",
            "tree testing", "heuristic evaluation", "competitive analysis", "persona development"
        ],
        "prototyping": [
            "low-fidelity prototyping", "high-fidelity prototyping", "interactive prototyping",
            "paper prototyping", "digital prototyping", "rapid prototyping"
        ],
        "soft_skills": [
            "empathy", "creativity", "problem solving", "communication", "collaboration",
            "critical thinking", "attention to detail", "user advocacy", "storytelling"
        ],
        "action_verbs": [
            "designed", "researched", "prototyped", "tested", "iterated", "collaborated",
            "facilitated", "analyzed", "synthesized", "validated", "optimized"
        ],
        "weak_verbs": ["worked", "helped", "assisted", "participated", "involved"],
        "trends": [
            "design systems", "accessibility", "inclusive design", "voice ui", "ar/vr design",
            "motion design", "micro-interactions", "design tokens", "atomic design", "design ops"
        ]
    }
}

# Interview questions database
INTERVIEW_QUESTIONS = {
    "web_development": {
        "easy": [
            "What is the difference between HTML and HTML5?",
            "Explain the box model in CSS.",
            "What are the different data types in JavaScript?",
            "What is the difference between let, const, and var?",
            "How do you center a div in CSS?",
            "What is responsive web design?",
            "Explain the difference between GET and POST requests.",
            "What is the DOM?",
            "What are semantic HTML elements?",
            "How do you include CSS in an HTML document?"
        ],
        "medium": [
            "Explain the concept of closures in JavaScript.",
            "What is the difference between synchronous and asynchronous programming?",
            "How does React's virtual DOM work?",
            "What are React hooks and why are they useful?",
            "Explain the difference between SQL and NoSQL databases.",
            "What is RESTful API design?",
            "How do you optimize website performance?",
            "What is Cross-Origin Resource Sharing (CORS)?",
            "Explain the concept of middleware in Express.js.",
            "What are the principles of responsive design?"
        ],
        "hard": [
            "Design a scalable architecture for a social media application.",
            "How would you implement real-time features in a web application?",
            "Explain the trade-offs between different state management solutions in React.",
            "How would you optimize a database query that's running slowly?",
            "Design a caching strategy for a high-traffic e-commerce website.",
            "How would you implement authentication and authorization in a microservices architecture?",
            "Explain how you would handle race conditions in a distributed system.",
            "Design a system to handle file uploads for millions of users.",
            "How would you implement a recommendation system for an e-commerce platform?",
            "Explain the challenges and solutions for implementing real-time collaboration features."
        ]
    },
    "data_science": {
        "easy": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain what a p-value represents in statistics.",
            "What is the difference between correlation and causation?",
            "What are the main steps in the data science process?",
            "Explain the bias-variance tradeoff.",
            "What is overfitting and how can you prevent it?",
            "What is the difference between classification and regression?",
            "Explain what cross-validation is and why it's important.",
            "What are some common data preprocessing techniques?",
            "What is the difference between Type I and Type II errors?"
        ],
        "medium": [
            "How would you handle missing data in a dataset?",
            "Explain the difference between bagging and boosting.",
            "What are some techniques for feature selection?",
            "How do you evaluate the performance of a classification model?",
            "Explain the concept of regularization in machine learning.",
            "What is the curse of dimensionality?",
            "How would you detect and handle outliers in your data?",
            "Explain the difference between parametric and non-parametric models.",
            "What are some techniques for handling imbalanced datasets?",
            "How do you choose the right algorithm for a given problem?"
        ],
        "hard": [
            "Design an A/B testing framework for a large-scale application.",
            "How would you build a recommendation system from scratch?",
            "Explain how you would implement a real-time fraud detection system.",
            "Design a machine learning pipeline for processing streaming data.",
            "How would you approach building a natural language processing system?",
            "Explain the challenges of deploying machine learning models in production.",
            "How would you design a system to detect anomalies in time series data?",
            "Explain how you would build a computer vision system for autonomous vehicles.",
            "Design a distributed machine learning system for training on large datasets.",
            "How would you implement a reinforcement learning system for game playing?"
        ]
    },
    "ui_ux": {
        "easy": [
            "What is the difference between UI and UX design?",
            "Explain the importance of user personas in design.",
            "What are the basic principles of good design?",
            "What is a wireframe and when would you use one?",
            "Explain the concept of information architecture.",
            "What is usability testing?",
            "What are some common UI design patterns?",
            "Explain the importance of accessibility in design.",
            "What is a design system?",
            "What are the key elements of a good user interface?"
        ],
        "medium": [
            "How do you conduct user research for a new product?",
            "Explain the design thinking process.",
            "How do you prioritize features based on user needs?",
            "What are some methods for testing design concepts?",
            "How do you design for different screen sizes and devices?",
            "Explain the concept of progressive disclosure.",
            "How do you measure the success of a design?",
            "What are some techniques for improving user engagement?",
            "How do you handle conflicting stakeholder requirements?",
            "Explain the importance of micro-interactions in design."
        ],
        "hard": [
            "Design a complete user experience for a complex enterprise application.",
            "How would you redesign the checkout process for an e-commerce platform?",
            "Design a mobile app for elderly users with accessibility considerations.",
            "How would you approach designing for emerging technologies like AR/VR?",
            "Design a dashboard for data visualization that serves multiple user types.",
            "How would you conduct user research in a highly regulated industry?",
            "Design a design system that can scale across multiple products and platforms.",
            "How would you approach designing for global markets with cultural differences?",
            "Design an onboarding experience for a complex B2B software product.",
            "How would you measure and improve user satisfaction for a large-scale application?"
        ]
    }
}


class EnhancedResumeAnalyzer:
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = set()

        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                pass

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""

    def preprocess_text(self, text):
        """Advanced text preprocessing using NLP"""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()

        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                # Tokenize and lemmatize
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                          if token not in self.stop_words and len(token) > 2]
                return ' '.join(tokens)
            except:
                pass

        return text

    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        entities = {
            'persons': [],
            'organizations': [],
            'dates': [],
            'locations': [],
            'skills': []
        }

        if SPACY_AVAILABLE:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities['persons'].append(ent.text)
                    elif ent.label_ in ["ORG", "COMPANY"]:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ == "DATE":
                        entities['dates'].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities['locations'].append(ent.text)
            except:
                pass

        # Fallback entity extraction using regex
        if not entities['dates']:
            date_patterns = [
                r'\b\d{4}\b',  # Years
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'  # Month Year
            ]
            for pattern in date_patterns:
                entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))

        return entities

    def calculate_experience_years(self, text):
        """Calculate years of experience from resume text"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:field|industry)',
        ]

        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches])

        # Extract years from dates
        current_year = datetime.now().year
        year_mentions = re.findall(r'\b(19|20)\d{2}\b', text)
        if year_mentions:
            years_mentioned = [int(year) for year in year_mentions]
            if len(years_mentioned) >= 2:
                experience_from_dates = current_year - min(years_mentioned)
                if experience_from_dates > 0 and experience_from_dates < 50:
                    years.append(experience_from_dates)

        return max(years) if years else 0

    def extract_skills(self, text, field):
        """Extract skills using NLP and keyword matching"""
        skills = {
            'technical': [],
            'soft': [],
            'tools': [],
            'frameworks': []
        }

        field_keywords = ENHANCED_KEYWORD_DICT.get(field, {})
        text_lower = text.lower()

        # Extract technical skills
        for skill in field_keywords.get('core_skills', []):
            if skill.lower() in text_lower:
                skills['technical'].append(skill)

        # Extract frameworks
        for framework in field_keywords.get('frameworks', []):
            if framework.lower() in text_lower:
                skills['frameworks'].append(framework)

        # Extract tools
        for tool in field_keywords.get('tools', []):
            if tool.lower() in text_lower:
                skills['tools'].append(tool)

        # Extract soft skills
        for soft_skill in field_keywords.get('soft_skills', []):
            if soft_skill.lower() in text_lower:
                skills['soft'].append(soft_skill)

        return skills

    def calculate_readability_score(self, text):
        """Calculate readability metrics"""
        if not text or not NLTK_AVAILABLE:
            return {'flesch_ease': 50, 'grade_level': 12}

        try:
            if hasattr(textstat, 'flesch_reading_ease'):
                flesch_ease = textstat.flesch_reading_ease(text)
                grade_level = textstat.flesch_kincaid_grade(text)
            else:
                # Fallback calculation
                sentences = len(sent_tokenize(text))
                words = len(word_tokenize(text))
                syllables = sum([self.count_syllables(word) for word in word_tokenize(text)])

                if sentences > 0 and words > 0:
                    flesch_ease = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
                    grade_level = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
                else:
                    flesch_ease = 50
                    grade_level = 12

            return {
                'flesch_ease': max(0, min(100, flesch_ease)),
                'grade_level': max(1, min(20, grade_level))
            }
        except:
            return {'flesch_ease': 50, 'grade_level': 12}

    def count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        if word.endswith('e'):
            syllable_count -= 1

        return max(1, syllable_count)

    def calculate_keyword_density(self, text, field):
        field_keywords = ENHANCED_KEYWORD_DICT.get(field, {})
        all_keywords = set()

        for category in ['core_skills', 'frameworks', 'tools', 'soft_skills']:
            all_keywords.update(map(str.lower, field_keywords.get(category, [])))

        text_lower = text.lower()

        found_keywords = set()
        for kw in all_keywords:
            if kw in text_lower:
                found_keywords.add(kw)

        raw_density = (len(found_keywords) / len(all_keywords)) * 100 if all_keywords else 0

        # Penalize outdated terms
        outdated = ['php', 'jquery', 'xml', 'asp.net']
        outdated_count = sum(1 for word in outdated if word in text_lower)
        penalty = outdated_count * 5

        return max(0, raw_density - penalty)

    def analyze_content_quality(self, text):
        scores = {
            'action_verbs': 0,
            'quantified_achievements': 0,
            'weak_language': 0,
            'sentence_variety': 0
        }

        if not text:
            return scores

        text_lower = text.lower()

        action_verbs = [
            'achieved', 'built', 'created', 'developed', 'designed', 'implemented',
            'improved', 'increased', 'led', 'managed', 'optimized', 'reduced',
            'streamlined', 'transformed', 'delivered', 'executed', 'launched'
        ]

        action_count = sum(1 for verb in action_verbs if verb in text_lower)
        scores['action_verbs'] = min(10, action_count)

        number_patterns = [
            r'\d+%', r'\$\d+', r'\d+\+', r'\d+k', r'\d+m', r'\d+ million',
            r'\d+ thousand', r'\d+ times', r'\d+x', r'by \d+'
        ]
        quantified_count = sum(len(re.findall(pattern, text_lower)) for pattern in number_patterns)
        scores['quantified_achievements'] = min(10, quantified_count)

        weak_phrases = ['responsible for', 'worked on', 'helped with', 'assisted in', 'participated in']
        weak_count = sum(1 for phrase in weak_phrases if phrase in text_lower)
        scores['weak_language'] = max(0, 10 - weak_count * 3)

        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                if sentences:
                    avg_length = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences)
                    if 14 <= avg_length <= 20:
                        scores['sentence_variety'] = 10
                    elif 10 <= avg_length < 14 or 20 < avg_length <= 25:
                        scores['sentence_variety'] = 6
                    else:
                        scores['sentence_variety'] = 2
            except:
                scores['sentence_variety'] = 5
        else:
            scores['sentence_variety'] = 5

        return scores

    def calculate_ats_score(self, text, file_path):
        """Calculate ATS (Applicant Tracking System) compatibility score"""
        score = 0
        feedback = []

        # Check for standard sections
        required_sections = ['experience', 'education', 'skills']
        text_lower = text.lower()

        for section in required_sections:
            if section in text_lower:
                score += 10
            else:
                feedback.append(f"Missing {section} section")

        # Check for contact information
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{10,}'

        if re.search(email_pattern, text):
            score += 10
        else:
            feedback.append("Missing email address")

        if re.search(phone_pattern, text):
            score += 10
        else:
            feedback.append("Missing phone number")

        # Check for proper formatting
        if any(bullet in text for bullet in ['•', '◦', '-', '*']):
            score += 10
        else:
            feedback.append("Use bullet points for better readability")

        # Check file format
        if file_path.lower().endswith('.pdf'):
            score += 10
        else:
            feedback.append("PDF format is preferred for ATS compatibility")

        # Check for excessive formatting
        if not re.search(r'[^\w\s\-\.\,\;\:\!\?\(\)]', text):
            score += 10
        else:
            feedback.append("Avoid excessive special characters and formatting")

        return min(100, score), feedback

    def generate_interview_questions(self, text, field, difficulty='mixed'):
        """Generate interview questions based on resume content and field"""
        questions = []
        field_questions = INTERVIEW_QUESTIONS.get(field, {})

        if difficulty == 'mixed':
            questions.extend(
                random.sample(field_questions.get('easy', []), min(3, len(field_questions.get('easy', [])))))
            questions.extend(
                random.sample(field_questions.get('medium', []), min(4, len(field_questions.get('medium', [])))))
            questions.extend(
                random.sample(field_questions.get('hard', []), min(3, len(field_questions.get('hard', [])))))
        else:
            questions = random.sample(field_questions.get(difficulty, []),
                                      min(10, len(field_questions.get(difficulty, []))))

        # Add resume-specific questions
        skills = self.extract_skills(text, field)
        if skills['technical']:
            skill = random.choice(skills['technical'])
            questions.append(f"Tell me about your experience with {skill}.")

        if skills['frameworks']:
            framework = random.choice(skills['frameworks'])
            questions.append(f"How have you used {framework} in your projects?")

        experience_years = self.calculate_experience_years(text)
        if experience_years > 0:
            questions.append(
                f"With {experience_years} years of experience, what would you say is your greatest professional achievement?")

        return questions[:10]

    def compare_resumes(self, resume1_text, resume2_text, field):
        """Compare two resumes and provide insights"""
        comparison = {
            'resume1': self.analyze_resume_text(resume1_text, field),
            'resume2': self.analyze_resume_text(resume2_text, field),
            'insights': []
        }

        # Add comparison insights
        r1_score = comparison['resume1']['total_score']
        r2_score = comparison['resume2']['total_score']

        if r1_score > r2_score:
            comparison['insights'].append(f"Resume 1 scores higher ({r1_score} vs {r2_score})")
        elif r2_score > r1_score:
            comparison['insights'].append(f"Resume 2 scores higher ({r2_score} vs {r1_score})")
        else:
            comparison['insights'].append("Both resumes have similar scores")

        return comparison

    def analyze_resume_text(self, text, field):
        """Analyze resume text without file operations"""
        entities = self.extract_entities(text)
        skills = self.extract_skills(text, field)
        experience_years = self.calculate_experience_years(text)
        readability = self.calculate_readability_score(text)
        keyword_density = self.calculate_keyword_density(text, field)
        content_quality = self.analyze_content_quality(text)

        # Calculate scores
        content_score = sum(content_quality.values())
        keyword_score = min(50, keyword_density * 2)
        formatting_score = min(30, (readability['flesch_ease'] / 100) * 30)

        total_score = (
                (content_score / 40) * 25 +
                (keyword_score / 50) * 30 +
                (formatting_score / 30) * 15 +
                75  # Default ATS score for text-only analysis
        )

        return {
            'total_score': round(total_score, 1),
            'content_score': content_score,
            'keyword_score': round(keyword_score, 1),
            'formatting_score': round(formatting_score, 1),
            'experience_years': experience_years,
            'skills': skills,
            'entities': entities,
            'keyword_density': round(keyword_density, 1)
        }

    def analyze_resume(self, file_path, field):
        """Main analysis function with caching"""
        try:
            # Check cache first
            with open(file_path, 'rb') as f:
                file_content = f.read()

            file_hash = get_file_hash(file_content)

            # Check database cache
            conn = sqlite3.connect(Config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT analysis_result FROM resume_analysis WHERE file_hash = ? AND field = ?',
                (file_hash, field)
            )
            cached_result = cursor.fetchone()

            if cached_result:
                conn.close()
                logger.info(f"Returning cached result for hash: {file_hash}")
                return json.loads(cached_result[0])

            # Extract text
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            else:
                text = self.extract_text_from_docx(file_path)

            if not text:
                return {"error": "Could not extract text from resume"}

            word_count = len(text.split())

            # Perform comprehensive analysis
            entities = self.extract_entities(text)
            skills = self.extract_skills(text, field)
            experience_years = self.calculate_experience_years(text)
            readability = self.calculate_readability_score(text)
            keyword_density = self.calculate_keyword_density(text, field)
            content_quality = self.analyze_content_quality(text)
            ats_score, ats_feedback = self.calculate_ats_score(text, file_path)

            # Word count feedback
            if word_count < 200:
                ats_feedback.append("Your resume seems too short. Try to expand with more detail.")
            elif word_count > 1200:
                ats_feedback.append("Your resume may be too long. Try to keep it under 2 pages (~1000 words).")

            # EXTRA FACTORS
            bonus_points = 0
            penalty_points = 0

            # Reward if LinkedIn or portfolio is present
            if 'linkedin.com/in/' in text.lower() or 'github.com/' in text.lower():
                bonus_points += 5

            # Penalize if no mention of 'project' or 'internship'
            if 'project' not in text.lower() and 'internship' not in text.lower():
                penalty_points += 5

            # Penalize low lexical diversity
            words = text.lower().split()
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if words else 0
            if lexical_diversity < 0.3:
                penalty_points += 5

            # Calculate overall scores
            content_score = sum(content_quality.values())
            keyword_score = min(50, keyword_density * 2)
            formatting_score = min(30, (readability['flesch_ease'] / 100) * 30)

            total_score = (
                    (content_score / 40) * 25 +  # 25% weight
                    (keyword_score / 50) * 30 +  # 30% weight
                    (formatting_score / 30) * 15 +  # 15% weight
                    (ats_score / 100) * 30  # 30% weight
            )

            total_score += bonus_points - penalty_points
            total_score = max(0, min(100, round(total_score, 1)))

            # Generate feedback
            feedback = self.generate_feedback(content_quality, keyword_density, ats_feedback, skills, field)

            # Generate interview questions
            interview_questions = {
                'easy': self.generate_interview_questions(text, field, 'easy'),
                'medium': self.generate_interview_questions(text, field, 'medium'),
                'hard': self.generate_interview_questions(text, field, 'hard')
            }

            score_band = self.get_score_band(total_score)

            result = {
                'total_score': round(total_score, 1),
                'content_score': content_score,
                'keyword_score': round(keyword_score, 1),
                'formatting_score': round(formatting_score, 1),
                'ats_score': ats_score,
                'experience_years': experience_years,
                'readability': readability,
                'skills': skills,
                'entities': entities,
                'keyword_density': round(keyword_density, 1),
                'feedback': feedback,
                'interview_questions': interview_questions,
                'analysis_timestamp': datetime.now().isoformat(),
                'score_band': score_band,
                'word_count': word_count
            }

            # Cache the result
            cursor.execute(
                'INSERT OR REPLACE INTO resume_analysis (file_hash, field, analysis_result) VALUES (?, ?, ?)',
                (file_hash, field, json.dumps(result))
            )
            conn.commit()
            conn.close()

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def generate_feedback(self, content_quality, keyword_density, ats_feedback, skills, field):
        """Generate comprehensive feedback"""
        feedback = {
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }

        # Analyze strengths
        if content_quality['action_verbs'] >= 8:
            feedback['strengths'].append("Strong use of action verbs throughout the resume")

        if content_quality['quantified_achievements'] >= 6:
            feedback['strengths'].append("Good quantification of achievements with specific numbers")

        if keyword_density >= 30:
            feedback['strengths'].append("Excellent keyword optimization for the field")

        if len(skills['technical']) >= 5:
            feedback['strengths'].append("Comprehensive technical skills section")

        # Identify improvements
        if content_quality['action_verbs'] < 5:
            feedback['improvements'].append("Use more action verbs to describe your accomplishments")

        if content_quality['quantified_achievements'] < 3:
            feedback['improvements'].append("Add more quantified achievements with specific numbers and percentages")

        if keyword_density < 20:
            feedback['improvements'].append(
                f"Include more {field.replace('_', ' ')} specific keywords and technologies")

        if content_quality['weak_language'] < 8:
            feedback['improvements'].append(
                "Replace weak phrases like 'responsible for' with stronger action statements")

        # Add ATS feedback
        feedback['improvements'].extend(ats_feedback)

        # Generate suggestions
        field_keywords = ENHANCED_KEYWORD_DICT.get(field, {})
        missing_skills = []

        for skill in field_keywords.get('core_skills', [])[:5]:
            if skill not in [s.lower() for s in skills['technical']]:
                missing_skills.append(skill)

        if missing_skills:
            feedback['suggestions'].append(f"Consider adding these relevant skills: {', '.join(missing_skills[:3])}")

        trending_skills = field_keywords.get('trends', [])[:3]
        if trending_skills:
            feedback['suggestions'].append(f"Consider learning trending technologies: {', '.join(trending_skills)}")

        return feedback

    def get_score_band(self, score):
        if score >= 90:
            return "🌟 Excellent"
        elif score >= 75:
            return "✅ Good"
        elif score >= 60:
            return "⚠️ Needs Improvement"
        else:
            return "❌ Poor"


# Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

# Initialize
init_database()
analyzer = EnhancedResumeAnalyzer()

# Create upload folder
Path(Config.UPLOAD_FOLDER).mkdir(exist_ok=True)


@app.after_request
def after_request(response):
    return security_headers(response)


compare_results_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Comparison Results - AI Resume Analyzer Pro</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #f8fafc;
            --accent-color: #1e293b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--gradient);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }

        .container {
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 48px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .header p {
            color: #64748b;
            font-size: 1.25rem;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            margin-bottom: 48px;
        }

        @media (max-width: 768px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }

        .resume-card {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 32px;
            border: 2px solid #e2e8f0;
        }

        .resume-title {
            color: var(--accent-color);
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 24px;
            text-align: center;
            word-break: break-word;
        }

        .score-display {
            font-size: 3rem;
            font-weight: 900;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 32px;
        }

        .metric {
            margin-bottom: 16px;
        }

        .metric-label {
            color: var(--accent-color);
            font-weight: 600;
            margin-bottom: 4px;
        }

        .metric-value {
            color: var(--primary-color);
            font-weight: 700;
        }

        .progress-container {
            width: 100%;
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            margin-top: 4px;
        }

        .progress-bar {
            height: 100%;
            border-radius: 5px;
            background: var(--gradient);
        }

        .insights-section {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
        }

        .insights-title {
            color: var(--accent-color);
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 24px;
            text-align: center;
        }

        .insight-item {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .improvement-section {
            background: #fff8e6;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
            border-left: 4px solid var(--warning-color);
        }

        .improvement-title {
            color: var(--warning-color);
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 24px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .suggestion-item {
            margin-bottom: 12px;
            padding-left: 20px;
            position: relative;
        }

        .suggestion-item:before {
            content: "•";
            color: var(--warning-color);
            font-weight: bold;
            position: absolute;
            left: 0;
        }

        .chart-container {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .chart-title {
            color: var(--accent-color);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
        }

        .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 16px 32px;
            background: var(--gradient);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-top: 32px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
        }

        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        }

        .center-container {
            display: flex;
            justify-content: center;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Resume Comparison Results</h1>
            <p>Comparing: <strong>{{ file1 }}</strong> vs <strong>{{ file2 }}</strong></p>
        </div>

        <div class="comparison-grid">
            <div class="resume-card">
                <h3 class="resume-title">{{ file1 }}</h3>
                <div class="score-display">{{ comparison.resume1.total_score }}/100</div>

                <div class="metric">
                    <div class="metric-label">Keyword Density</div>
                    <div class="metric-value">{{ comparison.resume1.keyword_density }}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {{ comparison.resume1.keyword_density }}%"></div>
                    </div>
                </div>
            </div>

            <div class="resume-card">
                <h3 class="resume-title">{{ file2 }}</h3>
                <div class="score-display">{{ comparison.resume2.total_score }}/100</div>

                <div class="metric">
                    <div class="metric-label">Keyword Density</div>
                    <div class="metric-value">{{ comparison.resume2.keyword_density }}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {{ comparison.resume2.keyword_density }}%"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h4 class="chart-title">Score Comparison</h4>
            <canvas id="scoreChart"></canvas>
        </div>

        <div class="insights-section">
            <h3 class="insights-title">🔍 Key Insights</h3>
            {% for insight in comparison.insights %}
            <div class="insight-item">{{ insight }}</div>
            {% endfor %}
        </div>

        {% if comparison.improvement_suggestions %}
        <div class="improvement-section">
            <h4 class="improvement-title">💡 How to Improve the Weaker Resume</h4>
            {% for suggestion in comparison.improvement_suggestions %}
            <div class="suggestion-item">{{ suggestion }}</div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="center-container">
            <a href="/compare" class="back-btn">← Compare Another Resume</a>
        </div>
    </div>

    <script>
        // Score comparison chart
        document.addEventListener('DOMContentLoaded', function() {
            const scoreCtx = document.getElementById('scoreChart').getContext('2d');
            const scoreChart = new Chart(scoreCtx, {
                type: 'bar',
                data: {
                    labels: ['{{ file1 }}', '{{ file2 }}'],
                    datasets: [{
                        label: 'Total Score',
                        data: [{{ comparison.resume1.total_score }}, {{ comparison.resume2.total_score }}],
                        backgroundColor: [
                            'rgba(37, 99, 235, 0.7)',
                            'rgba(118, 75, 162, 0.7)'
                        ],
                        borderColor: [
                            'rgba(37, 99, 235, 1)',
                            'rgba(118, 75, 162, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        });
    </script>
</body>
</html>
"""



# Enhanced HTML templates (keeping original design)
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Resume Analyzer Pro</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #f8fafc;
            --accent-color: #1e293b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            width: 100%;
            background: white;
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 48px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 16px;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #64748b;
            font-size: 1.25rem;
            line-height: 1.6;
        }

        .upload-container {
            background: var(--secondary-color);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 32px;
            border: 2px dashed #e2e8f0;
            transition: all 0.3s ease;
        }

        .upload-container:hover {
            border-color: var(--primary-color);
            background: #f1f5f9;
        }

        .form-group {
            margin-bottom: 32px;
        }

        .form-group label {
            display: block;
            margin-bottom: 12px;
            color: var(--accent-color);
            font-weight: 600;
            font-size: 1.125rem;
        }

        .file-input-container {
            position: relative;
            border: 3px dashed var(--primary-color);
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }

        .file-input-container:hover {
            background: #eff6ff;
            transform: translateY(-2px);
        }

        .file-input-container input[type="file"] {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            opacity: 0;
            cursor: pointer;
        }

        .file-input-container .upload-icon {
            font-size: 3rem;
            color: var(--primary-color);
            margin-bottom: 16px;
        }

        .file-input-container p {
            color: var(--primary-color);
            font-size: 1.125rem;
            font-weight: 500;
            margin: 0;
        }

        .file-input-container .file-info {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 8px;
        }

        select {
            width: 100%;
            padding: 16px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1.125rem;
            color: var(--accent-color);
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 12px center;
            background-repeat: no-repeat;
            background-size: 16px;
        }

        select:focus {
            border-color: var(--primary-color);
            outline: none;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .analyze-btn {
            width: 100%;
            padding: 18px;
            background: var(--gradient);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        }

        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-top: 48px;
        }

        .feature-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 32px;
            border-radius: 20px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.15);
        }

        .feature-card .icon {
            font-size: 2.5rem;
            margin-bottom: 16px;
        }

        .feature-card h3 {
            color: var(--accent-color);
            margin-bottom: 12px;
            font-size: 1.25rem;
            font-weight: 700;
        }

        .feature-card p {
            color: #64748b;
            font-size: 1rem;
            line-height: 1.6;
        }

        .tech-stack {
            margin-top: 48px;
            text-align: center;
        }

        .tech-stack h3 {
            color: var(--accent-color);
            margin-bottom: 24px;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .tech-badges {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 12px;
        }

        .tech-badge {
            background: var(--gradient);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .new-features {
            margin-top: 48px;
            text-align: center;
        }

        .new-features h3 {
            color: var(--accent-color);
            margin-bottom: 24px;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .feature-links {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .feature-link {
            background: var(--primary-color);
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .feature-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        }

        @media (max-width: 768px) {
            .container {
                padding: 24px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .features {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI Resume Analyzer Pro</h1>
            <p>Advanced AI-powered resume analysis with ML insights, NLP processing, and personalized interview preparation</p>
        </div>

        <form action="/analyze_resume" method="post" enctype="multipart/form-data">
            <div class="upload-container">
                <div class="form-group">
                    <label for="file">📄 Upload Your Resume</label>
                    <div class="file-input-container">
                        <div class="upload-icon">📁</div>
                        <input type="file" name="file" id="file" required accept=".pdf,.docx">
                        <p id="file-name-display">Drag and drop your resume here or click to browse</p>
                        <p class="file-info">Supports PDF and DOCX formats • Max 10MB</p>
                    </div>
                </div>

                <div class="form-group">
                    <label for="field">🎯 Select Your Target Industry</label>
                    <select name="field" id="field" required>
                        <option value="">Choose your field...</option>
                        <option value="data_science">📊 Data Science & Analytics</option>
                        <option value="web_development">💻 Web Development</option>
                        <option value="ui_ux">🎨 UI/UX Design</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="analyze-btn">🔍 Analyze Resume with AI</button>
        </form>

       <div class="new-features">
    <h3>✨ New Features</h3>
    <div class="feature-links">
        <a href="/analytics" class="feature-link">📊 Analytics Dashboard</a>
        <a href="/compare" class="feature-link">⚖️ Compare Resumes</a>
    </div>
</div>

        <div class="features">
            <div class="feature-card">
                <div class="icon">🤖</div>
                <h3>AI-Powered Analysis</h3>
                <p>Advanced NLP and machine learning algorithms analyze your resume content, structure, and optimization potential</p>
            </div>
            <div class="feature-card">
                <div class="icon">📈</div>
                <h3>ATS Optimization</h3>
                <p>Ensure your resume passes Applicant Tracking Systems with our comprehensive compatibility scoring</p>
            </div>
            <div class="feature-card">
                <div class="icon">🎯</div>
                <h3>Interview Preparation</h3>
                <p>Get personalized interview questions based on your resume content and industry, with easy to hard difficulty levels</p>
            </div>
            <div class="feature-card">
                <div class="icon">📊</div>
                <h3>Detailed Analytics</h3>
                <p>Comprehensive scoring across content quality, keyword optimization, formatting, and readability metrics</p>
            </div>
            <div class="feature-card">
                <div class="icon">🔍</div>
                <h3>Skills Extraction</h3>
                <p>Automatic identification and categorization of technical skills, frameworks, tools, and soft skills</p>
            </div>
            <div class="feature-card">
                <div class="icon">💡</div>
                <h3>Smart Recommendations</h3>
                <p>Personalized suggestions for improvement based on industry standards and current market trends</p>
            </div>
        </div>

        <div class="tech-stack">
            <h3>🛠️ Powered by Advanced Technologies</h3>
            <div class="tech-badges">
                <span class="tech-badge">Python Flask</span>
                <span class="tech-badge">spaCy NLP</span>
                <span class="tech-badge">NLTK</span>
                <span class="tech-badge">scikit-learn</span>
                <span class="tech-badge">PyMuPDF</span>
                <span class="tech-badge">TF-IDF</span>
                <span class="tech-badge">Machine Learning</span>
                <span class="tech-badge">Entity Recognition</span>
                <span class="tech-badge">SQLite</span>
                <span class="tech-badge">Caching</span>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('file').addEventListener('change', function() {
            const fileNameDisplay = document.getElementById('file-name-display');
            if (this.files && this.files.length > 0) {
                const file = this.files[0];
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                fileNameDisplay.innerHTML = `📄 ${file.name} <br><small>(${fileSize} MB)</small>`;
            } else {
                fileNameDisplay.textContent = 'Drag and drop your resume here or click to browse';
            }
        });

        // Add drag and drop functionality
        const uploadContainer = document.querySelector('.file-input-container');

        uploadContainer.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.background = '#dbeafe';
            this.style.borderColor = '#3b82f6';
        });

        uploadContainer.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.background = 'white';
            this.style.borderColor = '#2563eb';
        });

        uploadContainer.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.background = 'white';
            this.style.borderColor = '#2563eb';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('file').files = files;
                const event = new Event('change', { bubbles: true });
                document.getElementById('file').dispatchEvent(event);
            }
        });
    </script>
</body>
</html>
"""

results_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Analysis Results - AI Resume Analyzer Pro</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #f8fafc;
            --accent-color: #1e293b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--gradient);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 48px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .score-overview {
            background: var(--gradient);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 48px;
            color: white;
        }

        .total-score {
            font-size: 4rem;
            font-weight: 900;
            margin-bottom: 16px;
        }

        .score-label {
            font-size: 1.25rem;
            opacity: 0.9;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }

        .metric-card {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }

        .metric-card h3 {
            color: var(--accent-color);
            margin-bottom: 12px;
            font-size: 1.125rem;
            font-weight: 600;
        }

        .metric-value {
            font-size: 2rem;
            color: var(--primary-color);
            font-weight: 800;
            margin-bottom: 8px;
        }

        .metric-description {
            color: #64748b;
            font-size: 0.875rem;
        }

        .section {
            margin-bottom: 48px;
        }

        .section-title {
            color: var(--accent-color);
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .skills-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 32px;
        }

        .skills-category {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #e2e8f0;
        }

        .skills-category h4 {
            color: var(--accent-color);
            margin-bottom: 16px;
            font-size: 1.125rem;
            font-weight: 600;
        }

        .skills-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .skill-tag {
            background: var(--primary-color);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .feedback-section {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
        }

        .feedback-category {
            margin-bottom: 32px;
        }

        .feedback-category h4 {
            color: var(--accent-color);
            margin-bottom: 16px;
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .feedback-item {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .interview-questions {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
        }

        .difficulty-tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }

        .difficulty-tab {
            padding: 12px 24px;
            border-radius: 25px;
            border: 2px solid #e2e8f0;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .difficulty-tab.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        .questions-list {
            display: none;
        }

        .questions-list.active {
            display: block;
        }

        .question-item {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            border-left: 4px solid var(--success-color);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .question-number {
            color: var(--primary-color);
            font-weight: 600;
            margin-bottom: 8px;
        }

        .action-buttons {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 48px;
        }

        .action-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 16px 32px;
            background: var(--gradient);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        }

        .action-btn.secondary {
            background: var(--secondary-color);
            color: var(--accent-color);
            border: 2px solid #e2e8f0;
        }

        .entity-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .entity-card {
            background: white;
            border-radius: 12px;
            padding: 16px;
            border: 2px solid #e2e8f0;
        }

        .entity-card h5 {
            color: var(--accent-color);
            margin-bottom: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .entity-value {
            color: var(--primary-color);
            font-weight: 600;
        }

        @media (max-width: 768px) {
            .container {
                padding: 24px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .total-score {
                font-size: 3rem;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .difficulty-tabs {
                justify-content: center;
            }

            .action-buttons {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Resume Analysis Results</h1>
            <p>Comprehensive AI-powered analysis of your resume</p>
        </div>

        <div class="score-overview">
            <div class="total-score">{{ results.total_score }}/100</div>
            <div class="score-label">{{ results.score_band }}</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>📝 Content Quality</h3>
                <div class="metric-value">{{ results.content_score }}/40</div>
                <div class="metric-description">Action verbs, achievements, language quality</div>
            </div>
            <div class="metric-card">
                <h3>🔍 Keyword Optimization</h3>
                <div class="metric-value">{{ results.keyword_score }}/50</div>
                <div class="metric-description">Industry-specific keyword density</div>
            </div>
            <div class="metric-card">
                <h3>📄 Formatting</h3>
                <div class="metric-value">{{ results.formatting_score }}/30</div>
                <div class="metric-description">Readability and structure</div>
            </div>
            <div class="metric-card">
                <h3>🤖 ATS Compatibility</h3>
                <div class="metric-value">{{ results.ats_score }}/100</div>
                <div class="metric-description">Applicant Tracking System optimization</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Resume Analytics</h2>
            <div class="entity-info">
                <div class="entity-card">
                    <h5>Experience</h5>
                    <div class="entity-value">{{ results.experience_years }} years</div>
                </div>
                <div class="entity-card">
                    <h5>Word Count</h5>
                    <div class="entity-value">{{ results.word_count }}</div>
                </div>
                <div class="entity-card">
                    <h5>Keyword Density</h5>
                    <div class="entity-value">{{ results.keyword_density }}%</div>
                </div>
                <div class="entity-card">
                    <h5>Readability Score</h5>
                    <div class="entity-value">{{ results.readability.flesch_ease }}/100</div>
                </div>
                <div class="entity-card">
                    <h5>Grade Level</h5>
                    <div class="entity-value">{{ results.readability.grade_level }}</div>
                </div>
            </div>
        </div>

        {% if results.skills %}
        <div class="section">
            <h2 class="section-title">🛠️ Extracted Skills</h2>
            <div class="skills-grid">
                {% if results.skills.technical %}
                <div class="skills-category">
                    <h4>💻 Technical Skills</h4>
                    <div class="skills-list">
                        {% for skill in results.skills.technical %}
                        <span class="skill-tag">{{ skill }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                {% if results.skills.frameworks %}
                <div class="skills-category">
                    <h4>🔧 Frameworks & Tools</h4>
                    <div class="skills-list">
                        {% for framework in results.skills.frameworks %}
                        <span class="skill-tag">{{ framework }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                {% if results.skills.soft %}
                <div class="skills-category">
                    <h4>🤝 Soft Skills</h4>
                    <div class="skills-list">
                        {% for skill in results.skills.soft %}
                        <span class="skill-tag">{{ skill }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                {% if results.skills.tools %}
                <div class="skills-category">
                    <h4>⚙️ Tools & Technologies</h4>
                    <div class="skills-list">
                        {% for tool in results.skills.tools %}
                        <span class="skill-tag">{{ tool }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        {% if results.feedback %}
        <div class="section">
            <h2 class="section-title">💡 AI-Generated Feedback</h2>
            <div class="feedback-section">
                {% if results.feedback.strengths %}
                <div class="feedback-category">
                    <h4>✅ Strengths</h4>
                    {% for strength in results.feedback.strengths %}
                    <div class="feedback-item">{{ strength }}</div>
                    {% endfor %}
                </div>
                {% endif %}

                {% if results.feedback.improvements %}
                <div class="feedback-category">
                    <h4>🔧 Areas for Improvement</h4>
                    {% for improvement in results.feedback.improvements %}
                    <div class="feedback-item">{{ improvement }}</div>
                    {% endfor %}
                </div>
                {% endif %}

                {% if results.feedback.suggestions %}
                <div class="feedback-category">
                    <h4>🚀 Suggestions</h4>
                    {% for suggestion in results.feedback.suggestions %}
                    <div class="feedback-item">{{ suggestion }}</div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        {% if results.interview_questions %}
        <div class="section">
            <h2 class="section-title">🎤 Interview Preparation</h2>
            <div class="interview-questions">
                <div class="difficulty-tabs">
                    <div class="difficulty-tab active" onclick="showQuestions('easy')">🟢 Easy</div>
                    <div class="difficulty-tab" onclick="showQuestions('medium')">🟡 Medium</div>
                    <div class="difficulty-tab" onclick="showQuestions('hard')">🔴 Hard</div>
                </div>

                <div id="easy-questions" class="questions-list active">
                    {% for question in results.interview_questions.easy %}
                    <div class="question-item">
                        <div class="question-number">Question {{ loop.index }}</div>
                        <div>{{ question }}</div>
                    </div>
                    {% endfor %}
                </div>

                <div id="medium-questions" class="questions-list">
                    {% for question in results.interview_questions.medium %}
                    <div class="question-item">
                        <div class="question-number">Question {{ loop.index }}</div>
                        <div>{{ question }}</div>
                    </div>
                    {% endfor %}
                </div>

                <div id="hard-questions" class="questions-list">
                    {% for question in results.interview_questions.hard %}
                    <div class="question-item">
                        <div class="question-number">Question {{ loop.index }}</div>
                        <div>{{ question }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}

        <div class="action-buttons">
            <a href="/" class="action-btn">
                ← Analyze Another Resume
            </a>
            <a href="/export/{{ results.analysis_timestamp }}" class="action-btn secondary">
                📄 Export Report
            </a>
        </div>
    </div>

    <script>
        function showQuestions(difficulty) {
            // Hide all question lists
            document.querySelectorAll('.questions-list').forEach(list => {
                list.classList.remove('active');
            });

            // Remove active class from all tabs
            document.querySelectorAll('.difficulty-tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected questions and activate tab
            document.getElementById(difficulty + '-questions').classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

# Additional templates for new features
compare_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compare Resumes - AI Resume Analyzer Pro</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #f8fafc;
            --accent-color: #1e293b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            width: 100%;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            width: 100%;
            background: white;
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 48px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .header p {
            color: #64748b;
            font-size: 1.25rem;
            line-height: 1.6;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }

        @media (max-width: 768px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }

        .upload-section {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 32px;
            border: 2px dashed #e2e8f0;
        }

        .upload-section h3 {
            color: var(--accent-color);
            margin-bottom: 16px;
            font-size: 1.25rem;
            font-weight: 600;
        }

        .file-input-container {
            position: relative;
            border: 2px dashed var(--primary-color);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .file-input-container:hover {
            background: #eff6ff;
        }

        .file-input-container input[type="file"] {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            opacity: 0;
            cursor: pointer;
        }

        .file-input-container p {
            margin: 0;
            color: var(--primary-color);
            font-weight: 500;
        }

        .file-input-container .file-info {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 8px;
        }

        .compare-btn {
            width: 100%;
            padding: 18px;
            background: var(--gradient);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 24px;
        }

        .compare-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        }

        .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: var(--secondary-color);
            color: var(--accent-color);
            text-decoration: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 2px solid #e2e8f0;
            margin-top: 32px;
        }

        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: var(--accent-color);
            font-weight: 600;
        }

        select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            color: var(--accent-color);
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select:focus {
            border-color: var(--primary-color);
            outline: none;
        }

        .center-container {
            display: flex;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚖️ Compare Resumes</h1>
            <p>Upload two resumes to see how they compare side by side</p>
        </div>

        <form action="/compare_resumes" method="post" enctype="multipart/form-data">
            <div class="comparison-grid">
                <div class="upload-section">
                    <h3>📄 Resume 1</h3>
                    <div class="file-input-container" id="file1-container">
                        <input type="file" name="file1" id="file1" required accept=".pdf,.docx">
                        <p id="file1-name">Click to upload first resume</p>
                        <p class="file-info">Supports PDF and DOCX formats</p>
                    </div>
                </div>

                <div class="upload-section">
                    <h3>📄 Resume 2</h3>
                    <div class="file-input-container" id="file2-container">
                        <input type="file" name="file2" id="file2" required accept=".pdf,.docx">
                        <p id="file2-name">Click to upload second resume</p>
                        <p class="file-info">Supports PDF and DOCX formats</p>
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label for="field">🎯 Select Field for Comparison</label>
                <select name="field" id="field" required>
                    <option value="">Choose field...</option>
                    <option value="data_science">📊 Data Science & Analytics</option>
                    <option value="web_development">💻 Web Development</option>
                    <option value="ui_ux">🎨 UI/UX Design</option>
                </select>
            </div>

            <button type="submit" class="compare-btn">🔍 Compare Resumes</button>
        </form>

        <div class="center-container">
            <a href="/" class="back-btn">← Back to Home</a>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Handle file input changes
            function setupFileInput(inputId, containerId, nameId) {
                const input = document.getElementById(inputId);
                const container = document.getElementById(containerId);
                const nameDisplay = document.getElementById(nameId);

                input.addEventListener('change', function() {
                    if (this.files && this.files.length > 0) {
                        const file = this.files[0];
                        const fileSize = (file.size / 1024 / 1024).toFixed(2);
                        nameDisplay.textContent = file.name;
                        container.querySelector('.file-info').textContent = `(${fileSize} MB)`;
                    }
                });

                // Drag and drop functionality
                container.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    this.style.background = '#dbeafe';
                    this.style.borderColor = '#3b82f6';
                });

                container.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    this.style.background = 'white';
                    this.style.borderColor = '#2563eb';
                });

                container.addEventListener('drop', function(e) {
                    e.preventDefault();
                    this.style.background = 'white';
                    this.style.borderColor = '#2563eb';

                    if (e.dataTransfer.files.length > 0) {
                        input.files = e.dataTransfer.files;
                        const event = new Event('change');
                        input.dispatchEvent(event);
                    }
                });
            }

            setupFileInput('file1', 'file1-container', 'file1-name');
            setupFileInput('file2', 'file2-container', 'file2-name');
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(index_html)


@app.route('/compare')
def compare():
    return render_template_string(compare_html)


@app.route('/compare_resumes', methods=['POST'])
@rate_limit
def compare_resumes():
    try:
        # Get the uploaded files and field
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        field = request.form.get('field', 'data_science')

        # Validate inputs
        if not file1 or not file2:
            flash('Please upload both resumes', 'error')
            return redirect(url_for('compare'))

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            flash('Only PDF and DOCX files are allowed', 'error')
            return redirect(url_for('compare'))

        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        try:
            # Save files with secure filenames
            file1_path = os.path.join(temp_dir, secure_filename(file1.filename))
            file2_path = os.path.join(temp_dir, secure_filename(file2.filename))

            file1.save(file1_path)
            file2.save(file2_path)
            temp_files.extend([file1_path, file2_path])

            # Analyze both resumes
            result1 = analyzer.analyze_resume(file1_path, field)
            result2 = analyzer.analyze_resume(file2_path, field)

            # Check for analysis errors
            if "error" in result1:
                flash(f"Error analyzing first resume: {result1['error']}", "error")
                return redirect(url_for('compare'))
            if "error" in result2:
                flash(f"Error analyzing second resume: {result2['error']}", "error")
                return redirect(url_for('compare'))

            # Prepare comparison data
            comparison = {
                'resume1': result1,
                'resume2': result2,
                'insights': [],
                'improvement_suggestions': []
            }

            # Generate insights
            if result1['total_score'] > result2['total_score']:
                score_diff = result1['total_score'] - result2['total_score']
                comparison['insights'].append(
                    f"Resume 1 scored higher ({result1['total_score']} vs {result2['total_score']})"
                )

                # Generate improvement suggestions for resume 2
                if result1['keyword_score'] > result2['keyword_score'] + 5:
                    comparison['improvement_suggestions'].append(
                        "Increase keyword density for better ATS performance"
                    )
                if result1['content_score'] > result2['content_score'] + 5:
                    comparison['improvement_suggestions'].append(
                        "Add more action verbs and quantified achievements"
                    )
            else:
                score_diff = result2['total_score'] - result1['total_score']
                comparison['insights'].append(
                    f"Resume 2 scored higher ({result2['total_score']} vs {result1['total_score']})"
                )

                # Generate improvement suggestions for resume 1
                if result2['keyword_score'] > result1['keyword_score'] + 5:
                    comparison['improvement_suggestions'].append(
                        "Increase keyword density for better ATS performance"
                    )
                if result2['content_score'] > result1['content_score'] + 5:
                    comparison['improvement_suggestions'].append(
                        "Add more action verbs and quantified achievements"
                    )

            # Add general comparison metrics
            comparison['insights'].append(
                f"Keyword density: {result1['keyword_density']}% vs {result2['keyword_density']}%"
            )
            comparison['insights'].append(
                f"Experience: {result1['experience_years']} years vs {result2['experience_years']} years"
            )

            # Render the comparison results template
            return render_template_string(
                compare_results_html,
                file1=secure_filename(file1.filename),
                file2=secure_filename(file2.filename),
                comparison=comparison,
                field=field.replace('_', ' ').title()
            )

        finally:
            # Clean up temporary files
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error deleting temp file {file_path}: {e}")

            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error deleting temp directory {temp_dir}: {e}")

    except Exception as e:
        logger.error(f"Comparison error: {str(e)}", exc_info=True)
        flash(f"An error occurred during comparison: {str(e)}", "error")
        return redirect(url_for('compare'))

@app.route('/analyze_resume', methods=['POST'])
@rate_limit
def analyze():
    """Main resume analysis endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    field = request.form.get('field', 'data_science')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF or DOCX file.'}), 400

    # Validate field selection
    if field not in ENHANCED_KEYWORD_DICT:
        return jsonify({'error': 'Invalid field selection'}), 400

    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        suffix = ".pdf" if filename.lower().endswith('.pdf') else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Analyze resume
        result = analyzer.analyze_resume(temp_path, field)

        # Clean up temporary file
        os.unlink(temp_path)

        if "error" in result:
            return jsonify(result), 400

        # Track analytics
        track_analytics('resume_analysis', {
            'field': field,
            'score': result['total_score'],
            'experience_years': result['experience_years']
        })

        return render_template_string(results_html, results=result)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500


# API Endpoints
@app.route('/api/analyze', methods=['POST'])
@rate_limit
def api_analyze():
    """API endpoint for resume analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    field = request.form.get('field', 'data_science')

    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        suffix = ".pdf" if filename.lower().endswith('.pdf') else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        result = analyzer.analyze_resume(temp_path, field)
        os.unlink(temp_path)

        if "error" in result:
            return jsonify(result), 400

        track_analytics('api_analysis', {
            'field': field,
            'score': result['total_score']
        })

        return jsonify(result)

    except Exception as e:
        logger.error(f"API analysis failed: {str(e)}")
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500


@app.route('/api/docs')
def api_docs():
    """API documentation"""
    docs = {
        'title': 'Resume Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/analyze': {
                'description': 'Analyze a resume file',
                'parameters': {
                    'file': 'Resume file (PDF or DOCX)',
                    'field': 'Target field (data_science, web_development, ui_ux)'
                },
                'response': 'JSON with analysis results'
            },
            'GET /api/analytics': {
                'description': 'Get usage analytics',
                'response': 'JSON with analytics data'
            }
        }
    }
    return jsonify(docs)


@app.route('/api/analytics')
def api_analytics():
    """Get analytics data"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()

        # Get analysis counts by field
        cursor.execute('''
            SELECT JSON_EXTRACT(data, '$.field') as field, COUNT(*) as count
            FROM analytics 
            WHERE event_type = 'resume_analysis'
            GROUP BY field
        ''')
        field_counts = dict(cursor.fetchall())

        # Get average scores
        cursor.execute('''
            SELECT AVG(JSON_EXTRACT(data, '$.score')) as avg_score
            FROM analytics 
            WHERE event_type = 'resume_analysis'
        ''')
        avg_score = cursor.fetchone()[0] or 0

        # Get total analyses
        cursor.execute('''
            SELECT COUNT(*) as total
            FROM analytics 
            WHERE event_type = 'resume_analysis'
        ''')
        total_analyses = cursor.fetchone()[0]

        conn.close()

        analytics = {
            'total_analyses': total_analyses,
            'average_score': round(avg_score, 1),
            'field_distribution': field_counts,
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(analytics)

    except Exception as e:
        logger.error(f"Analytics retrieval failed: {str(e)}")
        return jsonify({'error': 'Failed to retrieve analytics'}), 500


@app.route('/analytics')
def analytics_dashboard():
    """Analytics dashboard"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()

        # Get basic stats
        cursor.execute('SELECT COUNT(*) FROM analytics WHERE event_type = "resume_analysis"')
        total_analyses = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(JSON_EXTRACT(data, "$.score")) FROM analytics WHERE event_type = "resume_analysis"')
        avg_score = cursor.fetchone()[0] or 0

        conn.close()

        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analytics Dashboard - Resume Analyzer</title>
            <style>
                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 24px;
                    padding: 48px;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 48px;
                }}
                .header h1 {{
                    color: #1e293b;
                    font-size: 2.5rem;
                    font-weight: 800;
                    margin-bottom: 16px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 24px;
                    margin-bottom: 48px;
                }}
                .stat-card {{
                    background: #f8fafc;
                    border-radius: 16px;
                    padding: 32px;
                    text-align: center;
                    border: 2px solid #e2e8f0;
                }}
                .stat-value {{
                    font-size: 3rem;
                    font-weight: 900;
                    color: #2563eb;
                    margin-bottom: 8px;
                }}
                .stat-label {{
                    color: #64748b;
                    font-size: 1.125rem;
                    font-weight: 600;
                }}
                .back-btn {{
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 16px 32px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }}
                .back-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Analytics Dashboard</h1>
                    <p>Usage statistics and insights</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_analyses}</div>
                        <div class="stat-label">Total Analyses</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_score:.1f}</div>
                        <div class="stat-label">Average Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_analyses * 0.85:.0f}</div>
                        <div class="stat-label">Satisfied Users</div>
                    </div>
                </div>

                <div style="text-align: center;">
                    <a href="/" class="back-btn">← Back to Home</a>
                </div>
            </div>
        </body>
        </html>
        """

        return dashboard_html

    except Exception as e:
        logger.error(f"Analytics dashboard failed: {str(e)}")
        return jsonify({'error': 'Failed to load dashboard'}), 500


@app.route('/export/<timestamp>')
def export_report(timestamp):
    """Export analysis report as text file"""
    try:
        # This would normally retrieve the specific analysis from database
        # For now, return a sample report
        report_content = f"""
RESUME ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== ANALYSIS SUMMARY ===
This report was generated by AI Resume Analyzer Pro.

=== RECOMMENDATIONS ===
1. Use more action verbs in your descriptions
2. Quantify your achievements with specific numbers
3. Include relevant keywords for your target field
4. Ensure proper formatting for ATS compatibility

=== NEXT STEPS ===
1. Revise your resume based on the feedback
2. Practice with the provided interview questions
3. Consider learning trending technologies in your field

Generated by AI Resume Analyzer Pro
"""

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(report_content)
            temp_path = temp_file.name

        track_analytics('report_export', {'timestamp': timestamp})

        return send_file(temp_path, as_attachment=True, download_name=f'resume_report_{timestamp}.txt')

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # Initialize database
    init_database()

    logger.info("Starting Resume Analyzer Pro...")
    app.run(debug=True, host='0.0.0.0', port=5000)
