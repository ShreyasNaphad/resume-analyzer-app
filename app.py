from flask import Flask, request, jsonify, render_template_string
import os
import tempfile
import fitz  # Changed from PyPDF2 and added to requirements
from docx import Document
import re
import random

# Define keyword lists for different industries
KEYWORD_DICT = {
    "web_development": {
        "core": ["html", "css", "javascript", "react", "node.js", "sql", "angular", "git", "mongodb"],
        "soft_skills": ["leadership", "problem-solving", "communication", "teamwork",
                        "critical thinking"],
        "action_verbs": ["managed", "developed", "led", "designed", "improved",
                        "implemented", "achieved"],
        "weak_verbs": ["worked", "helped", "assisted"],
        "tools": ["react", "node.js", "aws", "mongodb", "docker", "kubernetes", "jenkins"],
        "jargon": ["responsive design", "api integration", "web services",
                    "ci/cd", "version control"],
        "trends": ["next.js", "typescript", "tailwind css", "web3"]
    },
    "data_science": {
        "core": ["python", "R", "machine learning", "deep learning", "data analysis", "pandas",
                    "numpy", "sql", "matplotlib", "nlp"],
        "soft_skills": ["problem-solving", "communication",
                        "critical thinking", "research", "data visualization"],
        "action_verbs": ["analyzed", "developed", "implemented", "researched", "optimized",
                        "predicted", "deployed"],
        "weak_verbs": ["worked", "helped", "assisted"],
        "tools": ["sql", "power bi", "tableau", "python", "r", "tensorflow", "scikit-learn", "excel"],
        "jargon": ["etl", "data cleaning", "statistical analysis",
                    "data mining", "feature engineering", "data visualization"],
        "trends": ["mlops", "automl", "snowflake", "real-time analytics"]
    },
    "ui_ux": {
    "core": ["user research", "wireframing", "prototyping", "usability testing", "figma",
             "adobe xd", "ux design", "ui design", "design systems", "interaction design"],
    "soft_skills": ["empathy", "problem-solving", "communication", "creativity", 
                    "critical thinking"],
    "action_verbs": ["designed", "developed", "researched", "implemented", "optimized",
                     "prototyped", "iterated"],
    "weak_verbs": ["worked", "helped", "assisted"],
    "tools": ["figma", "adobe xd", "sketch", "invision", "axure", "balsamiq", "photoshop"],
    "jargon": ["user personas", "heuristic evaluation", "affinity mapping", "a/b testing", 
               "usability heuristics", "design thinking", "human-centered design"],
    "trends": ["dark mode design", "neumorphism", "glassmorphism", "microinteractions",
               "voice UI", "AI-driven UX", "accessible design"]
}
}
# Required resume sections
REQUIRED_SECTIONS = ["Experience", "Education", "Skills", "Projects"]

# Allowed fonts for readability
ALLOWED_FONTS = {
    "Arial", "Arial Narrow", "Arial Black",
    "Calibri", "Calibri Light", "Calibri Bold",
    "Verdana", "Verdana Bold",
    "Tahoma", "Tahoma Bold",
    "Garamond", "Garamond Bold",
    "Helvetica", "Helvetica Light", "Helvetica Bold",
    "Georgia", "Georgia Bold",
    "Times New Roman", "Times New Roman Bold"
}

# Feedback pool
FEEDBACK_POOL = {
    "core_keywords": [
        "You scored {score}/10 for core keywords. Try adding key terms like {examples}.",
        "Your core keywords section is good, but it could be stronger. Mention technologies like {examples}.",
        "Consider adding more relevant skills to align with the role, such as {examples}.",
        "Recruiters look for specific terms. Include {examples} to improve visibility.",
        "Highlight your experience with {examples} to boost your core skills."
    ],
    "soft_skills": [
        "Elaborate on teamwork or leadership in your roles.",
        "Including examples of communication and critical thinking could strengthen your soft skills.",
        "Great work so far! Add more emphasis on problem-solving or adaptability.",
        "Recruiters value interpersonal abilities. Highlight specific scenarios where your soft skills shine.",
        "Soft skills are essential. Incorporate terms like 'collaboration' and 'decision-making.'"
    ],
    "weak_verbs": [
        "Replace weak verbs like 'helped' or 'worked' with impactful ones such as 'implemented' or 'coordinated.'",
        "Using strong action verbs like 'developed' or 'optimized' makes your resume more persuasive.",
        "Your resume contains weak verbs. Strengthen it by focusing on dynamic action terms.",
        "Recruiters prefer assertive language. Swap 'assisted' with 'led' or 'executed.'",
        "Avoid passive language. Assert your achievements with words like 'achieved' or 'delivered.'"
    ],
    "tools": [
        "Add more technologies like {examples} to improve this section.",
        "Your tools section is solid, but adding technologies such as {examples} could make it stronger.",
        "Consider emphasizing experience with tools like {examples} to align with industry standards.",
        "Adding trending tools like {examples} can highlight your technical adaptability.",
        "Mention tools like {examples} to showcase your technical versatility."
    ],
    "trends": [
        "Your resume could benefit from including trending technologies like {examples}.",
        "Consider exploring and mentioning industry trends such as {examples} to stay current.",
        "Adding modern concepts like {examples} will demonstrate your awareness of emerging trends.",
        "To stand out, integrate trending topics like {examples} into your resume.",
        "Keep your resume future-focused by mentioning {examples} in your projects."
    ],
    "certifications": [
        "Certifications are essential. Highlight courses or certifications in {field} to add credibility.",
        "Adding  more certificates could enhance your profile.",
        "Showcase relevant certifications to emphasize your technical expertise.",
        "Consider pursuing certifications in {examples} to boost your score.",
        "Certifications from your field can validate your skills and improve recruiter interest."
    ],
    "fonts": [
        "Your resume uses non-standard fonts like {examples}. Switch to fonts such as Arial or Calibri for readability.",
        "Readable fonts like Helvetica or Verdana can improve your resume's presentation. Avoid using {examples}.",
        "Using inconsistent fonts can distract recruiters. Stick to professional ones like Times New Roman."
    ],
    "bullet_points": [
        "Adding bullet points can make your resume easier to skim and more recruiter-friendly.",
        "Consider using bullet points to clearly highlight your achievements and responsibilities.",
        "Structured resumes with bullet points are easier to read. Try incorporating them."
    ],
    "spacing": [
        "Ensure consistent spacing between sections to improve visual appeal.",
        "Adjust the spacing to make each section distinct and well-organized.",
        "Proper spacing enhances readability. Avoid large gaps or crowded text."
    ],
    "page_length": [
        "Your resume exceeds the recommended length. Condense it to 1-2 pages for better impact.",
        "Short and concise resumes are effective. Keep it within 1-2 pages.",
        "Consider trimming excess information to make your resume more concise."
    ],
    "sections": [
        "The {section} section is missing. Include it to provide a complete overview of your experience.",
        "Adding a detailed {section} section will make your resume more comprehensive.",
        "Your resume lacks a proper {section} section. Incorporate it to strengthen the structure."
    ]
}


def get_dynamic_feedback(category, score=None, examples=None, section=None, field=None):
    """Retrieves and formats dynamic feedback."""
    feedback_list = FEEDBACK_POOL.get(category, [])
    if feedback_list:
        feedback_template = random.choice(feedback_list)
        return feedback_template.format(
            score=score if score is not None else "N/A",
            examples=", ".join(examples) if examples else "N/A",
            section=section if section else "N/A",
            field=field if field else "N/A"
        )
    return "No feedback available."


def parse_pdf(file_path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()  # Ensure file is closed
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""



def parse_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""



def check_fonts(file_path):
    """Validate font usage"""
    fonts_used = set()
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                for font in page.get_fonts(full=True):
                    fonts_used.add(font[3])
            doc.close()  # Close PDF document
        else:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                for run in paragraph.runs:
                    if run.font.name:
                        fonts_used.add(run.font.name)
    except Exception as e:
        print(f"Error checking fonts: {e}")
        return [], 0

    invalid_fonts = [font for font in fonts_used if font and font not in ALLOWED_FONTS]
    font_penalty = min(len(invalid_fonts), 3)
    return invalid_fonts, font_penalty



def calculate_keyword_score(resume_text: str, field: str) -> tuple:
    """Calculate keyword score (50 points total)"""
    resume_text_lower = resume_text.lower()
    field_keywords = KEYWORD_DICT.get(field, {})
    scores = {
        'core_keywords': 0,
        'soft_skills': 0,
        'action_verbs': 0,
        'weak_verbs_penalty': 0,
        'tools': 0,
        'jargon': 0,
        'trends': 0,
        'certifications': 0
    }

    feedback = []

    # Core keywords (15 points)
    core_keywords_list = field_keywords.get('core', [])
    core_matches = sum(1 for kw in core_keywords_list if kw.lower() in resume_text_lower)
    total_core_keywords = max(len(core_keywords_list), 1)
    scores['core_keywords'] = int((core_matches / total_core_keywords) * 15)

    # Soft skills (10 points)
    soft_skills_list = field_keywords.get('soft_skills', [])
    soft_skills_matches = sum(1 for skill in soft_skills_list if skill.lower() in resume_text_lower)
    total_soft_skills = max(len(soft_skills_list), 1)
    scores['soft_skills'] = int((soft_skills_matches / total_soft_skills) * 10)

    # Action verbs (5 points)
    action_verbs_list = field_keywords.get('action_verbs', [])
    action_verbs_matches = sum(1 for verb in action_verbs_list if verb.lower() in resume_text_lower)
    total_action_verbs = max(len(action_verbs_list), 1)
    scores['action_verbs'] = int((action_verbs_matches / total_action_verbs) * 5)

    # Weak verbs penalty (up to -5 points)
    weak_verbs_list = field_keywords.get('weak_verbs', [])
    weak_verbs_matches = sum(1 for verb in weak_verbs_list if verb.lower() in resume_text_lower)
    scores['weak_verbs_penalty'] = min(5, weak_verbs_matches)

    # Tools (10 points)
    tools_list = field_keywords.get('tools', [])
    tools_matches = sum(1 for tool in tools_list if tool.lower() in resume_text_lower)
    total_tools = max(len(tools_list), 1)
    scores['tools'] = int((tools_matches / total_tools) * 10)

    # Jargon (5 points)
    jargon_list = field_keywords.get('jargon', [])
    jargon_matches = sum(1 for term in jargon_list if term.lower() in resume_text_lower)
    total_jargon = max(len(jargon_list), 1)
    scores['jargon'] = int((jargon_matches / total_jargon) * 5)

    # Trends (5 points)
    trends_list = field_keywords.get('trends', [])
    trends_matches = sum(1 for trend in trends_list if trend.lower() in resume_text_lower)
    total_trends = max(len(trends_list), 1)
    scores['trends'] = int((trends_matches / total_trends) * 5)

    # Certifications (5 points)
    cert_patterns = [
        r'certified\s+\w+',
        r'certification in\s+\w+',
        r'\w+\s+certificate',
        r'course(s)? in\s+\w+'
    ]
    cert_matches = sum(len(re.findall(pattern, resume_text, re.IGNORECASE)) for pattern in cert_patterns)
    total_certifications = 4
    scores['certifications'] = int((cert_matches / total_certifications) * 5)

    # Feedback generation
    if scores['core_keywords'] < 10:
        unused_core_keywords = [kw for kw in core_keywords_list if kw.lower() not in resume_text_lower]
        if unused_core_keywords:
            feedback.append(
                get_dynamic_feedback(
                    category="core_keywords",
                    score=scores['core_keywords'],
                    examples=unused_core_keywords[:3]
                )
            )
    if scores['soft_skills'] < 6:
        unused_soft_skills = [skill for skill in soft_skills_list if skill.lower() not in resume_text_lower]
        if unused_soft_skills:
            feedback.append(
                get_dynamic_feedback(
                    category="soft_skills",
                    score=scores['soft_skills'],
                    examples=unused_soft_skills[:3]
                )
            )
    if scores['tools'] < 5:
        unused_tools = [tool for tool in tools_list if tool.lower() not in resume_text_lower]
        if unused_tools:
            feedback.append(
                get_dynamic_feedback(
                    category="tools",
                    score=scores['tools'],
                    examples=unused_tools[:3]
                )
            )
    if scores['trends'] < 3:
        unused_trends = [trend for trend in trends_list if trend.lower() not in resume_text_lower]
        if unused_trends:
            feedback.append(
                get_dynamic_feedback(
                    category="trends",
                    score=scores['trends'],
                    examples=unused_trends[:2]
                )
            )
    if scores['certifications'] < 3:
        feedback.append(
            get_dynamic_feedback(
                category="certifications",
                score=scores['certifications'],
                field=field
            )
        )

    # Calculate total score
    total_score = (
        scores['core_keywords'] +
        scores['soft_skills'] +
        scores['action_verbs'] -
        scores['weak_verbs_penalty'] +
        scores['tools'] +
        scores['jargon'] +
        scores['trends'] +
        scores['certifications']
    )

    return total_score, feedback, scores



def calculate_section_score(resume_text: str) -> tuple:
    """Calculate section score (20 points total)"""
    resume_text_lower = resume_text.lower()
    scores = {section: 0 for section in REQUIRED_SECTIONS}
    feedback = []

    for section in REQUIRED_SECTIONS:
        if section.lower() in resume_text_lower:
            scores[section] = 5
        else:
            feedback.append(
                get_dynamic_feedback(
                    category="sections",
                    section=section
                )
            )

    total_score = sum(scores.values())
    return total_score, feedback, scores



def calculate_formatting_score(resume_text: str, file_path: str) -> tuple:
    """Calculate formatting score (30 points total)"""
    feedback = []
    scores = {
        'bullet_points': 0,
        'page_length': 0,
        'contact_info': 0,
        'fonts': 0,
        'spacing': 0,
        'sections': 0
    }

    # Bullet points check
    if any(bullet in resume_text for bullet in ["•", "-", "*"]):
        scores['bullet_points'] = 5
    else:
        feedback.append(
            get_dynamic_feedback(
                category="bullet_points"
            )
        )

    if file_path.endswith(".pdf"):
        try:
            doc = fitz.open(file_path)  # open the pdf
            num_pages = len(doc)
            doc.close()
        except Exception as e:
            print(f"Error reading PDF for page count: {e}")
            num_pages = 0

        if num_pages <= 2:
            scores['page_length'] = 5
        else:
            feedback.append(
                get_dynamic_feedback(
                    category="page_length"
                )
            )
    else:  # handle docx
        try:
            doc = Document(file_path)
            num_pages = len(doc.paragraphs) # this is a rough estimate for docx
        except Exception as e:
            print(f"Error reading docx for page count: {e}")
            num_pages = 0
        if num_pages <= 2:
            scores['page_length'] = 5
        else:
            feedback.append(
                get_dynamic_feedback(
                    category="page_length"
                )
            )

    # Contact info check
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    if re.search(email_pattern, resume_text) and re.search(phone_pattern, resume_text):
        scores['contact_info'] = 5
    else:
        feedback.append("Include complete contact information")

    # Font check
    invalid_fonts, font_penalty = check_fonts(file_path)
    scores['fonts'] = 5 - font_penalty
    if invalid_fonts:
        feedback.append(
            "Your resume uses non-standard fonts. Switch to professional fonts such as Arial, Calibri, or Helvetica for better readability."
        )

    # Spacing check
    if not re.search(r'\n{3,}', resume_text):
        scores['spacing'] = 5
    else:
        feedback.append(
            get_dynamic_feedback(
                category="spacing"
            )
        )

    # Section headers check
    section_pattern = r'^[A-Z][A-Za-z\s]+:?$'
    if re.search(section_pattern, resume_text, re.MULTILINE):
        scores['sections'] = 5
    else:
        feedback.append("Use clear section headers")

    total_score = sum(scores.values())
    return total_score, feedback, scores



def analyze_resume(file_path: str, field: str) -> dict:
    """Main function to analyze resume and calculate scores"""
    try:
        # Read resume
        resume_text = parse_pdf(file_path) if file_path.lower().endswith(".pdf") else parse_docx(file_path)
        if not resume_text:
            return {"error": "Could not extract text from resume"}

        # Calculate scores
        keyword_score, keyword_feedback, keyword_details = calculate_keyword_score(resume_text, field)
        section_score, section_feedback, section_details = calculate_section_score(resume_text)
        formatting_score, formatting_feedback, formatting_details = calculate_formatting_score(resume_text, file_path)

        # Calculate total score
        total_score = keyword_score + section_score + formatting_score

        return {
            "total_score": total_score,
            "keyword_score": keyword_score,
            "section_score": section_score,
            "formatting_score": formatting_score,
            "feedback": {
                "keywords": keyword_feedback,
                "sections": section_feedback,
                "formatting": formatting_feedback
            },
        }

    except Exception as e:
        return {"error": str(e)}




app = Flask(__name__)

# Enhanced HTML template with modern UI
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Analyzer Pro</title>
    <style>
        :root {
            --primary-color: #4a90e2;
            --secondary-color: #f5f7fa;
            --accent-color: #34495e;
            --success-color: #2ecc71;
            --error-color: #e74c3c;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            width: 100%;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.5em;
        }

        .upload-container {
            background: var(--secondary-color);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            margin-bottom: 10px;
            color: var(--accent-color);
            font-weight: 600;
            font-size: 1.2em;
        }

        .file-input-container {
            position: relative;
            border: 2px dashed var(--primary-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .file-input-container:hover {
            background: rgba(74, 144, 226, 0.1);
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
            color: var(--primary-color);
            font-size: 1.1em;
            margin: 0;
        }

        select {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1.1em;
            color: var(--accent-color);
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select:focus {
            border-color: var(--primary-color);
            outline: none;
        }

        button {
            width: 100%;
            padding: 15px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        button:hover {
            background: #357abd;
            transform: translateY(-2px);
        }

        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }

        .feature-card {
            background: var(--secondary-color);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .feature-card h3 {
            color: var(--accent-color);
            margin-bottom: 10px;
        }

        .feature-card p {
            color: #666;
            font-size: 1.1em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Resume Analyzer Pro</h1>
            <p>Get instant feedback on your resume and improve your chances of landing that dream job</p>
        </div>

        <form action="/analyze_resume" method="post" enctype="multipart/form-data">
            <div class="upload-container">
                <div class="form-group">
                    <label for="file">Upload Your Resume</label>
                    <div class="file-input-container">
                        <input type="file" name="file" id="file" required accept=".pdf,.docx">
                        <p id="file-name-display">Drag and drop your resume here or click to browse</p>
                        <p style="font-size: 0.9em; margin-top: 10px; color: #666;">(PDF or DOCX format)</p>
                    </div>
                </div>

                <div class="form-group">
                    <label for="field">Select Your Industry</label>
                    <select name="field" id="field" required>
                        <option value="data_science">Data Science</option>
                        <option value="web_development">Web Development</option>
                        <option value="ui_ux">UI/UX Design</option>
                    </select>
                </div>
            </div>

            <button type="submit">Analyze Resume</button>
        </form>

        <div class="features">
            <div class="feature-card">
                <h3>Keyword Analysis</h3>
                <p>Check if your resume includes industry-specific keywords</p>
            </div>
            <div class="feature-card">
                <h3>Format Check</h3>
                <p>Ensure your resume follows professional formatting standards</p>
            </div>
            <div class="feature-card">
                <h3>Smart Feedback</h3>
                <p>Get personalized suggestions to improve your resume</p>
            </div>
        </div>
    </div>
    <script>
    document.getElementById('file').addEventListener('change', function() {
        const fileNameDisplay = document.getElementById('file-name-display');
        if (this.files && this.files.length > 0) {
            fileNameDisplay.textContent = this.files[0].name;
        } else {
            fileNameDisplay.textContent = 'Drag and drop your resume here or click to browse';
        }
    });
</script>
</body>
</html>
"""

# Enhanced Results Template
results_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Analysis Results</title>
    <style>
        :root {
            --primary-color: #4a90e2;
            --secondary-color: #f5f7fa;
            --accent-color: #34495e;
            --success-color: #2ecc71;
            --warning-color: #f1c40f;
            --error-color: #e74c3c;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: var(--accent-color);
            font-size: 2.5em;
            margin-bottom: 20px;
        }

        .score-overview {
            background: var(--secondary-color);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 40px;
        }

        .total-score {
            font-size: 3em;
            color: var(--primary-color);
            font-weight: bold;
            margin-bottom: 20px;
        }

        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .score-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }

        .score-card h3 {
            color: var(--accent-color);
            margin-bottom: 10px;
        }

        .score-value {
            font-size: 1.8em;
            color: var(--primary-color);
            font-weight: bold;
        }

        .feedback-section {
            margin-bottom: 40px;
        }

        .feedback-section h2 {
            color: var(--accent-color);
            margin-bottom: 20px;
            text-align: center;
        }

        .feedback-category {
            margin-bottom: 30px;
        }

        .feedback-category h3 {
            color: var(--primary-color);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--secondary-color);
        }

        .feedback-item {
            background: var(--secondary-color);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            position: relative;
            padding-left: 40px;
        }

        .feedback-item:before {
            content: "💡";
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
        }

        .back-button {
            display: inline-block;
            padding: 15px 30px;
            background: var(--primary-color);
            color: white;
            text-decoration: none;
            border-radius: 10px;
            margin-top: 20px;
            transition: all 0.3s ease;
        }

        .back-button:hover {
            background: #357abd;
            transform: translateY(-2px);
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            .header h1 {
                font-size: 2em;
            }

            .total-score {
                font-size: 2.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Resume Analysis Results</h1>
        </div>

        <div class="score-overview">
            <div class="total-score">{{ results.total_score }}/100</div>
            <p>Overall Resume Score</p>
        </div>

        <div class="feedback-section">
            <h2>Detailed Feedback</h2>
            
            {% if results.feedback.keywords %}
            <div class="feedback-category">
                <h3>Keyword Suggestions</h3>
                {% for item in results.feedback.keywords %}
                    <div class="feedback-item">{{ item }}</div>
                {% endfor %}
            </div>
            {% endif %}

            {% if results.feedback.sections %}
            <div class="feedback-category">
                <h3>Section Improvements</h3>
                {% for item in results.feedback.sections %}
                    <div class="feedback-item">{{ item }}</div>
                {% endfor %}
            </div>
            {% endif %}

            {% if results.feedback.formatting %}
            <div class="feedback-category">
                <h3>Formatting Tips</h3>
                {% for item in results.feedback.formatting %}
                    <div class="feedback-item">{{ item }}</div>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <div style="text-align: center;">
            <a href="/" class="back-button">Analyze Another Resume</a>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/analyze_resume', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    field = request.form.get('field', 'data_science')

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            if file.filename.lower().endswith(('.pdf', '.PDF')):
                suffix = ".pdf"
            elif file.filename.lower().endswith(('.docx', '.DOCX')):
                suffix = ".docx"
            else:
                return jsonify({'error': 'Invalid file type. Please upload a PDF or DOCX file.'})

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                file.save(temp_path)

            result = analyze_resume(temp_path, field)
            os.unlink(temp_path)

            if "error" in result:
                return jsonify(result)

            return render_template_string(results_html, results=result)

        except Exception as e:
            return jsonify({'error': f"Exception occurred during analysis: {str(e)}"})

    return jsonify({'error': 'An error occurred'})

if __name__ == '__main__':
    app.run(debug=True)
