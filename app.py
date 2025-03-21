from flask import Flask, request, jsonify
import os
import tempfile
import fitz
from PyPDF2 import PdfReader
from docx import Document
import re
import random

# ... (Your KEYWORD_DICT, REQUIRED_SECTIONS, ALLOWED_FONTS, FEEDBACK_POOL,
#      get_dynamic_feedback, parse_pdf, parse_docx, check_fonts,
#      calculate_keyword_score, calculate_section_score,
#      calculate_formatting_score, analyze_resume functions) ...
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
        "Your soft skills section scored {score}/10. Elaborate on teamwork or leadership in your roles.",
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
        "You have {score}/5 for tools. Add more technologies like {examples} to improve this section.",
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
        "You scored {score}/10 for certifications. Adding programs like {examples} could enhance your profile.",
        "Showcase relevant certifications to emphasize your technical expertise.",
        "Consider pursuing certifications in {examples} to boost your score.",
        "Certifications like {examples} can validate your skills and improve recruiter interest."
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
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""


def parse_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
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
        'core_keywords': 0,  # 15 points
        'soft_skills': 0,  # 10 points
        'action_verbs': 0,  # 5 points
        'weak_verbs_penalty': 0,  # -5 points max
        'tools': 0,  # 10 points
        'jargon': 0,  # 5 points
        'trends': 0,  # 5 points
        'certifications': 0  # 5 points
    }

    feedback = []

    # Core keywords (15 points)
    core_keywords_list = field_keywords.get('core', [])
    core_matches = sum(1 for kw in core_keywords_list if kw.lower() in resume_text_lower)
    total_core_keywords = max(len(core_keywords_list), 1)  # Avoid division by zero
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
    scores['weak_verbs_penalty'] = min(5, weak_verbs_matches)  # Keep penalty capped

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
    total_certifications = 4  # Assume a max of 4 certification patterns for dynamic scaling
    scores['certifications'] = int((cert_matches / total_certifications) * 5)

    # Feedback generation
    if scores['core_keywords'] < 10:
        feedback.append(
            get_dynamic_feedback(
                category="core_keywords",
                score=scores['core_keywords'],
                examples=core_keywords_list[:3]
            )
        )
    if scores['soft_skills'] < 6:
        feedback.append(
            get_dynamic_feedback(
                category="soft_skills",
                score=scores['soft_skills'],
                examples=soft_skills_list[:3]
            )
        )
    if scores['tools'] < 5:
        feedback.append(
            get_dynamic_feedback(
                category="tools",
                score=scores['tools'],
                examples=tools_list[:3]
            )
        )
    if scores['trends'] < 3:
        feedback.append(
            get_dynamic_feedback(
                category="trends",
                score=scores['trends'],
                examples=trends_list[:2]
            )
        )
    if scores['certifications'] < 3:
        feedback.append(
            get_dynamic_feedback(
                category="certifications",
                score=scores['certifications'],
                field=field  # Pass field to avoid N/A
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
        'bullet_points': 0,  # 5 points
        'page_length': 0,  # 5 points
        'contact_info': 0,  # 5 points
        'fonts': 0,  # 5 points
        'spacing': 0,  # 5 points
        'sections': 0  # 5 points
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
        num_pages = len(PdfReader(file_path).pages)
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
            "detailed_scores": {
                "keywords": keyword_details,
                "sections": section_details,
                "formatting": formatting_details
            }
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    # Example usage
    file_path = r"C:\Users\My Pc\Downloads\Resume Checker\Shreyas_Naphad_Resume1.pdf"  # Replace with actual path
    field = "data_science"  # or "data_science"

    if not os.path.exists(file_path):
        print("File not found.")
        return

    results = analyze_resume(file_path, field)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n=== Resume Analysis Results ===")
    print(f"\nTotal Score: {results['total_score']}/100")
    print(f"- Keyword Score: {results['keyword_score']}/50")
    print(f"- Section Score: {results['section_score']}/20")
    print(f"- Formatting Score: {results['formatting_score']}/30")

    print("\nFeedback:")
    for category, feedback_list in results['feedback'].items():
        if feedback_list:
            print(f"\n{category.title()} Feedback:")
            for feedback in feedback_list:
                print(f"- {feedback}")
        else:
            print(f"\n{category.title()} Feedback: Great job here!")


#if __name__ == "__main__":
 #   main()

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Welcome to the Resume Analyzer</h1>
        <p>Use the <strong>/analyze_resume</strong> endpoint to upload and analyze your resume.</p>
        <form action="/analyze_resume" method="post" enctype="multipart/form-data">
            <label for="file">Upload Resume (PDF only):</label><br>
            <input type="file" name="file" id="file" required><br><br>
            <label for="field">Select Field:</label><br>
            <select name="field" id="field">
                <option value="data_science">Data Science</option>
                <option value="web_development">Web Development</option>
            </select><br><br>
            <button type="submit">Analyze Resume</button>
        </form>
    </body>
    </html>
    '''

@app.route('/analyze_resume', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    field = request.form.get('field', 'data_science') # Get field from form. Default to data science.

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            result = analyze_resume(temp_file.name, field)
            os.unlink(temp_file.name) # Delete temp file.
        return jsonify(result)

    return jsonify({'error': 'An error occurred'})

if __name__ == '__main__':
    app.run(debug=True)
