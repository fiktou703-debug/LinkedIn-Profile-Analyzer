import streamlit as st
import re
import os
from collections import Counter
import nltk
import spacy

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # dotenv not installed, relying on environment or manual input

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Install spaCy model: python -m spacy download en_core_web_sm")
    st.stop()

# Gemini AI setup
def configure_genai(api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash"), True
    except Exception as e:
        return None, False

# API Key Handling
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    with st.sidebar:
        api_key = st.text_input("🔑 Enter Gemini API Key", type="password")
        st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")

model = None
AI_ENABLED = False

if api_key:
    model, AI_ENABLED = configure_genai(api_key)

# Setup
st.set_page_config(page_title="LinkedIn Profile Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 LinkedIn Psychology & Archetype Analyzer")
st.markdown("### AI-powered Digital Psychology & Personal SEO Insights")

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    if AI_ENABLED:
        st.success("🤖 Gemini AI Connected")
    else:
        st.warning("🤖 AI Disconnected (Add Key)")
with col2:
    st.info("🧠 NLP Analysis Active")
with col3:
    st.info("📊 Psychology Engine Ready")

# Input
input_type = st.radio("Input Method:", ["Paste Text", "Upload PDF", "LinkedIn URL"])

profile_text = ""

if input_type == "Upload PDF":
    st.info("📄 Export your LinkedIn profile: Profile → More → Save to PDF")
    uploaded_file = st.file_uploader("Upload LinkedIn Profile PDF:", type="pdf", help="Upload the PDF file exported from your LinkedIn profile")
    if uploaded_file:
        try:
            try:
                import PyPDF2
            except ImportError:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
                import PyPDF2
            
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + "\\n"
            profile_text = pdf_text.strip()
            if profile_text:
                st.success("✅ PDF text extracted successfully!")
                st.text_area("Extracted PDF Content:", value=profile_text, height=200)
            else:
                st.warning("⚠️ No text found in PDF. Try a different file.")
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            st.info("💡 Try restarting the app or use manual text input")
elif input_type == "LinkedIn URL":
    url = st.text_input("LinkedIn Profile URL:")
    st.info("💡 LinkedIn blocks scraping. Copy profile text manually.")
    profile_text = st.text_area("LinkedIn Profile Content:", height=300)
else:
    profile_text = st.text_area("LinkedIn Profile Content:", height=300)

def analyze_profile(text):
    if not text:
        return {}, [], 0, [], []
    
    # Use spaCy for NLP processing
    doc = nlp(text)
    
    # Extract entities and technical terms
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "PERSON"]]
    
    # Use NLTK for tokenization and POS tagging
    from nltk.corpus import stopwords
    
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can'}
    
    # Extract meaningful tokens using spaCy
    meaningful_tokens = [token.lemma_.lower() for token in doc 
                        if token.is_alpha and not token.is_stop and len(token.text) > 2]
    
    # Extract noun phrases for skills
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    
    # Technical terms database
    tech_terms = {'python', 'java', 'javascript', 'react', 'nodejs', 'aws', 'azure', 'docker', 'sql', 'mongodb', 'git', 'machine learning', 'data science', 'analytics', 'tensorflow', 'pytorch', 'kubernetes', 'devops', 'agile', 'scrum', 'artificial intelligence', 'deep learning'}
    
    # Combine all extracted terms
    all_terms = meaningful_tokens + noun_phrases + entities
    
    # Filter for technical keywords
    tech_keywords = [term for term in all_terms if any(tech in term for tech in tech_terms)]
    other_keywords = [term for term in meaningful_tokens if term not in stop_words and len(term) > 3]
    
    # Prioritize technical terms
    keywords = tech_keywords + other_keywords
    keyword_freq = Counter(keywords)
    
    # Enhanced skills detection using spaCy entities and patterns
    skills_db = ["python", "java", "javascript", "sql", "react", "aws", "machine learning", "data science", "project management", "leadership", "communication", "teamwork", "agile", "scrum", "docker", "kubernetes"]
    
    # Use both exact matching and spaCy similarity
    matched_skills = []
    text_lower = text.lower()
    
    for skill in skills_db:
        if skill in text_lower:
            matched_skills.append(skill.title())
    
    matched_skills = list(set(matched_skills))
    
    # Score calculation
    score = min(100, len(text.split()) // 10 + len(matched_skills) * 5 + len(set(keywords)) // 5)
    
    # Suggestions
    tips = []
    if len(text.split()) < 100: tips.append("Add more content")
    if len(matched_skills) < 3: tips.append("Include more skills")
    if not any(w in text.lower() for w in ["achieved", "led", "improved"]): tips.append("Add achievements")
    
    # Career roles
    roles = []
    if any(s in text.lower() for s in ["python", "java", "javascript"]): roles.append("Software Developer")
    if any(s in text.lower() for s in ["data", "machine learning"]): roles.append("Data Scientist")
    if any(s in text.lower() for s in ["management", "leadership"]): roles.append("Project Manager")
    
    return keyword_freq, matched_skills, score, tips, roles

# Analysis
if st.button("🔍 Analyze Profile Psychology"):
    if profile_text:
        keyword_freq, skills, score, tips, roles = analyze_profile(profile_text)
        
        # Original Metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Base Profile Score", f"{score}/100")
        with col2: st.metric("Detected Skills", len(skills))
        with col3: st.metric("Word Count", len(profile_text.split()))
        
        st.divider()
        
        # AI insights with Gemini
        if AI_ENABLED and model:
            with st.spinner("🤖 Consulting the Digital Psychologist... Analyzing Archetypes & Trust Signals..."):
                try:
                    prompt = f"""
                    Act as a Digital Psychologist and Personal Branding Expert (like Cialdini meets a LinkedIn Strategist). 
                    Analyze this LinkedIn profile to reveal hidden psychological signals and personal brand strength.

                    Profile Content:
                    {profile_text[:3500]}

                    Detected Skills: {', '.join(skills)}

                    Provide a deep psychological analysis with the following structured sections:

                    ### 1. 🎭 Professional Archetype
                    Identify the user's primary and secondary Brand Archetypes (e.g., The Sage, The Ruler, The Hero, The Creator, The Outlaw, The Magician, etc.).
                    - **Primary Archetype:** [Name] - [Brief explanation of why]
                    - **Secondary Archetype:** [Name] - [Brief explanation]
                    - **Psychological Tone:** Describe the mood of their writing (e.g., Authoritative, Empathetic, Disruptive).

                    ### 2. 🧠 Big 5 Personality Insights (Implied)
                    Analyze the writing style to infer where they likely stand on the Big 5 traits:
                    - **Openness:** [High/Medium/Low] - [Evidence from text]
                    - **Conscientiousness:** [High/Medium/Low] - [Evidence]
                    - **Extraversion:** [High/Medium/Low] - [Evidence]
                    - **Agreeableness:** [High/Medium/Low] - [Evidence]
                    - **Neuroticism:** [Implied Emotional Stability level]

                    ### 3. 🛡️ Trust Signals & Persuasion (Cialdini's Principles)
                    Evaluate how well the profile builds trust using Robert Cialdini's principles. Give a score (0-10) for each:
                    - **Authority:** [Score/10] - Do they demonstrate expertise? (e.g., quantified results, prestigious roles).
                    - **Social Proof:** [Score/10] - Are there mentions of well-known clients, publications, or large numbers?
                    - **Consistency:** [Score/10] - Does the narrative align with their detailed experience?
                    - **Liking:** [Score/10] - Is the "About" section personable and relatable?

                    ### 4. 🚀 "A+" Headline & Positioning Audit
                    - **Headline Grade:** [A/B/C/D]
                    - **Critique:** Does it follow the 'What I do + Who I do it for + How' formula?
                    - **Improvement:** Specific rewrite suggestions to make it punchier and more psychologically compelling.

                    ### 5. 💡 Strategic Recommendations
                    - **Psychology-Based Improvements:** How to tweak the language to better match their target archetype.
                    - **Missing Trust Signals:** What specific elements (numbers, testmonials, case studies) are missing?

                    Format the output in clean, structured Markdown with emojis.
                    """
                    
                    response = model.generate_content(prompt)
                    st.markdown("## 🧠 Digital Psychology Analysis")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
                    st.info("💡 Check your API key and internet connection")
        elif not AI_ENABLED:
            st.error("⚠️ Gemini API Key Required for Psychology Analysis")
            st.info("Add GEMINI_API_KEY to .env or enter it in the sidebar.")
            
    else:
        st.warning("Please enter profile content first.")

# Instructions
with st.expander("ℹ️ About Archetypes & Trust Signals"):
    st.markdown(\"""
    **Professional Archetypes:**
    - **The Sage**: Seeks truth (Teachers, Researchers).
    - **The Ruler**: Seeks control & order (CEOs, Managers).
    - **The Creator**: Seeks innovation (Designers, Artists).
    
    **Trust Signals (Cialdini):**
    - **Authority**: Showing you are an expert.
    - **Social Proof**: Showing others trust you.
    \""")
