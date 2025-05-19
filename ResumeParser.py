import streamlit as st
import openai
import io
import json
import pandas as pd
import numpy as np
import PyPDF2
import docx
import altair as alt
from rapidfuzz import fuzz  # For fuzzy matching

# --------------------------
# Helper Functions
# --------------------------

def extract_text_from_pdf(file_obj):
    """Extracts text from a PDF file-like object."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_obj):
    """Extracts text from a DOCX file-like object."""
    try:
        doc = docx.Document(file_obj)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text(uploaded_file):
    """
    Determines the file type (PDF or DOCX) and extracts text accordingly.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        return ""

# --- Semantic Similarity Functions ---
def get_embedding(text, openai_api_key, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try:
        openai.api_key = openai_api_key
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding API error: {e}")
        return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_semantic_similarity(jd_text, resume_text, openai_api_key):
    jd_embedding = get_embedding(jd_text, openai_api_key)
    resume_embedding = get_embedding(resume_text, openai_api_key)
    if jd_embedding is None or resume_embedding is None:
        return 0
    sim = cosine_similarity(jd_embedding, resume_embedding)
    # Cosine similarity is usually between 0 and 1 for these embeddings.
    return round(sim * 100, 2)

def parse_resume(resume_text, jd_text, weights, openai_api_key):
    """
    Use OpenAI to extract candidate details and calculate a match score,
    taking into account recruiter-defined weights.
    """
    prompt = f"""
You are a resume parsing and matching assistant.

Job Description:
----------------
{jd_text}

Candidate Resume:
-----------------
{resume_text}

The recruiter has set the following weightings for evaluating the candidate:
- Skills weight: {weights['skills']}
- Experiences weight: {weights['experiences']}
- Certifications weight: {weights['certifications']}

Based on the job description and the provided resume, extract the following details and output a valid JSON object (without markdown formatting):

- "name": The candidate's full name.
- "address": The candidate's address (if available).
- "location": The candidate's current location (if mentioned; if not available, output "Not provided").
- "contact": The candidate's email address or other contact information.
- "phone": The candidate's phone number.
- "companies": A list of companies the candidate has worked for.
- "experiences": A brief summary of the candidate's experiences.
- "skills": A list of skills mentioned in the resume.
- "certifications": A list of certifications (if mentioned) or an empty list.
- "match_percentage": A number between 0 and 100 representing the candidate’s fit for the job, taking into account the above weightings.
- "good_things": The positive aspects of the candidate.
- "gaps": The drawbacks or gaps in the candidate's profile.
- "final_decision": The overall recommendation (Recommended, Good to have, or Rejected).

Ensure that the JSON is well-formatted.
"""
    try:
        openai.api_key = openai_api_key
        response = openai.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant for resume parsing."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=0  # Use 0 for more deterministic output.
        )
        result_text = response.choices[0].message.content
        parsed_result = json.loads(result_text)
        return parsed_result
    except Exception as e:
        st.error(f"Error during OpenAI API call: {e}")
        return None

def generate_interview_questions(candidate_data, jd_text, openai_api_key):
    """
    Generate 5 competency-based HR screening questions in multiple-choice or true/false format.
    These questions are designed to evaluate whether the candidate actually possesses the skills
    and experiences claimed in their resume and meets the key requirements of the job.
    The questions should be scenario or competency based, rather than simply recalling resume details.
    Each question must include:
        - The question text (phrased so an HR person can ask it),
        - Answer options (for multiple-choice, provide 4 options labeled A, B, C, D; for true/false, use "True/False"),
        - The correct answer clearly indicated.
    Please output the 5 questions as a numbered list in plain text.
    """
    prompt = f"""
You are an expert HR screening specialist. Your task is to generate 5 competency-based screening questions that help verify whether a candidate truly possesses the skills and experience they claim, and meets the key requirements outlined in the job description. The questions should be formulated in a way that:
- They are scenario or competency-based (e.g., “What would you do if …?” or “True or False: …”),
- They require the candidate to explain or choose an answer that reflects practical competence,
- They are easy for a non-technical HR professional to administer and evaluate.

Use the information below to craft questions that are directly relevant to the candidate's background and the job requirements.

Candidate Details:
-------------------
Name: {candidate_data.get('name', 'N/A')}
Experiences: {candidate_data.get('experiences', 'N/A')}
Skills: {', '.join(candidate_data.get('skills', [])) if candidate_data.get('skills') else 'N/A'}
Certifications: {', '.join(candidate_data.get('certifications', [])) if candidate_data.get('certifications') else 'None'}
Good Aspects: {candidate_data.get('good_things', 'N/A')}
Gaps: {candidate_data.get('gaps', 'N/A')}
Final Decision: {candidate_data.get('final_decision', 'N/A')}

Job Description:
----------------
{jd_text}

Based on the above, generate 5 screening questions. For each question, include:
1. A scenario or competency-based question that tests the candidate's real-world application of their skills and experience.
2. The answer options. For multiple-choice questions, provide 4 options labeled A, B, C, and D; for true/false questions, simply list "True/False".
3. The correct answer clearly indicated (e.g., "Correct Answer: B" or "Correct Answer: True").

Please output the questions as a numbered list in plain text without markdown formatting.
    """
    try:
        openai.api_key = openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR screening specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        questions_text = response.choices[0].message.content
        return questions_text
    except Exception as e:
        st.error(f"Error during OpenAI API call (interview questions): {e}")
        return "Could not generate interview questions."

def parse_search_query(query, openai_api_key):
    """
    Use an LLM to convert a natural language search query into structured filters.
    Candidate profiles include many fields such as overall_experience, location, skills, companies,
    certifications, match_percentage, semantic_similarity, adjusted_match, good_things, gaps, final_decision, etc.
    Output the filters in JSON format, using null for fields not mentioned.
    For numeric fields, output an operator with a number (e.g., ">=5").
    """
    prompt = f"""
You are an assistant that extracts search filters from natural language queries about candidate profiles.
Candidate profiles have the following fields:

- location (string)
- skills (list of strings)
- companies (list of strings)
- certifications (list of strings)
- match_percentage (numeric)
- semantic_similarity (numeric)
- adjusted_match (numeric)
- good_things (string)
- gaps (string)
- final_decision (string)

Extract any filters mentioned in the query. For numeric filters, include an operator (>, >=, <, <=, =) followed by a number.
For list or string filters, output the desired value or a list of values.
Output the result in JSON format like:
{{
    "location": "<location>" or null,
    "skills": [list of skills] or null,
    "companies": [list of companies] or null,
    "certifications": [list of certifications] or null,
    "match_percentage": "<operator><number>" or null,
    "semantic_similarity": "<operator><number>" or null,
    "adjusted_match": "<operator><number>" or null,
    "good_things": "<text>" or null,
    "gaps": "<text>" or null,
    "final_decision": "<text>" or null
}}

Query: {query}
"""
    try:
        openai.api_key = openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts search filters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        filters_text = response.choices[0].message.content.strip()
        filters = json.loads(filters_text)
        return filters
    except Exception as e:
        st.error(f"Error parsing search query: {e}")
        return {}

def apply_filters(candidates, filters):
    """
    Filter the candidate list based on arbitrary filters provided.
    For each key in filters (if not null), apply filtering.
    For numeric fields, parse the operator and compare numbers.
    For string fields and list fields, use fuzzy matching to allow for approximate matches.
    Also support a simple negation if the filter value starts with an exclamation mark ("!").
    """
    threshold = 70  # Fuzzy matching threshold (0-100)
    filtered = []
    for cand in candidates:
        meets = True
        for key, filter_value in filters.items():
            if filter_value is None:
                continue  # No filter for this field
            cand_value = cand.get(key, "")
            # Numeric fields:
            if key in ["overall_experience", "match_percentage", "semantic_similarity", "adjusted_match"]:
                try:
                    op = filter_value[0:2] if len(filter_value) > 1 and filter_value[1] in "=<>!" else filter_value[0]
                    num_filter = float(filter_value[len(op):])
                    try:
                        cand_num = float(cand_value)
                    except:
                        meets = False
                        break
                    if op == ">=" and not (cand_num >= num_filter):
                        meets = False
                        break
                    elif op == ">" and not (cand_num > num_filter):
                        meets = False
                        break
                    elif op == "<=" and not (cand_num <= num_filter):
                        meets = False
                        break
                    elif op == "<" and not (cand_num < num_filter):
                        meets = False
                        break
                    elif (op == "=" or op == "==") and not (cand_num == num_filter):
                        meets = False
                        break
                except Exception as e:
                    continue

            # For fields that are lists (e.g., skills, companies, certifications):
            elif key in ["skills", "companies", "certifications"]:
                # Assume candidate field is a comma-separated string
                if isinstance(cand_value, str):
                    cand_list = [item.strip().lower() for item in cand_value.split(",")]
                else:
                    cand_list = []
                # Ensure filter is a list
                if isinstance(filter_value, str):
                    # Check for negation: if filter starts with "!", then we require a low match score.
                    negate = filter_value.startswith("!")
                    filter_term = filter_value[1:] if negate else filter_value
                    match_found = any(fuzz.token_set_ratio(item, filter_term.lower()) >= threshold for item in cand_list)
                    if negate and match_found:
                        meets = False
                        break
                    elif not negate and not match_found:
                        meets = False
                        break
                else:
                    # If filter is a list, check that at least one element approximately matches
                    match_found = False
                    for term in filter_value:
                        term = term.lower()
                        if any(fuzz.token_set_ratio(item, term) >= threshold for item in cand_list):
                            match_found = True
                            break
                    if not match_found:
                        meets = False
                        break

            # For other string fields (e.g., location, final_decision, good_things, gaps):
            else:
                if isinstance(cand_value, str):
                    # Check for negation
                    negate = False
                    filter_term = filter_value
                    if isinstance(filter_value, str) and filter_value.startswith("!"):
                        negate = True
                        filter_term = filter_value[1:]
                    # Use fuzzy matching between candidate value and filter term
                    score = fuzz.token_set_ratio(cand_value.lower(), filter_term.lower())
                    if negate:
                        # For negation, if score is high, candidate should be excluded
                        if score >= threshold:
                            meets = False
                            break
                    else:
                        if score < threshold:
                            meets = False
                            break
                else:
                    meets = False
                    break
        if meets:
            filtered.append(cand)
    return filtered

def semantic_search_tab(openai_api_key):
    st.subheader("Semantic Search")
    query = st.text_input("Enter your search query (e.g., 'Show me candidates with at least 5 years of ML experience, not from Bangalore, who have worked in TCS'):")
    if st.button("Search"):
        if "results" not in st.session_state:
            st.info("No candidate results available. Please process resumes first.")
        elif not query.strip():
            st.info("Please enter a valid query.")
        else:
            with st.spinner("Parsing query and filtering candidates..."):
                filters = parse_search_query(query, openai_api_key)
                st.write("Derived filters:", filters)
                candidates = st.session_state["results"]
                filtered_candidates = apply_filters(candidates, filters)
            if filtered_candidates:
                st.write(f"Found {len(filtered_candidates)} matching candidate(s):")
                # Display filtered candidates as a table of key details
                flattened = [flatten_candidate(cand) for cand in filtered_candidates]
                df = pd.DataFrame(flattened)
                st.dataframe(df, use_container_width=True)
            else:
                st.write("No candidates match the specified criteria.")

# --------------------------
# Processing Functions
# --------------------------
def process_resumes(resume_files, jd_file, weights, openai_api_key):
    """
    Process each resume file: extract text, compute semantic similarity,
    parse details via OpenAI, and combine scores.
    """
    jd_text = extract_text(jd_file)

    results = []
    for uploaded_file in resume_files:
        resume_text = extract_text(uploaded_file)
        st.info(f"Processing {uploaded_file.name} ...")
        parsed = parse_resume(resume_text, jd_text, weights, openai_api_key)
        if parsed:
            # Compute semantic similarity score
            sem_sim = compute_semantic_similarity(jd_text, resume_text, openai_api_key)
            parsed["semantic_similarity"] = sem_sim
            try:
                parsed_match = float(parsed.get("match_percentage", 0))
            except:
                parsed_match = 0
            # Use a weighted average: 70% match_percentage and 30% semantic similarity
            adjusted_match = round((0.7 * parsed_match + 0.3 * sem_sim), 2)
            parsed["adjusted_match"] = adjusted_match

            # Set final decision using stricter thresholds
            if adjusted_match >= 85:
                parsed["final_decision"] = "Highly Recommended"
            elif adjusted_match >= 70:
                parsed["final_decision"] = "Recommended"
            elif adjusted_match >= 50:
                parsed["final_decision"] = "Good to Have"
            else:
                parsed["final_decision"] = "Rejected"

            parsed["filename"] = uploaded_file.name
            results.append(parsed)
    return results

def build_skill_frequency_chart(results):
    """Build a chart for the frequency of skills across all resumes."""
    skill_counts = {}
    for cand in results:
        for skill in cand.get("skills", []):
            skill = skill.strip().lower()
            if skill:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    if skill_counts:
        df_skills = pd.DataFrame(list(skill_counts.items()), columns=["Skill", "Count"])
        chart = alt.Chart(df_skills).mark_bar().encode(
            x=alt.X("Skill:N", sort='-y'),
            y="Count:Q",
            tooltip=["Skill", "Count"]
        ).properties(width=600, height=300)
        return chart
    return None

def flatten_candidate(candidate):
    """Converts candidate dict values to strings if they are lists or other non-scalar types."""
    flat = {}
    for key, value in candidate.items():
        if isinstance(value, list):
            flat[key] = ", ".join([str(v) for v in value])
        else:
            flat[key] = value
    return flat

# --------------------------
# Main Streamlit UI
# --------------------------
def main():
    st.title("Resume Insights")

    if "openai_api_key" not in st.session_state:
        st.session_state["logged_in"] = False
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "show_upload" not in st.session_state:
        st.session_state["show_upload"] = False

    if not st.session_state["logged_in"]:
        st.write("Please enter your OpenAI API key to proceed.")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        login_button = st.button("Login")
        if login_button and openai_api_key:
            try:
                openai.api_key = openai_api_key
                openai.models.list()  # Try a simple API call to verify the key
                st.session_state["openai_api_key"] = openai_api_key
                st.session_state["logged_in"] = True
                st.rerun()
            except openai.AuthenticationError:
                st.error("Invalid OpenAI API key. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif not st.session_state["show_upload"]:
        # --- Introduction Screen ---
        st.subheader("Welcome to the Resume Parsing & Matching Tool!")
        st.write("""
        This tool helps you efficiently analyze and compare candidate resumes against job descriptions. Here's a quick overview:
        """)
        st.markdown("""
        * **Evaluation Weights (Sidebar):** Use the sliders on the left to adjust the importance of Skills, Experiences, and Certifications in the matching process.
        * **Upload & Process Tab:** Upload multiple resume files (PDF or DOCX) and a single Job Description file (PDF or DOCX). Click 'Process Resumes' to analyze them.
        * **Dashboard Tab:** View a summary of processed candidates, their match scores, key information extracted from their resumes, and the final recommendation. You can also filter candidates by the final decision.
        * **Export Data Tab:** Download the processed candidate data as a CSV file for further analysis or record-keeping.
        * **Semantic Search Tab:** Perform natural language searches across the processed candidate profiles to find specific qualifications or experiences.
        * **Interview Questions Tab:** Select a processed candidate to generate competency-based interview questions tailored to their profile and the job description.
        * **Feedback Tab:** Provide and review feedback on individual candidate recommendations.
        """)
        proceed_button = st.button("Proceed to Upload Files")
        if proceed_button:
            st.session_state["show_upload"] = True
            st.rerun()
    else:
        st.write("---")
        # --- Inject Custom CSS for a Trendy, Colorful Look ---
        st.markdown("""
        <style>
        .candidate-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        }
        .candidate-card h2 {
            color: #007bff;
            margin-bottom: 0.5rem;
        }
        .candidate-summary {
            font-size: 14px;
            color: #343a40;
            margin: 2px 0;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- Sidebar: Recruiter-Defined Weights ---
        st.sidebar.header("Set Evaluation Weights")
        skills_weight = st.sidebar.slider("Skills Weight", 0, 10, 5)
        experiences_weight = st.sidebar.slider("Experiences Weight", 0, 10, 5)
        certifications_weight = st.sidebar.slider("Certifications Weight", 0, 10, 3)
        weights = {
            "skills": skills_weight,
            "experiences": experiences_weight,
            "certifications": certifications_weight
        }

        # --- Create Tabs ---
        upload_tab, dashboard_tab_tab, export_data_tab, semantic_search_tab_ui, interview_tab, feedback_tab = st.tabs(
            ["Upload & Process", "Dashboard", "Export Data", "Semantic Search", "Interview Questions", "Feedback"]
        )

        # ----- Upload & Process Tab -----
        with upload_tab:
            st.subheader("Upload Files")
            resume_files = st.file_uploader("Select Resume Files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
            jd_file = st.file_uploader("Select Job Description File (PDF or DOCX)", type=["pdf", "docx"])
            if st.button("Process Resumes", key="process_button"):
                if not resume_files or not jd_file:
                    st.error("Please upload both resume files and a job description file.")
                else:
                    with st.spinner("Processing resumes..."):
                        results = process_resumes(resume_files, jd_file, weights, st.session_state["openai_api_key"])
                        if results:
                            st.session_state["results"] = results  # Store results in session state
                            st.session_state["jd_file"] = jd_file # Store JD file
                            st.success("Processing complete!")
                        else:
                            st.error("No candidate results obtained. Please check your inputs and OpenAI API key.")

        # ----- Dashboard Tab -----
        with dashboard_tab_tab:
            st.subheader("Candidate Profiles Dashboard")
            if "results" not in st.session_state:
                st.info("No candidate results available. Please process resumes first.")
            else:
                results = st.session_state["results"]
                total_profiles = len(results)
                recommended = len([cand for cand in results if cand.get("final_decision", "").lower() in ["recommended", "highly recommended"]])
                rejected = len([cand for cand in results if cand.get("final_decision", "").lower() == "rejected"])
                good_to_have = len([cand for cand in results if cand.get("final_decision", "").lower() == "good to have"])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Profiles", total_profiles)
                col2.metric("Recommended", recommended)
                col3.metric("Rejected", rejected)
                col4.metric("Good To Have", good_to_have)

                st.markdown("---")
                st.write("### Candidate Profiles")
                final_decision_options = ["All"]
                for cand in results:
                    decision = cand.get("final_decision", "N/A")
                    if decision not in final_decision_options:
                        final_decision_options.append(decision)
                selected_decision = st.selectbox("Filter by Final Decision", options=final_decision_options)
                filtered_candidates = [cand for cand in results if cand.get("final_decision") == selected_decision] if selected_decision != "All" else results

                for cand in filtered_candidates:
                    candidate_flat = flatten_candidate(cand)
                    st.markdown(f"""
                    <div class="candidate-card">
                        <h2>{candidate_flat.get('name', 'N/A')}</h2>
                        <p class="candidate-summary"><strong>Location:</strong> {candidate_flat.get('location', candidate_flat.get('address', 'Not provided'))}</p>
                        <p class="candidate-summary"><strong>Contact:</strong> {candidate_flat.get('contact', 'Not provided')} | <strong>Phone:</strong> {candidate_flat.get('phone', 'Not provided')}</p>
                        <p class="candidate-summary"><strong>Companies:</strong> {candidate_flat.get('companies', 'Not provided')}</p>
                        <p class="candidate-summary"><strong>Match Percentage:</strong> {candidate_flat.get('match_percentage', 'N/A')}</p>
                        <p class="candidate-summary"><strong>Semantic Similarity:</strong> {candidate_flat.get('semantic_similarity', 'N/A')}</p>
                        <p class="candidate-summary"><strong>Adjusted Match:</strong> {candidate_flat.get('adjusted_match', 'N/A')}</p>
                        <p class="candidate-summary"><strong>Good Aspects:</strong> {candidate_flat.get('good_things', 'N/A')}</p>
                        <p class="candidate-summary"><strong>Gaps:</strong> {candidate_flat.get('gaps', 'N/A')}</p>
                        <p class="candidate-summary"><strong>Final Decision:</strong> {candidate_flat.get('final_decision', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ----- Export Data Tab -----
        with export_data_tab:
            st.subheader("Export Candidate Data")
            if "results" in st.session_state:
                results = st.session_state["results"]
                flattened_candidates = [flatten_candidate(cand) for cand in results]
                df = pd.DataFrame(flattened_candidates)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(label="Export data as CSV",
                                    data=csv,
                                    file_name="candidates.csv",
                                    mime="text/csv")
            else:
                st.info("No candidate results available. Please process resumes first.")

        # ----- Semantic Search Tab -----
        with semantic_search_tab_ui:
            semantic_search_tab(st.session_state["openai_api_key"])

        # ----- Interview Questions Tab -----
        with interview_tab:
            if "results" in st.session_state and st.session_state.get("jd_file") is not None:
                results = st.session_state["results"]
                jd_file = st.session_state["jd_file"]
                candidate_options = [cand.get("filename", f"Candidate {i}") for i, cand in enumerate(results)]
                with st.form("interview_questions_form"):
                    candidate_for_interview = st.selectbox("Select a candidate for interview question generation", candidate_options)
                    generate_submitted = st.form_submit_button("Generate Questions")
                if generate_submitted:
                    candidate = next((cand for cand in results if cand.get("filename") == candidate_for_interview), None)
                    if candidate:
                        jd_text = extract_text(jd_file)
                        questions = generate_interview_questions(candidate, jd_text, st.session_state["openai_api_key"])
                        st.text_area("Interview Questions", value=questions, height=300)
            else:
                st.info("No candidate results available or Job Description missing. Please process resumes first.")

        # ----- Feedback Tab -----
        with feedback_tab:
            if "results" in st.session_state:
                results = st.session_state["results"]
                st.subheader("Provide Feedback on Recommendations")
                feedback_data = {}
                for cand in results:
                    with st.expander(f"Feedback for {cand.get('name', cand.get('filename'))}"):
                        feedback = st.text_area(f"Enter feedback for {cand.get('name', cand.get('filename'))}", key=cand.get('filename'))
                        if st.button(f"Submit Feedback for {cand.get('filename')}", key=f"submit_{cand.get('filename')}"):
                            feedback_data[cand.get('filename')] = feedback
                            st.success("Feedback submitted!")
                if feedback_data:
                    st.write("### Collected Feedback")
                    st.write(feedback_data)
                else:
                    st.info("No candidate results available. Please process resumes first.")

if __name__ == "__main__":
    main()