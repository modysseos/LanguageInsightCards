import re, math, json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, LangDetectException
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# --- UI TRANSLATIONS ---
UI_TEXT = {
    'en': {
        'page_title': 'Language Insight Cards',
        'text_area_label': 'Paste your text here...',
        'button_en': 'Show Results in English',
        'button_el': 'Show Results in Greek',
        'results_title': 'Sentences: {sents} | Tokens: {toks} | Generated: {ts}',
        'plot_glide_title': 'Plot Glide vs. Plot Twists',
        'plot_glide_y_axis': 'Coherence (adjacent cosine)',
        'megaphone_title': 'Megaphone vs. Microscope',
        'megaphone_series_claims': 'Claims/k',
        'megaphone_series_specifics': 'Specifics/k',
        'confidence_title': 'Confidence Dial',
        'confidence_series_abs': 'Absolutist/k',
        'confidence_series_hedges': 'Hedges/k',
        'chorus_title': 'Chorus Lines',
        'chorus_y_axis': 'Repetitive Phrases',
        'audience_pressure_title': 'Audience Pressure',
        'audience_pressure_series_imp': 'Direct Address/k',
        'audience_pressure_series_excl': 'Exclamations/k',
        'thinking_aloud_title': 'Thinking Aloud',
        'thinking_aloud_series': 'Cognitive Words/k',
        'me_vs_we_title': 'Me vs. We',
        'me_vs_we_series_i': 'I-words/k',
        'me_vs_we_series_we': 'We-words/k',
        'diversity_title': 'Lexical Diversity',
        'diversity_series': 'Type-Token Ratio',
        'questions_title': 'Question Rate',
        'questions_series': 'Questions Asked',
        'downloads_title': 'Downloads',
        'download_csv': 'Summary (CSV)',
        'download_json': 'Metrics (JSON)',
        'ai_interpretation_title': 'See AI Interpretation',
    },
    'el': {
        'page_title': 'Κάρτες Γλωσσικής Ανάλυσης',
        'text_area_label': 'Επικολλήστε το κείμενό σας εδώ...',
        'button_en': 'Εμφάνιση στα Αγγλικά',
        'button_el': 'Ανάλυση στα Ελληνικά',
        'results_title': 'Προτάσεις: {sents} | Λέξεις: {toks} | Δημιουργία: {ts}',
        'plot_glide_title': 'Ομαλή Ροή vs. Απότομες Στροφές',
        'plot_glide_y_axis': 'Συνοχή (γειτονικό συνημίτονο)',
        'megaphone_title': 'Μεγάφωνο vs. Μικροσκόπιο',
        'megaphone_series_claims': 'Ισχυρισμοί/k',
        'megaphone_series_specifics': 'Λεπτομέρειες/k',
        'confidence_title': 'Δείκτης Αυτοπεποίθησης',
        'confidence_series_abs': 'Απόλυτοι Όροι/k',
        'confidence_series_hedges': 'Επιφυλάξεις/k',
        'chorus_title': 'Επαναλαμβανόμενες Φράσεις',
        'chorus_y_axis': 'Επαναλαμβανόμενες Φράσεις',
        'audience_pressure_title': 'Πίεση προς το Ακροατήριο',
        'audience_pressure_series_imp': 'Άμεση Προσφώνηση/k',
        'audience_pressure_series_excl': 'Θαυμαστικά/k',
        'thinking_aloud_title': 'Ελεύθερη Σκέψη',
        'thinking_aloud_series': 'Γνωστικές Λέξεις/k',
        'me_vs_we_title': 'Εγώ vs. Εμείς',
        'me_vs_we_series_i': 'Λέξεις «Εγώ»/k',
        'me_vs_we_series_we': 'Λέξεις «Εμείς»/k',
        'diversity_title': 'Λεξιλογική Ποικιλία',
        'diversity_series': 'Δείκτης Λέξεων-Τύπων (TTR)',
        'questions_title': 'Ρυθμός Ερωτήσεων',
        'questions_series': 'Αριθμός Ερωτήσεων',
        'downloads_title': 'Λήψεις',
        'download_csv': 'Περίληψη (CSV)',
        'download_json': 'Μετρικές (JSON)',
        'ai_interpretation_title': 'Δείτε την AI Ερμηνεία',
    }
}

# --- API & MODEL CONFIGURATION ---
# The OpenAI API key is loaded from the .env file.
# See the .env.example file and the README for instructions.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────────────────────
# 1.  AI COMMENTARY
# ─────────────────────────────────────────────────────────────
def generate_commentary(card_title, metrics, transcript, output_lang):
    """Generate AI commentary for a specific analysis card."""
    lang_map = {'en': 'English', 'el': 'Greek'}
    output_lang_name = lang_map.get(output_lang, 'English')

    system_prompt = (
        "You are a helpful and insightful communication coach. Your goal is to analyze linguistic metrics from a transcript and explain them to a user in a simple, friendly, and encouraging way. "
        "You will receive the title of an 'Insight Card,' the metrics, and the transcript. "
        "Your task is to provide a short, easy-to-understand interpretation (2-4 sentences). Avoid jargon. "
        "Do not just list the metrics. Instead, explain what they suggest about the communication style. For example, what does high coherence feel like for a listener? Or what is the effect of using many 'absolutist' words? "
        "Your tone should be neutral and educational, focusing on the language, not judging the speaker."
    )

    # Create a summary of the metrics to pass to the model
    metric_summary = {k: v for k, v in metrics.items() if k not in ['sentences', 'cos', 'top_rep']}
    metric_summary['top_rep_str'] = 'None'
    if metrics.get('top_rep'):
        top_phrase = ' '.join(metrics['top_rep'][0][0])
        count = metrics['top_rep'][0][1]
        metric_summary['top_rep_str'] = f"'{top_phrase}' (x{count})"

    transcript_snippet = transcript[:2000]
    user_prompt = f"""Insight Card: {card_title}

Metrics: {json.dumps(metric_summary, indent=2)}

Transcript: 
{transcript_snippet}...

Please provide your interpretation of the '{card_title}' card based on these metrics and the transcript. IMPORTANT: Your entire response must be in {output_lang_name}."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI commentary: {e}"

# ─────────────────────────────────────────────────────────────
# 1.  BASIC CLEAN-UP & UTILITIES
# ─────────────────────────────────────────────────────────────
def clean_srt(text: str) -> str:
    """Remove indices, timestamps, tags, speaker labels, stage directions."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rules = [
        r"(?m)^\s*\d+\s*$",                                             # numeric indices
        r"(?im)^\s*\d{1,2}:\d{2}(:\d{2})?[,\.]\d{3}\s*-->\s*[^\n]+$",   # 00:00:00,000 -->
        r"<[^>]+>",                                                     # <vtt tags>
        r"\{\\[^}]+\}",                                                 # ASS/SSA tags
        r"(?m)^\s*>>?[^:\n]+:\s*",                                      # NAME: or >> NAME:
        r"(?m)^\s*(WEBVTT|NOTE).*?$",                                   # headers
        r"[\(\[]\s*[A-Za-zΑ-Ωα-ω \.]+?\s*[\)\]]"                        # [music] etc.
    ]
    for rx in rules: text = re.sub(rx, " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()

def split_sentences(txt: str):
    parts = re.split(r"(?<=[\.\!\?;…])\s+", txt)
    return [p.strip(" «»“”\"'()[]{}") for p in parts if p.strip()]

def tokens(txt: str, stop_words: set) -> list:
    return [t for t in re.findall(r'[a-zA-Zα-ωΑ-Ωάέήίόύώ]+', txt.lower()) if t not in stop_words]

def tfidf_adjacent(sentences):
    if len(sentences) < 2: return []
    X = TfidfVectorizer(ngram_range=(1, 2), analyzer="word").fit_transform(sentences)
    return [(X[i] @ X[i+1].T).toarray()[0, 0] for i in range(len(sentences) - 1)]

# ─────────────────────────────────────────────────────────────
# 2.  LOAD LEXICONS
# ─────────────────────────────────────────────────────────────
def load_lexicons(path='lexicons.json'):
    """Load lexicons from a JSON file into a dictionary of sets for fast lookups."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lexicons = json.load(f)
        # Convert lists to sets for efficient 'in' operations
        return {key: set(values) for key, values in lexicons.items()}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading lexicons from {path}: {e}")
        # Fallback to empty lexicons to prevent crashes
        return {
            'claims': set(), 'specifics': set(), 'absolutist': set(), 'hedges': set(),
            'you_imperative': set(), 'stop_words': set(), 'cognitive_process': set(),
            'first_person_singular': set(), 'first_person_plural': set()
        }

# ─────────────────────────────────────────────────────────────
# 3.  MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────
def calculate_coherence(sentences):
    """Calculate TF-IDF cosine similarity between adjacent sentences and find valleys."""
    if len(sentences) < 2:
        return {'cos': [], 'mean': np.nan, 'p10': np.nan, 'p90': np.nan, 'twists': []}

    cos = tfidf_adjacent(sentences)
    if not cos:
        return {'cos': [], 'mean': np.nan, 'p10': np.nan, 'p90': np.nan, 'twists': []}

    mean, p10, p90 = (float(v) for v in (
        np.mean(cos), np.percentile(cos, 10), np.percentile(cos, 90)))

    std_dev = np.std(cos)
    if std_dev > 0:
        thr = mean - 0.5 * std_dev
        twists = [i for i, v in enumerate(cos) if v < thr and
                  v <= cos[max(i - 1, 0)] and v <= cos[min(i + 1, len(cos) - 1)]]
    else:
        twists = []

    return {'cos': cos, 'mean': mean, 'p10': p10, 'p90': p90, 'twists': twists}

def calculate_lexical_metrics(txt, toks, n_tok, lexicons):
    """Calculate metrics based on predefined lexicons (claims, specifics, etc.)."""
    claim_k = sum(p in txt for p in lexicons['claims']) * 1000 / n_tok if n_tok > 0 else 0
    spec_k = sum(p in txt for p in lexicons['specifics']) * 1000 / n_tok if n_tok > 0 else 0
    abs_k = sum(1 for w in toks if w in lexicons['absolutist']) * 1000 / n_tok if n_tok > 0 else 0
    hedge_k = sum(1 for w in toks if w in lexicons['hedges']) * 1000 / n_tok if n_tok > 0 else 0

    ratio = claim_k / spec_k if spec_k > 0 else float('inf') if claim_k > 0 else 0

    return {
        'claim_k': claim_k,
        'spec_k': spec_k,
        'abs_k': abs_k,
        'hedge_k': hedge_k,
        'ratio': ratio
    }

def calculate_repetition_metrics(toks, lexicons):
    """Calculate n-gram repetition rates."""
    if not toks:
        return {'big_r': 0, 'tri_r': 0, 'top_rep': []}

    bigr = Counter(tuple(toks[i:i+2]) for i in range(len(toks) - 1))
    trig = Counter(tuple(toks[i:i+3]) for i in range(len(toks) - 2))
    rep = lambda c: sum(v for v in c.values() if v > 1) / max(1, sum(c.values()))

    top_rep = [(ng, c) for ng, c in trig.most_common(20) if c >= 3 and all(w not in lexicons['stop_words'] for w in ng)][:5]
    if not top_rep:
        top_rep = [(ng, c) for ng, c in bigr.most_common(20) if c >= 3 and all(w not in lexicons['stop_words'] for w in ng)][:5]

    return {'big_r': rep(bigr), 'tri_r': rep(trig), 'top_rep': top_rep}

def calculate_pressure_metrics(txt, sents, n_tok, lexicons):
    excl = txt.count("!")
    youimp_k = sum(p in txt for p in lexicons['you_imperative']) * 1000 / n_tok if n_tok > 0 else 0
    return {'excl': excl, 'youimp_k': youimp_k}

def calculate_cognitive_metrics(toks, n_tok, lexicons):
    cog_k = sum(1 for w in toks if w in lexicons['cognitive_process']) * 1000 / n_tok if n_tok > 0 else 0
    return {'cog_k': cog_k}

def calculate_social_metrics(toks, n_tok, lexicons):
    i_k = sum(1 for w in toks if w in lexicons['first_person_singular']) * 1000 / n_tok if n_tok > 0 else 0
    we_k = sum(1 for w in toks if w in lexicons['first_person_plural']) * 1000 / n_tok if n_tok > 0 else 0
    return {'i_k': i_k, 'we_k': we_k}

def calculate_diversity_metrics(toks, n_tok):
    """Calculate lexical diversity (TTR) and question rate."""
    if n_tok == 0:
        return {'ttr': 0}
    ttr = len(set(toks)) / n_tok
    return {'ttr': ttr}

def calculate_question_rate(sents):
    """Calculate the number of questions asked."""
    q_count = sum(1 for s in sents if s.endswith('?'))
    return {'q_count': q_count}

def get_language_lexicons(lang='en'):
    """Selects the correct lexicon set based on the detected language."""
    with open('lexicons.json', 'r', encoding='utf-8') as f:
        all_lexicons = json.load(f)

    lexicon_set = {}
    suffix = f"_{lang}" if lang != 'en' else "_en" if 'stop_words_en' in all_lexicons else ""
    
    base_keys = [
        'claims', 'specifics', 'absolutist', 'hedges', 'you_imperative', 
        'cognitive_process', 'first_person_singular', 'first_person_plural', 'stop_words'
    ]
    
    for key in base_keys:
        lang_key = f"{key}_{lang}"
        # Fallback to English key if the language-specific one doesn't exist
        if lang_key in all_lexicons:
            lexicon_set[key] = set(all_lexicons[lang_key])
        elif key in all_lexicons:
            lexicon_set[key] = set(all_lexicons[key])
        elif f"{key}_en" in all_lexicons:
             lexicon_set[key] = set(all_lexicons[f"{key}_en"])
        else:
            lexicon_set[key] = set()

    return lexicon_set

def analyse(text: str):
    """Run a full analysis pipeline on the input text."""
    txt = clean_srt(text)
    sents = split_sentences(txt)
    
    try:
        lang = detect(txt)
    except LangDetectException:
        lang = 'en' # Default to English if detection fails

    lexicons = get_language_lexicons(lang)
    
    toks = tokens(" ".join(sents), lexicons.get('stop_words', set()))
    n_tok = len(toks)

    if not sents:
        return None # Or return a default empty structure

    results = {'sentences': sents, 'n_tok': n_tok}
    results.update(calculate_coherence(sents))
    results.update(calculate_lexical_metrics(txt, toks, n_tok, lexicons))
    results.update(calculate_repetition_metrics(toks, lexicons))
    results.update(calculate_pressure_metrics(txt, sents, n_tok, lexicons))
    results.update(calculate_cognitive_metrics(toks, n_tok, lexicons))
    results.update(calculate_social_metrics(toks, n_tok, lexicons))
    results.update(calculate_diversity_metrics(toks, n_tok))
    results.update(calculate_question_rate(sents))

    return results

# ─────────────────────────────────────────────────────────────
# 4.  STREAMLIT UI
# ─────────────────────────────────────────────────────────────

# Initialize session state variables if they don't exist
if 'display_lang' not in st.session_state:
    st.session_state['display_lang'] = 'en'
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ""

# Get the UI text for the current display language
T = UI_TEXT[st.session_state['display_lang']]

st.set_page_config(page_title=T['page_title'], layout="wide")
st.title(T['page_title'])

# File uploader and text area
upl = st.file_uploader(" ", label_visibility="collapsed", type=["srt", "vtt", "txt"])
txt_input = upl.read().decode("utf-8", "ignore") if upl else st.text_area(T['text_area_label'], height=150, key="text_input")

# --- Analysis Buttons ---
col1, col2, _, _ = st.columns(4)

analysis_triggered = False
if col1.button(UI_TEXT['en']['button_en']):
    st.session_state['display_lang'] = 'en'
    analysis_triggered = True

if col2.button(UI_TEXT['el']['button_el']):
    st.session_state['display_lang'] = 'el'
    analysis_triggered = True

if analysis_triggered and txt_input:
    with st.spinner('Analyzing...'):
        st.session_state['results'] = analyse(txt_input)
        st.session_state['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.rerun()

# --- Display Results ---
if st.session_state['results']:
    results = st.session_state['results']
    display_lang = st.session_state['display_lang']
    T = UI_TEXT[display_lang]

    st.success(T['results_title'].format(
        sents=len(results['sentences']),
        toks=results['n_tok'],
        ts=st.session_state['timestamp']
    ))

    col1, col2 = st.columns(2)

    # --- Column 1: Coherence, Repetition, Cognitive --- 
    with col1:
        # Plot Glide Card
        with st.container(border=True):
            card_title = T['plot_glide_title']
            st.subheader(f"🎢 {card_title}")
            if results.get('cos'):
                fig, ax = plt.subplots()
                ax.plot(results['cos'])
                ax.set_ylabel(T['plot_glide_y_axis'])
                st.pyplot(fig)
                plt.close(fig)
                with st.expander(T['ai_interpretation_title']):
                    st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Chorus Lines Card
        with st.container(border=True):
            card_title = T['chorus_title']
            st.subheader(f"🔁 {card_title}")
            df_rep = pd.DataFrame({'Bigram Reps': [results['big_r']], 'Trigram Reps': [results['tri_r']]})
            st.bar_chart(df_rep.T)
            if results.get('top_rep'):
                top_phrase = ' '.join(results['top_rep'][0][0])
                count = results['top_rep'][0][1]
                st.markdown(f"Top repeated phrase: **'{top_phrase}'** (x{count})")
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Thinking Aloud Card
        with st.container(border=True):
            card_title = T['thinking_aloud_title']
            st.subheader(f"🧠 {card_title}")
            st.bar_chart(pd.DataFrame({T['thinking_aloud_series']: [results['cog_k']]}))
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Lexical Diversity Card
        with st.container(border=True):
            card_title = T['diversity_title']
            st.subheader(f"📚 {card_title}")
            st.metric(label=T['diversity_series'], value=f"{results['ttr']:.2f}")
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

    # --- Column 2: Lexical, Social --- 
    with col2:
        # Megaphone vs. Microscope Card
        with st.container(border=True):
            card_title = T['megaphone_title']
            st.subheader(f"📣 {card_title}")
            df_lex = pd.DataFrame({T['megaphone_series_claims']: [results['claim_k']], T['megaphone_series_specifics']: [results['spec_k']]})
            st.bar_chart(df_lex.T)
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Confidence Dial Card
        with st.container(border=True):
            card_title = T['confidence_title']
            st.subheader(f"🎚️ {card_title}")
            df_conf = pd.DataFrame({T['confidence_series_abs']: [results['abs_k']], T['confidence_series_hedges']: [results['hedge_k']]})
            st.bar_chart(df_conf.T)
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Audience Pressure Card
        with st.container(border=True):
            card_title = T['audience_pressure_title']
            st.subheader(f"💥 {card_title}")
            df_press = pd.DataFrame({T['audience_pressure_series_imp']: [results['youimp_k']], T['audience_pressure_series_excl']: [results['excl']]})
            st.bar_chart(df_press.T)
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Me vs. We Card
        with st.container(border=True):
            card_title = T['me_vs_we_title']
            st.subheader(f"💬 {card_title}")
            df_social = pd.DataFrame({T['me_vs_we_series_i']: [results['i_k']], T['me_vs_we_series_we']: [results['we_k']]})
            st.bar_chart(df_social.T)
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

        # Question Rate Card
        with st.container(border=True):
            card_title = T['questions_title']
            st.subheader(f"❓ {card_title}")
            st.metric(label=T['questions_series'], value=results['q_count'])
            with st.expander(T['ai_interpretation_title']):
                st.markdown(generate_commentary(card_title, results, txt_input, display_lang))

    # --- Downloads Section ---
    with st.expander(T['downloads_title']):
        # Create a summary DataFrame for CSV
        summary_data = {
            'Metric': list(results.keys()),
            'Value': [str(v) for v in results.values()]
        }
        df_summary = pd.DataFrame(summary_data)

        st.download_button(
            label=T['download_csv'],
            data=df_summary.to_csv(index=False).encode('utf-8'),
            file_name='language_insights_summary.csv',
            mime='text/csv',
        )

        st.download_button(
            label=T['download_json'],
            data=json.dumps(results, indent=2, default=str).encode('utf-8'),
            file_name='language_insights_metrics.json',
            mime='application/json',
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>This app was developed under MIT License by Michalis Odysseos in 2025</p>", unsafe_allow_html=True)
