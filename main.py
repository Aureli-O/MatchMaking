from supabase import create_client
import streamlit as st
from deep_translator import GoogleTranslator
from together import Together
import pandas as pd
import numpy as np
from pyvis.network import Network
import umap
from sklearn.preprocessing import StandardScaler
from scipy import spatial
import json
import time
import os
from typing import List, Dict
from dotenv import load_dotenv
from textwrap import wrap
from transformers import pipeline
import re
from streamlit_supabase_auth import login_form, logout_button
import streamlit.components.v1 as components
import ast

load_dotenv()

def get_secret(key: str, default=None):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY")
TOGETHER_API_KEY = get_secret("TOGETHER_API_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL não encontrado. Configure .env ou st.secrets.")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY não encontrado. Configure .env ou st.secrets.")
if not SUPABASE_SERVICE_KEY:
    st.warning("SUPABASE_SERVICE_KEY não encontrado. Operações admin (upsert) podem falhar por RLS.")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

supabase_admin = None
if SUPABASE_SERVICE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

client = Together(api_key=TOGETHER_API_KEY)

@st.cache_data(show_spinner=False)
def translate_to_english(text: str) -> str:
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        # fallback: retorna texto original
        print("Translation failed:", e)
        return text


def text_to_embeddings(text: str) -> List[float]:
    if not text:
        return []
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-32k-retrieval",
        input=text,
    )
    try:
        return [x.embedding for x in response.data][0]
    except Exception:
        try:
            return response[0]
        except Exception as e:
            print("Embedding parse error:", e)
            return []

SENSITIVE_KEYWORDS = [
    "suicídio", "ódio", "racista", "assassinato", "violência",
    "homofobia", "terrorismo", "droga", "se matar", "matar", "arma",
    "negro", "preto", "viado", "judeu", "árabe"
]

def contains_sensitive_keywords(text: str) -> bool:
    text_lower = text.lower()
    if any(re.search(rf"\b{re.escape(word)}\b", text_lower) for word in SENSITIVE_KEYWORDS):
        return True
    return False

@st.cache_resource
def load_multilingual_classifier():
    return pipeline("text-classification", model="martin-ha/toxic-comment-model")

@st.cache_resource
def load_english_classifier():
    return pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

multilingual_classifier = load_multilingual_classifier()
english_classifier = load_english_classifier()

def is_toxic_with_model(text: str, classifier, threshold: float = 0.7) -> bool:
    try:
        results = classifier(text)
        for r in results:
            if r["label"].lower() in ["toxic", "hate", "insult", "offensive"] and r["score"] > threshold:
                return True
        return False
    except Exception as e:
        print("⚠️ Erro no classificador:", e)
        return False

def is_safe_input(text: str) -> bool:
    if contains_sensitive_keywords(text):
        print("❌ Bloqueado por keywords")
        return False
    if is_toxic_with_model(text, multilingual_classifier):
        print("❌ Bloqueado pelo classificador multilíngue")
        return False
    translated = translate_to_english(text)
    if is_toxic_with_model(translated, english_classifier):
        print("❌ Bloqueado pelo classificador inglês (via tradução)")
        return False

    return True

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text,
  email text UNIQUE,
  photo_url text,
  preferences text,
  embedding vector(768),
  groups text[] DEFAULT ARRAY['#global'],
  created_at timestamptz DEFAULT now()
);
'''

def ensure_table_exists():
    try:
        supabase.rpc("sql", {"q": CREATE_TABLE_SQL}).execute()
    except Exception as e:
        print("Could not create table programmatically. Please create the 'users' table manually in Supabase SQL Editor.", e)

def upsert_user(user_id: str, name: str, email: str, photo_url: str, preferences: str,
                embedding: List[float], groups: List[str], user_color: str, consent: bool = True):
    if not user_id:
        raise ValueError("⚠️ user_id (auth.uid) está vazio — verifique autenticação.")

    try:
        emb_to_save = list(embedding) if embedding is not None else None

        data = {
            "id": user_id,
            "name": name,
            "email": email,
            "photo_url": photo_url,
            "preferences": preferences,
            "embedding": emb_to_save,
            "groups": groups,
            "user_color": user_color,
            "consent": bool(consent)
        }

        if supabase_admin:
            resp = supabase_admin.table("users").upsert(data, on_conflict="id").execute()
        else:
            resp = supabase.table("users").upsert(data, on_conflict="id").execute()

        if hasattr(resp, 'error') and resp.error:
            print("Upsert returned error:", resp.error)
        else:
            print("Upsert resp:", getattr(resp, "data", resp))

        return resp
    except Exception as e:
        print("Upsert error:", e)
        return None

def get_all_users():
    try:
        resp = supabase.table('users').select(
            'id,name,email,photo_url,preferences,embedding,groups,user_color'
        ).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        print("Get users error:", e)
        return []

    
def filter_users_by_group(users, group: str):
    return [u for u in users if group in (u.get("groups") or [])]

def parse_embedding(emb):
    if isinstance(emb, str):
        try:
            return np.array(json.loads(emb), dtype=float).flatten()
        except json.JSONDecodeError:
            return np.array(ast.literal_eval(emb), dtype=float).flatten()
    elif isinstance(emb, list) or isinstance(emb, np.ndarray):
        return np.array(emb, dtype=float).flatten()
    return None

def compute_similarities(target_embedding: List[float], others: List[Dict], top_k: int = 5):
    distances = []
    a = parse_embedding(target_embedding)
    if a is None:
        return []

    for other in others:
        emb = other.get('embedding')
        b = parse_embedding(emb)
        if b is None or a.shape[0] != b.shape[0]:
            continue

        d = spatial.distance.cosine(a, b)
        distances.append({
            'id': other.get('id'),
            'name': other.get('name'),
            'email': other.get('email'),
            'photo_url': other.get('photo_url'),
            'preferences': other.get('preferences'),
            'distance': d
        })

    distances = sorted(distances, key=lambda x: x['distance'])
    return distances[:top_k]

def build_pyvis_graph(users: List[Dict], edges: List[Dict], notebook: bool = False) -> Network:
    net = Network(
        height='700px',
        width='100%',
        bgcolor="#0e1117",
        font_color='white',
        notebook=notebook,
        cdn_resources='remote'
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)

    for u in users:
        uid = u.get('id')
        name = u.get('name') or 'Sem nome'
        email = u.get('email') or ''
        preferences = u.get('preferences') or ''
        node_color = u.get('user_color') or "#1f77b4"

        matches = compute_similarities(u["embedding"], [v for v in users if v["id"] != uid], top_k=5)

        matches_text = "\n".join(
            [f"- {m['name']} ({round((1 - m['distance']) * 100)}%)" for m in matches]
        ) or "Nenhum match encontrado"

        wrapped_prefs = "\n".join(wrap(preferences, width=50))

        title_text = (
            f"👤 {name}\n"
            f"📧 {email}\n\n"
            f"🎯 Gostos:\n{wrapped_prefs}\n\n"
            f"🔗 Top 5 matches:\n{matches_text}"
        )

        net.add_node(
            uid,
            label=name,
            title=title_text,
            color=node_color
        )

    for e in edges:
        net.add_edge(e['source'], e['target'], value=e.get('weight', 1))

    return net


def compute_all_edges(users: List[Dict], per_user_k: int = 5):
    edges = []
    id_map = {u['id']: u for u in users}
    for u in users:
        emb = u.get('embedding')
        if not emb:
            continue
        matches = compute_similarities(emb, [v for v in users if v['id'] != u['id']], top_k=per_user_k)
        for m in matches:
            weight = max(0.01, 1.0 - m['distance'])
            edges.append({'source': u['id'], 'target': m['id'], 'weight': float(weight)})
    seen = set()
    unique_edges = []
    for e in edges:
        pair = (e['source'], e['target'])
        if pair not in seen:
            seen.add(pair)
            unique_edges.append(e)
    return unique_edges

st.set_page_config(
    page_title='Matchmaking',
    layout='wide',
    page_icon="✨"
)
st.title('Matchmaking')

with st.sidebar:
    st.header('Login')

    session = login_form(
        url=SUPABASE_URL,
        apiKey=SUPABASE_ANON_KEY,
        providers=["google"],
    )

    if session:
        user = session.get("user") or {}

        user_id = (
            user.get("id") or
            user.get("sub") or
            user.get("user_metadata", {}).get("provider_id") or
            None
        )

        if not user_id:
            st.error("Erro: não foi possível recuperar o ID do usuário (auth.uid).")
        else:
            user_email = user.get("email")
            metadata = user.get("user_metadata", {}) if isinstance(user, dict) else {}
            display_name = metadata.get("full_name") or metadata.get("name") or user_email or "Usuário"
            avatar = metadata.get("avatar_url") or user.get("avatar_url") or None

            st.success(f"Conectado: {display_name}")
            if avatar:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(avatar, width=80)

            logout_button(apiKey=SUPABASE_ANON_KEY, url=SUPABASE_URL)

    else:
        pass

if not session:
    st.info("Por favor, faça login com o Google no menu lateral (à esquerda) para continuar.")
    st.stop()

user = session.get("user") or {}

if 'user' not in st.session_state:
    st.session_state['user'] = user

if 'consent_given' not in st.session_state:
    consent_flag = False
    try:
        resp = supabase.table("users").select("consent").eq("id", user_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if data:
            row = data[0] if isinstance(data, list) else data
            consent_flag = bool(row.get('consent')) if row else False
    except Exception as e:
        print("Could not read consent from Supabase:", e)
        consent_flag = False
    st.session_state['consent_given'] = consent_flag

if 'user' in st.session_state:
    session_user = st.session_state['user']

    if 'existing_preferences' not in st.session_state:
        st.session_state['existing_preferences'] = ""
        try:
            uid = session_user.get("id") or session_user.get("sub") or session_user.get("user_metadata", {}).get("provider_id")
            if uid:
                resp = supabase.table("users").select("preferences").eq("id", uid).execute()
                data = resp.data if hasattr(resp, 'data') else resp
                if data:
                    row = data[0] if isinstance(data, list) else data
                    prefs = row.get("preferences") if row else None
                    if prefs:
                        st.session_state['existing_preferences'] = prefs
        except Exception as e:
            print("Could not fetch existing preferences:", e)

    preferences_input = st.text_area(
        "Escreva seus gostos (ex: filmes, hobbies, comidas, interesses)",
        value=st.session_state.get('existing_preferences', ""),
        help="Dê preferência em texto corrido",
        height=150,
        placeholder="Gosto de futebol, videogames e música eletrônica, mas não sou fã de leitura extensa ou dançar.",
    )

    user_color = st.color_picker("Escolha sua cor no grafo", "#1f77b4")

    col_groups_input, col_selected_group = st.columns([3,1])
    with col_groups_input:
        groups_input = st.text_input(
            "Grupos",
            help="Use vírgulas para separar os grupos.\nCaso deixe esse campo vazio, aparecerá todos os usuários",
            placeholder="#global, #trabalho",
        )

        user_other_groups = [g.strip() for g in groups_input.split(",") if g.strip() and g.strip() != "#global"]

        user_groups_candidate = ["#global"] + user_other_groups
        seen = set()
        user_groups = []
        for g in user_groups_candidate:
            if g not in seen:
                seen.add(g)
                user_groups.append(g)

    with col_selected_group:
        selected_group = st.selectbox("Selecione o grupo", user_groups)

    consent_given = st.session_state.get('consent_given', False)
    consent_checkbox_checked = False
    if not consent_given:
        col_termos_text, col_checkbox = st.columns([3,1])
        with col_termos_text:
            with st.expander("📜 Termos de uso e consentimento (clique para ver)"):
                st.markdown("""
                Ao aceitar, você concorda que seu nome, email, foto e gostos serão utilizados
                para gerar conexões e exibir o grafo de afinidades nesta aplicação de cunho educacional.
                """)
        with col_checkbox:
            consent_checkbox_checked = st.checkbox("✅ Aceito o uso!", key="consent_checkbox")
        send_disabled = not consent_checkbox_checked
    else:
        send_disabled = False

    if st.button(
        'Enviar e Gerar conexões',
        help ="As conexões são um grafo que conecta pessoas com interesses semelhantes.",
        disabled=send_disabled
    ):
        with st.spinner('Processando e gerando conexões...'):
            try:
                if not preferences_input or not preferences_input.strip():
                    st.error("⚠️ O campo de gostos não pode estar vazio.")
                elif not is_safe_input(preferences_input):
                    st.error("⚠️ O texto contém termos sensíveis ou tóxicos e não pode ser enviado.")
                else:
                    translated = translate_to_english(preferences_input)
                    emb = text_to_embeddings(translated)

                    user_id = session_user.get("id") or session_user.get("sub") or session_user.get("user_metadata", {}).get("provider_id")
                    user_email = session_user.get("email") or user.get("email")
                    user_name = session_user.get("name") or display_name
                    user_photo = session_user.get("photo_url") or avatar

                    if not user_email:
                        st.error("Erro: email do usuário não encontrado. Faça login novamente.")
                        st.stop()

                    try:
                        resp = upsert_user(
                            user_id=user_id,
                            name=user_name,
                            email=user_email,
                            photo_url=user_photo,
                            preferences=preferences_input,
                            embedding=emb,
                            groups=user_groups,
                            user_color=user_color,
                            consent=True
                        )
                    except TypeError:
                        resp = upsert_user(
                            name=user_name,
                            email=user_email,
                            photo_url=user_photo,
                            preferences=preferences_input,
                            embedding=emb,
                            groups=user_groups,
                            user_color=user_color,
                            consent=True
                        )

                    if resp is None:
                        st.error("Falha ao salvar os dados", icon ="❌")
                    else:
                        success_box = st.empty()
                        success_box.success("Dados salvos!", icon ="✅")
                        st.session_state['consent_given'] = True
                        time.sleep(2)
                        success_box.empty()

                    progress = st.progress(0)
                    progress.progress(20)

                    resp = supabase.table("users").select(
                        "id,name,email,photo_url,preferences,embedding,groups,user_color,consent"
                    ).eq("consent", True).execute()
                    users = resp.data if hasattr(resp, 'data') else resp
                    progress.progress(40)

                    filtered_users = filter_users_by_group(users, selected_group)
                    progress.progress(60)

                    clean_users = [u for u in filtered_users if u.get('embedding')]
                    edges = compute_all_edges(clean_users, per_user_k=5)
                    progress.progress(80)

                    net = build_pyvis_graph(clean_users, edges, notebook=False)

                    tmpfile = 'graph_tmp.html'
                    net.save_graph(tmpfile)
                    with open(tmpfile, 'r', encoding='utf-8') as f:
                        html = f.read()

                    progress.progress(100)
                    components.html(html, height=710, scrolling=True)
                    progress.empty()
            except Exception as e:
                st.error(f'❌ Ocorreu um erro: {e}')
                st.stop()
else:
    st.info('Faça login para acessar o formulário e gerar o grafo.')