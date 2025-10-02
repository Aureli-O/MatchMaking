# %%
# Bibliotecas usadas

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

# %%
# Chaves de API (carregadas do .env) e clientes do supabase e together AI
load_dotenv()

def get_secret(key: str, default=None):
    # 1) tenta pegar do ambiente (.env ou variáveis do sistema)
    val = os.getenv(key)
    if val:
        return val
    # 2) tenta pegar do st.secrets (usado no Streamlit Cloud)
    try:
        return st.secrets[key]
    except Exception:
        return default

# 🔹 Segredos (agora incluindo SERVICE_KEY)
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY")  # <-- necessário para upserts/admin
TOGETHER_API_KEY = get_secret("TOGETHER_API_KEY")

# Validar se as keys existem (ajuda no debug)
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL não encontrado. Configure .env ou st.secrets.")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY não encontrado. Configure .env ou st.secrets.")
if not SUPABASE_SERVICE_KEY:
    # não falha: avisa. Idealmente configure SUPABASE_SERVICE_KEY para permitir upserts via admin.
    st.warning("SUPABASE_SERVICE_KEY não encontrado. Operações admin (upsert) podem falhar por RLS.")

# 🔹 Clientes
# Cliente público (anon) — usado para selects que respeitam RLS
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Cliente admin (service role) — usado apenas para operações administrativas/upserts
supabase_admin = None
if SUPABASE_SERVICE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

client = Together(api_key=TOGETHER_API_KEY)

# %%
# Função de tradução e criação de embeddings

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
    """
    Gera embedding usando Together AI. Retorna vector (lista de floats).
    Ajuste o model se necessário.
    """
    if not text:
        return []
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-32k-retrieval",
        input=text,
    )
    # A resposta pode ter estrutura diferente dependendo da versão do SDK
    # Esperamos response.data -> list of objects with .embedding
    try:
        return [x.embedding for x in response.data][0]
    except Exception:
        # tentar alternativa: response[0]
        try:
            return response[0]
        except Exception as e:
            print("Embedding parse error:", e)
            return []

# %%
# Detector de toxixidade usando keywords e modelos multilinguais
    
# 🚨 Lista mínima de palavras-chave proibidas (multi-idioma)
SENSITIVE_KEYWORDS = [
    "suicídio", "ódio", "racista", "assassinato", "violência",
    "homofobia", "terrorismo", "droga", "se matar", "matar", "arma",
    "negro", "preto", "viado", "judeu", "árabe"
]

def contains_sensitive_keywords(text: str) -> bool:
    """Verifica se contém palavras-chave proibidas (anti-bypass)."""
    text_lower = text.lower()
    if any(re.search(rf"\b{re.escape(word)}\b", text_lower) for word in SENSITIVE_KEYWORDS):
        return True
    return False

# 🚀 Carrega dois classificadores: multilíngue + inglês
@st.cache_resource
def load_multilingual_classifier():
    # Multilíngue → pega direto em PT-BR, ES etc.
    return pipeline("text-classification", model="martin-ha/toxic-comment-model")

@st.cache_resource
def load_english_classifier():
    # Inglês → mais robusto para hate speech explícito
    return pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

multilingual_classifier = load_multilingual_classifier()
english_classifier = load_english_classifier()

def is_toxic_with_model(text: str, classifier, threshold: float = 0.7) -> bool:
    """Classifica o texto com um modelo específico."""
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
    """
    Proteção em 3 camadas:
    1. Regras manuais (keywords)
    2. Modelo multilíngue
    3. Modelo inglês (após tradução)
    """
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

# %%
# Helpers DB (Supabase): criação de tabela (executar apenas uma vez) e funções CRUD

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
    """Tenta criar a tabela users no Supabase (executar uma vez)."""
    try:
        # Supabase SQL via RPC (execute_raw é método do supabase-py)
        supabase.rpc("sql", {"q": CREATE_TABLE_SQL}).execute()
    except Exception as e:
        # Muitos projetos Supabase não usam essa rota; como alternativa, documente para criar via SQL Editor no dashboard
        print("Could not create table programmatically. Please create the 'users' table manually in Supabase SQL Editor.", e)

def upsert_user(user_id: str, name: str, email: str, photo_url: str, preferences: str,
                embedding: List[float], groups: List[str], user_color: str):
    """Insere ou atualiza usuário no Supabase.

    Usa o cliente admin (service role) se estiver disponível para evitar RLS ao gravar
    embeddings/metadata. Se `supabase_admin` não estiver configurado, tentará usar
    o cliente anon (que pode falhar devido a RLS).
    """
    if not user_id:
        raise ValueError("⚠️ user_id (auth.uid) está vazio — verifique autenticação.")

    try:
        # garante que embedding seja lista serializável (evita "invalid input" no Postgres)
        emb_to_save = list(embedding) if embedding is not None else None

        data = {
            "id": user_id,
            "name": name,
            "email": email,
            "photo_url": photo_url,
            "preferences": preferences,
            "embedding": emb_to_save,
            "groups": groups,
            "user_color": user_color
        }

        # Usa cliente admin se disponível (ignora RLS)
        if supabase_admin:
            resp = supabase_admin.table("users").upsert(data, on_conflict="id").execute()
        else:
            # fallback: usa cliente anon — pode levantar erro por RLS
            resp = supabase.table("users").upsert(data, on_conflict="id").execute()

        # debug minimal
        if hasattr(resp, 'error') and resp.error:
            print("Upsert returned error:", resp.error)
        else:
            print("Upsert resp:", getattr(resp, "data", resp))

        return resp
    except Exception as e:
        print("Upsert error:", e)
        return None

def get_all_users():
    """Retorna todos usuários com embeddings, grupos e cor."""
    try:
        resp = supabase.table('users').select(
            'id,name,email,photo_url,preferences,embedding,groups,user_color'
        ).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        print("Get users error:", e)
        return []

    
def filter_users_by_group(users, group: str):
    """Filtra usuários que participam de um grupo específico."""
    return [u for u in users if group in (u.get("groups") or [])]

# %%
# Similaridade e construção de grafo

def parse_embedding(emb):
    """Converte embedding vindo como string/lista para lista de floats."""
    if isinstance(emb, str):
        try:
            # tenta via json
            return np.array(json.loads(emb), dtype=float).flatten()
        except json.JSONDecodeError:
            # fallback se for formato python
            return np.array(ast.literal_eval(emb), dtype=float).flatten()
    elif isinstance(emb, list) or isinstance(emb, np.ndarray):
        return np.array(emb, dtype=float).flatten()
    return None

def compute_similarities(target_embedding: List[float], others: List[Dict], top_k: int = 5):
    """Calcula distância coseno entre target e lista de outros usuários (cada other tem 'id' e 'embedding'). Retorna top_k matches."""
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

        # 🔹 Calcula top 5 matches do usuário
        matches = compute_similarities(u["embedding"], [v for v in users if v["id"] != uid], top_k=5)

        # 🔹 Monta texto dos matches
        matches_text = "\n".join(
            [f"- {m['name']} ({round((1 - m['distance']) * 100)}%)" for m in matches]
        ) or "Nenhum match encontrado"

        # 🔹 Quebra gostos em várias linhas (wrap)
        wrapped_prefs = "\n".join(wrap(preferences, width=50))

        # 🔹 Monta tooltip completo
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

    # Adiciona arestas
    for e in edges:
        net.add_edge(e['source'], e['target'], value=e.get('weight', 1))

    return net


# %%
# Função para gerar arestas a partir de todos usuários (exemplo: conectar top 5 de cada usuário)

def compute_all_edges(users: List[Dict], per_user_k: int = 5):
    edges = []
    # preparar mapa id->embedding
    id_map = {u['id']: u for u in users}
    for u in users:
        emb = u.get('embedding')
        if not emb:
            continue
        matches = compute_similarities(emb, [v for v in users if v['id'] != u['id']], top_k=per_user_k)
        for m in matches:
            weight = max(0.01, 1.0 - m['distance'])  # converte distance->score simples
            edges.append({'source': u['id'], 'target': m['id'], 'weight': float(weight)})
    # deduplicar (source,target) ordenado
    seen = set()
    unique_edges = []
    for e in edges:
        pair = (e['source'], e['target'])
        if pair not in seen:
            seen.add(pair)
            unique_edges.append(e)
    return unique_edges

# %% 
# Streamlit app

st.set_page_config(
    page_title='Matchmaking',
    layout='wide',
    page_icon="✨"
    )
st.title('Matchmaking — Demo')

with st.sidebar:
    st.header('Login')

    st.markdown('''
    **Autenticação:** use o login com Google (via Supabase).
    ''')

    # login só com Google
    session = login_form(
        url=SUPABASE_URL,
        apiKey=SUPABASE_ANON_KEY,
        providers=["google"],
    )

    if not session:
        st.info("Faça login com Google para continuar.")
        st.stop()

    user = session.get("user") or {}

    # 🔑 garante o auth.uid (tenta várias chaves possíveis)
    user_id = (
        user.get("id") or                      # padrão Supabase Auth
        user.get("sub") or                     # fallback para alguns OAuth
        user.get("user_metadata", {}).get("provider_id") or  # fallback Google
        None
    )
    if not user_id:
        st.error("Erro: não foi possível recuperar o ID do usuário (auth.uid).")
        st.stop()

    user_email = user.get("email")
    metadata = user.get("user_metadata", {}) if isinstance(user, dict) else {}
    display_name = metadata.get("full_name") or metadata.get("name") or user.get("email") or "Usuário"
    avatar = metadata.get("avatar_url") or user.get("avatar_url") or None

    st.success(f"Conectado: {display_name}")
    if avatar:
        st.image(avatar, width=80)

    logout_button(apiKey=SUPABASE_ANON_KEY, url=SUPABASE_URL)
# Main panel: formulário
# Nota: o restante do código usa st.session_state['user'] no original; aqui usamos a 'user' obtida via session
# para compatibilidade podemos popular st.session_state['user'] se desejar:
# popula session_state sem sobrescrever o ID
if 'user' not in st.session_state:
    # guarda o user completo (não só nome/email), para não perder o "id"
    st.session_state['user'] = user


if 'user' in st.session_state:
    # agora pega o ID direto do session_state
    session_user = st.session_state['user']
    preferences_input = st.text_area(
        'Escreva seus gostos (ex: filmes, hobbies, comidas, interesses)',
        height=150
    )
    user_color = st.color_picker("Escolha sua cor no grafo", "#1f77b4")

    col_groups_input, col_selected_group = st.columns([3,1])
    with col_groups_input:
        groups_input = st.text_input(
            "Grupos (use vírgulas, ex: #global, #turma1, #trabalho1)",
            "#global"
        )
        user_groups = [g.strip() for g in groups_input.split(",") if g.strip()]
        if not user_groups:
            user_groups = ["#global"]

    with col_selected_group:
        selected_group = st.selectbox("Selecione o grupo para gerar o grafo", user_groups)

    if st.button('Enviar'):
        with st.spinner('Processando...'):
            try:
                if not is_safe_input(preferences_input):
                    st.error("⚠️ O texto contém termos sensíveis ou tóxicos e não pode ser enviado.")
                else:
                    translated = translate_to_english(preferences_input)
                    emb = text_to_embeddings(translated)

                    user = st.session_state['user']

                    # 🔑 pega o id do Supabase Auth (auth.uid)
                    user_id = user.get("id") or user.get("sub") or user.get("user_metadata", {}).get("provider_id")
                    user_email = user.get("email") or session_user.get("email")

                    if not user_id:
                        st.error("Erro: ID do usuário não encontrado (auth.uid). Faça login novamente.")
                        st.stop()

                    # chama upsert (usa supabase_admin se for configurado)
                    resp = upsert_user(
                        user_id=user_id,
                        name=display_name,
                        email=user_email,
                        photo_url=avatar,
                        preferences=preferences_input,
                        embedding=emb,
                        groups=user_groups,
                        user_color=user_color
                    )

                    # debug simples
                    if resp is None:
                        st.error("❌ Falha ao salvar os dados (veja logs do servidor).")
                    else:
                        st.success('Dados salvos!')
                    time.sleep(1)
            except Exception as e:
                st.error(f'❌ Ocorreu um erro: {e}')
                st.stop()

    if st.button('Gerar grafo'):
        with st.spinner('Buscando usuários e gerando grafo...'):
            users = get_all_users()

            # 🔎 filtra usuários pelo grupo selecionado
            filtered_users = filter_users_by_group(users, selected_group)

            clean_users = [u for u in filtered_users if u.get('embedding')]

            edges = compute_all_edges(clean_users, per_user_k=5)
            net = build_pyvis_graph(clean_users, edges, notebook=False)

            tmpfile = 'graph_tmp.html'
            net.save_graph(tmpfile)
            with open(tmpfile, 'r', encoding='utf-8') as f:
                html = f.read()
            components.html(html, height=710, scrolling=True)
else:
    st.info('Faça login para acessar o formulário e gerar o grafo.')

# %%
# 9) Observações finais e próximos passos (executar manualmente ou adaptar)
# - Autenticação real: configure Supabase Auth (Google) no dashboard do Supabase.
#   Em produção, não use a SUPABASE_SERVICE_KEY no frontend; use rotas server-side para operações sensíveis.
# - Schema: ajuste o tamanho do vector(1536) conforme o modelo. O exemplo usa 1536 por compatibilidade com modelos maiores.
# - Se preferir busca vetorial nativa, habilite pgvector no Supabase e use consultas SQL para `ORDER BY embedding <-> new_embedding`.
# - Em produção, adicione cache para embeddings e tratamento de erros mais robusto.

# Fim do notebook
