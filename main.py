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
                embedding: List[float], groups: List[str], user_color: str, consent: bool = True):
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
            "user_color": user_color,
            "consent": bool(consent)
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

def get_all_users(consent_only: bool = True):
    """Retorna todos usuários com embeddings, grupos e cor.
    Se consent_only=True, retorna apenas usuários com consent = true.
    """
    try:
        q = supabase.table('users').select('id,name,email,photo_url,preferences,embedding,groups,user_color,consent')
        if consent_only:
            q = q.eq('consent', True)
        resp = q.execute()
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

    st.markdown("""
    **Autenticação:** use o login com Google (via Supabase).
    """)

    st.write("Antes de logar, por favor leia e aceite os termos de uso das suas informações (nome, email, preferências e uso para geração de matches).")

    # Expander com texto resumido dos termos (personalize com teu texto legal)
    with st.expander("Ver termos de uso / privacidade"):
        st.markdown("""
        **Resumo:** ao aceitar, você autoriza que seu nome, email, avatar e as preferências informadas sejam usadas para gerar conexões (matches) e a visualização em grafo.
        - Os dados serão salvos na nossa base (Supabase).
        - Você pode revogar o consentimento removendo seu registro ou solicitando exclusão.
        """)

    # Checkbox de consentimento pré-login
    prelogin_consent = st.checkbox("✅ Eu li e aceito o uso das minhas informações para gerar matches (necessário para fazer login)")

    session = None
    if prelogin_consent:
        # chama o widget de login - só aparece após o checkbox
        session = login_form(
            url=SUPABASE_URL,
            apiKey=SUPABASE_ANON_KEY,
            providers=["google"],
        )
    else:
        st.info("Marque a caixa de consentimento para prosseguir com o login.")

if session:
    user = session.get("user") or {}

    # extrai user_id (auth.uid) robustamente
    user_id = (
        user.get("id") or
        user.get("sub") or
        user.get("user_metadata", {}).get("provider_id") or
        None
    )
    if not user_id:
        st.sidebar.error("Erro: não foi possível recuperar o ID do usuário (auth.uid).")
        st.stop()

    # guarda a sessão em session_state para uso posterior
    if 'user' not in st.session_state:
        st.session_state['user'] = user

    # verifica no Supabase se esse usuário já consentiu
    try:
        resp = supabase.table("users").select("consent").eq("id", user_id).maybe_single().execute()
        consent_in_db = False
        if hasattr(resp, "data") and resp.data:
            # resp.data pode ser dict ou objeto; tentamos extrair
            if isinstance(resp.data, dict):
                consent_in_db = bool(resp.data.get("consent", False))
            else:
                try:
                    consent_in_db = bool(resp.data[0].get("consent", False))
                except Exception:
                    consent_in_db = False
        else:
            consent_in_db = False
    except Exception as e:
        print("Erro ao consultar consent no Supabase:", e)
        consent_in_db = False

    # Se não consentiu antes no DB, mostramos uma mensagem e pedimos aceite pós-login
    if not consent_in_db:
        st.warning("Você ainda não registrou consentimento no sistema. Para usar o Matchmaking, é necessário aceitar os termos de uso.")
        if st.button("Aceitar termos e continuar"):
            # grava upsert mínimo para registrar consentimento
            metadata = user.get("user_metadata", {}) if isinstance(user, dict) else {}
            display_name = metadata.get("full_name") or metadata.get("name") or user.get("email") or "Usuário"
            avatar = metadata.get("avatar_url") or user.get("avatar_url") or None
            user_email = user.get("email")

            try:
                upsert_resp = upsert_user(
                    user_id=user_id,
                    name=display_name,
                    email=user_email,
                    photo_url=avatar,
                    preferences="",      # sem preferências ainda
                    embedding=[],        # sem embedding agora
                    groups=["#global"],  # default
                    user_color="#1f77b4",
                    consent=True
                )
                if upsert_resp is None:
                    st.error("Erro ao registrar consentimento. Tente novamente.")
                else:
                    st.success("Consentimento registrado — obrigado! Você já pode usar o sistema.")
                    # opcionalmente atualizar variável local
                    consent_in_db = True
            except Exception as e:
                st.error(f"Erro ao salvar consentimento: {e}")

# Se o usuário não logou, interrompe o app (comportamento conforme tua versão original)
if 'user' not in st.session_state:
    st.info("Faça login para continuar.")
    st.stop()

# ---------------------------
# Main panel: formulário (mantive teu layout, apenas adicionei verificação de consent antes de permitir enviar)
# ---------------------------
session_user = st.session_state['user']
metadata = session_user.get("user_metadata", {}) if isinstance(session_user, dict) else {}
display_name = metadata.get("full_name") or metadata.get("name") or session_user.get("email") or "Usuário"
avatar = metadata.get("avatar_url") or session_user.get("avatar_url") or None

st.subheader('Conte-nos seus gostos / preferências')

preferences_input = st.text_area(
    "Escreva seus gostos (ex: filmes, hobbies, comidas, interesses)",
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
    user_groups = [g.strip() for g in groups_input.split(",") if g.strip()]
    if not user_groups:
        user_groups = ["#global"]

with col_selected_group:
    selected_group = st.selectbox("Selecione o grupo", user_groups)

# Antes de permitir enviar, confirmamos que o usuário logado tem consent no DB
# (caso o usuário tenha aceitado pré-login, ou aceitado no botão pós-login, isso já estará true)
def user_has_consent(user_id: str) -> bool:
    try:
        resp = supabase.table("users").select("consent").eq("id", user_id).maybe_single().execute()
        if hasattr(resp, "data") and resp.data:
            if isinstance(resp.data, dict):
                return bool(resp.data.get("consent", False))
        return False
    except Exception:
        return False

logged_user_id = (session_user.get("id") or session_user.get("sub") or session_user.get("user_metadata", {}).get("provider_id"))
if not user_has_consent(logged_user_id):
    st.error("Você precisa aceitar os termos (consentimento) antes de enviar suas preferências. Veja a barra lateral para aceitar.", icon="⚠️")
    st.stop()

# substitui os dois botões anteriores por este único fluxo combinado
if st.button(
    'Enviar e Gerar grafo',
    help ="O grafo conecta pessoas com interesses semelhantes.",
    ):
    with st.spinner('Processando e gerando grafo...'):
        try:
            # Validações
            if not preferences_input.strip():
                st.error("⚠️ O campo de preferências não pode estar vazio.")
            elif not is_safe_input(preferences_input):
                st.error("⚠️ O texto contém termos sensíveis ou tóxicos e não pode ser enviado.")
            else:
                # 1) traduz + gera embedding
                translated = translate_to_english(preferences_input)
                emb = text_to_embeddings(translated)

                # 2) informações do usuário (robustas)
                session_user = st.session_state.get('user', {})
                user_id = session_user.get("id") or session_user.get("sub") or session_user.get("user_metadata", {}).get("provider_id")
                user_email = session_user.get("email")
                user_name = session_user.get("name") or display_name
                user_photo = session_user.get("photo_url") or avatar

                if not user_email:
                    st.error("Erro: email do usuário não encontrado. Faça login novamente.")
                    st.stop()

                # 3) upsert com consent=True
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
                    # assinatura alternativa sem user_id
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

                # 4) resultado do upsert
                if resp is None:
                    st.error("Falha ao salvar os dados", icon ="❌")
                else:
                    success_box = st.empty()
                    success_box.success("Dados salvos!", icon ="✅")
                    time.sleep(2)
                    success_box.empty()

                # 5) gerar grafo (feedback com progress)
                progress = st.progress(0)
                progress.progress(20)

                # busca usuários SOMENTE com consentimento
                users = get_all_users(consent_only=True)
                progress.progress(40)

                # filtra pelo grupo selecionado
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


# %%
# 9) Observações finais e próximos passos (executar manualmente ou adaptar)
# - Autenticação real: configure Supabase Auth (Google) no dashboard do Supabase.
#   Em produção, não use a SUPABASE_SERVICE_KEY no frontend; use rotas server-side para operações sensíveis.
# - Schema: ajuste o tamanho do vector(1536) conforme o modelo. O exemplo usa 1536 por compatibilidade com modelos maiores.
# - Se preferir busca vetorial nativa, habilite pgvector no Supabase e use consultas SQL para `ORDER BY embedding <-> new_embedding`.
# - Em produção, adicione cache para embeddings e tratamento de erros mais robusto.

# Fim do notebook
