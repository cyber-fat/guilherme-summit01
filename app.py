import streamlit as st
import os
import time

# Imports do LangChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Botão na barra lateral para limpar memória
with st.sidebar:
    if st.button("🔄 Atualizar Cérebro (Limpar Cache)"):
        st.cache_resource.clear()
        st.success("Memória limpa! Recarregando...")
        time.sleep(1)
        st.rerun()


# ============================
# CONFIGURAÇÃO DA PÁGINA
# ============================
st.set_page_config(page_title="Summit IA Assistant", page_icon="🎓")

st.title("🎓 Assistente Summit IA na Educação")
st.markdown("Pergunte sobre palestrantes, temas e conteúdos do evento.")

# ============================
# CACHING (Para não recarregar o banco a cada clique)
# ============================
@st.cache_resource
def carregar_banco():
    # Verifica se a pasta existe
    if not os.path.exists("vector_store"):
        st.error("❌ Erro: A pasta 'vector_store' não foi encontrada no repositório.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    # allow_dangerous_deserialization é necessário para FAISS local
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def gerar_variacoes_pergunta(llm, pergunta_original):
    template = """
    Você é um assistente de busca. O usuário está perguntando sobre o "Summit Explore a IA na Educação".
    Gere 4 versões diferentes da pergunta do usuário para encontrar a resposta correta.
    
    Diretrizes:
    1. Se disser "evento", substitua por "Summit Explore a IA".
    2. Se perguntar de "palestrantes", inclua: "Lista oficial de nomes", "Relação completa de convidados".
    
    Pergunta original: {question}
    Retorne apenas as 4 perguntas, uma por linha.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    resultado = chain.invoke({"question": pergunta_original})
    return [p.strip() for p in resultado.split('\n') if p.strip()]

# ============================
# LÓGICA PRINCIPAL
# ============================

# 1. Configura API Key (Pega dos Segredos do Streamlit Cloud)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # Fallback para rodar local se tiver no .env ou variável de sistema
    pass

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Chave de API não configurada. O Chat não funcionará.")
    st.stop()

# 2. Carrega o Banco
vectorstore = carregar_banco()
if not vectorstore:
    st.stop()

# 3. Inicializa Histórico de Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou a IA do Summit. Pode me perguntar sobre palestrantes, temas ou horários."}]

# 4. Exibe mensagens anteriores na tela
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Caixa de Entrada do Usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona msg do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Lógica de Resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔎 *Pesquisando nos documentos...*")
        
        try:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            
            # A. Gera Variações (Multi-Query Manual)
            variacoes = gerar_variacoes_pergunta(llm, prompt)
            
            # B. Busca Robusta (High Recall)
            docs_encontrados = []
            for p in variacoes:
                docs = vectorstore.similarity_search(p, k=10)
                docs_encontrados.extend(docs)
            
            # Deduplicação
            docs_unicos = {d.page_content: d for d in docs_encontrados}
            lista_final = list(docs_unicos.values())
            
            contexto_texto = "\n\n".join([f"FONTE: {d.metadata.get('source')}\nCONTEÚDO: {d.page_content}" for d in lista_final])

            # C. Gera Resposta Final
            template_resposta = """
            Você é um assistente especialista no Summit 'Explore a IA na Educação'.
            
            GLOSSÁRIO:
            - "Evento" = "Summit Explore a IA na Educação".
            - "Palestrantes" = Use a LISTA MESTRA prioritariamente.
            
            INSTRUÇÕES:
            Use o contexto abaixo. Se houver listas divididas, junte-as.
            Se não souber, diga que não sabe.

            CONTEXTO:
            {context}

            PERGUNTA: {question}
            """
            
            chain = ChatPromptTemplate.from_template(template_resposta) | llm | StrOutputParser()
            
            resposta_final = chain.invoke({"context": contexto_texto, "question": prompt})
            
            # Exibe resposta final
            message_placeholder.markdown(resposta_final)
            
            # Salva no histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro: {e}")
