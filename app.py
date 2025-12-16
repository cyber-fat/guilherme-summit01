import streamlit as st
import os
import time

# Imports do LangChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================
# CONFIGURAÇÃO DA PÁGINA
# ============================
st.set_page_config(page_title="Summit IA Assistant", page_icon="🎓")

st.title("🎓 Assistente Summit IA na Educação")
st.markdown("Pergunte sobre palestrantes, temas e conteúdos do evento.")

# Botão na barra lateral para limpar memória
with st.sidebar:
    st.header("Controles")
    if st.button("🔄 Atualizar Cérebro (Limpar Cache)"):
        st.cache_resource.clear()
        st.session_state.messages = [] # Limpa também o chat visual
        st.success("Memória limpa! Recarregando...")
        time.sleep(1)
        st.rerun()

# ============================
# CACHING (Carregamento do Banco)
# ============================
@st.cache_resource
def carregar_banco():
    if not os.path.exists("vector_store"):
        st.error("❌ Erro: A pasta 'vector_store' não foi encontrada. Verifique se você fez o upload dos arquivos.")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        # allow_dangerous_deserialization é necessário para FAISS local confiável
        vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao carregar o banco de vetores: {e}")
        return None

def gerar_variacoes_pergunta(llm, pergunta_original):
    """Gera múltiplas versões da pergunta para melhorar a busca (Multi-Query Retrieval)"""
    template = """
    Você é um assistente de busca. O usuário está perguntando sobre o "Summit Explore a IA na Educação".
    Gere 4 versões diferentes da pergunta do usuário para encontrar a resposta correta nos documentos.
    
    Diretrizes:
    1. Se disser "evento", substitua por "Summit Explore a IA".
    2. Se perguntar de "palestrantes", inclua: "Lista oficial de nomes".
    
    Pergunta original: {question}
    Retorne apenas as 4 perguntas, uma por linha. Sem numeração.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    resultado = chain.invoke({"question": pergunta_original})
    return [p.strip() for p in resultado.split('\n') if p.strip()]

def formatar_historico(messages):
    """Transforma o histórico de chat do Streamlit em texto para a IA"""
    # Pega as últimas 6 mensagens para dar contexto sem estourar tokens
    # Ignora a primeira mensagem se for apenas a saudação do sistema
    historico_recente = messages[-6:-1] 
    texto_historico = ""
    for msg in historico_recente:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        texto_historico += f"{role}: {msg['content']}\n"
    return texto_historico if texto_historico else "Nenhum histórico anterior."

# ============================
# LÓGICA PRINCIPAL
# ============================

# 1. Configura API Key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Chave de API não configurada. Configure o .streamlit/secrets.toml")
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
    # Adiciona msg do usuário ao histórico visual
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Lógica de Resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔎 *Pesquisando...*")
        
        try:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            
            # A. Preparação do Contexto
            # 1. Recupera histórico para entender referências ("ele", "ela", "o evento")
            historico_str = formatar_historico(st.session_state.messages)

            # 2. Gera Variações para busca
            # Combinamos a pergunta atual com um pingo de contexto se necessário
            variacoes = gerar_variacoes_pergunta(llm, prompt)
            
            # B. Busca Robusta Otimizada (k=7)
            docs_encontrados = []
            for p in variacoes:
                docs = vectorstore.similarity_search(p, k=7)
                docs_encontrados.extend(docs)
            
            # Deduplicação
            docs_unicos = {d.page_content: d for d in docs_encontrados}
            lista_final = list(docs_unicos.values())
            
            contexto_texto = "\n\n".join([f"FONTE: {d.metadata.get('source', 'Desconhecida')}\nCONTEÚDO: {d.page_content}" for d in lista_final])

            # Debug (Opcional - visível apenas se expandir)
            with st.expander("🛠️ Ver Contexto e Memória (Debug)"):
                st.write("**Histórico enviado:**")
                st.text(historico_str)
                st.write(f"**Documentos recuperados:** {len(lista_final)}")

            # C. Gera Resposta Final com Memória
            template_resposta = """
            Você é um assistente especialista no Summit 'Explore a IA na Educação'.
            
            HISTÓRICO DA CONVERSA:
            {history}
            
            GLOSSÁRIO:
            - "Evento" = "Summit Explore a IA na Educação".
            
            DIRETRIZES DE RESPOSTA:
            1. Use o CONTEXTO abaixo para responder à PERGUNTA ATUAL.
            2. Se a pergunta usar pronomes como "ele", "ela", "disso", use o HISTÓRICO para entender a quem se refere.
            3. Se perguntarem sobre PALESTRANTES: 
               - Liste nomes únicos. NÃO repita nomes.
               - Se a lista for longa, cite os principais ou peça para especificar.
            4. Se a informação não estiver no contexto, diga que não sabe.
            
            CONTEXTO DOS DOCUMENTOS:
            {context}

            PERGUNTA ATUAL: {question}
            """
            
            chain = ChatPromptTemplate.from_template(template_resposta) | llm | StrOutputParser()
            
            # Passamos prompt, contexto E histórico
            resposta_final = chain.invoke({
                "context": contexto_texto, 
                "question": prompt,
                "history": historico_str
            })
            
            # Exibe resposta final
            message_placeholder.markdown(resposta_final)
            
            # Salva no histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
