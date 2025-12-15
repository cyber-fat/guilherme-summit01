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
        st.success("Memória limpa! Recarregando...")
        time.sleep(1)
        st.rerun()

# ============================
# CACHING (Carregamento do Banco)
# ============================
@st.cache_resource
def carregar_banco():
    # Verifica se a pasta existe
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
    2. Se perguntar de "palestrantes", inclua: "Lista oficial de nomes", "Relação completa de convidados".
    
    Pergunta original: {question}
    Retorne apenas as 4 perguntas, uma por linha. Sem numeração.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    resultado = chain.invoke({"question": pergunta_original})
    return [p.strip() for p in resultado.split('\n') if p.strip()]

# ============================
# LÓGICA PRINCIPAL
# ============================

# 1. Configura API Key
# Tenta pegar dos segredos do Streamlit ou variável de ambiente local
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Chave de API não configurada. Configure o .streamlit/secrets.toml ou o arquivo .env")
    st.stop()

# 2. Carrega o Banco
vectorstore = carregar_banco()
if not vectorstore:
    st.stop()

# 3. Inicializa Histórico de Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou a IA do Summit. Pode me perguntar sobre palestrantes, temas ou horários."}]

# 4. Exibe mensagens anteriores na tela (Re-renderiza o histórico)
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Caixa de Entrada do Usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona msg do usuário ao histórico visual e ao estado
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Lógica de Resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔎 *Pesquisando nos documentos...*")
        
        try:
            # Inicializa LLM
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            
            # A. Gera Variações (Multi-Query Manual)
            variacoes = gerar_variacoes_pergunta(llm, prompt)
            
            # B. Busca Robusta Otimizada
            docs_encontrados = []
            
            # ALTERAÇÃO IMPORTANTE: Reduzido k de 25 para 7 para evitar poluição de contexto
            # 4 variações x 7 docs = ~28 docs totais (gerenciável)
            for p in variacoes:
                docs = vectorstore.similarity_search(p, k=7)
                docs_encontrados.extend(docs)
            
            # Deduplicação baseada no conteúdo exato
            docs_unicos = {d.page_content: d for d in docs_encontrados}
            lista_final = list(docs_unicos.values())
            
            # Monta o contexto final
            contexto_texto = "\n\n".join([f"FONTE: {d.metadata.get('source', 'Desconhecida')}\nCONTEÚDO: {d.page_content}" for d in lista_final])

            # FERRAMENTA DE DEBUG (Visível apenas se clicar)
            # Isso ajuda a ver se o RAG está trazendo lixo ou repetindo texto
            with st.expander("🛠️ Ver Contexto Recuperado (Debug)"):
                st.write(f"Variações geradas: {variacoes}")
                st.write(f"Total de documentos únicos recuperados: {len(lista_final)}")
                st.text_area("Conteúdo Bruto enviado para a IA:", contexto_texto, height=200)

            # C. Gera Resposta Final
            template_resposta = """
            Você é um assistente especialista no Summit 'Explore a IA na Educação'.
            
            GLOSSÁRIO:
            - "Evento", "Conferência" = "Summit Explore a IA na Educação".
            - "Palestrantes" = Use a LISTA MESTRA prioritariamente.
            
            DIRETRIZES DE RESPOSTA (RÍGIDAS):
            1. Use APENAS as informações fornecidas no CONTEXTO abaixo.
            2. Se perguntarem sobre PALESTRANTES: 
               - Liste TODOS os nomes únicos encontrados.
               - DEDUPLIQUE: Se o nome "Ana" aparece 3 vezes no texto, escreva apenas uma vez.
               - Organize em ordem alfabética.
               - Não invente nomes que não estão no texto.
            3. Se a informação não estiver no contexto, diga: "Não encontrei essa informação nos documentos oficiais."
            
            CONTEXTO DOS DOCUMENTOS:
            {context}

            PERGUNTA DO USUÁRIO: {question}
            """
            
            chain = ChatPromptTemplate.from_template(template_resposta) | llm | StrOutputParser()
            
            # Executa a chain
            resposta_final = chain.invoke({"context": contexto_texto, "question": prompt})
            
            # Exibe resposta final
            message_placeholder.markdown(resposta_final)
            
            # Salva no histórico para manter a conversa
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})
            
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
