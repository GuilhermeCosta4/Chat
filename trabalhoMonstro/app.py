from flask import Flask, render_template, request, session ,redirect, url_for
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
import os
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, SystemMessage

app = Flask(__name__)
app.secret_key = 'supersecretkey'  

api_key = "AIzaSyB_zk1mqpX6oYf_Uz8_jhi4-s4SjdoG_a4"
genai.configure(api_key=api_key)

PASTA_DOCS = "Premier"

def carregar_documentos(pasta):
    docs = []
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())
    return docs

documentos = carregar_documentos(PASTA_DOCS)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_divididos = splitter.split_documents(documentos)

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=api_key,
    model="models/embedding-001"
)
db = FAISS.from_documents(docs_divididos, embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

juiz = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=api_key
)

prompt_juiz = '''
Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA para uma pessoa que gosta muito de futebol.

Critérios:
- A resposta está tecnicamente correta?
- Está clara para o nível da Premier league?
- O próximo passo sugerido está bem formulado?

Se a resposta for boa, diga “✅ Aprovado” e explique por quê.
Se tiver problemas, diga “⚠️ Reprovado” e proponha uma versão melhorada.
'''

def avaliar_resposta(pergunta, resposta_tutor):
    mensagens = [
        SystemMessage(content=prompt_juiz),
        HumanMessage(content=f"Pergunta do fã: {pergunta}\n\nResposta do tutor: {resposta_tutor}")
    ]
    return juiz.invoke(mensagens).content

# @app.route('/', methods=['GET', 'POST'])
# def duvida():
#     if 'conversa' not in session:
#         session['conversa'] = []

#     if request.method == 'POST':
#         pergunta = request.form['duvida']
#         try:
#             resultado = rag_chain({"query": pergunta})
#             resposta = resultado['result']

#             origem = "Arquivos" if resultado.get('source_documents') else "Gemini AI"

#             avaliacao = avaliar_resposta(pergunta, resposta)

#             entrada_usuario = ("Você", pergunta)
#             entrada_assistente = ("Assistente", resposta)
#             entrada_origem = ("Origem", origem)
#             entrada_avaliacao = ("Juiz", avaliacao)

#             session['conversa'].extend([
#                 entrada_usuario,
#                 entrada_assistente,
#                 entrada_origem,
#                 entrada_avaliacao
#             ])

#             # Log da conversa
#             

#         except Exception as e:
#             session['conversa'].append(("Erro", str(e)))

#     return render_template('duvida.html', conversa=session.get('conversa', []))
@app.route("/", methods=["GET", "POST"])
def chat():
    if "conversa" not in session:
        session["conversa"] = []

    if request.method == "POST":
        pergunta = request.form["duvida"]
        resultado = rag_chain({"query": pergunta})
        resposta = resultado['result']
        origem = "Arquivos" if resultado.get('source_documents') else "Gemini AI"
        avaliacao = avaliar_resposta(pergunta, resposta)
        session['conversa'].append(('Você', pergunta))
        session['conversa'].append(('Assistente', resposta))
        session['conversa'].append(('Origem', origem))  
        session['conversa'].append(('Juiz', avaliacao))
        session.modified = True

        with open("log.txt", "a", encoding="utf-8") as log:
                log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                log.write(f"Pergunta: {pergunta}\n")
                log.write(f"Resposta: {resposta}\n")
                log.write(f"Avaliação do Juiz: {avaliacao}\n")
                log.write("---------------------------\n")
        
        return redirect(url_for("chat"))

    return render_template("duvida.html", conversa=session["conversa"])

@app.route('/limpar', methods=['POST'])
def limpar():
    session.pop('conversa', None)
    return redirect(url_for('chat'))

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True)
