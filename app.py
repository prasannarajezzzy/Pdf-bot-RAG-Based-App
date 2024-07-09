from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Define vector_db as a global variable
vector_db = None

# List of available embedding models
embedding_models = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "msmarco-distilbert-base-tas-b",
    "paraphrase-xlm-r-multilingual-v1",
    "multi-qa-mpnet-base-dot-v1",
    "stsb-roberta-base-v2",
    "nli-roberta-base-v2",
    "nli-mpnet-base-v2",
]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        embedding = request.form.get("embedding", "all-MiniLM-L6-v2")
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return redirect(
                url_for("display_text", filename=filename, embedding=embedding)
            )
    return render_template("upload.html")


@app.route("/display/<filename>")
def display_text(filename):
    embedding = request.args.get("embedding", "all-MiniLM-L6-v2")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    text = extract_text_from_pdf(filepath, embedding)
    return render_template("display.html", text=text)


def extract_text_from_pdf(filepath, embedding):
    global vector_db  # Indicate that we're using the global variable
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    if filepath:
        loader = UnstructuredPDFLoader(file_path=filepath)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(data)

        embeddings = SentenceTransformerEmbeddings(model_name=embedding)
        vector_db = None

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="rag-collection",
        )

        print("-------------done-------------")
    else:
        print("Upload a PDF file")
    return text


def get_llm_response(query, context):
    prompt = (
        "Answer the question based on the Context , Question : Context:"
        + context
        + "EOF"
    )

    # Generate text
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )
    outputs = model.generate(input_ids, max_length=1995)
    print("--------------------------------llm------------------------")
    print(prompt)
    print(tokenizer.decode(outputs[0]))
    llm_resposne_text = tokenizer.decode(outputs[0])
    llm_resposne_text = llm_resposne_text.replace(prompt, "")
    with open("llm_resposne_text.txt", "w") as file:
        file.write(llm_resposne_text)
    with open("prompt.txt", "w") as file:
        file.write(prompt)
    with open("context.txt", "w") as file:
        file.write(context)

    return llm_resposne_text.split("EOF")[1]


@app.route("/query", methods=["POST"])
def query():
    global vector_db  # Indicate that we're using the global variable
    data = request.json
    query_text = data.get("query", "")
    print("vec db", vector_db)
    # Query the vector database
    if vector_db and query_text:
        results = vector_db.similarity_search(query_text, k=1)
        page_content_arr = [i.to_json()["kwargs"]["page_content"] for i in results]
        response = str(results) if results else "No relevant results found."
    else:
        response = "Vector database not initialized or query is empty."

    context = ".".join(page_content_arr)
    llm_res = get_llm_response(query_text, context)
    final_res = jsonify({"response": page_content_arr, "llm_response": llm_res})
    print("response", response)
    print("llm", llm_res)

    return final_res


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
