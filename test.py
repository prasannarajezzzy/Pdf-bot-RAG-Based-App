from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
from flask import jsonify
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}
vector_db = None


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
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return redirect(url_for("display_text", filename=filename))
    return render_template("upload.html")


@app.route("/display/<filename>")
def display_text(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    text = extract_text_from_pdf(filepath)
    return render_template("display.html", text=text)


def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    if filepath:
        loader = UnstructuredPDFLoader(file_path=filepath)
        data = loader.load()
        print("-------------done-------------")
    else:
        print("Upload a PDF file")

    return text


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query", "")
    print("vecdb", vector_db)
    # Query the vector database
    if vector_db and query_text:

        results = vector_db.similarity_search(query_text, k=3)
        response = results[0].text if results else "No relevant results found."
    else:
        response = "Vector database not initialized or query is empty."

    return jsonify({"response": response})


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
