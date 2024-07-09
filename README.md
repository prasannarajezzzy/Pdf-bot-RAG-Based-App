# Pdf-bot-RAG-Based-app-

Demo:


https://github.com/prasannarajezzzy/RAG-Based-PDf-bot/assets/30752161/8b180a55-09a2-4043-ba3c-2824134231fb


## Overview

This project is a web application built with Flask that allows users to upload PDF files and perform queries on the content using embeddings and a pre-trained language model. The application supports various embedding models for generating embeddings of the PDF content and utilizes a transformer model for generating responses based on queries.

## Features

- **PDF Upload**: Users can upload PDF files to the application.
- **Embedding Selection**: Users can select from multiple embedding models for processing the PDF content.
- **Content Extraction**: The application extracts text from the uploaded PDF files.
- **Vector Database**: The application uses a vector database to store and retrieve document embeddings.
- **Query Interface**: Users can query the content of the uploaded PDF files, and the application provides responses using a pre-trained language model.

## Technologies Used

- **Flask**: Web framework for building the application.
- **Bootstrap**: Front-end framework for styling the web pages.
- **PyPDF2**: Library for extracting text from PDF files.
- **LangChain**: Library for handling document loaders and text splitters.
- **Chroma**: Vector database for storing and querying document embeddings.
- **SentenceTransformerEmbeddings**: Library for generating embeddings from text.
- **GPT-Neo**: Pre-trained language model for generating responses to queries.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pdf-upload-query-app.git
    cd pdf-upload-query-app
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained models**:
    - Make sure to have the necessary models downloaded for `GPTNeoForCausalLM` and `GPT2Tokenizer`.

5. **Run the application**:
    ```bash
    flask run
    ```

## Usage

1. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:5000/`.

2. **Upload a PDF**:
   - Click on the "Choose File" button to select a PDF file from your computer.
   - Select an embedding type from the dropdown menu.
   - Click the "Upload" button to upload the file.

3. **View Extracted Text**:
   - After uploading, you will be redirected to a page displaying the extracted text from the PDF.

4. **Query the Content**:
   - Send a POST request to `/query` endpoint with your query.
   - The application will return a JSON response with the relevant context and the generated response.

## Project Structure

```
pdf-upload-query-app/
│
├── templates/
│   ├── upload.html          # HTML template for file upload
│   ├── display.html         # HTML template for displaying extracted text
│
├── uploads/                 # Directory to store uploaded PDF files
│
├── app.py                   # Main Flask application file
│
├── requirements.txt         # List of required Python packages
│
└── README.md                # Project README file
```

## Embedding Models

The application supports the following embedding models:

- all-MiniLM-L6-v2
- paraphrase-MiniLM-L6-v2
- msmarco-distilbert-base-tas-b
- paraphrase-xlm-r-multilingual-v1
- multi-qa-mpnet-base-dot-v1
- stsb-roberta-base-v2
- nli-roberta-base-v2
- nli-mpnet-base-v2

## API Endpoints

- **`/` [GET, POST]**: Upload PDF file and select embedding type.
- **`/display/<filename>` [GET]**: Display extracted text from the uploaded PDF file.
- **`/query` [POST]**: Query the content of the uploaded PDF files and get a response.

## Example Query

```bash
curl -X POST http://127.0.0.1:5000/query \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the main topic of the document?"
         }'
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [EleutherAI](https://www.eleuther.ai/) for the GPT-Neo model.
- [Hugging Face](https://huggingface.co/) for providing the transformers library and pre-trained models.
- [LangChain](https://langchain.com/) for document handling utilities.
- [Bootstrap](https://getbootstrap.com/) for front-end components and styling.

---

Feel free to modify and expand upon this README to better suit the specifics of your project.
