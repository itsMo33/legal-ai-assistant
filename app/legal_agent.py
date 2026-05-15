from groq import Groq
from dotenv import load_dotenv
import os
import re
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("تحميل نموذج الـ embeddings...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("تم التحميل!")

def clean_arabic_text(text):
    text = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\s\d\.\،\؟\!\:\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_pdf():
    pdf_path = "data/labor_law.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return []
    print("Converting PDF pages to images...")
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    print(f"Found {len(images)} pages, running OCR...")
    documents = []
    for i, image in enumerate(images):
        print(f"  OCR page {i+1}/{len(images)}...")
        text = pytesseract.image_to_string(image, lang="ara")
        text = clean_arabic_text(text)
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"page": i + 1, "source": pdf_path}
            ))
    print(f"Extracted text from {len(documents)} pages")
    if len(documents) == 0:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    if len(chunks) == 0:
        return None
    print("Creating vector store...")
    db = Chroma.from_documents(
        chunks,
        embedding=EMBEDDINGS,
        persist_directory="data/chroma_db"
    )
    print(f"Vector store created with {db._collection.count()} documents!")
    return db

print("تحميل الـ vector store...")
DB = Chroma(persist_directory="data/chroma_db", embedding_function=EMBEDDINGS)
RETRIEVER = DB.as_retriever(search_kwargs={"k": 3})
print(f"جاهز! عدد المستندات: {DB._collection.count()}")

def ask_legal_question(question: str) -> str:
    docs = RETRIEVER.invoke(question)
    if not docs:
        return "عذراً، لم أجد معلومات كافية للإجابة على سؤالك."

    context = "\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""اسمك "Better Call Mohammed" وأنت مساعد قانوني ذكي متخصص في نظام العمل السعودي.
إذا سألك أحد عن اسمك قل: أنا Better Call Mohammed مساعدك القانوني المتخصص في نظام العمل السعودي.
استخدم السياق التالي للإجابة على السؤال.
تعليمات مهمة جداً:
- أجب باللغة العربية الفصحى فقط
- لا تستخدم أي كلمات أو حروف من لغات أخرى مثل الصينية أو اليابانية
- إذا وجدت كلمات غير عربية في السياق تجاهلها تماماً
- اكتب الأرقام بالعربية أو بالأرقام الإنجليزية فقط
- دائماً اذكر رقم المادة القانونية عند الإجابة مثل: وفقاً للمادة 74 من نظام العمل السعودي
- إذا كانت الإجابة تشمل أكثر من مادة اذكرها جميعاً
- نبّه المستخدم باستشارة محامٍ مرخص للمشورة الرسمية

السياق:
{context}"""
            },
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    # تنظيف أي رموز غير عربية من الرد
    answer = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\s\d\.\،\؟\!\:\-\(\)\n A-Za-z]', '', answer)
    answer = re.sub(r' +', ' ', answer).strip()
    return answer

if __name__ == "__main__":
    import shutil
    if os.path.exists("data/chroma_db"):
        count = DB._collection.count()
        print(f"Documents in store: {count}")
        if count == 0:
            print("Vector store فارغ! إعادة البناء...")
            shutil.rmtree("data/chroma_db")
            chunks = load_pdf()
            create_vector_store(chunks)
    else:
        print("Loading PDF and creating vector store...")
        chunks = load_pdf()
        if len(chunks) == 0:
            exit(1)
        create_vector_store(chunks)