from PyPDF2 import PdfReader

reader = PdfReader(
    "/home/hung/Public/leao/llm/applied-generative-ai-and-natural-language-processing-with-python/Applied Natural Language Processing with Python.pdf"
)
for page in reader.pages:
    extracted_text = page.extract_text()
    print(extracted_text)
