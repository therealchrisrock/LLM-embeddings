import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from tiktoken import get_encoding

from definitions import ROOT_DIR
from llm_embeddings.scripts.preprocess import preprocess_text, store_chunks_single_file

blacklist = ['[document]', 'noscript', 'header', 'html', 'meta', 'head', 'input', 'script', ]
EPUB_PATH = f'{ROOT_DIR}/data/debt-5000-years.epub'
max_tokens = 500
def epub2thtml(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())
    return chapters
def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output

def thtml2ttext(thtml):
    Output = []
    for html in thtml:
        text =  chap2text(html)
        Output.append(text)
    return Output

def epub2text(epub_path):
    chapters = epub2thtml(epub_path)
    ttext = thtml2ttext(chapters)
    return ttext

if __name__ == '__main__':
    chapters = epub2thtml(EPUB_PATH)
    ttext = thtml2ttext(chapters)

    all_chunks = []
    for idx, chapter in enumerate(ttext):
        chapter_chunks = []
        tokenizer = get_encoding("cl100k_base")
        tokens = tokenizer.encode(chapter)
        c = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        chunks = [tokenizer.decode(chunk) for chunk in c]
        for chunk in chunks:
            all_chunks.append({"source": f"Chapter {idx+1}", "chunk": chunk})
    store_chunks_single_file(all_chunks, f"{ROOT_DIR}/data/chunks/all_chunks.json")


