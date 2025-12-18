# Báo cáo Lab7 - Dependency Parsing

## Nguyễn Đức Đạt - 23000109

## 1. Mục đích chương trình

Phân tích phụ thuộc câu (dependency parsing) bằng spaCy và hiểu cấu trúc
cây phụ thuộc.

## 2. Nội dung chính

- Giới thiệu mô hình dependency parsing
- Thực nghiệm với spaCy
- Trích xuất thông tin như ROOT, phụ thuộc, cây cú pháp

## 3. Các bước tiến hành: chương trình nằm ở `notebook/lab7_dependency_parsing.ipynb`

- Cài đặt spaCy
- Tải mô hình ngôn ngữ
- Phân tích câu
- Trực quan hóa cây
- Viết hàm tùy chỉnh

## 4. Giải thích code và mô tả các bước

## Phần 2: Phân tích câu và trực quan hóa

### Tải mô hình và phân tích câu

```python
import spacy
from spacy import displacy

# Tải mô hình tiếng Anh đã cài đặt
# Sử dụng en_core_web_md vì nó chứa các vector từ và cây cú pháp đầy đủ
nlp = spacy.load("en_core_web_md")
# Câu ví dụ
text = "The quick brown fox jumps over the lazy dog."
# Phân tích câu với pipeline của spaCy
doc = nlp(text)
displacy.serve(doc, style="dep")
```

### Từ gốc của câu là từ "jumps"

### Từ "jumps" có các từ phụ thuộc là: "fox" và "over"

### "fox" là head của "The", "quick", "brown"

## Phần 3: Truy cập các thành phần trong cây phụ thuộc

```python
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)
print(f"{'TEXT':<12} | {'DEP':<10} | {'HEAD TEXT':<12} | {'HEAD POS':<8} | {'CHILDREN'}")
print("-" * 70)
for token in doc:
    # Trích xuất các thuộc tính
    children = [child.text for child in token.children]
    print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8}")
```

```terminal
TEXT         | DEP        | HEAD TEXT    | HEAD POS | CHILDREN
----------------------------------------------------------------------
Apple        | nsubj      | looking      | VERB
is           | aux        | looking      | VERB
looking      | ROOT       | looking      | VERB
at           | prep       | looking      | VERB
buying       | pcomp      | at           | ADP
U.K.         | compound   | startup      | NOUN
startup      | dobj       | buying       | VERB
for          | prep       | startup      | NOUN
$            | quantmod   | billion      | NUM
1            | compound   | billion      | NUM
billion      | pobj       | for          | ADP
```

## Phần 4: Duyệt cây phụ thuộc để trích xuất thông tin

### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của một động từ

```python
text = "The cat chased the mouse and the dog watched them."
doc = nlp(text)
for token in doc:
    # Chỉ tìm các động từ
    if token.pos_ == "VERB":
        verb = token.text
        subject = ""
        obj = ""
        # Tìm chủ ngữ (nsubj) và tân ngữ (dobj) trong các con của động từ
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = child.text
            if child.dep_ == "dobj":
                obj = child.text
        if subject and obj:
            print(f"Found Triplet: ({subject}, {verb}, {obj})")
```

```terminal
Found Triplet: (cat, chased, mouse)
Found Triplet: (dog, watched, them)
```

### 4.2. Bài toán: Tìm các tính từ bổ nghĩa cho một danh từ

```python
text = "The big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)
for token in doc:
    # Chỉ tìm các danh từ
    if token.pos_ == "NOUN":
        adjectives = []
        # Tìm các tính từ bổ nghĩa (amod) trong các con của danh từ
        for child in token.children:
            if child.dep_ == "amod":
                adjectives.append(child.text)
        if adjectives:
            print(f"Danh từ '{token.text}' được bổ nghĩa bởi các tính từ: {adjectives}")
```

```terminal
Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']
```

## Phần 5: Bài tập tự luyện

### Bài 1: Tìm động từ chính của câu

```python
def find_main_verb(doc):
    # Tìm token ROOT
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if root is None:
        return None  # không tìm thấy ROOT (hiếm)
    # Nếu ROOT là động từ thì trả về luôn
    if root.pos_ in ("VERB", "AUX"):
        return root
    # Nếu ROOT không phải động từ (vd: “There is a book”), tìm động từ trong các con của ROOT
    for child in root.children:
        if child.pos_ in ("VERB", "AUX"):
            return child
    # Không tìm thấy động từ ⇒ trả None
    return None
```

### Bài 2: Trích xuất các cụm danh từ (Noun Chunks)

```python
def extract_noun_phrases(doc):
    noun_phrases = []
    # Các phụ thuộc bổ nghĩa cho danh từ
    valid_deps = {"det", "amod", "compound", "poss", "nummod"}
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):  # token là danh từ
            left_indices = []
            right_indices = []
            # Duyệt children bên trái
            for child in token.lefts:
                if child.dep_ in valid_deps:
                    left_indices.append(child.i)
            # Duyệt children bên phải (ít hơn nhưng vẫn có)
            for child in token.rights:
                if child.dep_ in valid_deps:
                    right_indices.append(child.i)
            # Tạo khoảng bao phủ cụm danh từ
            start = min([token.i] + left_indices)
            end = max([token.i] + right_indices)
            span = doc[start : end + 1]
            noun_phrases.append(span)
    return noun_phrases
```

### Bài 3: Tìm đường đi ngắn nhất trong cây

```python
def get_path_to_root(token):
    path = []
    current = token
    while True:
        path.append(current)
        if current.dep_ == "ROOT":   # đã đến gốc
            break
        current = current.head       # đi lên cha
    return path
```

## 5. Phân tích kết quả

Chương trình đã phân tích chính xác quan hệ phụ thuộc, xác định ROOT và
các quan hệ như nsubj, dobj, amod,... Các hàm tùy chỉnh hoạt động đúng.

## 6. Khó khăn và giải pháp

- Việc hiểu cấu trúc dependency đòi hỏi thực hành → giải pháp: trực
  quan hóa và in cây.
- Một số câu phức tạp khó phân tích → dùng mô hình lớn hơn hoặc
  fine-tuning.

## 7. Nguồn tham khảo

- spaCy Documentation
- Stanford NLP dependency guidelines
