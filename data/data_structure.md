# MÔ TẢ CẤU TRÚC THƯ MỤC DỮ LIỆU

Thư mục chính chứa dữ liệu là `data/`.

## Thư mục `data/`

Thư mục này là nơi chứa các nguồn dữ liệu thô (raw data) và dữ liệu đã được tiền xử lý, phục vụ cho các bài Lab về NLP
và các ứng dụng liên quan.

---

### Bộ dữ liệu `hwu` (HWU - Leeds University): nằm ở `data/data/`

Bộ dữ liệu này thường được sử dụng cho các bài toán Phân loại Ý định (Intent Classification) hoặc các tác vụ liên quan
đến Hội thoại/Trợ lý ảo.

| Tên File    | Mô tả                           | Ứng dụng/Mục đích                                                  |
| :---------- | :------------------------------ | :----------------------------------------------------------------- |
| `test.csv`  | Dữ liệu kiểm thử.               | Đánh giá hiệu năng cuối cùng của mô hình.                          |
| `train.csv` | Dữ liệu huấn luyện chính.       | Xây dựng và tối ưu hóa mô hình.                                    |
| `val.csv`   | Dữ liệu validation (thẩm định). | Theo dõi overfitting và điều chỉnh siêu tham số (hyperparameters). |

---

### Bộ dữ liệu `UD_English-EWT` (Universal Dependencies - English Web Treebank): nằm ở `data/UD_English-EWT/UD_English-EWT/`

Bộ dữ liệu tiêu chuẩn này được sử dụng rộng rãi trong các tác vụ **Phân tích Cú pháp (Parsing)** và **Gán nhãn Chuỗi** (
Sequence Labeling), thường là POS Tagging (Gán nhãn Từ loại) hoặc Dependency Parsing.

| Tên File                 | Mô tả                                 | Định dạng | Ứng dụng/Mục đích                                          |
| :----------------------- | :------------------------------------ | :-------- | :--------------------------------------------------------- |
| `en_ewt-ud-dev.conllu`   | Tập dữ liệu Phát triển (Development). | CoNLL-U   | Dùng cho Validation (kiểm thử trong quá trình phát triển). |
| `en_ewt-ud-train.conllu` | Tập dữ liệu Huấn luyện.               | CoNLL-U   | Dữ liệu chính để huấn luyện mô hình.                       |

---

### Bộ dữ liệu `conll2003`: nằm ở `data/conll2003/`

Bộ dữ liệu dùng để: Gán nhãn thực thể có tên (NER) trong văn bản tiếng Anh

- Các loại thực thể:
  PER: Person (người)
  ORG: Organization (tổ chức)
  LOC: Location (địa điểm)
  MISC: Thực thể khác (quốc tịch, sự kiện, v.v.)
- | Cột       | Ý nghĩa                             |
  | --------- | ----------------------------------- |
  | **WORD**  | Từ trong câu                        |
  | **POS**   | Nhãn Part-of-Speech (Penn Treebank) |
  | **CHUNK** | Nhãn cú pháp nông (chunking)        |
  | **NER**   | Nhãn thực thể (BIO format)          |

- | Nhãn      | Ý nghĩa                     |
  | --------- | --------------------------- |
  | **B-XXX** | Bắt đầu thực thể loại XXX   |
  | **I-XXX** | Bên trong thực thể loại XXX |
  | **O**     | Không thuộc thực thể nào    |

- Phân biệt 3 file
  📘 eng.train

        Tập huấn luyện

        Lớn nhất

        Dùng để train mô hình NER

  📗 eng.testa

        Tập validation / development

        Dùng để:

        Tune hyperparameters

        Early stopping

        Đánh giá trong quá trình huấn luyện

  📕 eng.testb

        Tập test chính thức

        Chỉ dùng để:

        Đánh giá cuối cùng

        Không được dùng trong huấn luyện

### Các File và Dữ liệu khác trong `data/`

| Tên File                          | Mô tả                                                                          | Loại                 |
| :-------------------------------- | :----------------------------------------------------------------------------- | :------------------- |
| `c4-train.00000-of-01024.json.gz` | Có khả năng là một phần của bộ dữ liệu C4 (Colossal Clean Crawled Corpus) nén. | Dữ liệu ngôn ngữ lớn |
| `sentiments.csv`                  | Dữ liệu liên quan đến bài toán Phân tích Cảm xúc (Sentiment Analysis).         | Dữ liệu phân loại    |

---
