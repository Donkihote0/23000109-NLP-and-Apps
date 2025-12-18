# Lab3: Word Embeddings

## Nguyễn Đức Đạt - 23000109

### Nội dung chính

- Giảm chiều: Sử dụng các kỹ thuật như PCA hoặc t-SNE để giảm chiều
  các word vector (từ Word2Vec, GloVe, hoặc fastText) xuống còn 2 hoặc 3 chiều.
- Trực quan hóa: Vẽ biểu đồ (scatter plot) để hiển thị các từ trong không gian 2D/3D, qua
  đó quan sát mối quan hệ ngữ nghĩa giữa chúng.

### Các bước tiến hành: mở file `notebook/lab3_embedding_part1.ipynb` để chạy kết quả: chạy từng cell của notebook

- Tải mô hình Word Embedding (GloVe): Đây là mô hình embedding GloVe 100 chiều. Được huấn luyện trên Wikipedia + Gigaword. Chứa 400k từ tiếng Anh.
- Thực hiện các phép kiểm tra: Khi chạy, bạn sẽ nhận vector embeddings cho từ king (chỉ in 10 phần tử đầu).
- Biểu diễn câu: Tách câu thành từ; Lấy vector từng từ; Tính trung bình → vector biểu diễn câu. Câu được biểu diễn bằng “trung bình ngữ nghĩa”. Câu càng dài → càng trung bình → mất thông tin thứ tự → nhưng vẫn phản ánh nội dung chung.
- Trực quan hóa Word Embeddings: Giảm chiều vector từ 100D → 2D để vẽ scatter plot. Từ cùng loại đứng gần nhau: king, queen, prince nằm gần nhau; man, woman gần nhau; car, vehicle, truck tạo nhóm.

### Kết quả và giải thích

```
Vector cho từ 'king':
 [ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012 ]

Độ tương đồng giữa 'king' và 'queen': 0.7839043
Độ tương đồng giữa 'king' và 'man': 0.53093773

10 từ tương tự 'computer':
  computers  -> 0.9165
  software   -> 0.8815
  technology -> 0.8526
  electronic -> 0.8126
  internet   -> 0.8060
  computing  -> 0.8026
  devices    -> 0.8016
  digital    -> 0.7992
  applications -> 0.7913
  pc         -> 0.7883
```

Đây là vector nhúng 100 chiều.

Mỗi số thể hiện 1 chiều ngữ nghĩa.

Các từ giống nhau về nghĩa → vector gần nhau.

```
Vector biểu diễn câu:
[ 0.06438001  0.43381    -0.779435    0.0075025   0.07915     0.20077899
 -0.2454325  -0.05369498 -0.00951262 -0.68774253]
```

Câu được biểu diễn bằng “trung bình ngữ nghĩa”.

Câu càng dài → càng trung bình → mất thông tin thứ tự → nhưng vẫn phản ánh nội dung chung.

Ảnh trực quan hóa: nằm ở `notebook/output3.png`

### Nhận xét kết quả

Các cụm từ như king, queen, prince, princess, royal nằm gần nhau → đúng kỳ vọng về ngữ nghĩa.

Nhóm computer, software, technology, internet tạo cụm riêng → phản ánh đúng lĩnh vực công nghệ.

Nhóm động vật cũng có sự gần gũi (dog, cat, lion, tiger, wolf).
→ Mô hình GloVe học được mối quan hệ ngữ nghĩa khá tốt.

Nếu so sánh với mô hình Word2Vec tự huấn luyện, các cụm có thể kém tách biệt hơn do dữ liệu nhỏ hơn.

### Khó khăn thường gặp

- Tải mô hình GloVe từ gensim mất thời gian hoặc lỗi mạng
- Không phải từ nào cũng có trong mô hình (Out-of-Vocabulary – OOV)
- TSNE khi trực quan hóa chạy chậm và khó ổn định
- Việc biểu diễn câu bằng trung bình vector làm mất thông tin
- Chọn từ để trực quan hóa cần cẩn thận
- Lẫn lộn giữa cosine similarity và khoảng cách Euclidean

### Tài liệu tham khảo

- **Gensim Documentation:** [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- **GloVe Pre-trained Models:** [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
