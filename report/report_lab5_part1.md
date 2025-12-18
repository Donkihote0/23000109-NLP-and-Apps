# Lab5: (Nhập môn): Làm quen với PyTorch; Giới thiệu về Mạng Nơ-ron Hồi quy (RNNs) và Bài toán Phân loại Token

## Phần 1: Làm quen với PyTorch

### Mục tiêu chính

Bài thực hành này là bước đệm để bạn làm quen với PyTorch, một trong những thư
viện Deep Learning mạnh mẽ và phổ biến nhất. Trước khi xây dựng các mô hình phức
tạp như RNN, chúng ta cần nắm vững các khái niệm nền tảng.
Sau bài lab này, bạn sẽ có thể: - Hiểu và thao tác với đối tượng quan trọng nhất trong
PyTorch: Tensor. - Hiểu cách PyTorch tự động tính toán đạo hàm (gradient) thông
qua autograd. - Biết cách xây dựng một mạng nơ-ron đơn giản bằng cách kế thừa lớp
torch.nn.Module. - Làm quen với hai lớp (layer) cơ bản: nn.Linear và nn.Embedding.

### Các bước thực hiện

- Phần 1: Khám phá Tensor: Tensor là cấu trúc dữ liệu cốt lõi của PyTorch, tương tự như ndarray của NumPy nhưng
  có thêm khả năng chạy trên GPU và tự động tính đạo hàm.
  Task 1.1: Tạo Tensor
  Task 1.2: Các phép toán trên Tensor: Thực hiện các phép toán sau và in kết quả: 1. Cộng x_data với chính nó. 2. Nhân x_data với 5. 3. Nhân ma trận x_data với x_data.T (ma trận chuyển vị của nó). Sử dụng toán
  tử @.
  Task 1.3: Indexing và Slicing: Từ tensor x_data, hãy: 1. Lấy ra hàng đầu tiên. 2. Lấy ra cột thứ hai. 3. Lấy ra giá trị ở hàng thứ hai, cột thứ hai.
  Task 1.4: Thay đổi hình dạng Tensor: Sử dụng torch.rand để tạo một tensor có shape (4, 4). Sau đó, sử dụng hàm view hoặc reshape để biến nó thành một tensor có shape (16, 1).
- Phần 2: Tự động tính Đạo hàm với autograd: Đây là tính năng “ma thuật” của PyTorch. Khi bạn thực hiện các phép toán trên các tensor có requires_grad=True, PyTorch sẽ xây dựng một biểu đồ tính toán và tự động tính đạo hàm cho bạn.
  Task 2.1: Thực hành với autograd
- Phần 3: Xây dựng Mô hình đầu tiên với torch.nn: torch.nn là module cung cấp các lớp và công cụ để xây dựng mạng nơ-ron.
  Task 3.1: Lớp nn.Linear: Lớp nn.Linear thực hiện một phép biến đổi tuyến tính y = xA^T + b.
  Task 3.2: Lớp nn.Embedding: Lớp nn.Embedding là một bảng tra cứu, dùng để ánh xạ các chỉ số của từ thành các
  vector embedding.
  Task 3.3: Kết hợp thành một nn.Module: Đây là cách chuẩn để định nghĩa một mô hình trong PyTorch.

### Cách thực hiện: mở `notebook/lab5_pytorch_introduction.ipynb` và chạy từng cell để nhận kết quả

### Kết quả

```
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])

Tensor từ NumPy array:
 tensor([[1, 2],
        [3, 4]])

Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.2031, 0.6969],
        [0.0932, 0.8614]])

Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu
```

Both list và NumPy array đều được PyTorch chuyển thành tensor 2×2 giống nhau.
Điều này cho thấy PyTorch hỗ trợ tạo tensor linh hoạt từ nhiều kiểu dữ liệu.
Ý nghĩa: tạo tensor thành công và hiểu được các thuộc tính quan trọng (shape, dtype, device).

```
Kết quả cộng:
tensor([[2, 4],
        [6, 8]])

Kết quả nhân với 5:
tensor([[ 5, 10],
        [15, 20]])

Kết quả nhân ma trận:
tensor([[ 5, 11],
        [11, 25]])
```

Ý nghĩa: đã áp dụng chính xác 3 loại phép toán thường gặp trong deep learning:
element-wise, scalar operation, matrix multiplication.

```
Hàng đầu tiên:
tensor([1, 2])

Cột thứ hai:
tensor([2, 4])

Giá trị ở hàng 2, cột 2: 4
```

Ý nghĩa: đã truy cập dữ liệu trong tensor đúng chuẩn như NumPy, rất quan trọng khi chuẩn bị dữ liệu input/output.

```
Tensor ban đầu (4x4):
tensor([[0.6348, 0.4452, 0.6727, 0.2113],
        [0.8879, 0.6614, 0.0608, 0.4072],
        [0.2309, 0.9062, 0.9057, 0.5458],
        [0.4064, 0.2287, 0.2569, 0.0478]])

Tensor sau khi reshape (16x1):
tensor([[0.6348],
        [0.4452],
        [0.6727],
        [0.2113],
        [0.8879],
        [0.6614],
        [0.0608],
        [0.4072],
        [0.2309],
        [0.9062],
        [0.9057],
        [0.5458],
        [0.4064],
        [0.2287],
        [0.2569],
        [0.0478]])

Shape mới: torch.Size([16, 1])
```

Ý nghĩa: Reshape giúp thay đổi cấu trúc dữ liệu mà không đổi số phần tử → dùng liên tục trong flatten, chuẩn hóa dữ liệu, làm input cho layer linear.

```
x: tensor([1.], requires_grad=True)
y: tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at 0x785a6e5a5150>
Đạo hàm của z theo x: tensor([18.])
```

```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
Output:
 tensor([[ 0.7390, -0.2183],
        [-0.3359,  0.1085],
        [-0.0024,  1.0385]], grad_fn=<AddmmBackward0>)
```

Ý nghĩa: đã chạy fully connected layer đúng cách → bước quan trọng của mô hình neural network.

```
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings:
 tensor([[ 0.6538,  0.5125, -1.1340],
        [ 1.8118, -2.0129, -0.1645],
        [ 0.2063, -0.7799,  0.3810],
        [-1.0897, -0.5039,  0.8265]], grad_fn=<EmbeddingBackward0>)
```

Ý nghĩa: Embedding chuyển index thành vector dense – phần lõi của NLP.

```
Model output shape: torch.Size([1, 4, 2])
```

Diễn giải:
batch = 1
sequence length = 4
output features = 2

Có thể hiểu đây là mô hình dạng:
LSTM/GRU + Linear
hoặc
Embedding + Linear
cho mỗi timestep.

Ý nghĩa: Mô hình xử lý một chuỗi dài 4 token, và trả về 2 đặc trưng cho mỗi phần tử.

### Khó khăn thường gặp và giải pháp

- Khó khăn trong việc tạo và quản lý Tensor
- Lỗi shape khi thực hiện phép toán
- Khó khăn trong indexing và slicing
- Reshape dễ gây lỗi hoặc hiểu sai
- Khó hiểu về Autograd và gradient
- Linear layer thường gây lỗi shape
- Embedding khó vì yêu cầu input đặc biệt

## Phần 2: Giới thiệu về Mạng Nơ-ron Hồi quy (RNNs) và Bài toán Phân loại Token

### Mục tiêu chính: kết hợp RNN vào bài toán phân loại token

### Các bước thực hiện

1. Embedding: Mỗi từ trong câu đầu vào được chuyển đổi thành một vector
   embedding.
2. RNN Processing: Chuỗi các vector embedding này được đưa vào RNN. Tại mỗi
   từ, RNN tính toán một trạng thái ẩn h_t, chứa thông tin về từ hiện tại và ngữ cảnh
   từ các từ đứng trước.
3. Prediction: Trạng thái ẩn h_t (là một vector) được đưa qua một lớp Linear
   (Fully-Connected) để ánh xạ nó sang một vector có số chiều bằng số lượng
   nhãn (ví dụ: 17 nhãn cho bài toán POS).
4. Softmax: Cuối cùng, hàm Softmax được áp dụng lên vector này để tạo ra một
   phân phối xác suất, cho biết xác suất mỗi nhãn là đúng cho từ hiện tại. Chúng ta
   sẽ chọn nhãn có xác suất cao nhất.

### Cách thực hiện: mở file `notebook/lab5_rnn_token_classification.ipynb` và chạy từng cell để nhận kết quả

### Kết quả

```
Đọc thành công 30000 dòng văn bản.
Tách thành 30000 câu.
Top 10 từ tương đồng với 'reapply':
dosing          : 0.7752
grafting        : 0.7707
sterile         : 0.7386
misalignment    : 0.7335
exogenous       : 0.7309
awcsl           : 0.7288
inhalation      : 0.7272
liposuction     : 0.7242
confirmatory    : 0.7231
anemia          : 0.7228
```

Ảnh trực quan hóa: `notebook/output5.png`

### Giải thích kết quả và đánh giá

Dữ liệu có 30.000 đoạn text. Đây là quy mô vừa đủ để học các mối quan hệ từ cơ bản, nhưng chưa đủ lớn để học ngữ nghĩa sâu.

Word2Vec là mô hình shallow (chỉ 1 lớp ẩn). Nó học nhanh, dễ hiểu, nhưng không bắt được ngữ cảnh dài hoặc đa nghĩa như BERT / GPT.

Với min_count=2 và vector_size=100, mô hình đủ để biểu diễn ngữ nghĩa cơ bản (ví dụ: "grill" gần "bbq"). Nhưng cần clean dữ liệu tốt và đa dạng chủ đề để vector chất lượng cao hơn.

Hiệu suất và tốc độ huấn luyện: rất nhanh – đặc biệt nhờ workers=4 và kích thước từ vựng nhỏ (~10k từ). Huấn luyện chỉ vài giây đến vài chục giây.

Word2Vec + t-SNE cho biểu đồ trực quan tuyệt vời: các từ có ngữ nghĩa gần nhau tạo cụm dễ nhìn.

Hạn chế:

Không phân biệt nghĩa từ trong các ngữ cảnh khác nhau (VD: "bank" = ngân hàng và bờ sông).

Không tận dụng ngữ cảnh dài (chỉ nhìn vài từ xung quanh).

Không cập nhật được sau khi huấn luyện.

### Khó khăn và cách khắc phục

- Lỗi “ModuleNotFoundError” (thiếu thư viện)
- Lỗi mismatch giữa số lượng token và số lượng label
