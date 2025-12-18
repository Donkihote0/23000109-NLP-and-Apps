## Lab X: more_research

### Nguyễn Đức Đạt - 23000109

### Bối cảnh

- Khả năng tự học là quan trọng, đặc biệt khi các em học xong và không còn ai hướng dẫn.
  Với sự phát triển của các hệ thống tìm kiếm, truy vấn thông tin, các AI/Agent và Internet, việc tự học đang dễ dàng hơn bao giờ hết trong lịch sử.
- Trong phần nội dung thêm này, các em sẽ thử sức làm một nghiên cứu để tìm hiểu về chủ đề Text To Speech (TTS).
- Yêu cầu : tìm hiểu tổng quan về tình hình nghiên cứu, các hướng phát triển hiện tại, các cách triển khai và ưu–nhược điểm của từng cách.

### Bức tranh toàn cảnh

- Level 1 — TTS dựa trên quy tắc:
  Khởi đầu của TTS là các luật âm tiết cơ bản.
  Chạy rất nhanh và hỗ trợ tốt cho đa dạng ngôn ngữ.
  Tuy nhiên tính tự nhiên thấp, nghe khá “máy”.
  - Ưu điểm: nhanh, nhẹ, ít dữ liệu
  - Nhược điểm: âm thanh thiếu tự nhiên.
- Level 2 — Deep Learning truyền thống: Các mô hình Deep Learning phát triển giúp tạo âm thanh tự nhiên hơn nhiều.
  Ý tưởng chính:Dùng dữ liệu lớn để học cách sinh âm thanh tự nhiên.Thách thức lớn đảm bảo đa dạng ngôn ngữ và yêu cầu nhiều dữ liệu huấn luyện. Nhiều nghiên cứu hiện thực hóa pipeline cá nhân hoá: mỗi người sẽ ghi âm một bộ dữ liệu nhỏ, mô hình được tinh chỉnh nhẹ để phù hợp giọng từng người.
  Ưu điểm: Giọng tự nhiên hơn Level 1; Hỗ trợ cá nhân hoá; Tiết kiệm tài nguyên hơn so với Level 3
  Nhược điểm: Cần dữ liệu ghi âm; Tốn tài nguyên huấn luyện; Chưa đạt độ tự nhiên hoàn hảo trong mọi ngữ cảnh.
- Level 3 — Few-shot Voice Cloning: Đây là hướng tân tiến: chỉ cần vài giây âm thanh vẫn có thể tạo ra giọng nói mang đặc trưng cá nhân.
  Ưu điểm: Cực kỳ tiện lợi; Độ tự nhiên cao; Phù hợp ứng dụng thời gian thực
  Nhược điểm: Mô hình rất phức tạp; Tốn tài nguyên; Dễ bị lạm dụng (deepfake).

### Tối ưu hóa

#### Pipeline tối ưu cho Level 1 – Rule-based TTS

- Nhược điểm cần giảm thiểu: Giọng thô, không tự nhiên; Không biểu cảm; Khó mở rộng sang nhiều ngôn ngữ nếu phải viết nhiều luật thủ công.
- Cách pipeline được cải tiến:
  Chuẩn hóa văn bản (Text Normalization) tự động
  → giảm lỗi đọc sai số, viết tắt, ký hiệu.
  Bổ sung mô-đun gán trọng âm (Prosody Prediction)
  → thêm nhấn giọng, ngữ điệu → nghe tự nhiên hơn.
  Tách riêng phần phát âm (G2P – Grapheme to Phoneme)
  → giúp xử lý ngôn ngữ khó (như tiếng Việt, tiếng Trung).
  Kết hợp dữ liệu thống kê (Statistical Prosody Models)
  → không cần dữ liệu lớn nhưng tăng độ tự nhiên.
  Hậu xử lý âm thanh (Post-filtering)
  → làm âm thanh đỡ “máy”.
- Tối đa hóa ưu điểm
  Tận dụng tốc độ → chạy real-time, nhúng vào thiết bị nhỏ (IoT, embedded).
  Cực rẻ tài nguyên → phù hợp chatbot, callbot giá rẻ.

#### Pipeline tối ưu cho Level 2 – Deep Learning truyền thống

- Nhược điểm cần giảm: Cần nhiều dữ liệu huấn luyện; Tốn tài nguyên nếu tinh chỉnh từng người; Khó mở rộng đa ngôn ngữ.
- Cách nghiên cứu xây dựng pipeline tối ưu:
  Sử dụng mô hình hai giai đoạn tiêu chuẩn (Tacotron, FastSpeech + Vocoder):
  - Encoder → dự đoán spectrogram
  - Vocoder → biến spectrogram thành sóng âm
    → giúp giảm yêu cầu dữ liệu và tăng độ tự nhiên.
    Transfer Learning / Pretrained Backbone
    Pretrain trên tập lớn → fine-tune cho từng người
    Giảm dữ liệu cá nhân từ vài giờ → vài phút
    Multi-speaker Training
    Train model với hàng trăm giọng khác nhau
    Model học được không gian giọng (speaker embedding)
    Cần ít dữ liệu hơn khi thêm giọng mới
    Speaker Encoder tách biệt
    Encoder rút ra đặc trưng giọng
    Decoder tạo giọng
    → Giảm chi phí tinh chỉnh, giúp cá nhân hóa nhanh.
    Lightweight Vocoder (HiFi-GAN, WaveRNN)
    Đảm bảo thời gian thực mà không tốn GPU.
- Tối đa hóa ưu điểm:
  Độ tự nhiên tăng cao
  Khả năng cá nhân hoá mạnh
  Pipeline modular → dễ nâng cấp từng phần (prosody, vocoder…)

#### Pipeline tối ưu cho Level 3 – Few-shot / One-shot Voice Cloning

- Nhược điểm cần giảm
  Cần model lớn, phức tạp
  Dễ bị lạm dụng (deepfake)
  Chất lượng giảm nếu sample input quá ngắn
- Cách nghiên cứu thiết kế pipeline để khắc phục:
  Large Foundation Models (VALL-E, MetaVoice, CosyVoice...)
  Pretrain trên hàng chục ngàn giờ voice
  Cho phép few-shot chỉ từ 3–10 giây âm thanh
  → Giảm nhu cầu tinh chỉnh, vì model đã quá mạnh.
  Prompt-based Voice Cloning
  Giọng được xem là “prompt”
  Không cần training thêm
  → Tối đa hóa sự tiện lợi.
  Self-supervised Speech Representation (HuBERT, Wav2Vec2)
  Encoder học đặc trưng giọng và ngữ điệu
  Cho chất lượng ổn định ngay cả khi audio input ngắn
  Speaker normalization + noise reduction
  Làm sạch voice sample đầu vào → tăng chất lượng clone.
  Watermarking & Voice Shield
  Gắn dấu ẩn vào audio tạo ra
  Model phát hiện deepfake hoặc lạm dụng
  → Giảm rủi ro đạo đức.
- Tối đa hóa ưu điểm
  Thời gian tạo giọng cực nhanh
  Lấy giọng người dùng chỉ từ 5–10 giây
  Rất phù hợp sản phẩm cá nhân hóa (AI avatar, AI assistant)
