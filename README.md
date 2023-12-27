# CS221.O11 Detect AI Generated Text
![https://github.com/dauvannam321/CS221.O11/blob/main/images/demo_results.jpeg](https://github.com/dauvannam321/CS221.O11/blob/main/images/demo_results.jpeg)

## Giới Thiệu

Dự án này tập trung vào việc phát hiện văn bản được tạo ra bởi trí tuệ nhân tạo (AI). Trong môi trường ngày nay, AI có thể tạo ra văn bản rất giống với văn bản được tạo ra bởi con người, và điều này có thể gây ra nhiều vấn đề, bao gồm sự lan truyền tin giả mạo và làm giả mạo thông tin.

## Cách Sử Dụng

### Yêu Cầu Hệ Thống

- Python 3.x
- Các thư viện cần thiết khác (liệt kê trong file `requirements.txt`)

### Cài Đặt

1. Clone repository về máy của bạn:

   ```bash
   git clone https://github.com/dauvannam321/CS221.O11.git

2. Tải pretrained các model sau và giải nén chúng trong cùng thư mục CS221.O11:
   - [BertBase](https://drive.google.com/drive/folders/1eevQU8FX1a7Zdu1bOgqryP7gELpqdTu3?usp=sharing)
   - [DebertaV3Base](https://drive.google.com/drive/folders/14gIGjSUi8FfpGsB0ih5lUR1bDSopDYoP?usp=sharing)
   
3. Cài đặt các dependencies:

   ```bash
   pip install -r requirements.txt

4. Chạy ứng dụng:

   ```bash
   streamlit run main.py
