# C4 Global + Six ROI — Colab

## Kiến trúc

Mỗi `image_id` có bảy view theo thứ tự cố định:

1. Global X-ray.
2. Carpal/wrist.
3. MCP ngón cái.
4. MCP ngón trỏ.
5. MCP ngón giữa.
6. MCP ngón áp út.
7. MCP ngón út.

Một ConvNeXt-Tiny chia sẻ trọng số xử lý bảy view. Các feature được nối với sex
embedding và đưa qua regression head. ROI không được xem là mẫu độc lập.

## File trên Google Drive

Đặt ba file sau trực tiếp trong `MyDrive/data`:

- `data_dev_v1.zip` — dữ liệu development global đã có từ thí nghiệm trước.
- `C4_MULTI_ROI_V1_DATA.zip` — cache 84.216 ROI và manifest khóa.
- `C4_MULTI_ROI_COLAB_CODE.zip` — code và năm config khóa.

## Chạy

Mở đúng notebook của fold cần train:

- `C4_MULTI_ROI_COLAB_FOLD_1.ipynb`
- `C4_MULTI_ROI_COLAB_FOLD_2.ipynb`
- `C4_MULTI_ROI_COLAB_FOLD_3.ipynb`
- `C4_MULTI_ROI_COLAB_FOLD_4.ipynb`
- `C4_MULTI_ROI_COLAB_FOLD_5.ipynb`

Chọn T4 GPU và nhấn **Runtime → Run all**. Không sửa cell hoặc đường dẫn.

Notebook tự động:

- mount Drive;
- kiểm tra đúng tên ba ZIP;
- xóa riêng workspace tạm `/content/c4_multi_roi_workspace`;
- giải nén và tự sửa root nếu ZIP global có thêm thư mục bao ngoài;
- kiểm tra đủ 14.036 global images và 84.216 ROI images;
- cài dependencies và chạy full preflight;
- tự resume checkpoint tương thích;
- bảo toàn checkpoint cũ nếu hash không tương thích;
- lưu checkpoint/log/prediction về Drive.

Kết quả Fold X nằm tại:

`MyDrive/data/c4_multi_roi_runs/C4_MULTI_ROI_V1_FOLD_X/`

Nếu Colab ngắt, mở lại đúng notebook và Run all. Không chạy cùng một fold trên
hai runtime cùng lúc.

## Cảnh báo khoa học

ROI V1 là các vùng hình học gần đúng dựa trên hand mask/orientation; không được
gọi là segmentation chính xác từng xương. Kết quả chỉ được kết luận sau khi đủ
OOF 5-fold và paired bootstrap CI. Bộ test 200 đã từng được truy cập nên chỉ là
locked exploratory re-evaluation.
