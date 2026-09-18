# Audit C3-ROI V2 và kế hoạch hướng tới MAE ≤ 4,2 tháng

Ngày kiểm tra: 18-09-2026. Đây là audit mã, cấu hình và artifact **đã có trong project**; không phải một lượt train mới. Kết quả test 200 ảnh đã được xem nhiều lần, nên mọi phân tích trên test dưới đây là **thăm dò**. Không dùng nhãn test để chọn checkpoint, TTA, trọng số ensemble hoặc siêu tham số mới.

## 1. Baseline thực sự là gì?

Prompt ban đầu gọi kiến trúc hiện tại là “ConvNeXt V2 + Multi-Scale Label Distribution Learning”. Điều này **không đúng với C3-Z26 C3-ROI V2**, mô hình đang có test MAE tốt nhất trong nhánh này. [Config đóng băng](../c3_roi/outputs/C3_Z26_C3_ROI_T4_CODE_V2/C3_Z26_C3_ROI_V2/configs_t4_b36/fold_1.toml) ghi `architecture = "convnext_tiny"`, `sex_mode = "embedding"`, `regression_loss = "smooth_l1"`. [Model](../p1_baseline/model.py) trả về một giá trị hồi quy; ConvNeXt V2, multi-scale và LDL là các nhánh khác trong repository. Nếu báo cáo khoa học, phải dùng tên **ConvNeXt-Tiny + sex embedding + direct regression** cho C3-V2.

Đầu vào của C3-V2 là ảnh ROI bàn tay 512×512 grayscale từ C3-Z26, lặp thành ba kênh, chuẩn hóa ImageNet. ROI lấy từ mask bàn tay; khi không tìm được mask hợp lệ thì dùng ảnh toàn cục. Train dùng ImageNet weights, augmentation nhẹ (lật, xoay ±7°, dịch, scale, brightness/contrast/gamma), AdamW, cosine LR, Smooth L1 với beta 3 tháng, batch 36, tối đa 35 epoch, early stopping 8 epoch. Checkpoint được chọn theo **validation MAE sạch**, không theo test. TTA khóa 10 view: xoay −10, −5, 0, 5, 10 độ nhân lật/không lật; 5 fold tạo 50 dự đoán/ảnh rồi lấy trung bình. [Mã inference và MAE](../c3_roi/tta_test.py) giải chuẩn hóa dự đoán về tháng trước khi lấy `mean(abs(prediction − target))`.

| Tập đánh giá | Mô hình/recipe | Số ảnh | MAE tháng | Diễn giải |
|---|---|---:|---:|---|
| Validation Fold 1 | C3-V2 best checkpoint | 2.808 | 6,253462 | Mốc cho các pilot một fold trước đây |
| OOF clean | C3-V2 best checkpoint | 14.036 | 6,3301 (báo cáo baseline cũ: 6,330082) | Mốc phát triển để chọn hướng tiếp theo |
| RSNA test | C3-V2 5 fold, không TTA | 200 | 4,382809 | Thăm dò |
| RSNA test | C3-V2 5 fold, TTA 10 view | 200 | **4,251692** | Mốc test hiện tại, cách 4,2 là **0,051692 tháng ≈ 1,57 ngày** |
| RSNA test | Pilot C 5 fold, TTA 10 view | 200 | 4,402925 | Kém baseline +0,151233; CI delta [−0,0553; +0,3584] |
| RSNA test | Pilot B fold 1+5, TTA 10 view | 200 | 4,412837 | Kém baseline fold 1+5 +0,207568; CI [−0,0508; +0,4635] |

Test report gốc cho C3-V2 ở [đây](../c3_roi/artifacts/RSNA_DATA_BACKUP_20260909/C3_Z26_C3_ROI_V2_TTA_TEST/C3_Z26_C3_ROI_V2_TTA_TEST_report.json): MAE bootstrap CI [3,8014; 4,7187]. Delta TTA − raw là −0,1311 nhưng CI [−0,3115; +0,0461] cắt 0. Không có bằng chứng đủ mạnh rằng TTA luôn tốt hơn. Hai con số B/C test lấy từ output Colab được ghi trong [báo cáo B/C](BAO_CAO_PILOT_B_C_DEN_2026-09-18.md); CSV/report B/C test gốc còn trên Drive, chưa có bản sao trong project tại thời điểm audit. Không thể đối chiếu lại từng ảnh B/C test từ local artifact hiện có.

## 2. Audit dữ liệu, split và phép đo

Đã chạy [công cụ audit chỉ đọc](../c3_roi/audit_v2_protocol.py); kết quả máy đọc được ở [JSON](C3_V2_PROTOCOL_AUDIT_2026-09-18.json). Năm fold có validation 2.808/2.807 ảnh, tổng **14.036 ID duy nhất** và **14.036 SHA-256 duy nhất** trong manifest. Mỗi fold có 0 ID và 0 hash trùng giữa train/validation; hash manifest và số lượng đều khớp config đóng băng. Handoff `output_sha256` khớp cả 14.036 dòng validation. Metadata test không có cột nhãn tuổi; 200 ID test không trùng ID OOF. Đây là kiểm tra cấp ID/hash manifest, **chưa chứng minh không có cùng bệnh nhân hoặc ảnh gần trùng** vì không có patient ID, và cũng chưa tự mở mọi pixel train/test để so đối chiếu cảm nhận.

Phân phối OOF không đều: 0–59 tháng **895 ảnh (6,38%)**, 60–119 **3.881 (27,65%)**, 120–179 **8.033 (57,23%)**, 180–228 **1.227 (8,74%)**; nữ **6.430 (45,81%)**, nam **7.606 (54,19%)**. Do đó MAE chung bị chi phối bởi nhóm 120–179. Cần luôn báo cáo MAE từng nhóm, **MAE cân bằng bốn nhóm tuổi** và sai lệch có dấu, không chỉ MAE theo ảnh. Chưa thực hiện audit pixel đầy đủ về mờ, tương phản, xoay, clipping, marker, padding hay nguồn thiết bị; đây là khoảng trống kiểm tra, không phải bằng chứng rằng ảnh đã sạch.

Hai điểm sai lệch metadata/protocol được xác nhận:

1. **Target normalization không độc lập theo fold.** Cấu hình fold 2–5 đều dùng mean `127,2383327` và sample std `41,2489744`, đúng bằng thống kê **train Fold 1**. Mean train riêng của fold 2/3/4/5 là `127,4147297 / 127,3529255 / 127,2581708 / 127,2562116`; sample std cũng khác. Tập train Fold 1 chứa nhãn validation của các fold khác, nên cấu hình fold 2–5 đã dùng một thống kê được fit ngoài tập train của chính fold đó. Đây là **rò rỉ thống kê nhãn nhẹ ở mức protocol**, không phải train trực tiếp bằng ảnh validation; chưa định lượng tác động lên MAE. Không sửa config/checkpoint cũ sau khi train. Recipe mới phải tính mean/std trong từng fold train, đóng băng config/hashes rồi train lại nếu muốn đo tác động.
2. **`roi_mode` sau rescue không phản ánh ảnh thực dùng.** Manifest vẫn ghi `global_fallback` cho **1.974** ảnh phát triển, nhưng builder V2 đã thay pixel của **588** ảnh bằng candidate rescue. Số fallback thực sau rescue còn **1.386**; nếu nhóm theo cột `roi_mode` cũ sẽ trộn ảnh đã rescue với fallback thật. Test có 56/200 `global_fallback` theo handoff; rescue audit 588 ảnh chỉ thuộc train/validation phát triển. Nên phân tích subgroup bằng `rescue_status` ghép theo `image_id`, không dùng đơn độc `roi_mode` cũ. Giữ nguyên artifact lịch sử để tái lập; sửa metadata ở **phiên bản dataset mới** nếu cần.

## 3. Những thí nghiệm đã cho biết gì?

| Nhánh | Kết quả liên quan | Quyết định hiện tại |
|---|---|---|
| SWA/best+late2 | OOF 6,2881 so với best 6,3301; gain 0,0420 dưới gate 0,1 tháng | Không nâng làm model chính |
| LDL auxiliary Fold 1 | Tốt nhất 6,3421 so baseline 6,2535 | Không ưu tiên lặp lại cùng recipe |
| Đổi seed 2026 Fold 1 | 6,3597; ensemble 50/50 6,1875, gain 0,0659 dưới gate 0,1 | Không mở rộng chỉ vì một ensemble pilot |
| Ordinal auxiliary Fold 1 | 6,3374; ensemble baseline+ordinal 6,1630, gain 0,0905 dưới gate | Chưa chứng minh lợi ích đủ lớn |
| C3+E1+D3 1/3 | OOF 6,0442, nhưng test 4,3482 so C3 solo 4,2517 | Không chọn ensemble bằng test |
| Pilot C | OOF clean 6,3707 vs baseline 6,3300; artifact tổng hợp 6,5227 vs 7,2378; nhóm 0–59 clean +0,4976 | Giá trị về robustness tổng hợp; không thay baseline clean |
| Blend baseline 27% + C 73% | OOF clean 6,2725; artifact 6,4879; nhưng 0–59 clean +0,1960 | Chỉ kết quả thăm dò OOF; không dùng test để tinh chỉnh weight |

Kết quả C trên ảnh artifact là thật trong **cùng họ biến đổi tổng hợp** và ổn định qua ba seed biến đổi, nhưng không chứng minh khả năng khái quát sang bệnh viện/thiết bị khác. B và C test clean 200 ảnh đều không thắng baseline trong phép so sánh tương ứng, CI delta đều cắt 0. Fold 5 B/C làm xấu nhóm trẻ nhỏ; chi tiết và CI xem [báo cáo B/C](BAO_CAO_PILOT_B_C_DEN_2026-09-18.md). Vì vậy các hướng “thêm LDL/ordinal/consistency/SWA/TTA” không nên được tuyên bố là đường chắc chắn xuống 4,2.

## 4. Nghiên cứu gốc liên quan và giới hạn áp dụng

| Paper gốc | Bài học có thể kiểm tra ở đây | Giới hạn/rủi ro |
|---|---|---|
| [Deeplasia: deep learning for bone age assessment validated on skeletal dysplasias](https://pmc.ncbi.nlm.nih.gov/articles/PMC10776485/) (2024, online 2023) | Ensemble các điều kiện train, gồm độ phân giải cao; xác nhận RSNA test và bộ DHA/bệnh lý riêng. RSNA 200 ảnh báo cáo MAD 3,87 tháng. | Họ có bộ ngoài và nhiều mô hình; khác recipe/nguồn nhãn và mức tiếp xúc test. Không chuyển thẳng số 3,87 thành kỳ vọng cho C3. |
| [Bone age assessment by multi-granularity and multi-attention feature encoding (2M-Net)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11320534/) (2024) | Đa độ hạt và vùng xương có thể hữu ích; paper báo cáo 3,98 tháng trên benchmark của họ. Có thể thử nhánh local/global hoặc crop bàn tay/cổ tay có kiểm soát. | Kiến trúc và quy trình chia/train khác; nhiều nhánh tăng chi phí/overfit. Cần đối chứng cùng split và budget. |
| [Evaluating the Robustness of a Deep Learning Bone Age Algorithm to Clinical Image Variation Using Computational Stress Testing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140516/) (2024) | Khóa ma trận biến đổi brightness, contrast, xoay, lật, marker, độ phân giải; paper thấy nhiều biến đổi làm giảm độ ổn định. | Stress tổng hợp không thay cho bệnh viện ngoài; B/C ở đây cũng cho thấy robustness và clean accuracy có thể đổi chỗ. |
| [Population-specific calibration and validation of an open-source bone age AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC12457598/) (2025) | Calibration sex-specific trên tập fitting tách riêng có thể giảm bias khi chuyển miền; paper cải thiện 6,57→5,69 trên cohort Georgia held-out. | Hiệu quả phụ thuộc domain shift; fit calibration trên test 200 sẽ là leakage. Cần OOF cross-fit hoặc tập calibration độc lập. |
| [MSADCN: Multi-Scale Attentional Densely Connected Network for Automated Bone Age Assessment](https://www.techscience.com/cmc/v78n2/55580) (2024) | Kết hợp multi-scale với LDL + expectation regression; gợi ý ablation decode và sigma nếu thật sự quay lại LDL. | Pilot LDL hiện tại đã kém baseline; thêm phức tạp chưa có bằng chứng sẽ giúp C3. |
| [ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html) (2023) | ConvNeXt V2 có GRN và recipe pretraining khác; là một **architecture ablation** hợp lệ nếu giữ nguyên data/split/loss. | Paper không phải nghiên cứu bone age, nên không có cam kết giảm MAE; thay backbone có chi phí lớn. |
| [Generalizability and Bias in a Deep Learning Pediatric Bone Age Prediction Model Using Hand Radiographs](https://pubs.rsna.org/doi/10.1148/radiol.220505) (2022) | Ngoài MAE chung cần xem subgroup và external validation; paper báo cáo bias lâm sàng dù điểm tổng thể tốt. | Không được suy diễn hiệu quả trên DHA/bệnh viện từ RSNA test 200. |

Các số paper chỉ là bối cảnh, **không phải đối chứng trực tiếp công bằng** với C3-V2. Cần kiểm tra cùng test ID, cùng reference labels, điều kiện train, tiêu chí chọn model và mức sử dụng test trước khi nói “vượt paper”.

## 5. Kế hoạch thí nghiệm, theo thứ tự ưu tiên

Gate chung trước khi bắt đầu: chốt một baseline OOF, split/hash, seed, metric và chi phí; chỉ thay **một yếu tố** mỗi thí nghiệm. So sánh dự đoán ghép cặp trên đủ 14.036 ảnh, report CI bootstrap theo ảnh, MAE theo tuổi/giới/ROI hiệu lực, MAE cân bằng tuổi, bias có dấu, thất bại segmentation và chi phí. Giữ ứng viên khi clean OOF gain ít nhất **0,10 tháng**, CI 95% của delta ứng viên − baseline có cận trên <0, **không gây hại rõ ràng** cho nhóm 0–59 hoặc 60–119 (CI cận dưới của harm >0), và ổn định qua ít nhất hai fold/seed trước khi mở rộng. Gate 0,10 là ngưỡng quyết định thực nghiệm, không phải khẳng định ý nghĩa lâm sàng. Test 200 chỉ báo cáo sau khi recipe đã khóa; xác nhận thật cần cohort mới chưa từng xem.

| Ưu tiên / chi phí | Giả thuyết và ablation một biến | File liên quan / thiết lập | Giữ hoặc loại |
|---|---|---|---|
| **Cao / CPU, không train** | Audit từng ảnh lỗi lớn, tuổi 0–23/24–59/60–119, rescue thật và fallback thật; kiểm tra marker, mờ, padding, crop cắt xương, contrast. Đối chiếu ảnh gốc và ảnh ROI trên train/OOF; không chọn theo test. | `c3_roi/segmentation.py`, handoff/rescue audit, OOF CSV; lấy mẫu theo lỗi và ngẫu nhiên đối chứng, reviewer blind với model. | Chỉ chuyển thành thay đổi preprocessing khi xác định được lỗi ảnh/ROI có quy luật và QC tái lập; đây là chẩn đoán, không có MAE “sau” chưa train. |
| **Cao / 1–2 fold pilot, rồi 5 fold nếu qua gate** | Trẻ nhỏ thiếu mẫu và dễ bias: thử **một** mức age-balanced sampling nhẹ so permutation, hoặc age-weighted Smooth L1, không đồng thời. Đối chứng được train lại với mean/std riêng fold. | `p1_baseline/data.py`, config mới; ví dụ weight clip 1,5–2× hoặc sampling mix 25% balanced/75% gốc; cố định tổng update. | Xem toàn OOF, MAE 0–23/24–59/60–119, age-macro và sex; loại nếu gain chung do hy sinh nhóm khác. Rủi ro overfit trẻ nhỏ cao; Fold 1+5 pilot trước. |
| **Cao / 1–2 fold pilot** | Test preprocessing thuần túy: ROI V2 vs crop khác biên 8%/12% hoặc intensity normalization/CLAHE nhẹ **từng biến**, cùng checkpoint? Nếu đổi ảnh đầu vào khi inference, trước hết kiểm tra paired OOF; nếu có tín hiệu, train cùng recipe để công bằng. | `c3_roi/segmentation.py`, builder dataset mới, `p1_baseline/data.py`; lưu mask/bbox/QC và pixel SHA theo phiên bản. | Gate clean OOF, age groups và fallback hiệu lực; loại nếu tăng artifact robustness mà hại clean hoặc che cấu trúc xương. Không dùng inpainting mặc định. |
| **Trung bình / 5 fold inference, ít GPU** | Cross-fit bias correction hoặc sex-specific affine calibration từ OOF training folds; kiểm tra liệu bias theo tuổi/giới thật sự nhất quán qua fold. So với identity, không tune trên test. | OOF prediction CSV, script calibration mới; ridge/linear có regularization, ràng buộc monotonic; fit 4 fold, đánh giá fold thứ 5. | Chỉ giữ nếu cross-fit gain qua gate và không hại trẻ nhỏ; báo cáo độ biến thiên tham số. Rủi ro học nhiễu và nhãn test nếu làm sai protocol. |
| **Trung bình–thấp / GPU cao** | ConvNeXt V2 **hoặc** multi-scale/global-local, mỗi lần một thay đổi, giữ preprocessing/loss/split/budget như đối chứng; chỉ thử sau khi pilot dữ liệu/ROI rõ ràng. | `p1_baseline/model.py`, config nhánh mới; dùng pretrained tương ứng, 512 px, same optimizer/update budget, 1 fold pilot rồi 5 fold. | Không giữ vì paper có MAE thấp; giữ nếu OOF và subgroup qua gate. Rủi ro lệch pretraining, nhiều tham số, overfit và khó quy kết. |

Các thử nghiệm thấp ưu tiên: quét Gaussian sigma/adaptive LDL, EMD/Wasserstein/KL, ordinal, sex auxiliary head, EMA/SWA, thêm TTA view, ensemble nhiều backbone. Nên làm **sau** khi có bằng chứng riêng rằng bottleneck là label uncertainty hoặc model variance. Với dữ liệu hiện tại, LDL/ordinal/SWA/TTA đã có kết quả không đủ mạnh. Mỗi phép thử mới phải có ablation `regression-only`, `auxiliary-only` nếu hợp lý, `combined`, decode expectation vs regression, cùng budget và split. Không lặp lại grid tìm MAE trên 200 test.

## 6. Thay đổi đã thực hiện và trạng thái mục tiêu

Đã thêm công cụ audit chỉ đọc [audit_v2_protocol.py](../c3_roi/audit_v2_protocol.py) và [kết quả JSON](C3_V2_PROTOCOL_AUDIT_2026-09-18.json). Công cụ không sửa ảnh, checkpoint, manifest hay kết quả train cũ. Đã chạy nó trên package đóng băng, đối chiếu hash/split/config và tìm ra hai mismatch ở trên. **Không có train mới, không có MAE mới, chưa đạt ≤4,2 bằng chứng thực nghiệm.** Từ 4,2517 xuống 4,2 chỉ cần 0,0517 tháng về mặt số học, nhưng CI rộng và test đã được xem nhiều lần; đạt một số 4,19 trên cùng test bằng thử tiếp cũng không chứng minh cải thiện đáng tin cậy. Ưu tiên tiếp theo là sửa protocol cho thí nghiệm mới, chạy OOF pilot có gate và tìm bộ xác nhận độc lập.
