# Xây dựng Mini Search Engine cho báo tiếng Việt bằng Hybrid Retrieval

Repository này chứa mã nguồn cho đồ án Xử lý ngôn ngữ tự nhiên về truy hồi thông tin trên dữ liệu báo tiếng Việt. Hệ thống nhận truy vấn tiếng Việt và trả về danh sách bài báo hoặc đoạn văn liên quan theo thứ hạng. Các hướng tiếp cận chính gồm TF-IDF, BM25, Semantic Search bằng `intfloat/multilingual-e5-base`, FAISS và Hybrid Search bằng Reciprocal Rank Fusion.

Kết quả đánh giá hiện có trong repository so sánh 16 cấu hình trên 20 truy vấn. Trong phạm vi tập đánh giá này, cấu hình `semanticcfg_accented_with_stopwords` đạt kết quả cao nhất với P@10 = 0.925, MRR@10 = 0.975 và NDCG@10 = 0.879833. Kết quả chỉ đúng trong phạm vi corpus, tập truy vấn và cấu hình thí nghiệm hiện tại.

## 1. Tổng quan bài toán

Đề tài thuộc bài toán truy hồi thông tin (Information Retrieval), không phải chatbot. Người dùng nhập một truy vấn, ví dụ "vụ cướp tiệm vàng ở Huế", hệ thống tìm các bài báo có khả năng liên quan và sắp xếp chúng theo điểm.

Tìm kiếm chỉ dựa vào từ khóa có thể bỏ sót tài liệu khi truy vấn và bài báo dùng cách diễn đạt khác nhau. Ví dụ, truy vấn "giá xe hơi" có thể liên quan đến bài viết dùng cụm "chi phí mua ô tô", dù hai chuỗi không trùng hoàn toàn. Vì vậy project kết hợp hai hướng:

- Lexical retrieval: ưu tiên sự trùng khớp token, phù hợp với tên riêng, địa danh và từ khóa cụ thể.
- Semantic retrieval: biểu diễn truy vấn và văn bản bằng vector để tìm nội dung gần nghĩa.
- Hybrid retrieval: hợp nhất danh sách lexical và semantic theo thứ hạng.

## 2. Mục tiêu của đề tài

Các mục tiêu đã được thể hiện trong mã nguồn và kết quả thí nghiệm:

- Xây dựng pipeline làm sạch, chuẩn hóa và nạp dữ liệu báo tiếng Việt.
- Xây dựng chỉ mục lexical cho TF-IDF và BM25.
- Sinh embedding bằng mô hình E5 và xây dựng chỉ mục vector bằng FAISS.
- Cung cấp backend tìm kiếm với các chế độ `keyword`, `semantic` và `hybrid`.
- Cung cấp giao diện web Next.js để nhập truy vấn, lọc kết quả và hiển thị top-k.
- Đánh giá 16 cấu hình gồm TF-IDF, BM25, Semantic và Hybrid trên cùng tập truy vấn.
- Phân tích metric, độ trễ và lỗi truy hồi từ các file CSV thực nghiệm.

## 3. Chức năng chính

| Chức năng | Trạng thái trong repository | Ghi chú |
| --- | --- | --- |
| Tìm kiếm TF-IDF | Có trong code | `backend/app/services/retrieval/tfidf_engine.py` và evaluation 16 cấu hình. |
| Tìm kiếm BM25 | Có trong code | `backend/app/services/retrieval/bm25_engine.py`. |
| Semantic Search bằng E5 + FAISS | Có trong code | Cần artefact FAISS, metadata và model E5 khi chạy thật. |
| Hybrid Search bằng RRF | Có trong code | Hợp nhất theo thứ hạng, không cộng trực tiếp score BM25 và cosine. |
| Cross-Encoder reranking | Có code tùy chọn | `RERANKER_ENABLED=false` mặc định; chưa có bảng benchmark riêng trong CSV 16 cấu hình. |
| Lọc metadata | Có trong API/UI | Lọc theo category, author, source và khoảng ngày. |
| Gợi ý truy vấn | Có trong API | Endpoint `/suggest` tìm gợi ý từ tiêu đề trong corpus keyword. |
| Đánh giá tự động | Có script và CSV | Nằm trong `notebooks/evaluation_code/` và `notebooks/evaluation_csv/`. |

<img width="950" height="938" alt="image" src="https://github.com/user-attachments/assets/61e27353-72ed-432c-a815-45c99796718a" />


## 4. Kiến trúc hệ thống

Hệ thống tách thành hai giai đoạn: offline pipeline và online pipeline.

### 4.1. Quy trình offline

```text
Dữ liệu báo tiếng Việt
-> Làm sạch và chuẩn hóa văn bản
-> Tạo corpus bài báo hoặc chunk
-> Xây dựng TF-IDF/BM25
-> Sinh embedding E5 cho passage
-> Chuẩn hóa L2 embedding
-> Xây dựng FAISS index
-> Lưu index, metadata và cấu hình
```

Offline pipeline được chạy trước khi người dùng tìm kiếm. Mục tiêu là tạo sẵn các artefact nặng như pickle index, sparse matrix, embedding shard, FAISS index và metadata. Repository hiện tại có script để build nhưng không commit sẵn dữ liệu và artefact lớn trong `data/`.

### 4.2. Quy trình online

```text
Truy vấn người dùng
-> Chuẩn hóa truy vấn
-> Chọn keyword / semantic / hybrid
-> Lấy candidate top-k
-> Lọc metadata nếu có
-> Rerank bằng Cross-Encoder nếu bật
-> Gộp kết quả trùng doc_id
-> Trả kết quả cho frontend
```

Trong backend demo, endpoint chính là `GET /search`. Tham số `mode` nhận `keyword`, `semantic` hoặc `hybrid`. Reranking là tùy chọn qua tham số `rerank` hoặc biến môi trường `RERANKER_ENABLED`.

## 5. Dữ liệu và tiền xử lý

Repository hiện tại không chứa corpus đầy đủ trong thư mục `data/`; thư mục này chỉ có `.gitkeep`. Code mặc định mong đợi dữ liệu chunk tại:

```text
data/chunks/corpus_chunks_v2_e5.json
```

Các trường dữ liệu được code hỗ trợ gồm `doc_id`, `chunk_id`, `title`, `content`, `raw_text`, `url`, `source`, `topic`, `author`, `crawled_at`, `title_processed`, `content_processed`, `combined_processed`, `combined_unaccented`, `chunk_processed` và `chunk_unaccented`.

Tiền xử lý trong backend được triển khai tại `backend/app/services/preprocessing/text_normalizer.py`:

- Sửa một số lỗi mojibake thường gặp trong tiếng Việt.
- Chuyển chữ thường.
- Loại dấu tiếng Việt bằng Unicode decomposition.
- Loại URL và một số boilerplate.
- Tách token bằng biểu thức chính quy.
- Với token có dấu gạch dưới, giữ cả token ghép và các thành phần tách ra.

Notebooks trong `notebooks/` mô tả thêm các bước EDA, preprocessing, TF-IDF, BM25, E5 và FAISS. Tuy nhiên README này chỉ ghi số liệu đã có trong CSV hoặc code đọc được trực tiếp. Số bài báo ban đầu, số bài sau làm sạch và số chunk cuối cùng chưa xác định từ repository hiện tại vì corpus và artefact đầy đủ không được commit.

## 6. Các phương pháp truy hồi

### 6.1. TF-IDF

TF-IDF biểu diễn tài liệu và truy vấn thành vector thưa. TF đo mức xuất hiện của token trong tài liệu; IDF giảm trọng số của token xuất hiện quá phổ biến trong corpus. Engine trong backend tính cosine similarity giữa vector truy vấn và vector tài liệu, sau đó trả về các tài liệu có score cao nhất.

TF-IDF không phải mô hình học sâu và không hiểu ngữ nghĩa trực tiếp. Vai trò chính của phương pháp này là baseline lexical dễ kiểm tra.

### 6.2. BM25

BM25 là phương pháp lexical retrieval dựa trên tần suất từ, độ hiếm của từ và chuẩn hóa độ dài tài liệu. Code backend dùng các tham số mặc định:

| Tham số | Giá trị | Ý nghĩa |
| --- | ---: | --- |
| `k1` | 1.5 | Điều khiển mức bão hòa của tần suất từ. |
| `b` | 0.75 | Điều khiển mức chuẩn hóa theo độ dài tài liệu. |
| `title_weight` | 3 | Lặp lại tiêu đề để tăng ảnh hưởng của title trong chỉ mục. |

BM25 phù hợp khi truy vấn chứa từ khóa hoặc thực thể rõ ràng, nhưng có thể nhiễu khi token trùng khớp không phản ánh đúng chủ đề.

### 6.3. Semantic Search với E5

Semantic Search sử dụng mô hình `intfloat/multilingual-e5-base` để sinh embedding. Khi build index, passage được thêm tiền tố `passage:`; khi tìm kiếm, query được thêm tiền tố `query:`. Đây là quy ước phù hợp với cách E5 được huấn luyện.

Trong code build index, embedding được chuẩn hóa L2 trước khi lưu. Nếu build bằng script mặc định, mỗi vector có số chiều do model E5 trả về; với `multilingual-e5-base`, cấu hình thí nghiệm trước đây dùng embedding 768 chiều. Repository hiện tại không kèm `data/embeddings/e5_base/embedding_config.json`, nên số vector thực tế của artefact runtime chưa xác định từ repo con này.

### 6.4. FAISS và IndexFlatIP

FAISS là thư viện tìm kiếm vector. FAISS không tạo embedding; E5 sinh embedding, còn FAISS thực hiện tìm kiếm tương đồng trên các vector đã có.

`IndexFlatIP` là một loại chỉ mục trong FAISS:

- `Flat` nghĩa là so sánh trực tiếp với toàn bộ vector trong index.
- `IP` là Inner Product.
- Khi vector đã chuẩn hóa L2, Inner Product tương đương cosine similarity.
- Đây là exact search, không phải approximate search.

Code trong `backend/app/services/indexing/semantic_index.py` mặc định tạo `IndexFlatIP`. Hàm build cũng có tùy chọn nội bộ `use_ivf`, nhưng script CLI hiện tại chưa đưa tùy chọn này ra tham số dòng lệnh.

### 6.5. Hybrid Search và RRF

Hybrid Search kết hợp kết quả keyword và semantic. Project dùng Reciprocal Rank Fusion (RRF), tức là hợp nhất theo thứ hạng thay vì cộng trực tiếp score thô.

Công thức:

$$
\mathrm{RRF}(d)=\sum_{r \in R}\frac{1}{k+\mathrm{rank}_r(d)}
$$

Trong backend, `HYBRID_RRF_K` mặc định là 60. Trong evaluation 16 cấu hình, Hybrid dùng cùng ý tưởng RRF để kết hợp BM25 và Semantic theo từng cấu hình tiền xử lý.

Hybrid không mặc định tốt hơn Semantic. Nếu nhánh lexical đưa candidate nhiễu lên cao, RRF vẫn có thể kế thừa nhiễu đó.

### 6.6. Cross-Encoder Reranking

Cross-Encoder reranking được cài trong `backend/app/services/retrieval/reranker.py`. Reranker nhận các candidate đã được lấy từ keyword, semantic hoặc hybrid, rồi chấm lại từng cặp query-document bằng mô hình `BAAI/bge-reranker-v2-m3`.

Reranking có thể cải thiện thứ tự candidate nhưng làm tăng độ trễ vì phải chạy mô hình cho từng cặp. Trong `.env.example`, `RERANKER_ENABLED=false`, nên reranker không bật mặc định. Các file `metrics_16methods.csv` hiện không báo cáo riêng kết quả Cross-Encoder.

## 7. Cấu trúc thư mục

```text
vietnamese-news-search-engine/
├── backend/
│   └── app/
│       ├── api/routes/
│       ├── core/
│       └── services/
├── frontend/
│   └── app/
├── scripts/
├── notebooks/
│   ├── evaluation_code/
│   └── evaluation_csv/
├── docs/
├── reports/
├── data/
├── requirements.txt
├── .env.example
└── README.md
```

| Đường dẫn | Chức năng |
| --- | --- |
| `backend/app/main.py` | Entry point FastAPI. |
| `backend/app/api/routes/search.py` | API tìm kiếm, gợi ý, diagnostics và click event. |
| `backend/app/services/retrieval/` | TF-IDF, BM25, Semantic, Hybrid và Reranker. |
| `backend/app/services/indexing/` | Nạp tài liệu, build lexical index và semantic index. |
| `backend/app/services/preprocessing/` | Chuẩn hóa văn bản và tokenization cho backend. |
| `frontend/app/` | Giao diện Next.js. |
| `scripts/` | Script build và search thử lexical/semantic index. |
| `notebooks/` | Notebook EDA, preprocessing, retrieval và đánh giá. |
| `notebooks/evaluation_code/` | Script tạo ranking, qrels, metric và biểu đồ. |
| `notebooks/evaluation_csv/` | Kết quả đánh giá 16 cấu hình. |
| `docs/NLP_Project_Nhom06_ver6.docx` | Báo cáo Word của nhóm. |
| `data/` | Vị trí dự kiến cho corpus, index và embedding; repo hiện chỉ có `.gitkeep`. |

## 8. Yêu cầu hệ thống

| Nhóm | Yêu cầu |
| --- | --- |
| Python | Repository gợi ý dùng Python 3.10 trong comment của `requirements.txt`. |
| Backend | FastAPI, Uvicorn, NumPy, pandas, scikit-learn, SciPy, PyTorch, Transformers. |
| Semantic Search | Cần PyTorch, Transformers và FAISS. Windows nên cài PyTorch/FAISS bằng conda nếu dùng GPU. |
| Frontend | Node.js và npm để chạy Next.js. Repository không có `.nvmrc`. |
| GPU | Không bắt buộc cho keyword search; hữu ích khi build embedding hoặc chạy E5/Cross-Encoder. |
| Artefact | Cần dữ liệu chunk và index trong `data/` nếu muốn chạy demo đầy đủ. |

## 9. Cài đặt từng bước

### Bước 1: Clone repository

Repository chưa cung cấp URL trong file hiện tại, nên dùng placeholder:

```bash
git clone <repository-url>
cd vietnamese-news-search-engine
```

### Bước 2: Tạo môi trường Python

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux hoặc macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU trên Windows, comment trong `requirements.txt` khuyến nghị cài PyTorch và FAISS bằng conda trước, sau đó mới cài phần còn lại bằng pip.

### Bước 3: Cấu hình môi trường

Sao chép file mẫu:

```powershell
Copy-Item .env.example .env
```

Các biến quan trọng:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `SEARCH_BACKEND` | `keyword` trong `.env.example` | Mode mặc định khi API không truyền `mode`. |
| `LEXICAL_ENGINE` | `bm25` | Chọn engine lexical khi build/load keyword index. |
| `LEXICAL_INDEX_PATH` | `data/indexes/lexical_bm25.pkl` | Đường dẫn pickle index lexical. |
| `LEXICAL_DATA_PATH` | `data/chunks/corpus_chunks_v2_e5.json` | Dữ liệu chunk dùng để build lexical index. |
| `SEMANTIC_INDEX_PATH` | `data/embeddings/e5_base/semantic_e5_base.faiss` | FAISS index. |
| `SEMANTIC_METADATA_PATH` | `data/embeddings/e5_base/chunks_metadata.parquet` | Metadata của vector/chunk. |
| `SEMANTIC_CONFIG_PATH` | `data/embeddings/e5_base/embedding_config.json` | Cấu hình embedding. |
| `HYBRID_RRF_K` | `60` | Hằng số RRF. |
| `RERANKER_ENABLED` | `false` | Bật/tắt Cross-Encoder mặc định. |

### Bước 4: Chuẩn bị dữ liệu

Đặt corpus chunk tại:

```text
data/chunks/corpus_chunks_v2_e5.json
```

Repository hiện tại không kèm file này. Nếu chưa có corpus, các lệnh build index sẽ lỗi do thiếu dữ liệu đầu vào.

### Bước 5: Xây dựng chỉ mục lexical

Build BM25 mặc định:

```powershell
python scripts/build_lexical_index.py --engine bm25 --data-path data/chunks/corpus_chunks_v2_e5.json --output-path data/indexes/lexical_bm25.pkl
```

Build TF-IDF:

```powershell
python scripts/build_lexical_index.py --engine tfidf --data-path data/chunks/corpus_chunks_v2_e5.json --output-path data/indexes/lexical_tfidf.pkl
```

Chạy thử lexical index:

```powershell
python scripts/search_lexical_index.py --index-path data/indexes/lexical_bm25.pkl --query "cướp tiệm vàng ở Huế" --top-k 5
```

### Bước 6: Sinh embedding và xây dựng FAISS

```powershell
python scripts/build_semantic_index.py --data-path data/chunks/corpus_chunks_v2_e5.json --output-dir data/embeddings/e5_base --batch-size 16
```

Đầu ra:

```text
data/embeddings/e5_base/semantic_e5_base.faiss
data/embeddings/e5_base/chunks_metadata.parquet
data/embeddings/e5_base/embedding_config.json
```

Chạy thử semantic index:

```powershell
python scripts/search_semantic_index.py --query "cướp tiệm vàng ở Huế" --top-k 5
```

### Bước 7: Chạy backend

```powershell
uvicorn backend.app.main:app --reload
```

Backend mặc định chạy tại:

```text
http://127.0.0.1:8000
```

Các endpoint chính:

| Endpoint | Chức năng |
| --- | --- |
| `GET /health` | Kiểm tra backend. |
| `GET /search?q=...&mode=keyword` | Tìm kiếm keyword. |
| `GET /search?q=...&mode=semantic` | Tìm kiếm semantic. |
| `GET /search?q=...&mode=hybrid` | Tìm kiếm hybrid. |
| `GET /suggest?q=...` | Gợi ý từ tiêu đề. |
| `GET /diagnostics` | Kiểm tra trạng thái index và cấu hình. |

### Bước 8: Chạy frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend mặc định chạy tại:

```text
http://localhost:3000
```

`frontend/next.config.ts` proxy các request `/api/*` sang backend. Có thể đổi backend bằng biến môi trường `BACKEND_URL`, mặc định là `http://localhost:8000`.

### Bước 9: Kiểm tra hệ thống

Sau khi backend và frontend chạy, thử truy vấn:

```text
cướp tiệm vàng ở Huế
```

Nếu artefact keyword/semantic chưa có, `/diagnostics` sẽ báo thiếu đường dẫn tương ứng. Đây là lỗi cấu hình dữ liệu, không phải lỗi giao diện.

## 10. Cách sử dụng

1. Mở frontend tại `http://localhost:3000`.
2. Nhập truy vấn tiếng Việt.
3. Chọn chế độ tìm kiếm: keyword, semantic hoặc hybrid.
4. Chọn bộ lọc nếu cần: chuyên mục, tác giả, nguồn hoặc khoảng ngày.
5. Đọc danh sách kết quả, gồm tiêu đề, snippet, URL, score và metadata.
6. Bật rerank nếu giao diện/API truyền `rerank=true` và môi trường đã cài được Cross-Encoder.

Ví dụ gọi API:

```bash
curl "http://127.0.0.1:8000/search?q=c%C6%B0%E1%BB%9Bp%20ti%E1%BB%87m%20v%C3%A0ng%20%E1%BB%9F%20Hu%E1%BA%BF&mode=semantic&page_size=5"
```

## 11. Đánh giá thực nghiệm

Các file đánh giá nằm trong `notebooks/evaluation_csv/`.

| Nội dung | Giá trị kiểm chứng từ CSV |
| --- | ---: |
| Số truy vấn | 20 |
| Số chủ đề truy vấn | 9 |
| Số cấu hình/phương pháp được so sánh | 16 |
| Độ sâu đánh giá | Top-10 |
| Số dòng ranking | 3,200 |
| Số cặp qrels | 1,560 |
| Nhãn relevance | 0, 1, 2 |

Phân bố chủ đề truy vấn:

| Chủ đề | Số truy vấn |
| --- | ---: |
| Kinh tế | 4 |
| Công nghệ | 3 |
| Sức khỏe | 3 |
| Xã hội | 3 |
| Pháp luật | 2 |
| Thể thao | 2 |
| Du lịch | 1 |
| Giáo dục | 1 |
| Môi trường | 1 |

Ý nghĩa nhãn:

| Nhãn | Ý nghĩa |
| ---: | --- |
| 0 | Không liên quan. |
| 1 | Liên quan một phần. |
| 2 | Rất liên quan. |

Phân bố nhãn trong `annotation_candidates_16methods_labeled.csv`:

| Relevance | Số cặp |
| ---: | ---: |
| 0 | 1,008 |
| 1 | 180 |
| 2 | 372 |

Metric:

- P@10 là tỷ lệ tài liệu liên quan trong 10 kết quả đầu.
- MRR quan tâm vị trí của tài liệu liên quan đầu tiên.
- NDCG xét cả mức độ liên quan và thứ tự.
- Latency đo thời gian truy hồi trung bình theo giây/query trong file benchmark.

Trong code đánh giá, nhãn `1` và `2` được xem là liên quan khi tính P@10 và MRR@10. NDCG@10 dùng trực tiếp gain 0/1/2.

## 12. Kết quả

Bảng sau lấy cấu hình tốt nhất theo NDCG@10 của từng nhóm trong `metrics_16methods.csv`.

| Nhóm phương pháp | Cấu hình tốt nhất | P@10 | MRR@10 | NDCG@10 | Latency trung bình (s/query) |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF | `tfidf_accented_no_stopwords` | 0.220 | 0.251429 | 0.180357 | 0.263388 |
| BM25 | `bm25cfg_accented_with_stopwords` | 0.290 | 0.480000 | 0.257119 | 0.540651 |
| Semantic | `semanticcfg_accented_with_stopwords` | 0.925 | 0.975000 | 0.879833 | 0.375300 |
| Hybrid | `hybridcfg_accented_with_stopwords` | 0.630 | 0.741667 | 0.585535 | 3.434630 |

Nhận xét từ số liệu:

- Semantic với cấu hình có dấu và giữ stopword đạt NDCG@10 cao nhất.
- Hybrid tốt hơn BM25 và TF-IDF trong bảng tổng hợp, nhưng chưa vượt Semantic.
- Các cấu hình bỏ dấu giảm mạnh ở cả Semantic và Hybrid trong tập đánh giá này.
- Việc giữ stopword cho kết quả tốt hơn trong cấu hình tốt nhất của BM25, Semantic và Hybrid.
- Hybrid có độ trễ cao hơn vì phải chạy nhiều nhánh rồi hợp nhất kết quả.

Không nên diễn giải P@10 = 0.925 là tìm được 92.5% toàn bộ tài liệu liên quan trong corpus. P@10 chỉ xét tỷ lệ kết quả liên quan trong 10 vị trí đầu.

## 13. Phân tích lỗi

File lỗi chính:

```text
notebooks/evaluation_csv/error_cases_16methods.csv
```

Các nhóm lỗi cần chú ý:

- Khớp từ nhưng sai chủ đề: lexical search có thể đưa bài chứa token chung nhưng không đúng nhu cầu thông tin.
- Địa danh hoặc cụm từ phổ biến gây nhiễu: ví dụ một cụm như "miền Bắc" có thể xuất hiện trong nhiều ngữ cảnh khác nhau.
- Truy vấn quá rộng: query ngắn hoặc chung chung làm nhiều bài cùng có vẻ liên quan.
- Hybrid kế thừa nhiễu từ BM25: RRF vẫn cộng điểm theo thứ hạng cho candidate lexical đứng cao.
- Passage đúng cục bộ nhưng bài báo tổng thể chưa phù hợp: đặc biệt khi semantic search tìm theo chunk.

Các hướng cải thiện trong mục này cần được kiểm chứng lại bằng validation set, không nên chỉnh trực tiếp theo 20 query test.

## 14. Hạn chế

- Tập đánh giá có 20 truy vấn, còn nhỏ so với không gian truy vấn báo chí thực tế.
- Qrels được tạo từ pooling top-10, không bao phủ toàn bộ corpus.
- Nhãn chưa thể hiện kiểm tra chéo bởi nhiều người gán nhãn trong repository.
- Chưa có validation set riêng để tuning `rrf_k`, candidate pool hoặc trọng số kết hợp.
- Corpus và artefact runtime lớn không được commit trong repo con hiện tại.
- E5 chưa được fine-tune riêng cho báo chí tiếng Việt trong code hiện có.
- `IndexFlatIP` là exact search nên có thể khó mở rộng khi số vector tăng lớn.
- Cross-Encoder có code nhưng chưa có benchmark riêng trong bảng 16 cấu hình.

## 15. Hướng phát triển

- Mở rộng số lượng query và qrels để đánh giá ổn định hơn.
- Gán nhãn bởi nhiều người và đo agreement.
- Tách validation/test độc lập trước khi tuning.
- Fine-tune hoặc contrastive learning trên dữ liệu báo tiếng Việt.
- Thử Weighted RRF hoặc score fusion có chuẩn hóa.
- Tuning `rrf_k` và candidate pool trên validation set.
- Thử HNSW hoặc IVF khi corpus và số vector lớn hơn.
- Bổ sung incremental indexing để cập nhật bài báo mới.
- Benchmark Cross-Encoder theo chất lượng và latency.
- Theo dõi metric và latency trong môi trường triển khai thật.

## 16. Tái lập thí nghiệm

Các script đánh giá nằm trong:

```text
notebooks/evaluation_code/
```

Các file đầu ra đã có:

| File | Chức năng |
| --- | --- |
| `notebooks/evaluation_csv/rankings_16methods.csv` | Ranking top-10 của 16 cấu hình. |
| `notebooks/evaluation_csv/ranking_details_16methods.csv` | Ranking kèm metadata để gán nhãn. |
| `notebooks/evaluation_csv/annotation_candidates_16methods_labeled.csv` | Pool ứng viên đã có nhãn. |
| `notebooks/evaluation_csv/qrels_16methods.csv` | Qrels chuẩn hóa. |
| `notebooks/evaluation_csv/latency_16methods.csv` | Latency theo query và method. |
| `notebooks/evaluation_csv/metrics_16methods.csv` | Metric tổng hợp. |
| `notebooks/evaluation_csv/per_query_metrics_16methods.csv` | Metric theo từng query. |
| `notebooks/evaluation_csv/error_cases_16methods.csv` | Các trường hợp lỗi cần phân tích. |

Lệnh tính lại metric từ ranking, qrels và latency:

```powershell
python notebooks/evaluation_code/evaluate_retrieval.py --qrels notebooks/evaluation_csv/qrels_16methods.csv --rankings notebooks/evaluation_csv/rankings_16methods.csv --latency notebooks/evaluation_csv/latency_16methods.csv --output notebooks/evaluation_csv/metrics_16methods_recomputed.csv --per-query-output notebooks/evaluation_csv/per_query_metrics_16methods_recomputed.csv --k 10
```

Lệnh tạo lại ranking 16 cấu hình chỉ chạy được khi các artefact trong `data/indexes/`, `data/embeddings/` và `models/e5_base_offline/` đã tồn tại:

```powershell
python notebooks/evaluation_code/run_evaluation_rankings.py --queries data/evaluation/queries.jsonl --methods tfidf_accented_with_stopwords tfidf_accented_no_stopwords tfidf_unaccented_with_stopwords tfidf_unaccented_no_stopwords bm25cfg_accented_with_stopwords bm25cfg_accented_no_stopwords bm25cfg_unaccented_with_stopwords bm25cfg_unaccented_no_stopwords semanticcfg_accented_with_stopwords semanticcfg_accented_no_stopwords semanticcfg_unaccented_with_stopwords semanticcfg_unaccented_no_stopwords hybridcfg_accented_with_stopwords hybridcfg_accented_no_stopwords hybridcfg_unaccented_with_stopwords hybridcfg_unaccented_no_stopwords --top-k 10 --output notebooks/evaluation_csv/rankings_16methods.csv --details-output notebooks/evaluation_csv/ranking_details_16methods.csv --latency-output notebooks/evaluation_csv/latency_16methods.csv
```

