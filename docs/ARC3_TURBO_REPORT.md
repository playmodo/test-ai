# ARC-AGI-3: nghiên cứu và bản notebook TURBO (Qwen3.8-Flash-Next + Duck)

Notebook gốc: `taaf-flashnext-sheetu12b-0922`. Nó giống từng dòng code với notebook công khai tốt nhất của Scott
Le Grand (LB 5.19). Trên LB, bản của bạn đạt từ 3.18 đến 5.12.

Notebook mới: [`notebooks/arc3-flashnext-turbo.ipynb`](../notebooks/arc3-flashnext-turbo.ipynb), được sinh bằng
[`tools/build_turbo_notebook.py`](../tools/build_turbo_notebook.py). Phần patch chạy lúc runtime nằm ở
[`notebooks/src/turbo_patch.py`](../notebooks/src/turbo_patch.py).

Notebook mới vẫn dùng đúng model, đúng các dataset đang gắn (keithtyser bundle + runtime) và đúng harness Duck.
Không cần upload dataset nào mới.

---

## 1. Tóm tắt

- **Nút thắt số 1 là phần serving (vLLM), không phải chất lượng suy luận của model.**
  - Trong lần chạy validation 09-22 (25 game, 2 giờ), 97.6 % thời gian của mỗi game nằm trong request gọi LLM.
  - Khoảng 87 % của mỗi request là **thời gian xếp hàng** trong vLLM.
  - Nguyên nhân: MTP-3 chiếm thêm ~7.5 GiB trọng số, nên KV cache chỉ còn 5 GiB, tức khoảng 105k token.
  - Mỗi prompt dài ~22k token, nên chỉ ~4-5 request chạy cùng lúc cho 28 game. Log vLLM ghi 342 lần preemption.
  - Kết quả là mỗi game chỉ được **~42 lượt gọi LLM trong 2 giờ**. Trên tập ẩn 110 game ước tính chỉ còn ~37 lượt.
- **Điểm vẫn tăng tuyến tính khi bị cắt giờ**: 5.58 điểm ở phút 60, 10.02 điểm ở phút 120.
  - Các level đã qua thì hiệu quả đã tốt: số action trung vị bằng 0.7-0.8 lần baseline của người.
  - Cái thiếu là **độ sâu** (số level qua được), mà độ sâu đến từ số lượt suy nghĩ.
- **"ctx16k" trong notebook gốc không hề có tác dụng.**
  - Cell 3 đặt `LOCAL_ANALYZER_CONTEXT_WINDOW=16384`.
  - Nhưng `serving_setup.py` ghi lại giá trị 32768, và cell 9 nạp lại giá trị đó.
  - Log xác nhận: prompt trung vị là 21.2k token, lớn nhất 27k.
- **Watchdog không thể khởi động lại vLLM.**
  - Nguyên nhân: notebook đặt `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS`.
  - Khi đó đường dựng lại argv "no-chunk" của watchdog ném lỗi. Nếu vLLM chết giữa chừng, nó chết luôn.
- **Có ~1 điểm LB là nhiễu thuần.**
  - Cùng một code base keithtyser được nộp 12 lần: trung bình 3.29, độ lệch chuẩn 0.51, dao động 2.38-4.33.
  - Notebook này đã có các điểm 3.18, 5.12 và 5.19.
- **Không có notebook công khai nào ~12 điểm.**
  - Các đội 8-13 điểm dùng cùng model đều thay đổi **phần serving và cách chia thời gian**, không phải prompt.
  - Son Pham & Mark Barney (LB 7.36 → 8.23) chạy MTP tắt, prefix caching, context lớn hơn và T=1.0.
  - Trên public-25, cùng quỹ thời gian, họ đạt 13.6-19.2, còn notebook gốc đạt 10.02.
  - Tufa Labs (27.3): "phần lớn cải thiện đến từ việc vắt thêm token từ phần cứng và phân bổ chúng đúng cách".

**Kỳ vọng.** Ước lượng từ mô hình đã back-test trên log 09-22:
- Số lượt gọi LLM mỗi game tăng từ ~37 lên ~55-72, tức 1.5-2 lần.
- Public-25 tương ứng khoảng 14-20 (so với 10.02). LB có thể ~7-10 nếu tỷ lệ LB/local giữ ~0.5.
- Đây là ước lượng, chưa phải số đo. LB nhiễu ±1, nên cần nộp ≥2 lần để kết luận.

---

## 2. Các thay đổi (theo thứ tự tác động)

| # | Thay đổi | Ở đâu | Bằng chứng | Rủi ro / cách giảm rủi ro |
|---|---|---|---|---|
| 1 | **Tắt MTP, KV bf16 12 GiB, prefix caching (mamba align), `max_num_seqs=16`**. Chuỗi fallback: 12 GiB + prefix → 10 GiB không prefix (s14) → 7 GiB (s11) → profile MTP-3 gốc (s16, argv y hệt run 09-22). Giữ 28 lane game để batch luôn đầy | cell 3 (profiles), cell 9 (chuỗi setup) | Trọng số giảm từ 81.8 xuống 74.0 GiB khi tắt MTP. Mô hình block (800 token, 52.9 block/GiB, fit khớp 3 log thật) cho 12 GiB + prefix chứa ~16.7 request trung vị (~18.7 khi có F11), so với ~4 hiện nay. Thuitanium đo 722 tok/s (MTP0/KV7/c28) so với 359 tok/s stock. Chạy thật MTP0/KV7/s20 được 1,838 request so với ~1,400. Son chạy MTP0 + prefix ON trên đúng build vLLM này (GCP, không overlay): nhiều run 132 phút, không lỗi. Son cũng đo MTP-3 ở 7×32k chỉ 5.53, so với ~18 khi không MTP (public-25) | OOM lúc khởi động: 74 + 12 + ~6.8 overhead = 92.8 / 94.97 GiB (dư ~2 GiB; profile MTP-3 đã kiểm chứng chỉ dư 0.25 GiB). Nếu profile lỗi: kill mọi tiến trình vLLM/PLE (theo marker + session của vLLM, không động tới kernel), chờ GPU, RAM và cổng 1234 được giải phóng, xoá runtime /tmp, lưu log lần lỗi, khôi phục môi trường sạch rồi mới thử tiếp. **Quy tắc chọn profile tiếp theo**: lỗi ở profile có prefix → sang profile không prefix; OOM → profile nhỏ hơn; lỗi khác, setup bị treo (timeout), hoặc đã quá 45 phút → nhảy thẳng về profile MTP-3 đã kiểm chứng. Mỗi lần thử (trừ lần cuối) bị giới hạn bởi mốc 45 phút (sàn 5 phút). OOM chỉ được xét trên log và failure record **của chính lần thử đó** (chúng được đổi tên thành `*.attemptN.*` ngay khi dọn), và bỏ qua dòng cảnh báo vô hại của autotuner FlashInfer có trong log 09-22 khoẻ mạnh |
| 2 | **Kiểm tra sống sau khi server lên** | cell 9 `_turbo_live_check` | Request giống lượt Duck: tiền tố ~9k token (bảng 900 dòng) + 1 ảnh. Chạy tuần tự (cold, rồi lặp lại đúng câu đó = đường cache-hit), sau đó 4 request đồng thời. Với prefix caching: nếu cold đúng mà bản cache sai thì là trạng thái cache hỏng, và profile bị loại. **Thêm pha quá tải** (chỉ profile prefix): max_num_seqs+4 = 20 request đồng thời, mỗi cái ~26k token khác nhau + ảnh, thinking bật, đúng sampling của harness (T=1.0, top_p 0.95, top_k 20). Tổng vượt pool KV 12 GiB (~672 > 635 block) nên vLLM buộc phải preempt và evict block cache. Mọi request phải trả 200 và server còn sống; sau đó hỏi lại câu cold (cache của nó đã bị evict) và phải vẫn đúng | Kiểm tra ngắn tối đa ~6 phút; pha tải ước tính ~1.5-2 phút (prefill ~10k tok/s đo trên G4). Không bắt buộc số preemption phải tăng (chỉ ghi log), để không loại nhầm server tốt. HTTP 400 (ước lượng kích thước của chính probe sai) không bị tính là lỗi server |
| 2b | **Keepalive gateway** (chỉ khi nộp thật): gọi `GET api/games` mỗi 90 s trong lúc setup | cell 3 | BTC: run bị dừng nếu 15 phút không có tương tác. Startup >15 phút duy nhất từng có điểm (của Son) có keepalive | Không raise, không in log |
| 2c | **Tự kiểm tra hợp đồng watchdog** ngay sau khi load | cell 17 | Lỗi watchdog không restart được đã tồn tại âm thầm trong mọi lần nộp bản gốc | Validation: raise; nộp thật: chỉ in cảnh báo |
| 2d | **Watchdog restart thì tắt prefix caching** (chỉ khi profile đang chạy có prefix) | cell 17 | Nếu vLLM chết giữa run vì prefix caching, restart đúng argv cũ sẽ chết lại (tối đa 2 lần) và cả 8 giờ còn lại mất trắng. Bọc `start_server` của watchdog: đổi `TAAF_VLLM_ENABLE_PREFIX_CACHING=0` ngay trước khi dựng argv mới (hợp đồng argv cũ đã được kiểm trước đó), identity mới khớp với môi trường mới cho lần restart sau. T7 tự ngừng theo | Cùng pool KV 12 GiB, s16 (không prefix thì mỗi request tốn ít block hơn). Nếu `start_server` lỗi trước khi chạy argv mới thì môi trường được trả lại như cũ. Đã diễn tập trên CPU: mock vLLM chết giữa run → watchdog restart không prefix → game tiếp tục |
| 3 | **Không đặt `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS`** | cell 3 | Giá trị mặc định đã là 8192, nên argv không đổi. Bỏ nó thì watchdog restart được | Không có |
| 4 | **Ngân sách thời gian mỗi game tính lúc game bắt đầu**: (thời gian còn lại − 180 s) / số wave còn cần | cell 17 + `turbo_patch` T6 | Stock cố định 7920 s. Nếu setup chậm, fallback hay watchdog restart thì wave cuối bị cắt, và game chưa bắt đầu = 0 điểm. Không bao giờ assert 25 game trên đường nộp bài | Có sàn 600 s và trần 7920 s |
| 5 | **Giới hạn history trong runtime-state**: 64 action gần nhất (kèm frame đầu level nếu level bắt đầu trước đó), JSON gọn; ≤65 entry thì giữ nguyên như stock | T1 | Stock ghi lại TOÀN BỘ history (indent=2) sau mỗi action và nạp lại vào sandbox mỗi lần gọi `action()`: ~1.7 s mỗi action ở 300 entry, tranh GIL giữa 28 luồng. Tufa example-run: 26-38 % tool call bị timeout 30 s sau 200 action | Entry đầu của mỗi đoạn giữ lại có action rỗng, nên `transitions` không bao giờ nối frame qua chỗ bị cắt. Câu mô tả `history` trong system prompt được sửa cho đúng với hành vi này. Đã kiểm thử trong sandbox thật |
| 6 | **Builtins còn thiếu trong sandbox**: `class`, `object`, `super`, `KeyError`, `IndexError`, `AttributeError`, ...; cho import `dataclasses`, `typing`, `enum` | T2 | Stock báo `NameError: __build_class__` khi định nghĩa class, và không thể `except KeyError` | Bootstrap được compile-check. Nếu không tìm thấy anchor thì giữ bản gốc. Câu "The only importable standard-library modules are: …" trong system prompt được cập nhật để khớp sandbox |
| 7 | **analyze() lỗi thì retry**, không đánh dấu game `crashed` vĩnh viễn; **backoff khi request lỗi liên tiếp** | T3 | Stock: một exception bất kỳ làm game bị crashed cho đến hết run. Khi vLLM chết, stock còn retry mỗi giây mãi mãi, mỗi lần ghi transcript (~2 GB/giờ ở 16 lane) | Tối đa 6 lần lỗi liên tiếp rồi mới để game kết thúc (giải phóng lane). Request lỗi từ lần thứ 3 thì backoff 2 → 30 s |
| 8 | **Nhiệt độ 1.0** khi MTP tắt (giữ 0.6 nếu rơi về MTP-3) | T4 | Model card Qwen3.8 cho chế độ thinking là 1.0/0.95/20. Mọi cấu hình Flash-Next của Son có điểm đều dùng 1.0. T=0.3 đã được đo là tệ hơn | Chưa có A/B riêng cho 0.6 so với 1.0 |
| 9 | **Yield 60 → 120 s, tối đa 3 tool call mỗi lượt** | T4 | Với yield 60 s, 968/1026 lượt analyze chỉ có 1 lần gọi LLM, và mỗi lần yield lại gửi lại toàn bộ prompt + ảnh. Giới hạn 3 tool call đảm bảo một lượt dài không tự đẩy prompt của chính nó ra khỏi context | – |
| 10 | **Prompt: arm B + arm C của Son/Mark.** Xoá 4 câu sai hoặc gây nhiễu ("puzzle", "64 x 64", đoạn thanh HUD/timer); thêm 6 gạch đầu dòng "cơ chế có thể là…"; xoá câu "tối ưu ít action nhất" | T5 | Đo **trên đúng bundle keithtyser này, trên phần cứng Kaggle**, 7 game khó nhất × 4 lượt: control 0.864, B 1.216, B+C 1.550 (~2.9 SE). Level đã qua đã ở 0.7-0.8× baseline, nên không cần đẩy tiết kiệm action | Chênh C so với B chỉ 1.4 SE. Không thêm câu hành vi mới: các câu kiểu đó đo được −3 đến −5 điểm |
| 11 | **ACTION7 hiển thị là `UNDO` và thực thi được** | T5 | Stock quảng cáo ACTION7 nhưng từ chối nó ("Unknown action"). Tài liệu ARC ghi ACTION7 = undo. Dùng tên `UNDO` thay vì câu ghi chú trong system prompt (arm H của Son cho thấy ghi chú gửi cho mọi game có thể gây hại) | Chỉ 6/25 game public có ACTION7 |
| 12 | **Sửa dòng sai "The game is over."**: harness đã tự RESET level rồi | T5 | Dòng này xuất hiện trong 4.3 % prompt (Tufa example-run) và làm model hiểu sai trạng thái | – |
| 13 | **Dòng đồng hồ trung tính**: "Game clock: X min used, Y min left for this game. Levels cleared: a of b." | T5 | Chỉ chèn vào bản gửi đi của prompt lượt hiện tại (không lưu vào history), tính một lần mỗi lượt | Không kèm câu thúc giục hành động |
| 14 | **AGENTFIX F11 bật**: bỏ đoạn hướng dẫn lặp lại ở các lượt cũ trong bản gửi đi | cell 10 | ~2.6k ký tự (~700 token) lặp nguyên văn ở mỗi user message cũ. Prompt giảm ~10-17 % mà không mất thông tin | Việc cắt history không thay đổi |
| 15 | **Half-swap trimming** (chỉ khi prefix caching bật): khi tràn context, cắt về 60 % budget thay vì bỏ 1 block mỗi lượt | T7 | Giữ tiền tố ổn định cho vài lượt tiếp theo, nên cache hit tăng. Son dùng cách này trong mọi kernel từ LB 5.02 trở lên; đo trên GCP thấy trung tính về chất lượng | Chỉ áp dụng ở đầu lượt, nên không bao giờ đẩy prompt đang chạy ra ngoài. Kiểm tra mỗi lần gọi: nếu watchdog đã restart không prefix thì tự ngừng |
| 16 | **Dọn dẹp cuối run không bao giờ raise** (kể cả `bm._save_json`); audit validation chấp nhận game `crashed` (in cảnh báo) | cell 17 | `stop_background` và teardown có thể ném lỗi sau khi game đã xong. Bộ chấm điểm đóng băng chỉ tính pass mà mọi game kết thúc won/gave_up/cancelled, nên có game `crashed` thì nó raise `ValueError`: khi đó in `TURBO_SCORE_SKIPPED` và giữ file parquet placeholder, bản commit vẫn nộp được | – |

Giữ nguyên: AGENTFIX F1/F3, F13 (frame animation, là thay đổi duy nhất từng được ghi nhận làm tăng LB ẩn 3.20 → 3.71),
F19 (contact sheet), ARM P (ảnh ×12), 28 lane game, xhigh reasoning (mặc định của template).

**Về prefix caching (điểm tranh luận).**
- Phản biện serving ủng hộ bật: Son đã chạy MTP0 + prefix ON trên đúng build vLLM này; +9 % số lượt, và +24 % khi kèm half-swap.
- Bằng chứng tải thật: log `g4run-q38-kwvmeta-tool-r1` của Son (GCP G4, RTX PRO 6000) dùng **đúng build vLLM
  `0.1.dev20073+g8e685d198`**, có PLE offload worker, `enable_prefix_caching: True`, `speculative_config=None` (MTP tắt),
  mamba `align`, `max_num_seqs 22`, `--async-scheduling`. KV cache chạy ở 97-99.7 % với 18-21 request, tức là có preempt.
  Trong 37 phút có 879 request chat, **0 lỗi HTTP, không crash**, prefix hit ~25 %.
- Phản biện an toàn muốn tắt: chưa ai chạy tổ hợp này trên Kaggle, và một crash giữa run thì watchdog gốc chỉ khởi động lại
  đúng cấu hình đó.
- Tôi chọn **bật ở profile đầu**, kèm kiểm tra sống, phát hiện cache hỏng và **pha quá tải** (preempt + evict thật) trước
  khi nhận profile. Nếu vLLM vẫn chết giữa run thì **watchdog restart không prefix**. Profile thứ hai tắt prefix.
- Nếu muốn an toàn tối đa, đặt `os.environ['TURBO_PROFILE_START']='1'` ở đầu cell 3: bắt đầu từ 10 GiB không prefix.
  Nên dùng bản này cho **lần nộp thứ hai** để A/B.

## 3. Những thứ đã cân nhắc nhưng KHÔNG làm

- **KV fp8**: lớp QSA của runtime này báo lỗi "requires a BF16 main KV cache".
  - Overlay 4 file vLLM của Son mở khoá fp8, nhưng là code nghiên cứu, rủi ro khi nộp.
- **Context > 32k** (ví dụ 11 lane × 43k):
  - Cần monkeypatch `serving_setup` và watchdog.
  - Không có prefix hit cao thì phần prefill tăng, ước tính **−38 % số lượt gọi**. Để dành cho A/B sau.
- **Công thức điểm, "pace yourself", Loop-A, Play discipline**:
  - Các câu hành vi thêm vào đều đo được âm, hoặc bị gây nhiễu bởi game "trúng số" (sb26).
- **Tắt suy nghĩ hoặc giảm reasoning effort**: preserve_thinking=false tụt còn 2.68; effort medium còn 8.39 (so với ~18).
- **Kill game bị kẹt**: sau khi kẹt ≥60 phút, P(qua level trong 30 phút tới) vẫn ~26 %.
- **Tường thuật animation bằng text (F13b)**: LB 2.57.
- **REFUSE no-op**: gây hồi quy v3. **IMAGE_KEEP=0**: số action giảm 29 %.
- **analyzer_timeout 1200**: không có tác dụng, vì latency tối đa chỉ 325 s.

## 4. Cách dùng trên Kaggle

1. Mở notebook gốc của bạn trên Kaggle, **File → Import notebook**, chọn `notebooks/arc3-flashnext-turbo.ipynb`.
   - Metadata đã có sẵn accelerator và các data source.
   - Kiểm tra lại 3 input đã gắn: `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`,
     `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`, model `keithtyser/qwen3-8-flash-next-nvfp4`, và competition.
   - Chọn GPU **RTX Pro 6000**, internet **tắt**.
2. **Save & Run (commit)**: chạy validation 2 giờ trên 25 game public, giống notebook gốc.
   - Muốn tiết kiệm quota: đặt `os.environ['TURBO_SMOKE_GAMES']='ft09-0d8bbf25,ls20-9607627b,vc33-5430563c'` và
     `os.environ['AGENTFIX_VALIDATION_RUNTIME_S']='900'` ở đầu cell 3.
3. Đọc log commit (mục 5). Nếu `TURBO_SERVING_READY` và các dòng kiểm tra đều ổn → **Submit**.
4. Nên nộp **ít nhất 2 lần**, vì độ nhiễu của LB là ±1.

Các biến môi trường để thử nghiệm (đặt ở đầu cell 3):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TURBO_PROFILE_START` | 0 | Bỏ qua các profile đầu. Ví dụ `1` = bắt đầu từ 10 GiB, không prefix |
| `TURBO_TEMPERATURE` / `TURBO_TEMPERATURE_MTP` | 1.0 / 0.6 | Nhiệt độ khi MTP tắt / khi MTP bật |
| `TURBO_YIELD_SECONDS`, `TURBO_TOOL_STEPS` | 120, 3 | Yield và số tool call tối đa mỗi lượt |
| `TURBO_HISTORY_KEEP` | 64 | Số action giữ trong history của sandbox (0 = tắt giới hạn) |
| `TURBO_HALF_SWAP` | auto | `off` để tắt; `auto` = chỉ bật khi prefix caching bật |
| `TURBO_CLOCK`, `TURBO_UNDO`, `TURBO_PROMPT` | 1 | Tắt từng phần patch prompt |
| `TURBO_BUDGET_RESERVE_S`, `TURBO_MAX_RUNTIME_S_PER_GAME`, `TURBO_BUDGET_FLOOR_S` | 180, 7920, 600 | Ngân sách thời gian mỗi game |
| `TURBO_SETUP_DEADLINE_S`, `TURBO_SETUP_ATTEMPT_TIMEOUT_S` | 2700, 2100 | Sau mốc này chỉ thử profile đã kiểm chứng; timeout mỗi lần setup |
| `TURBO_GATEWAY_KEEPALIVE` | 1 | Keepalive gateway khi nộp thật |
| `TURBO_LIVE_LOAD`, `TURBO_LOAD_PROMPT_TOKENS`, `TURBO_LOAD_TIMEOUT_S` | 1, 26000, 300 | Pha quá tải của kiểm tra sống (profile prefix) |
| `TURBO_RESTART_NO_PREFIX` | 1 | Watchdog restart thì tắt prefix caching |
| `TURBO_SMOKE_GAMES` | (trống) | Chạy validation trên một tập con game |

## 5. Log cần kiểm tra sau commit

- **Cell 9**:
  - `TURBO_SERVING_TRY 0 mtp0-kv12-prefix-s16 …`
  - `TURBO_SERVING_LOG … Model loading took ~74 GiB` (MTP tắt)
  - `GPU KV cache size: ~4xx,xxx tokens`
  - `Mamba cache mode is set to 'align'`
  - `TURBO_LIVE_CHECK … ok=True … cold=True/True warm=True/True burst=4/4 prefix_hits=a->b` (b > a nếu có metric)
  - `TURBO_LIVE_LOAD profile=mtp0-kv12-prefix-s16 ok=True seconds=… done=20/20 statuses=['ok'] prompt_tokens=…
    preemptions=a->b` (b > a cho thấy đường preempt đã thực sự chạy), rồi `TURBO_LIVE_CHECK post_load … ok=True`
  - `TURBO_SERVING_READY mtp0-kv12-prefix-s16`
  - Nếu có dòng `TURBO_SERVING_SETUP_FAILED` hoặc `… ok=False`: xem `/kaggle/working/vllm-openai-server.attempt0.log`
    và `vllm-setup-failure.attempt0.json`; dòng `TURBO_FALLBACK … oom=… timed_out=… to=…` cho biết lý do chọn profile sau.
- **Cell 10**: `AGENTFIX LIVE: {… 'F11_dedup': True …}`.
- **Cell 11**: `TURBO_PATCH {… 'T5_system_edits': 7, 'T5_undo': True, 'T6_budget': True, 'T7_half_swap': 0.6 …}`.
  - Không được có khoá `*_error` hoặc `*_missing`; ở chế độ commit, notebook sẽ assert điều này.
- **Cell 17**:
  - `TURBO_WATCHDOG_CONTRACT ok` và (với profile prefix) `TURBO_WATCHDOG_RESTART_POLICY prefix_off_on_restart`.
    Nếu giữa run có `TURBO_WATCHDOG_RESTART_NO_PREFIX`, vLLM đã chết một lần và được dựng lại không prefix
    (chi tiết trong `vllm-watchdog-events.jsonl`).
  - `TURBO_WAVEFIT …`, `TURBO_BUDGET_CONFIG …`, và mỗi game có một dòng `TURBO_BUDGET game=… budget_s=…`.
  - `AGENTFIX_TIMING chat … dt=` phải giảm rõ so với ~176 s của bản gốc.
  - Tổng số request và số action phải tăng.
- **Cuối run**: `mean score` của public-25 (bản gốc: 10.02 trong 2 giờ).
  - Metric vLLM nằm trong `/kaggle/working/vllm-metrics-final.prom`: preemption, prefix-cache hit, queue time.

## 6. Kiểm thử đã làm (không có GPU)

- `tools/cpu_mock/`: dựng cây `/kaggle` giả để chạy **nguyên notebook** trên CPU với LLM giả.
  - `mock_serving_setup.py` import `serving_setup.py` **thật** để validate mọi biến `TAAF_VLLM_*` và in đúng argv vLLM.
  - Chạy với 25 game public offline (arc-agi 0.9.8 / arcengine 0.9.3) và bộ chấm điểm thật.
- **Các kịch bản đã chạy đạt** (`tools/cpu_mock/run_scenarios.sh`, mỗi kịch bản chạy trọn notebook rồi `check_turbo_run.py`):
  - **A**: profile prefix qua kiểm tra ngắn + **pha quá tải 20/20** (đúng sampling 1.0/0.95/20) + câu cold sau khi evict.
    Sau đó **vLLM giả chết giữa run** → watchdog phát hiện sau ~60 s, restart với **prefix tắt**
    (`TURBO_WATCHDOG_RESTART_NO_PREFIX`, `restart_complete`) → 529 request game được server mới phục vụ, không cell nào lỗi.
  - **B**: pha quá tải trả HTTP 500 → loại profile prefix → profile 10 GiB không prefix.
  - **C**: profile 0 lỗi **OOM**, profile 1 lỗi thường → nhảy về MTP-3 (s16), nhiệt độ tự về 0.6. Kiểm tra được rằng
    OOM của lần 0 không "rò" sang quyết định của lần 1 (bản trước sẽ đi nhầm sang 7 GiB).
  - **D**: setup **bị treo** → hết giới hạn lần thử → nhảy thẳng về MTP-3.
  - **E**: setup lỗi và để lại **tiến trình mồ côi giữ cổng 1234** → bị kill → profile kế tiếp lên.
  - **F**: prefix cache "hỏng" (trả lời sai ở đường cache-hit) → loại profile prefix.
  - Bộ phân loại OOM: kiểm thử đơn vị trên log 09-22 khoẻ mạnh (không bị coi là OOM) và các dạng OOM thật.
  - Hợp đồng watchdog (argv dựng lại khớp argv đang chạy) được kiểm tra với file identity thật.
  - **Diễn tập nhánh nộp bài thật** (`--competition`, `KAGGLE_IS_COMPETITION_RERUN=1`): gateway giả chế độ competition
    với 110 game, 1 scorecard, ngân sách 9 giờ thu nhỏ còn 12 phút. Kết quả trên bản cuối: keepalive chạy, profile prefix
    qua kiểm tra ngắn + pha quá tải 20/20 + câu cold sau evict, hợp đồng watchdog ok, chính sách restart không prefix được
    cài, 4 wave, 110/110 game có ngân sách riêng và kết thúc hợp lệ, 2421 request game ở T=1.0, teardown sạch, không cell
    nào lỗi.
  - Cả 25 game chạy xong, không game nào crashed; audit + scorer chạy; không cell nào lỗi.
- **Các hiệu ứng đã xác nhận trong request:** nhiệt độ 1.0; dòng đồng hồ; system prompt đã sửa (không còn "puzzle");
  `UNDO` có trong valid actions ở các game có ACTION7; dòng GAME_OVER đã sửa; F11 hoạt động.
- **Sandbox:** class, KeyError, dataclass, typing và enum đều chạy được; `os` vẫn bị chặn.
- **Giới hạn:** không kiểm được hành vi thật của vLLM (bộ nhớ, prefix caching trên GPU). Vì vậy mới cần chuỗi fallback và
  kiểm tra sống.

Lệnh tái lập:
```
python tools/build_turbo_notebook.py
python tools/cpu_mock/build_fake_kaggle.py --duck-src <Tufalabs/duck-harness> --bundle-ref <bundle dir> --env-files <environment_files>
tools/cpu_mock/run_scenarios.sh all          # A-F ở trên
python tools/cpu_mock/run_notebook_cpu.py notebooks/arc3-flashnext-turbo.ipynb --out /tmp/c.ipynb --competition \
    --competition-run-s 720 --taaf-src <duck-harness>/tufa-arc-agi-framework/src \
    --env-files /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files \
    --env MOCK_LOG=/tmp/c.jsonl --env TURBO_BUDGET_FLOOR_S=60 --env TURBO_BUDGET_RESERVE_S=30 --env TURBO_RELEASE_MIN_MEM_GIB=0
```

## 7. Bước tiếp theo (A/B, theo thứ tự nên thử)

1. Nộp TURBO 2 lần. So sánh với cụm điểm 3.2-5.2 của bản gốc.
2. Nếu log cho thấy profile 12 GiB + prefix chạy ổn và `prefix_hits` tăng, thử `TURBO_PROFILE_START=1`.
   Đây là A/B "không prefix caching" để đo riêng tác dụng của prefix caching.
3. Thử nghiệm context 40-48k với ~10-11 lane. Cần wrapper cho `serving_setup` (vá `ANALYZER_CONTEXT`) và cho watchdog.
   Chỉ nên thử khi đã chắc prefix hit cao.
4. Overlay runtime fp8 KV của Son (`research/minimal-harness-20260912` branch, `runtime_overlays/`): gấp đôi KV. Rủi ro cao.

## Nguồn

- Mã gốc Duck: https://github.com/Tufalabs/duck-harness
- Nghiên cứu cộng đồng về đúng notebook này (log, discussion, 456 notebook có điểm): https://github.com/Hisernberg/arc-agi-3
- Son Pham & Mark Barney (arm prompt, đo serving, overlay runtime): https://github.com/sonpham-org/arc-3
- RL LoRA Qwen3.8-27B với Duck (dữ liệu tokens/episode): https://github.com/U4AR/qwen38-arc3-rl
