# Changelog

All notable changes to the `alice-zip` Rust crate (repository root, crates.io)
are documented here. The CLI / FFI / Python native crate under `libalice/` has
its own [CHANGELOG](libalice/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Python package の `decompress_residual` は none / lzma / zlib の residual を float64 に変換せず、保存された float32 の bit のまま返す float64 を経由すると signaling NaN が quiet NaN に変わっていた (ランダムな bit パターンでおよそ 1000 個に 2 個) header の `shape` と `original_len` が食い違えば `ValueError`
- Python package の ResidualData の delta は、先頭の差分を 0 にしていたため先頭の値を失い、全体がその値だけずれて戻っていた ([5.0, 5.5, 6.0, 4.0] → [0.0, 0.5, 1.0, -1.0]) 書き手は新しい method `bitdelta` で書く: 隣り合う f32 の bit パターンの差を wrapping な uint32 で持ち (先頭は先頭の bit パターン)、xz で圧縮する 整数の演算なので NaN の payload・非正規化数・±0・無限大を含めてどの値も bit 単位で戻る (ランダムな bit パターン 10,000 個と特殊値の往復の試験で確かめている) 旧 `delta` は `base_value` の記録があれば (Rust の旧 writer) 読み、無ければ `ValueError` (復元できないので元データから圧縮し直す) 差分だけが要る場合は `decompress_delta_differences` で取り出せる `decompress_residual` は記録 dtype によらず float32 で返す 記録 dtype は元データの型で、適用するのは `reconstruct` だけ (偶数への丸め、型の範囲で飽和、cast) 記録 dtype が実数の 11 種以外の header は読む時に `ValueError` 移行: `decompress_residual` の戻り値が元の dtype であることを前提にしていた呼び出し側は `reconstruct` を使う (residual を整数 dtype へ cast すると切り捨てになり、int16 で約半数の値の復元が 1 ずれる)
- ResidualData の JSON header の `"version"` は整数だけを版として受け付け、float (`2.0`) や文字列 (`"2"`) は拒否する (Python は `ValueError`、Rust は `ResidualError::InvalidHeader`) 書き手は整数で書く これまで Python は `2.0` を、Rust は `"2"` を版 2 として読み、両者の判定が分かれていた
- Python package の procedural payload の復元が、書き手が payload に記録した dtype で返る 4 つの生成器 (Perlin / Fourier / sine / polynomial) が結果を常に float32 に変換していたので、float64 の入力が float32 で返り、byte 数が header の `original_size` の半分になっていた 形式は変わらない (dtype は以前から payload の `params.dtype` にある) payload の dtype は書き手が記録しうる実数の 11 種 (`float16/32/64`、`int8〜64`、`uint8〜64`) だけを受け付け、それ以外 (complex、文字列型、未知の名前) は `ValueError` 復元した byte 数が header の `original_size` と違えば、procedural でも `ValueError`

### Added
- CI: 追跡している Python の未定義の名前と上書きを ruff 0.14.0 (F821 / F811) で検査する (preflight も同じ、検査した file が 0 件なら失敗) 改名した試験関数の古い名前を `__main__` の block が呼んでいたのが見つかった 試験 file の独自の `__main__` runner は除き、試験関数の無かった `tests/test_quant_simple.py` は pytest の試験にした
- pytest は数値の警告 (範囲外の cast・桁あふれ)、閉じていない資源、報告できない例外も失敗にする (pyproject の filterwarnings) 写真の試験はネットワークを使わない決定的な画像にした 以前は外部の site から取得して 403 で失敗し、追跡していた `test_images/mandrill.jpg` と `tests/test_images/mandrill.jpg` は画像でなく HTML だった (両方外した) 画像の試験は `tests/test_image_shaped_data.py` で、正弦波を並べた画像で procedural の経路も確かめる
- CI: default feature を外した build で全 target に clippy をかける (preflight も同じ引数) `compression_ratio` example は `required-features = ["lzma"]` を宣言した (default を外すと compile できなかった) pytest は試験関数が判定を `return` すると失敗にする (`PytestReturnNotNoneWarning` を error に、pyproject の設定) 試験の警告 2 種 (範囲外の float を int16 に cast した入力、signaling NaN を float64 に広げる変換) を直した
- CI: Python の試験を 3.9 と 3.12 で走らせる (宣言している最小の版を試験する) `scripts/version_check.py` が libalice の版が Cargo.toml だけにあることと、`alice_zip.__version__` が pyproject の版と同じことを 3 OS で確かめる (比べた件数が 0 なら失敗)
- README / README_ja に Python の lossless モードの保証 (bit 単位の復元、residual の version 4、lzma への退避、以前の結果の扱い) を書き、`scripts/claim_check.py` は Python の試験名も照合する Python package は PyPI に未公開なので、入れ方を `pip install git+https://github.com/ext-sakamoro/ALICE-Zip` に直した (`pip install alice-zip` と書いていた)
- Python package の `ResidualCompressor.compress_original(original, generated)`: 元データと generated から、`reconstruct` で bit 単位で元に戻る residual を書く residual は元データの精度で持つ (float64 / int32 / uint32 / int64 / uint64 は float64) 元か generated が有限でない位置 (NaN・無限大) と、規則で復元できない位置は例外として元の要素をそのまま持ち、residual と一緒に圧縮する 差分では無限大や NaN の bit を運べず、inf - inf の NaN の符号のように環境で結果が変わる経路もあったため 例外か float64 の residual を持つ file は version 4 (`residual_dtype` と `exceptions`)、以前の読み手は version で拒否する `exceptions` は version 3 と 4、`residual_dtype` は version 4 でだけ読み、それ以外の version に付いていれば拒否する header は平らな object とし、同じ鍵が 2 回ある header、object の値、配列の中の配列・object は拒否する (鍵の重複を以前は後の値で読んでいた) 既知の鍵は使わない場面でも型を見る 鍵 × JSON の値の種類 × version の組 (375 行) を仕様から生成した `tests/data/residual/header_gating.txt` で両方の読み手を照合する `decompress_residual` は residual の精度 (float32 / float64) で返す libalice の `residual::compress_original` と同じ file を書く header の数の欄は整数 (base_value / min_val / scale は数) だけを受け付け、bool (Python では int の部分型) や `8.0` の形は拒否する 既知の挙動: 以前の版の読み手は、version 2 のまま exceptions の鍵と末尾の block を持つ file (どの書き手も作らない) を例外を落として読む この版の読み手は拒否する

- `container`: 複数の payload をそれぞれの SHA-256 で識別して 1 つにまとめるコンテナ 56 byte の header (非 ASCII の byte で始まる 8 byte の magic、major / minor 版、semantics id)、section の表 (tag、critical flag、offset、長さ、SHA-256)、file 全体の SHA-256 を末尾に置く `Container::id` は header と表を domain を分けて hash する 読み手は別の major 版、未知の critical section、予約 bit、隙間や余分な byte、合わない末尾や payload の hash を拒否し、未知の critical でない section は保つ `SREF` section は section を SHA-256 で参照し (無い参照は拒否)、`LIDS` section は法則の識別子を並べ、header の semantics id と、`ContainerView::verify_law_ids` で計算し直した識別子に照合する `read_any` は ALICE_ZIP の版 1.0 / 1.1 も読み、書き手が出したことのない値は拒否する `parse_legacy_alice_zip_header` は header を `LegacyHeader` で返し、`LegacyHeader::verify_original` は元データとして渡された byte を `original_size` と、記録があれば (全 0 でなければ) `original_hash` に照合する `ContainerView` は payload の hash を読む時だけ確かめる
- `tests/container_oracle.rs` (25 本): byte と識別子が独立した参照の書き手 `tests/data/container/container_ref.py` と一致する fixture の 1 bit の変更はすべて、その位置を受け持つ検査で拒否される 退化した入力は定めた error を返す
- `examples/container_roundtrip.rs`
- CI: コンテナの oracle を各 OS の `no_std` build と `wasm32-wasip1` 上の `wasmtime` で走らせる wasm の job は通った試験が 0 件なら失敗にする
- `container::LegacyHeader::original_hash_checkable` — ALICE_ZIP の payload を復元すると元データが再現されるか (`original_hash` を照合できるか) LZMA fallback (`0x30`) だけが true Python の `alice_zip.core.original_hash_checkable` と同じ規則で、両方の試験が同じ表 (`tests/container_oracle.rs` の `CHECKABLE`) を読む

### Changed (破壊的変更)
- libalice (Python の wheel) の版は `libalice/Cargo.toml` の版を使う (`libalice/pyproject.toml` は `dynamic = ["version"]`) pyproject に 2.4.0 と書いたまま Cargo.toml が 2.7.0 に進んでいたので、wheel の版は 2.7.0 になる (新しい番号を決めたのでなく、Cargo.toml と同じ番号に直した) libalice の `requires-python` は CI で試験している最小の 3.9 にした (`>=3.8` と書いていた)
- Python package の `ProceduralCompressionDesigner.compress(quantize_residual=8 / 16)` は residual を `ResidualCompressor` の量子化 (偶数への丸め、xz) で ResidualData の file にする 以前は切り捨て (`astype(uint8)`) で 1 step 近くずれていた 誤差は半 step に、float32 で持つ residual と出力 dtype への丸めを足した範囲 `residual_format` を持たない以前の量子化の結果は以前と同じく読む
- Python package の `ResidualCompressor.compute_residual` は差が NaN になる時の値を規則で決める (元データの NaN に quiet bit を立てたもの、なければ generated の NaN、inf - inf は正の quiet NaN) x86 では inf - inf が負の NaN になり、同じ入力で書く residual が環境で変わっていた
- Python package の `ProceduralCompressionDesigner.compress(enable_lossless=True)` は residual を元データから書き (`ResidualCompressor.compress_original`)、`decompress` は入力を bit 単位で返す 以前は float32 の residual を足すだけで、float32 の sin 2000 点で 17 点、float64 の sin で 1999 点、float64 の多項式で 1536 点が一致していなかった (`is_lossless` は True のまま) 小さい residual を捨てて lossless とする経路も除いた `residual_data` は ResidualData の file になり、`metadata["residual_format"]` が `"alice-residual"` 移行: 以前の版で作った `CompressionResult` (residual_format が無く量子化もしていない residual) は `decompress` が `ValueError` を返す 入力から圧縮し直すか、近似値でよければ `decompress(result, allow_approximate=True)` で読む 結果 (generator の parameter と residual) が入力の lzma 以上の大きさなら lzma の結果を返す (adaptive_fallback が True の時、既定) residual が運べない dtype (complex など) は lzma の結果を返す (以前は ValueError) 大きさは sin 2000 点で float32 3,291 B (lzma 5,172 B)、float64 10,375 B (lzma 13,692 B)
- Python package の `ResidualCompressor.reconstruct` は整数 dtype の飽和を整数の領域で決める 64 bit の最大値の float は 2^63 / 2^64 に丸まるので、そこへ clip して cast すると範囲外になり、結果が環境で変わっていた (arm64 は飽和、x86 は最小値や 0) 規則: 整数 dtype は偶数への丸めのあと型の範囲で飽和 (無限大も)、浮動小数 dtype は 1 回だけ丸め、桁あふれは無限大 和が NaN になる時の NaN は規則で決め (generated の NaN、なければ residual の NaN、inf + -inf は正の quiet NaN)、浮動小数 dtype では符号・quiet bit・payload の上位 bit を残す (x86 では inf + -inf の NaN が負になるなど、ハードウェアの結果が環境で違うため) 移行: 整数 dtype で和が NaN になる入力は `ValueError` になる (以前は環境依存の値を返していた)

- ResidualData の header の鍵を Python の書き手の形 (`method` / `shape` / `dtype` / `quant_bits` / `version`) にそろえ、libalice と互いの file を読み合える Python の読み手は libalice の旧 header (`original_len` だけ) も読む (shape は `[original_len]`、dtype は float32) libalice の旧量子化コンテナ (header に `min_val`、payload は marker 0xFC の残差コンテナ) も読む libalice の量子化の書き手は Python と同じ形式・同じ符号 (偶数への丸め) を出す 量子化はどちらも入力の dtype によらず float64 で計算する (float32 では 32 bit の最上位符号 2^32-1 が 2^32 に丸まり、uint32 への範囲外 cast になっていた) NaN と無限大を含む値は量子化しない (Python は `ValueError`、libalice の書き手は lossless の方式を選ぶ) どちらの読み手がどの書き手の file を受け付けるかは、各書き手の実際の出力で表 (`tests/data/residual/acceptance.txt`) にしている
- Python package (`alice_zip.core`) の ALICE_ZIP 読み手は、書き手が出したことのない値を別の値として読まず拒否する (`ValueError`) Rust の `container::parse_legacy_alice_zip_header` と同じ規則
  - 版 1.0 / 1.1 以外 (major 2 を 1.1 として、minor 2 以上を 1.1 として読んでいた)
  - 未知の `payload_type` (`PROCEDURAL` として読んでいた)
  - engine の index が 4 以上 (`IndexError` だった)
  - 66 byte 未満の 1.1 header (1.0 として読んでいた)
  - header の `compressed_size` と header の後ろの byte 数が違う file (後ろが長い file を読んでいた)
- Python package の `residual_compression.ResidualData.from_bytes` は JSON header の `"version"` が 1〜4 以外なら `ValueError` (以前は 3 以上を version 2 として読んでいた) version 3 / 4 は例外と residual の精度を持つ形 (Added の `compress_original` を参照) libalice の Rust 版と同じ規則で、両方の試験が同じ fixture (`tests/data/residual/`) を読む
- `ALICEZip.decompress` は lossless な payload (LZMA fallback) を復元した後、長さを `original_size` と、記録があれば (全 0 でなければ) SHA-256 を `original_hash` と照合し、違えば `ValueError` procedural / media / texture は生成パラメータから近似で復元するので照合しない (`original_hash_checkable`)
- 移行: Python package と本 crate の書き手が出した file (版 1.0 / 1.1、定義された値) はこれまでどおり読める 上の値を持つ file はどの書き手も出していないので、読めなくなった file は壊れているか別の形式 書き手の出力は変わらない (同じ入力で同じ bytes、header は参照実装の配置と一致)

## [0.8.0] - 2026-10-09

### Added

- `compression::compress_residual_xor` / `decompress_residual_xor` — a residual
  container that stores `original.to_bits() ^ model.to_bits()` instead of
  `original - model`. Reversible by construction, so every finite sample comes
  back unchanged: signed zeros, denormals, and values far smaller than their
  model all survive. Measured on a 100,000-sample sine fitted by
  `analyze_signal`, the subtraction form loses the **99 samples nearest the zero
  crossings** (where `|original| << |model|`, `fl(original - model)` rounds the
  original away and `model + residual` does not return it); the xor form loses
  none. It is also smaller on that signal (5,306 B against 8,450 B) because the
  xor of two close values is mostly zero bytes — on the degree-3 polynomial it
  is larger (3,916 B against 2,783 B), so the subtraction container stays for
  callers who prefer size over exactness
- `compression::ResidualCodec` + `residual_codec_default` +
  `*_with` variants of all three container constructors — the payload codec is
  now chosen explicitly and recorded in the container, so a reader never guesses
- `compression::residual_container_codec` — reads the codec back out of any
  container this module produces
- `scripts/claim_check.py` + a CI job: every `<!-- claim-test: NAME -->` in
  `README.md` / `README_ja.md` must resolve to a real `fn NAME`, the two
  documents must carry the same set of markers, and **0 markers fails**. The
  convention existed since the law module landed but nothing checked it
- `examples/compression_ratio.rs` — prints the benchmark table the READMEs
  quote, so the numbers there are reproducible rather than asserted

### Changed

- **Residual containers default to deflate instead of LZMA.** `lzma-rs` is pure
  Rust but its encoder is weak: on the same 400,000 bytes it produced
  **316,528 B where this crate's own zlib wrapper produced 8,742 B** (36x worse),
  and on the sine residual 46,916 B against 8,430 B. The README's promise that
  the fallback is "never worse than standard tools" was therefore wrong by 84x.
  Deflate comes from `flate2`, already a dependency, so nothing new is pulled in
  and the pure-Rust / `no_std`-friendly story is unchanged
- **The `level` argument now does something.** It is the deflate level `0..=9`
  (values above 9 are clamped to 9, and **0 means store**, the same as
  `zlib_compress`). How much the level matters depends on the input and its
  length — measured on the sine residual
  (`generate_sine_wave(n, 50.0, 1.0, 0.0, 0.0)` fitted by
  `analyze_signal(.., 8, 0.999)`, one coefficient):

  | n | non-zero residual | max | level 6 | level 9 | ratio |
  |---|---|---|---|---|---|
  | 4,096 | 3,008 | 1.192e-7 | 1,139 B | 1,009 B | 1.13x |
  | 100,000 | 74,100 | 1.788e-7 | 16,687 B | 8,430 B | 1.98x |

  ⚠️ An earlier draft of this entry quoted the 1.98x figure without the input
  it belongs to, which reads as a property of the level rather than of that
  signal at that length.
  The LZMA path still has fixed settings and ignores it, as before
- Containers are available with the `std` feature instead of requiring `lzma`;
  `ResidualCodec::Lzma` (and reading a version 0 container) still needs `lzma`
  and returns a clear error without it
- `compress_residual_lossless`'s documentation now states that the container
  stores the residual array exactly but that a pipeline built on the subtraction
  form is not bit-exact, and points at `compress_residual_xor`
- CI and `scripts/preflight.sh` build the library on the host without `std`
  under `-D warnings`, once as is and once with `alice-det-math/std` enabled.

### Fixed

- **`level 0` no longer silently becomes `level 1` in the residual containers.**
  The deflate level was floored with `clamp(1, 9)`, so "store it, do not
  compress" was unreachable through the containers while `zlib_compress` in the
  same module has always honoured 0: measured on the same 400,000 bytes,
  `zlib_compress(.., 0)` produced 400,071 (a store) and the container produced
  31,607 — byte-identical to level 1. ⚠️ Two public functions in one module gave
  the same argument two different meanings. The floor had no reason behind it
  and is gone; `tests/residual_container_oracle.rs` now pins that level 0
  produces more bytes than the input and differs from level 1 (the assertion
  goes red if the floor comes back)
- Both READMEs quoted parameters-only sizes under the heading "Lossless:
  Bit-perfect reconstruction" — 1400x for a sine that was in fact reconstructed
  with a 6e-8 error. The benchmark tables now separate **bit-exact**
  (75x / 102x / 806x), **lossy** (160x–1286x with the error stated) and
  **parameters only** (11,111x–20,000x with the error stated), and carry a
  `zlib alone` baseline column so the comparison is visible. `README_ja.md` was
  corrected in the same commit
- A build without the `std` feature failed under `-D warnings` with
  `trait FloatExt is never used` whenever another crate in the dependency graph
  enabled `alice-det-math`'s default `std` feature (for example a downstream
  crate that also depends on `alice-det-math`). Linking std made the float
  method calls (`x.sqrt()`, `x.floor()`, `x.round()`) resolve to std's inherent
  methods, so the `libm` shim was never used. The shim is now four functions in
  `math` (`floor_f32`, `sqrt_f32`, `sqrt_f64`, `round_f64`) that the call sites
  name explicitly, so the `std` feature of this crate alone decides which
  implementation runs. All four operations are exactly specified by IEEE 754,
  so the results are bit-identical either way.

### Format

- Containers written by 0.8.0 carry a version marker and a codec byte
  (`0xFD` lossless, `0xFE` xor, `0xFC` quantised). **Containers written by
  earlier releases still decode**; containers written by 0.8.0 need 0.8.0 or
  later to read

## [0.7.0] - 2026-10-08

### Changed

- **Breaking:** the reconstruction laws accumulate in `f64` and round to `f32`
  once on return, instead of accumulating in `f32` and adding the DC term
  last. `generators::{generate_sine_wave, generate_multi_sine,
  generate_from_coefficients}` therefore return different bits in the last
  places. Analytic behaviour is unchanged and `law_id` is unaffected (the
  polynomial path already evaluated in `f64`).

  The reason is that each law was implemented twice: here over whole arrays,
  and again in a consumer that answered single-sample queries without
  materialising a segment. On the same input at integer positions with
  `n = 1024` the two disagreed on **788 / 1024** samples for a sine,
  **881 / 1024** for a multi-sine and **962 / 1024** for a Fourier
  reconstruction. Against the `f64` closed form the `f32` accumulation was off
  by at most 9.16e-7 and the `f64` one by 5.95e-8, so the more accurate form
  became the law. `tests/determinism_golden.rs` re-records the two sinusoid
  digests for this reason.

### Added

- `generators::{sine_at, multi_sine_at, fourier_at, polynomial_at}` evaluate a
  law at one position, which may be fractional. These are **the** laws, and the
  array generators are now a `map` over them, so an array read and a point read
  cannot drift apart. A consumer that answers point queries calls these instead
  of keeping its own copy of the law.
- `tests/law_single_source.rs` pins both halves of that: the error of each
  array generator against the `f64` closed form stays inside one `f32` step
  (1.19e-7), and the array output equals the point output bit for bit over
  seven lengths including `n = 1`, `2`, `3` and `257`, with a byte-count gate so
  a comparison of nothing cannot pass. A separate case pins that the point
  evaluator does not round its position, which an integer-only comparison would
  miss.


## [0.6.0] - 2026-10-08

### Changed

- **Breaking:** the float transcendentals now come from `alice-det-math`
  instead of the platform `libm`, so `generators::{analyze_signal,
  generate_from_coefficients, generate_sine_wave, generate_multi_sine}` and
  `entropy::{shannon_entropy, theoretical_min_size}` return different bits in
  the last places. Analytic behaviour is unchanged (the analytic oracles pass
  untouched) and `law::SignalLaw::law_id` is unaffected, since evaluating a
  polynomial law uses only IEEE 754 basic operations.

  The reason is that `law_id` promises that two laws share an identifier only
  if evaluating them returns the same bits for every `x`, and IEEE 754 does not
  require the transcendentals to be correctly rounded: the platform
  implementation is free to differ between operating system, CPU and compiler.
  Measured on one machine by changing nothing but the `std` feature,
  `generate_multi_sine` differed in 1 sample of 64 and `analyze_signal` in 6
  fields of 9. After the move, all five recorded scenarios agree.

- `math::FloatExt`, the `no_std` shim, now carries only `sqrt`, `floor` and
  `round`. IEEE 754 specifies each of them exactly, so the shim and the
  platform agree bit for bit; the module documentation no longer claims that
  the transcendentals stay "within 1-2 ulp", which is not a statement about
  determinism.

- `.gitignore` を Rust / Python のビルド生成物に絞り、CI と test の comment を
  外部参照でなく理由そのものを書く形に直した (`.gitignore` は公開 package に
  同梱されるので、開発環境固有の除外は clone ごとの `.git/info/exclude` に置く)

### Added

- `law::SEMANTICS_ID`, re-exported from `alice-det-math`: the value to pass as
  `semantics_id` to `law::SignalLaw::law_id` for a law evaluated through this
  crate. Previously a caller had nothing to pass but an invented array, and an
  identifier whose arithmetic half is invented does not identify arithmetic.
- `tests/determinism_golden.rs` — records the bit patterns of the law
  identifier, the evaluation it stands for, the sinusoid generators, the
  spectrum, the reconstruction, the entropy and one path with no transcendental
  on it. Nothing in the crate pinned a single bit before: the 193 existing
  tests all stayed green while the outputs above changed, because they compare
  against closed forms with tolerances. Each scenario checks a minimum byte
  count first, so a scenario that stopped producing values cannot pass by
  comparing nothing against nothing. CI runs the file on three operating
  systems in the default build and again in the `no_std` build, and every run
  has to produce the same digests.
- `clippy.toml` — `disallowed-methods` for the 50 float transcendental methods
  of `f32` and `f64`, so a platform implementation cannot come back unnoticed.
  `sqrt`, `floor`, `round`, `trunc`, `ceil` and `mul_add` are deliberately
  absent (IEEE 754 specifies all of them exactly). The entropy tests keep the
  platform `log2` behind a scoped allow, because an oracle whose expected value
  comes from the implementation under test proves nothing.

  The two gates catch different things, measured by reverting each change one
  at a time: of 10 such reversions, the golden caught 8 and the lint 7, with 2
  caught only by the lint (where the platform and `alice-det-math` happen to
  agree at the sampled points) and 3 only by the golden (changes to the
  identifier's field order and to the evaluation itself, which no lint can
  see). None survived both.

## [0.5.2] - 2026-10-07

### Added

- `law::SignalLaw::law_id` — a 32-byte content identifier for a law, so a
  stored result can name the law it came from. SHA-256 over the valid range,
  the coefficients and a caller-supplied identifier for the numeric semantics;
  the evidence, residual, provenance and oracle cases are excluded, so the same
  law fitted from two measurement runs gets one identifier. Equal identifier
  implies bit-identical `evaluate`; the converse is not guaranteed and the
  identifier is not a deduplication key. The byte layout is documented on the
  method and pinned by a golden digest.
- `law::LAW_ID_DOMAIN` and `law::SIGNAL_LAW_KIND` — the two tags the encoding
  mixes in, published so an independent implementation can reproduce an
  identifier byte for byte.
- `tests/analytic_law_id.rs` — injectivity over each evaluation input, metadata
  independence, the observability of a zero's sign, rejection of the inputs the
  encoding cannot canonicalise, and a cross-platform golden.

### Changed

- New dependency `sha2` (`default-features = false`, so the `no_std` build is
  unaffected) for the digest above.

## [0.5.1] - 2026-10-07

### Added
- `law` module (`no_std`): `SignalLaw` keeps a fixed-degree polynomial fitted by
  least squares together with its evidence, measured `ResidualStats`, the closed
  `ValidRange` of the evidence, `Provenance` and `OracleCase`s `evaluate` refuses
  `x` outside the range (`LawError::OutOfRange`) instead of extrapolating,
  `check_oracles` reports pass / fail / out of range, and `ingest` judges new
  evidence as `Supports` / `ParameterUpdate` (with the refitted law) /
  `ResidualGrew` / `Breaks` / `OutOfRange` / `NoEvidence` by documented rules
  The fit reuses the crate's Householder QR solver on `x` normalised to `[0, 1]`
- `tests/analytic_law.rs`: closed-form recovery at unsampled conditions, residual
  measured against the evidence, range refusal, oracle outcomes, each verdict,
  and degenerate input (empty / too few points / single `x` / NaN / repeated `x`)
- `law::SignalLawParts` with `SignalLaw::to_parts` / `from_parts` / `coefficients`:
  every field public for storing a law in another format; `from_parts` validates
  the parts (coefficients, finite values, domain, evidence inside the domain) and
  measures the residual again from the stored evidence instead of trusting it

### Changed
- `scripts/preflight.sh`: `cargo audit` keeps its advisory database under the build
  directory

## [0.5.0] - 2026-09-16

### Added
- `quantize` module (`no_std`): `quantize_8bit` / `quantize_16bit` and their inverses —
  the min / max quantisation law of the `.alice` residual payloads and `alice-edge`
  coefficient batches, moved here from `libalice` (single home, like `generators`)
- `lzma` feature (implies `std`, `lzma-rs`): `compression::{lzma_compress,
  lzma_decompress}` and the `.alice` residual containers
  `compress_residual_quantized` / `decompress_residual_quantized` /
  `compress_residual_lossless` / `decompress_residual_lossless` (byte-identical layout
  to `libalice` ≤ 2.3, documented on the module); `decompress_*` reject length overflow
- `tests/analytic_oracle.rs`: quantisation error ≤ half a step with exact endpoints
  (8 / 16 bit, n = 1..1000), residual container layout + round trip
- `tests/analytic_oracle.rs`: 6 laws that mutation testing (cargo-mutants, 8 shards,
  93% score) found unmeasured — sine DC term, energy-threshold edge values / inclusive
  cutoff, FFT empty-input guards, fit error = normalised MSE of the returned fit, 1D
  value-noise lattice / midpoint / octave-composition laws, and the alice-db persisted
  `generate_fbm_1d` values (bit-identical to 0.3.1)

- CI `libalice-python` job: `cargo check` / clippy of the `python` feature, `maturin
  develop`, and the Python test suite twice (native accelerator enabled / pure-Python
  fallback)
- `libalice/README.md` (the maturin / cargo `readme` used to point outside the crate,
  which current maturin rejects)

### Changed
- CI `quality-deep.yml`: mutants run as 8 shards over `--lib --test analytic_oracle`,
  `ulimit -v 6 GiB` so an infinite-loop mutant aborts instead of OOM-killing the runner
- docs.rs / CI feature set is now `std,fft,parallel,lzma`
- `libalice/pyproject.toml`: distribution name `alice-zip` → `libalice` (the import
  name; the root `pyproject.toml` is the `alice-zip` Python package), SPDX license
  string, author e-mail placeholder replaced in both pyproject files

### Fixed
- `alice_zip.native_accelerator`: `fourier_generate` / `multi_sine` /
  `polynomial_generate` raised `TypeError` on the native path when coefficients came
  as lists (as decoded from a `.alice` container) because PyO3 extracts tuples only;
  they are now normalised before the call (pure-Python fallback was unaffected)
- `alice_zip.native_accelerator.is_available()` reported `True` without the extension:
  the repository's `libalice/` directory imports as an empty namespace package and a
  legacy fallback imported the pure-Python `alice_zip` package itself as "native";
  availability now requires the extension entry points to exist

## [0.4.0] - 2026-09-15

### Added
- Real `no_std + alloc` support: `#![cfg_attr(not(feature = "std"), no_std)]`,
  float math through `libm` under `no_std`, CI builds the rlib for
  `thumbv7em-none-eabihf`. Until 0.3 the `no_std` claim was documentation only
  (`flate2` was unconditional and no attribute existed)
- Cargo features: `std` (default; zlib wrappers + `std::error::Error`),
  `fft` (`generators::analyze_signal_fft`, rustfft, implies `std`),
  `parallel` (rayon rows for the 2D Perlin textures, implies `std`)
- `generators::PerlinNoise` / `generate_perlin_2d` / `generate_perlin_advanced`
  (2D gradient Perlin, seeded permutation table, moved here from `libalice` so
  every generator law has exactly one home)
- `generators::fit_polynomial_unit` / `generate_polynomial_unit` — the `.alice`
  container convention (`x ∈ [0, 1]`, descending coefficients) beside the
  `alice-db` convention (`x = 0..n-1`, ascending)
- `generators::generate_fbm_1d` — the 1D value-noise law formerly reachable as
  `generate_perlin_advanced(n, _dimension, …)`
- `lz77::MAX_WINDOW` (65535); window / lookahead above it are clamped instead of
  truncated to `u16`
- `ZipError::InvalidParameter`; `ZipError` is `Copy`, `Hash`,
  `#[non_exhaustive]` and implements `std::error::Error` under `std`
- `tests/analytic_oracle.rs` — 22 closed-form / reference-implementation tests
  (entropy of `k` equiprobable symbols = `log2 k`, LZ77 round trip swept over
  window / lookahead, exact polynomial recovery, single-bin Fourier
  reconstruction including Nyquist, naive-DFT vs FFT parity, Perlin lattice
  zeros + `libalice` 2.2 reference, zlib level sweep)
- CI: 9-job `ci.yml` (3-OS tests, clippy `-D warnings`, bare-metal `no_std`,
  MSRV 1.87 real compile, feature powerset, rustdoc `-D warnings`, `libalice`),
  `security-audit.yml` (audit × 3 lockfiles, cargo-deny, machete, coverage,
  semver-checks, stub / FFI-guard grep), `fuzz.yml` (7 targets incl. Fourier
  parity fuzz), `quality-deep.yml` (cargo-mutants), `scripts/preflight.sh`

### Changed
- **Breaking**: `lz77_decode` returns `Result<Vec<u8>, ZipError>`
  (`InvalidData` for tokens that reference bytes before the output start)
- **Breaking**: `Dictionary::add` returns `Result<u32, ZipError>`
  (`DictionaryFull` for capacity `0` or `u32` arena overflow)
- **Breaking**: `generate_perlin_advanced(n, _dimension, seed, …)` (1D, unused
  `_dimension`) is now `generate_fbm_1d(n, seed, …) -> Result`; the name
  `generate_perlin_advanced(width, height, …) -> Result` is the 2D texture law
- `fit_polynomial` solves the least squares by Householder QR on `x/(n-1)`
  (normal equations lost 4-5 digits at degree 5); returned coefficients are
  unchanged in meaning
- `analyze_signal` removes the DC offset before the transform, drops numerical
  zeros (`< 1e-10`), and normalises the energy threshold by the **total** non-DC
  energy as its documentation always said (it used the truncated top-k energy)
- Naive DFT accumulates in `f64` (the naive path is the precision reference for
  the FFT path)
- `[package] resolver = "3"`; `rust-toolchain.toml` 1.98.1 + `thumbv7em` target;
  crates.io `exclude` covers the Python / bindings / fuzz directories

### Fixed
- `shannon_entropy`: a 20-term `log2` series saturated at ≈ 6.74 bit for a
  uniform byte distribution (correct: 8.0); `theoretical_min_size` inherited
  the error. `log2` is now exact (`f64::log2` / `libm::log2`)
- `lz77_encode` emitted a phantom trailing `0` literal when the last match
  reached the end of the input, so decode output was one byte longer than the
  input (unit tests sliced the extra byte away)
- `generate_from_coefficients` weighted the Nyquist bin (`k = n/2`) by 2 like a
  mirrored bin; it is self-conjugate and is now weighted by 1
- `lz77_decode` / `Dictionary::add` panicked on malformed input

## [0.3.1] - 2026-09-15
### Changed
- `rust-version = "1.87"` declared (const `Vec::len` in `Dictionary`), `homepage`
  / `documentation` metadata, rustfmt

## [0.3.0] - 2026-09-13
### Changed
- **Breaking**: `generate_perlin_advanced` takes `u64` seed / `u32` octaves
  (`alice-db` type compatibility)

## [0.2.1] - 2026-09-13
### Added
- `generators::generate_multi_sine`, `generators::generate_perlin_advanced`

## [0.2.0] - 2026-09-13
### Added
- `compression` (zlib wrappers over `flate2`) and `generators` (polynomial /
  Fourier / sinusoid) modules for `alice-db`
### Changed
- License: ALICE-Zip Open Core License → `MIT OR Apache-2.0`

## [0.1.0] - 2026-07-04
### Added
- Initial crates.io release: `lz77`, `dictionary`, `entropy`, `bpe`, `error`,
  `prelude` (split from a single `lib.rs`, 102 tests)

[Unreleased]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.3.0...v0.3.1
[0.3.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.1...alice-zip-v0.3.0
[0.2.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.0...alice-zip-v0.2.1
[0.2.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.1.0...alice-zip-v0.2.0
[0.1.0]: https://github.com/ext-sakamoro/ALICE-Zip/releases/tag/alice-zip-v0.1.0
