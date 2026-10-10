<p align="center">
  <img src="assets/logo-on-light.png" alt="ALICE-Zip" width="400">
</p>

<h1 align="center">ALICE-Zip</h1>

<p align="center">
  <a href="https://crates.io/crates/alice-zip"><img src="https://img.shields.io/crates/v/alice-zip.svg" alt="crates.io"></a>
  <a href="https://docs.rs/alice-zip"><img src="https://docs.rs/alice-zip/badge.svg" alt="docs.rs"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/ci.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/security-audit.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/security-audit.yml/badge.svg" alt="Security"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/fuzz.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/fuzz.yml/badge.svg" alt="Fuzz"></a>
  <a href="#ライセンス"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.9+-yellow.svg" alt="Python"></a>
</p>

> **手続き的生成圧縮エンジン**
> *データではなく、アルゴリズムを保存する*

[English README](README.md)

---

ALICE-Zip は、データそのものではなく**「データの生成方法」**を保存する次世代圧縮ツール

パターン、波形、数学的データに対して**元のサンプルをビット単位で復元したまま 75 倍〜806 倍**、
1e-12 の誤差を許すなら 160 倍〜1286 倍 法則が当たらないデータは byte 圧縮に落ち、
素の zlib と同じサイズに着地します (悪化しません) 本ページの数値はすべて
`cargo run --release --example compression_ratio --features lzma` の出力

<!-- claim-test: the_lossless_container_is_not_worse_than_this_crates_own_zlib_wrapper -->

## 特徴

- **手続き的圧縮:** サイン波、多項式、数学的パターンを認識
- **適応型フォールバック:** 法則が当たらない時は byte をそのまま圧縮する
  素の zlib と比較する試験があるので「標準ツールより悪くならない」は約束でなく検査
- **構成上のビット一致:** 残差をモデルとの **bit の xor** で保存するため、符号付きゼロ・
  非正規化数・モデルよりはるかに小さい値を含めて全ての有限サンプルがそのまま戻る
  旧来の減算残差 (`original - model` を `f32` で計算) は**ビット一致しません** —
  下表のサイン波では零交差近傍の 99 サンプルが失われます
  <!-- claim-test: the_xor_container_is_bit_exact_on_every_law_family_the_readme_quotes -->
- **クロスプラットフォーム:** Python, Rust, C#/Unity, C++/UE5

## リポジトリ構成

| パス | 内容 | 配布先 |
|------|------|--------|
| `/` (`alice-zip`) | **Rust core crate** — 圧縮 primitive (LZ77 / 辞書 / BPE / エントロピー / 量子化 / zlib・LZMA residual container) + 全 generator 法則 (多項式 / Fourier / Perlin)、`no_std + alloc` | [crates.io](https://crates.io/crates/alice-zip) · [docs.rs](https://docs.rs/alice-zip) |
| `libalice/` (`alice-zip-cli`) | CLI `alice`、C FFI (`cdylib`)、PyO3 native module core generators / compression の thin re-export | C# / UE5 binding、`libalice/` から maturin で build する Python module |
| `alice_zip/` | Python package (`ALICEZip` analyzer + `.alice` container) | 本 repository から (PyPI には未公開) |
| `bindings/` | C++ / C# (Unity) / UE5 wrapper (`libalice/include/alice.h`) | |

## インストール

```bash
# Python (PyPI には未公開: 本 repository から入れる)
pip install git+https://github.com/ext-sakamoro/ALICE-Zip

# Rust
cargo add alice-zip                       # std (default): zlib wrapper 付き
cargo add alice-zip --features fft,parallel,lzma   # rustfft 解析、rayon texture、LZMA residual container
cargo add alice-zip --no-default-features     # no_std + alloc (sqrt / floor / round は libm)
```

### Rust crate

```rust
use alice_zip::prelude::*;

// 圧縮 primitive (no_std)
let tokens = lz77_encode(b"abcabcabcabc", 256, 32);
assert_eq!(lz77_decode(&tokens)?, b"abcabcabcabc");
assert_eq!(shannon_entropy(&(0..=255u8).collect::<Vec<_>>()), 8.0);

// generator 法則: 系列を fit → 係数だけ保持 → 再生成
let series: Vec<f32> = (0..64).map(|i| 3.0 + 2.0 * i as f32).collect();
let (coeffs, degree, err) = fit_polynomial(&series, 4, 1e-6).unwrap();
assert_eq!((degree, generate_polynomial(64, &coeffs) == series), (1, true));

let (bins, dc) = analyze_signal(&generate_sine_wave(32, 1.0, 2.0, 0.0, 0.5), 4, 0.99);
let regenerated = generate_from_coefficients(32, &bins, dc);
# Ok::<(), alice_zip::ZipError>(())
```

| feature | 追加されるもの | default |
|---------|----------------|---------|
| `std` | `compression` (flate2 zlib)、`ZipError` の `std::error::Error` | ✓ |
| `fft` | `generators::analyze_signal_fft` (rustfft、naive DFT と同一契約) | |
| `parallel` | `generate_perlin_2d` / `_advanced` の rayon 行並列 | |
| `lzma` | `compression::{lzma_compress, lzma_decompress}` + `.alice` の量子化 / lossless residual container (lzma-rs) | |
| *(なし)* | `no_std + alloc`、`sqrt` / `floor` / `round` は `libm` (超越関数は全 build で `alice-det-math` 経由)、CI が `thumbv7em-none-eabihf` で rlib build | |

永続化されている係数 convention は 2 つあり、別名で共存する (別の法則なので暗黙に
混用できない): `fit_polynomial` / `generate_polynomial` (`x = 0..n-1`、昇順 —
`alice-db` segment) と `fit_polynomial_unit` / `generate_polynomial_unit`
(`x ∈ [0, 1]`、降順 — `.alice` container) 全法則は閉形式解との突合
([`tests/analytic_oracle.rs`](tests/analytic_oracle.rs)) と fuzz (7 target、
naive DFT ↔ FFT parity 含む) で検証 MSRV 1.87

## クイックスタート

### コマンドライン

```bash
# 圧縮
alice-zip compress data.bin -o data.alice

# 解凍
alice-zip decompress data.alice -o restored.bin

# ファイル情報を表示
alice-zip info data.alice
```

### Python API

```python
from alice_zip import ALICEZip
import numpy as np

zipper = ALICEZip()

# サイン波データを圧縮
data = np.sin(np.linspace(0, 100*np.pi, 100000)).astype(np.float32)
compressed = zipper.compress(data)

print(f"元サイズ: {data.nbytes:,} bytes")
print(f"圧縮後: {len(compressed):,} bytes")
print(f"圧縮率: {data.nbytes / len(compressed):.1f}x")

# 解凍
restored = zipper.decompress(compressed)
```

### lossless モード (Python)

```python
from alice_zip import ProceduralCompressionDesigner

designer = ProceduralCompressionDesigner()
result = designer.compress(data, enable_lossless=True)
restored = designer.decompress(result)   # data と同じ dtype、同じ byte
```

- `decompress` は実数の 11 種の dtype (`float16/32/64`、`int8`〜`int64`、`uint8`〜`uint64`) で、NaN の payload や無限大を含めて入力を dtype ごと bit 単位で返す <!-- claim-test: test_lossless_gives_back_the_input_bit_for_bit -->
- residual は元データから書き、元データの精度で持つ (float64 / int32 / uint32 / int64 / uint64 の元データは `float64`、それ以外は `float32`) 生成値と residual で正確に復元できない値 (NaN、無限大、差が正確でない値) はそのまま持つ residual は `ResidualData` の file で version 4 (residual が `float32` でそのまま持つ値が無ければ version 2)
- parameter と residual の合計が入力の LZMA より小さくなければ LZMA の結果を返す residual が運べない dtype (complex など) も同じ <!-- claim-test: test_a_lossless_result_is_not_larger_than_lzma_of_the_input -->
- `quantize_residual=8` / `16` は lossy (`is_lossless` は `False`) で、各値の誤差は量子化の半 step に、`float32` で持つ residual の丸めと出力 dtype への丸め (それぞれ最後の桁の半分) を足した範囲に収まる residual の幅が小さい `float32` のデータでは丸めの項が半 step よりずっと大きくなり、半 step は上界にほとんど効かない <!-- claim-test: test_a_quantized_residual_is_off_by_at_most_half_a_step -->
- 以前の版で作った結果 (`float32` の residual で lossless でなかったもの) は `decompress` が拒否する 入力から圧縮し直すか、近似値でよければ `allow_approximate=True` で読む <!-- claim-test: test_a_result_of_the_earlier_residual_path_is_refused -->

## 仕組み

従来の圧縮は**バイト列**のパターンを探す ALICE は**数学的**パターンを探す

```
元データ = 生成関数(パラメータ) + 残差

ここで:
  - 生成関数() = 数学関数（多項式、サイン波など）
  - パラメータ = 小さな記述（〜100バイト）
  - 残差 = 圧縮された差分（多くの場合ほぼゼロ）
```

xor であることが厳密な復元の根拠です 差 (`original - model`) を `f32` で計算すると、
元の値がモデルよりはるかに小さい箇所で元の値が丸めで消えるので `model + residual` は
元に戻りませんが、`bits ^ bits` にはその経路がありません

### 例

```
入力:  サイン波、100,000サンプル (f32 で 400,000 バイト)
        ↓
分析: Fourier 係数 1 個 — 「周波数 bin 50、振幅 1.0、位相 0」
        ↓
出力: パラメータ 20 バイト + 圧縮した xor 残差 5,306 バイト
        ↓
結果: 400,000 → 5,326 バイト = 75 倍、サンプルはビット単位で一致
        (同じ byte 列に素の zlib: 8,742 バイト = 45.8 倍)
```

パラメータ 20 バイトだけを残せば 20,000 倍ですが、それは最大絶対誤差 1.8e-7 の
**近似**であって復元ではありません 下のベンチマーク表では両方を別の列に置いています
前者を「ロスレス」と並べて書くと結果を 2 桁以上大きく見せることになるためです

### 証拠つきの法則 (`law` module)

`law::SignalLaw` は当てはめた多項式を、元になった証拠 (点列)、測った残差、証拠が覆う `x` の範囲、出典、再現すべき参照値と一緒に持つ 新しい条件で再計算できるが範囲外へは外挿しない 新しい証拠は追記せず判定する

```rust
use alice_zip::law::{IngestPolicy, Provenance, SignalLaw, Verdict};

let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 1.0 + 2.0 * i as f64)).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))?;
assert!((law.evaluate(2.5)? - 6.0).abs() < 1e-12); // a condition that was not measured
assert!(law.evaluate(9.0).is_err());               // outside the measured range

let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
assert!(matches!(law.ingest(&[(0.5, 2.0), (3.5, 8.0)], &policy), Verdict::Supports { .. }));
# Ok::<(), alice_zip::law::LawError>(())
```

| 判定 | 条件 |
|------|------|
| `Supports` | 新しい点が許容幅の中で法則と一致する |
| `ParameterUpdate` | 同じ形で旧と新の点を合わせて当てはまり、パラメータだけが変わる (当てはめ直した法則を返す) |
| `ResidualGrew` | ずれが許容幅を超えるが、その `break_factor` 倍以内 |
| `Breaks` | 新しい点はこの形では説明できない |
| `OutOfRange` | 当てはめた範囲の外の点がある 何も判定しない |

規則と順序は module の doc に書き、[`tests/analytic_law.rs`](tests/analytic_law.rs) で固定している

#### 法則を内容で識別する

`SignalLaw::law_id` は法則の 32 byte の識別子を返す 保存した結果が「どの法則から出たか」を名指しできる

```rust
use alice_zip::law::{Provenance, SignalLaw};

let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 1.0 + 2.0 * i as f64)).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))?;

// 法則の評価に使う算術を識別する 32 byte 本 crate で評価する法則なら
// `law::SEMANTICS_ID` (`alice-det-math` からの再公開) 別の算術を使う場合だけ別の値を渡す
let id = law.law_id(&alice_zip::law::SEMANTICS_ID);
# Ok::<(), alice_zip::law::LawError>(())
```

digest に入るのは `evaluate` が読むもの (有効範囲と係数) と `semantics_id` だけ 証拠・残差・出典・oracle case は意図的に外してある それらは法則をどう得てどう正当化したかの記録で、法則が何を計算するかではないので、同じ法則を別の測定から当てはめても識別子は 1 つになる

**保証する**: 識別子が同じなら、IEEE 754 の基本演算が成り立つどの target でも `evaluate` は全ての `x` で同じ bit を返す
<!-- claim-test: equal_law_id_implies_bit_identical_evaluation -->

この保証があるため、float の超越関数は platform ではなく `alice-det-math` から取る IEEE 754 は `sin` / `cos` / `atan2` / `log2` に正確丸めを要求しないので、platform 版は OS / CPU / compiler で違いうる `std` feature だけを変えた同一機での実測では、0.5 系は `generate_multi_sine` が 64 sample のうち 1、`analyze_signal` が 9 field のうち 6 で別の bit を返した `tests/determinism_golden.rs` が bit 配列を記録し、CI が 3 OS と `no_std` build で走らせる `clippy.toml` が platform の method を拒否するので逆流しない `sqrt` / `floor` / `round` は IEEE 754 が結果を 1 通りに定めるので platform のままにしている
<!-- claim-test: sinusoid_generators_are_the_recorded_bits -->

**保証する**: 復元の法則は 1 つの実装しか持たない `sine_at` / `multi_sine_at` / `fourier_at` / `polynomial_at` が 1 点 (非整数でよい) での値を返し、配列生成器はその `map` なので、範囲読み出しと 1 sample の読み出しが食い違うことがない 単点問い合わせに答える側はこれを呼び、法則の写しを持たない 0.7.0 より前は crate の外に写しがあり、sine で 1024 sample のうち 788 が不一致だった 法則は `f64` で積んで 1 度だけ丸める形にした (`f64` の閉形式に対する最大誤差は 5.95e-8、`f32` で積む形は 9.16e-7)
<!-- claim-test: array_generators_are_a_map_over_the_point_law -->

**保証しない**: その逆 評価が同じでも識別子は分かれうる 末尾に 0 の係数を足した場合が最も単純な例 法則を正規形に落とすのは別の問題なので、識別子は重複排除の鍵には使えない
<!-- claim-test: evaluation_equivalent_laws_may_still_differ_in_id -->

byte 配置は `law_id` の doc に書き、CI が対応 target すべてで再現する golden digest で固定している 独立の実装が同じ識別子を計算できる
<!-- claim-test: law_id_golden -->

浮動小数は生の bit で入り、`-0.0` は `+0.0` に畳まない 1 次の係数が負なら、定数項の零の符号が範囲の下端の結果に現れるので、畳むと評価の異なる法則に同じ識別子を与えることになる
<!-- claim-test: negative_zero_is_observable_in_evaluation_so_it_changes_the_id -->

### 内容ハッシュつきコンテナ (`container` module)

`container::Container` は複数の payload を 1 つの file にまとめ、それぞれを
SHA-256 で識別する section は既存の形式の bytes をそのまま運ぶので、中身の
形式は自分の版を持ち続ける コンテナが決めるのは header / section 表 / file の
整合性だけ

- **識別子**: `Container::id` は header と section 表 (各 payload の SHA-256 を
  含む) の SHA-256 domain 分離と長さ前置は `law::SignalLaw::law_id` と同じ符号化
- **整合性**: 末尾 32 byte はそれより前の全 byte の SHA-256 で、読み手は必ず
  検証する 各 payload の SHA-256 は読む時に検証するので、`ContainerView` は
  他の section を hash し直さずに 1 つだけ返せる
- **版**: major が 1 以外は拒否、minor は何でも読む 未知の tag の section は
  critical なら拒否、そうでなければそのまま保持するので、読んで書き戻すと同じ
  bytes になる
- **参照**: `SREF` section は他の section を SHA-256 で指すので、何度使う
  payload も 1 回だけ置かれ、無い section への参照は拒否される `LIDS`
  section は法則の識別子を並べ、header の semantics id と、
  `ContainerView::verify_law_ids` を通して payload から再計算した識別子と照合する
- **以前の file**: `container::read_any` は版 1.0 / 1.1 の `ALICE_ZIP` file も
  読み (header と payload の 2 section)、書かれたことのない値は拒否する

<!-- claim-test: every_single_bit_flip_is_refused_by_the_expected_check -->
コンテナのどの 1 bit を変えても拒否される (4 section の fixture の全 bit で確認)
bytes と識別子は独立の参照実装と Linux / macOS / Windows / `wasm32-wasip1` で
一致する (`tests/container_oracle.rs`、`tests/data/container/container_ref.py`)

```rust
use alice_zip::container::{read_any, Container, Tag};

let mut c = Container::new([0; 32]);
c.push(Tag::RAW, true, b"payload".to_vec());
let bytes = c.to_bytes();
assert_eq!(read_any(&bytes)?, c);
# Ok::<(), alice_zip::container::ContainerError>(())
```

## ベンチマーク

100,000 個の `f32` サンプル (400,000 バイト)、deflate level 9
再現は `cargo run --release --example compression_ratio --features lzma`
`zlib 単体` 列は同じ byte 列を本 crate の zlib wrapper に同じ level で通したもので、
フォールバックが並ばなければならない床です

**ビット一致** (法則のパラメータ + xor 残差、サンプルがそのまま戻る):

| データタイプ | 圧縮後 | 圧縮率 | ビット一致 | zlib 単体 |
|-------------|--------|-------|-----------|----------|
| サイン波 | 5,326 B | **75倍** | はい | 8,742 B (45.8倍) |
| 多項式 (3次) | 3,916 B | **102倍** | はい | 330,123 B (1.2倍) |
| 線形グラデーション | 496 B | **806倍** | はい | 293,616 B (1.4倍) |
| ランダムデータ | 364,645 B | 1倍 | はい | 364,641 B (1.1倍) |

<!-- claim-test: the_xor_container_is_bit_exact_on_every_law_family_the_readme_quotes -->

**非可逆** (誤差の上限を受け入れられる場合):

| データタイプ | パラメータのみ | 圧縮率 | 最大絶対誤差 | 量子化残差 (16bit) | 圧縮率 | 最大絶対誤差 |
|-------------|--------------|-------|------------|------------------|-------|------------|
| サイン波 | 20 B | 20,000倍 | 1.8e-7 | 2,075 B | 193倍 | 1.4e-12 |
| 多項式 (3次) | 36 B | 11,111倍 | 6.0e-8 | 2,500 B | 160倍 | 9.1e-13 |
| 線形グラデーション | 20 B | 20,000倍 | 7.5e-9 | 311 B | 1,286倍 | 5.0e-14 |
| ランダムデータ | 4 B | 100,000倍 | 1.0e0 | 200,062 B | 2倍 | 1.5e-5 |

表の読み方:

- **ランダムデータはフォールバックが働いている状態** 法則が当たらないので信号全体が
  byte 圧縮を通り、結果は素の zlib と 4 バイト差で並ぶ フォールバックの目的は
  勝つことではなく負けないこと
- **パラメータのみの列は復元ではない** 20,000 倍は法則のサイズで、誤差列がその代償
- **xor 容器が買うのは厳密性でサイズではない** 旧来の減算残差も利用可能
  (`compress_residual_lossless`) で、多項式では小さく (2,783 B vs 3,916 B)、短い信号では
  サイン波でも逆転する (4,096 サンプルでは減算側が小さい) が、ビット一致しない
  サンプルをそのまま戻す必要があるなら `compress_residual_xor`、誤差の上限を許せて
  サイズが要るなら減算側を選ぶ

## 最適なユースケース

- **科学データ:** シミュレーション出力、センサー読み取り、波形
- **ゲームアセット:** 手続き的テクスチャ、地形ハイトマップ
- **IoT/エッジ:** 帯域制限のあるデバイスのセンサーログ
- **時系列:** テレメトリ、パターンのある監視ログ

## 使うべきでない場面

| データタイプ | 理由 |
|-------------|------|
| JPEG/PNG/MP3 | 既に圧縮済み |
| ランダム/暗号化データ | パターンがない |
| 小さなファイル (<1KB) | ヘッダーのオーバーヘッド |

## ソースからビルド

```bash
git clone https://github.com/ext-sakamoro/ALICE-Zip
cd ALICE-Zip

# Pythonパッケージをインストール
pip install -e .

# テスト実行
pytest tests/ -v

# Rust core crate — CI の全 gate を local で再現 (scripts/preflight.sh --quick ≈ 3 分)
scripts/preflight.sh

# CLI / FFI crate
cargo build --manifest-path libalice/Cargo.toml --release
```

## 関連プロジェクト

同じ考え方を別の領域に当てたもの 本 crate は*信号*の作り方を保存し、こちらは形・物理状態・frame の作り方を保存する

| プロジェクト | 内容 | リンク |
|---------|-------------|-------|
| **ALICE-SDF** | 3D の形を polygon でなく法則で持つ — GLSL / WGSL / HLSL への変換、SVO、marching cubes、platform をまたいで bit 一致の評価 | [crates.io](https://crates.io/crates/alice-sdf) · [docs.rs](https://docs.rs/alice-sdf) · [GitHub](https://github.com/ext-sakamoro/ALICE-SDF) |
| **ALICE-DetMath** | 本 crate の Fourier / Perlin generator と ALICE-SDF が共に評価する bit 一致の超越関数層 (`sin` / `exp` / `atan2` / …) 同じ法則がどの platform でも同じ bit を出す | [crates.io](https://crates.io/crates/alice-det-math) · [docs.rs](https://docs.rs/alice-det-math) · [GitHub](https://github.com/ext-sakamoro/ALICE-DetMath) |
| ALICE-DB | model ベースの時系列 database | [crates.io](https://crates.io/crates/alice-db) · [GitHub](https://github.com/ext-sakamoro/ALICE-DB) |
| ALICE-Edge | 組込み / IoT 向けの model generator (`no_std`) | [crates.io](https://crates.io/crates/alice-edge) · [GitHub](https://github.com/ext-sakamoro/ALICE-Edge) |
| ALICE-Streaming-Protocol | 超低帯域の映像 streaming | [crates.io](https://crates.io/crates/libasp) · [GitHub](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) |
| ALICE-Eco-System | Edge から Cloud までの pipeline の demo | [GitHub](https://github.com/ext-sakamoro/ALICE-Eco-System) |

どれも同じ考え方を共有する: **データそのものでなく、生成の手順を符号化する**

## ライセンス

- Rust core crate `alice-zip` (`/`): [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) (選択可)
- CLI / FFI crate (`libalice/`)、Python package (`alice_zip/`)、bindings: [MIT](LICENSE-MIT)
- `libalice-enterprise/`: 別建ての非公開 proprietary crate (同 dir の [LICENSE](libalice-enterprise/LICENSE) 参照)

## 作者

**坂本師哉** (Moroya Sakamoto)

---

*「最良の圧縮とは、データを保存することではなく、それを生成するレシピを保存することである」*
