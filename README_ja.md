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
> *データではなく、アルゴリズムを保存する。*

[English README](README.md)

---

ALICE-Zipは、データそのものではなく**「データの生成方法」**を保存する次世代圧縮ツールです。

パターン、波形、数学的データに対して**10倍〜1000倍**の圧縮率を実現。
それ以外のデータにはLZMAにフォールバックし、標準ツールより**悪くなることはありません**。

## 特徴

- **手続き的圧縮:** サイン波、多項式、数学的パターンを認識
- **適応型フォールバック:** 手続き的圧縮が効かない場合は自動的にLZMAを選択
- **ロスレス:** ビット完全な復元
- **クロスプラットフォーム:** Python, Rust, C#/Unity, C++/UE5

## リポジトリ構成

| パス | 内容 | 配布先 |
|------|------|--------|
| `/` (`alice-zip`) | **Rust core crate** — 圧縮 primitive (LZ77 / 辞書 / BPE / エントロピー / 量子化 / zlib・LZMA residual container) + 全 generator 法則 (多項式 / Fourier / Perlin)、`no_std + alloc` | [crates.io](https://crates.io/crates/alice-zip) · [docs.rs](https://docs.rs/alice-zip) |
| `libalice/` (`alice-zip-cli`) | CLI `alice`、C FFI (`cdylib`)、PyO3 native module core generators / compression の thin re-export | C# / UE5 binding、pip `libalice` |
| `alice_zip/` | Python package (`ALICEZip` analyzer + `.alice` container) | pip `alice-zip` |
| `bindings/` | C++ / C# (Unity) / UE5 wrapper (`libalice/include/alice.h`) | |

## インストール

```bash
# Python
pip install alice-zip

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

## 仕組み

従来の圧縮は**バイト列**のパターンを探します。ALICEは**数学的**パターンを探します。

```
元データ = 生成関数(パラメータ) + 残差

ここで:
  - 生成関数() = 数学関数（多項式、サイン波など）
  - パラメータ = 小さな記述（〜100バイト）
  - 残差 = 圧縮された差分（多くの場合ほぼゼロ）
```

### 例

```
入力:  サイン波、100,000サンプル (400 KB)
        ↓
分析: 「サイン波、周波数=50Hz、振幅=1.0」として検出
        ↓
出力: パラメータのみ (~280バイト)
        ↓
結果: 400 KB → 280バイト = 1400倍圧縮
```

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

## ベンチマーク

| データタイプ | 元サイズ | 圧縮後 | 圧縮率 |
|-------------|---------|--------|-------|
| サイン波 (100Kサンプル) | 400 KB | 〜280バイト | **1400倍** |
| 多項式 (3次) | 400 KB | 〜285バイト | **1400倍** |
| 線形グラデーション | 400 KB | 〜250バイト | **1600倍** |
| ランダムデータ | 400 KB | 〜370 KB | 1.08倍 (LZMAフォールバック) |

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

## ライセンス

- Rust core crate `alice-zip` (`/`): [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) (選択可)
- CLI / FFI crate (`libalice/`)、Python package (`alice_zip/`)、bindings: [MIT](LICENSE-MIT)
- `libalice-enterprise/`: 別建ての非公開 proprietary crate (同 dir の [LICENSE](libalice-enterprise/LICENSE) 参照)

## 作者

**坂本師哉** (Moroya Sakamoto)

---

*「最良の圧縮とは、データを保存することではなく、それを生成するレシピを保存することである。」*
