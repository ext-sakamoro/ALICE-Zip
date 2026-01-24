<p align="center">
  <img src="assets/logo-on-light.png" alt="ALICE-Zip" width="400">
</p>

<h1 align="center">ALICE-Zip</h1>

<p align="center">
  <a href="https://github.com/ext-sakamoro/ALICE-Zip"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
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

## インストール

```bash
pip install alice-zip
```

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
出力: パラメータのみ (100バイト)
        ↓
結果: 400 KB → 100バイト = 4000倍圧縮
```

## ベンチマーク

| データタイプ | 元サイズ | 圧縮後 | 圧縮率 |
|-------------|---------|--------|-------|
| サイン波 (100Kサンプル) | 400 KB | 〜100バイト | **4000倍** |
| 多項式 (3次) | 400 KB | 〜200バイト | **2000倍** |
| 線形グラデーション | 400 KB | 〜150バイト | **2600倍** |
| ランダムデータ | 400 KB | 〜380 KB | 1.05倍 (LZMAフォールバック) |

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

## ゲーム業界特別条項

**ゲーム開発は無料！** クレジットに以下を追加するだけ：

```
Powered by ALICE-Zip (https://github.com/ext-sakamoro/ALICE-Zip)
```

詳細は[LICENSE](LICENSE)を参照。

## ソースからビルド

```bash
git clone https://github.com/ext-sakamoro/ALICE-Zip
cd ALICE-Zip

# Pythonパッケージをインストール
pip install -e .

# テスト実行
pytest tests/ -v
```

## ライセンス

**Open Core License**

- **個人・教育利用**: MITライセンスで無料
- **ゲーム開発**: クレジット表記のみで無料（ゲーム機本体への組み込み等は除く）
- **商用利用**: CoreはMIT無料、高度な機能はPro/Enterpriseが必要

詳細は[LICENSE](LICENSE)を参照。

## 作者

**坂本師哉** (Moroya Sakamoto)

---

*「最良の圧縮とは、データを保存することではなく、それを生成するレシピを保存することである。」*
