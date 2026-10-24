# Changelog

All notable changes to the `alice-zip` Rust crate (repository root, crates.io)
are documented here. The CLI / FFI / Python native crate under `libalice/` has
its own [CHANGELOG](libalice/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### 破壊的変更

- `fft` feature: `generators::analyze_signal_fft` は rustfft をやめ、crate 内の f64 の FFT (`src/generators/fft.rs`、2 冪は radix-2、それ以外は Bluestein、twiddle は `alice_det_math::sin_cos64`、fused multiply-add を使わない) で計算する 関数の signature は同じで、返す bit は変わる (rustfft は target ごとに SIMD の butterfly を選び、aarch64 では fma の結果の符号を反転し、twiddle と norm / arg は platform の libm だった) 新しい出力は x86_64 と aarch64 で同じ bit `tests/fft_oracle.rs` (直接 DFT との誤差の上限) と `tests/determinism_golden.rs` の `fft_spectrum_is_the_recorded_bits` を追加し、その digest を `SEMANTICS_ID` の pin に加えた (`51ffeb07…`、それまでの `65ba78c1…` は `PREVIOUS_SEMANTICS_IDS` に入れた) 依存から rustfft を外した 計測 (arm64、release、1 回あたり): n=1024 15.5 → 32 µs、4096 67 → 114 µs、65536 1.3 → 2.8 ms、1000 21 → 101 µs、97 4.3 → 10 µs、100000 2.8 → 19.8 ms (2 冪でない長さは Bluestein で長さ 2n 以上の変換を 3 回行うため遅い) libalice の `generators::analyze_signal` (この関数の再 export) と `.alice` の Fourier 係数も同じく変わる

### Added
- `law_id`: Law の識別子 v2 `LawIdHasher` ALICE の全 crate が同じ encoding で Law の識別子を作る (domain `alice-law-id/v2`、kind と semantics の識別子の後に値を型 tag と長さ前置で入れる、長さと整数は big-endian) 独立実装 `scripts/law_id_v2_reference.py` (標準 library だけの Python) と試験 `tests/law_id_v2.rs` が `tests/data/law_id/v2_probes.txt` の 10 件で一致する
- `law_ref`: Law の参照 v2 `LawRef` kind・semantics の識別子・law id・名前付きの値の行で 1 つの Law を指す bytes (magic `LOLR`、版 2) law id は kind と semantics に行を (名前, 値) の順で `LawIdHasher` に入れた値で、読む側が計算し直して違えば両方の識別子を示して拒否する 行は名前の昇順 (重複なし) で持ち、作る側が並べ替え、読む側は順違いと重複を拒否するので 1 つの Law に bytes は 1 つ 長さは残りの bytes と照らしてから読み、行数は 64 まで f64 の tag は NaN の規則が入るまで予約 (拒否) 独立実装 `scripts/law_ref_v2_reference.py` と試験 `tests/law_ref_v2.rs` が `tests/data/law_ref/v2_probes.txt` (参照 11 件、拒否 19 件) で一致する
- CI: test job と preflight で 2 つの独立実装を `--check` で走らせ、`scripts/run_tests.sh` に `law_id_v2=3` / `law_ref_v2=7` を足した
- `container::stream` (`std` feature): 容器を stream で書く `StreamWriter` (`Read + Write + Seek`、書き先は容器の終わりで終わること、先に bytes が残る書き先は `finish` が `StreamError::TrailingBytes` で拒否) と読む `StreamReader` (`Read + Seek`) section を全部メモリに持たずに書け、同じ section に対して `Container::to_bytes` と同じ bytes を出す (永続形式・`FORMAT_ID`・識別子は変えない) `finish` は書いた section を読み直して表の SHA-256 と照合してから trailer を書き、宣言した数と違う section は `StreamError::SectionCount`、`finish` の前に捨てた section は `StreamError::Unfinished` で拒否する (`finish` しない writer の出力はどの reader も開かない) `StreamReader` は `ContainerView` と同じ検査を同じ順で行い、section の bytes は `UnverifiedSection::finish` が hash を確かめるまで未検証 (合わなかった section は後の読みと `finish` でも同じ error を返す) header と表の検査は `ContainerView` と共有する関数にした `tests/container_stream.rs` (10 件) は全 fixture で `to_bytes` と参照の bytes に一致すること、全 fixture と 2 つの fixture の 1 byte の全変更 (trailer を合わせた写しも) で `ContainerView` と同じ判定になること、書いた後に変えた payload を trailer に入れないことを確かめる `tests/container_count_bound.rs` は file に収まらない section 数を view と stream の両方が table を確保する前に拒否することを、確保の最大量で確かめる
- `container`: 容器の上に自分の必須 section を足す形式のための入口 `ContainerView::open_with_known` / `open_strict_with_known` と `Container::from_bytes_with_known` / `from_bytes_strict_with_known` reader が `Tag::KNOWN` 以外に知っている tag を宣言し、その tag の critical な section は拒否せず読む 宣言しない tag の critical な section は従来どおり `UnknownCritical` で拒否し、非 critical な section は従来どおり保持する 永続形式・`FORMAT_ID`・識別子・`FMID` / `SREF` / `LIDS` の規則は変えない (既存の `open` / `from_bytes` 系は宣言が空の同じ入口)
- CI (ALICE-Law G11): `scripts/changed_paths.py` (g3-ir.yml の path の絞り込み) の試験 `scripts/test_changed_paths.py` を test job と preflight で走らせる それまで CI のどこからも走っておらず、絞り込みの判定が壊れても red にならなかった
- CI (ALICE-Law G11): test job の default と docs.rs feature set の `scripts/run_tests.sh` に `zip_semantics_id` / `container_format_id` / `residual_semantics_id` を足した 識別子の試験の binary が 1 件も走らない時に step が落ちる (G6 / G8 の canary がその binary にある)
- CI (ALICE-Law G11 (c)): ほかの ALICE の crate と同じ bytes で持つ file を ALICE-Law の表と比べる `scripts/shared_files.sh` (試験 `scripts/test_shared_files.py`、ALICE-DetMath などと同じ中身) と `.github/shared-files-pin` を足し、test job (3 OS) と preflight で走らせる 表と検査器は pin の版で取得し SHA-256 を確かめる 差、表に無い共有の path、比較 0 件は失敗
- `conformance/g11-own.toml`: ALICE-Law G11 の表のうち、この crate 固有の行 (G4 / G5 / G6 / G7 / G8 の golden と識別子の試験、比較 0 件は `scripts/run_tests.sh` の試験) 共有の検査器の行は ALICE-Law の `conformance/g11-shared-rows.toml` から生成する
- CI: text の file の行末の検査 `scripts/check_eol.py` (試験 `scripts/test_check_eol.py`) と `.github/workflows/eol.yml` を足した `git ls-files --eol` で text の file が index か working tree で CRLF / 混在なら失敗 (`-text` と binary は除く、比べた数を出し 0 件は失敗)、workflow は ubuntu / macos / windows で走り、preflight にも同じ検査を足した (ALICE の crate で同じ中身の file)
- CI (ALICE-Law G9): `scripts/id_move_check.py` の規則を G9 の文言に合わせた 公開版の値が以前の識別子の一覧に残る移動 (同じ bit の再現) は版を上げなくてよく、その代わり `previous-tests` に挙げた試験が、そのままでは通り、旧値を一覧から抜いた写しでは落ちることを検査器が実際に走らせて確かめる (値を書いただけの試験は通らない) 一覧に無い移動は意味の変更として破壊的な版を求める CHANGELOG に両方の値を書くことはどちらでも求める
- CI: README の ALICE-Law 節の検査 (law-readme-block.yml) を preflight でも同じに走らせる `scripts/law_readme_block_check.py` と試験 `scripts/test_law_readme_block_check.py` を足した (ALICE-DB と同じ中身の共有 file、workflow の check の引数と節の pin の生成器を使い、比較 0 件・件数の行なし・pin が無い・fetch の失敗は失敗、未 commit の入力は検査していないと表示) 試験は CI と preflight で走る
- CI (ALICE-Law G9): `scripts/id_move_check.py` は release tag の file を一時の index で書き出す (`git archive` と違い export-ignore の file も含め、repository の index と checkout に触れない) 他の crate の定数の公開 re-export も載せ漏れの検査に数える (`DET_MATH_SEMANTICS_ID` は SEMANTICS_ID に畳み込まれるので理由付きで除外)
- CI (ALICE-Law G9): `scripts/id_move_check.py` (ALICE の識別子を持つ crate で共有) を追加 最新の release tag の公開版と作業 tree で `law::SEMANTICS_ID` と `container::FORMAT_ID` の値を読み (型が `[u8; 32]` に解決される const / static / 関連 const は可視性・参照・型の別名・長さの書き方・feature によらず全部、検査対象か理由付きの除外のどちらかに載っていないと fail、rustdoc の JSON で読む)、動いたのに破壊的な版になっていない / 公開版の値が `PREVIOUS_SEMANTICS_IDS` に無い / CHANGELOG が両方を書いていない時に fail する (ci.yml の clippy job と preflight、registry が読めない時は main で fail、ci/ branch と PR では警告)
- CI (ALICE-Law G4 / G5 / G13): release profile の golden の step に every feature (std / fft / parallel / lzma) の組を足した (x86_64 と aarch64、preflight にも同じ step) `clippy.toml` に HashMap / HashSet とその iterator、std::thread の Builder / JoinHandle / ScopedJoinHandle 型、`std::thread::spawn` / `scope` / `Builder::spawn` / `spawn_scoped` / `available_parallelism` を追加 `tests/order_determinism.rs` (parallel feature の 4 x 4096 の 2D Perlin を 1・2・4・8・16 thread で 4 回ずつ作り、直列の合成と bit で比べる) と `scripts/order_guard.py` (ALICE-Analytics / Edge と同じ file、許した行は本文の SHA-256 で固定し、書き換えると一覧を直すまで fail、rayon の並列 method は `par_` で始まるもの全部と rayon の trait 名を、dot 付きでも trait 経由 (UFCS) でも拾う) と理由付きの一覧を追加 libalice (alice-zip-cli) の residual header の map と試験の map を HashMap から BTreeMap にした (key での参照と削除だけで、挙動は同じ) `alice-law-conformance.toml` の G4・G5・G13 を pass にした (G4 / G5 の debug と release の段は既にあり、台帳が古かった)
- CI (ALICE-Law G3): fma の結果の符号反転を禁じる 2 段の検査を ALICE-LOL から同じ file で移した (構文木の `tools/no-outer-fneg-of-fma` を ci.yml の test job と preflight で、release の LLVM IR を読む `scripts/fneg_fma_ir_check.py` を `.github/workflows/g3-ir.yml` で x86_64 と aarch64 の native runner で実行、path の判定は changes job、走るべき job が走らなければ gate job が red) `scripts/no-outer-fneg-of-fma-allowlist.toml` は fma の呼び出し 5 件 (すべて tests/) と src/ の 0 件を固定する `alice-law-conformance.toml` の G3 を pass にした
- `int2d` module: 整数の格子 (存在の mask つき) か整数の点の集合の上の `i64` の値を、整数の法則に対する残差として保存する 法則は `IntLaw2` (`predict(i, j) -> i64`、`law_id`) で、`Zero` と `PlanePerTile` (tile ごとの最小二乗の平面を `i128` の Cramer の公式で解き、係数は `2^-k` の倍数、四捨五入は半分を上へ) を持ち、呼び出し側が自分の曲面を実装できる `encode` / `decode` (`std` feature) は `Mode::Lossless` と `Mode::Bounded { q }` (誤差は高々 `q / 2`、最大の誤差を返す) を持ち、header に法則の識別子・domain の digest・mode・`q`・件数・最大の誤差・元の値と復元した値の SHA-256 を書き、archive の末尾に全体の SHA-256 を置く 予測から符号化まで浮動小数点を使わず (module は `clippy::float_arithmetic` を deny、試験が source に `f32` / `f64` が無いことを確かめる)、overflow は error 識別子は `law::SignalLaw::law_id` と同じ符号化で semantics id を含まない 新しい依存は無い `tests/int2d_oracle.rs` (14 件) は可逆の往復 (±2^40、符号の交互、mask つきの格子、点)、`Bounded` の誤差の上限と半分の丸めの向き、平面の完全な予測、overflow、識別子の違い・domain の違い・全 bit の反転・payload の書き換えの拒否、固定の入力に対する archive の bytes (3 OS の CI) を確かめる
- `int2d`: `PlanePerTile::from_parts` / `origin` / `to_bytes` / `from_bytes` (magic `I2PT`、版、原点・tile・`k`・tile ごとの係数を big-endian で並べ、末尾に SHA-256、tile は添字の順で並びが違えば拒否) fit した法則を archive と別に保存し、別の process で復元して decode できる `Codec` と `encode_with` (`Codec::Raw` は zigzag + LEB128 の残差を圧縮せずに置き、呼び出し側の backend と比べられる、既定は `Codec::Deflate`) `PlanePerTile` の tile の上限 (1024) を外し、tile は座標の単位で指定する 各 tile は最初の値 `v0` を引いた値で解くので (係数は同じ)、成り立つ範囲は `12 n^3 T^4 max(D, 1) 2^k < 2^127` (tile の点数 `n`、辺 `T`、tile 内の値の幅 `D`) で、超えると `Int2dError::Overflow` decode は展開した残差が件数の上限 (1 値 10 byte) に達したら `BadPayload("payload expands past the count")` で止める 試験を 9 件足し (23 件)、header の誤差の上限・lossless の 2 つの digest・余分な残差・上限を超える展開・法則の bytes の全 bit の反転と tile の並びを、trailer を合わせて偽造した archive と法則で確かめる
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査する (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` と test) 静的は target/ 以外の設定・script・Rust を comment を除いて読み、`target-cpu=native`、素の literal でない target-cpu / target-feature / llvm-args の値、command や file から取った RUSTFLAGS、fast-math 系の flag、sse / sse2 を外す target-feature を検出する (意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く) 動的は新しい target dir で `cargo check -v` を走らせ、cargo が rustc に実際に渡した flag を読む 静的は test job (ubuntu / macOS / Windows)、動的は release-profile job (ubuntu / macOS、default / all-features / no-default) と preflight で実行
- `tests/float_environment.rs` (ALICE-DB と同一): round-to-nearest-even と subnormal が flush されないことを実行時に確かめる test test job の default と docs.rs feature set、release profile で `scripts/run_tests.sh` を通して実行 `alice-law-conformance.toml` の G14 を pass にした
- CI (docs): `README.md` と `README_ja.md` の対応を照合する `scripts/readme_parity_check.py` (試験 `scripts/test_readme_parity_check.py`、見出しの対は `scripts/readme_headings.toml`) を追加 見出しの並び・節ごとの段落 / 箇条書き / 表の行 / code block の数・code block の中身・inline code・数値・hash・`alice-*` 名・link と URL を照合し、何も比較できなければ失敗する ALICE-Law の同名の検査器を元に、日本語版の file 名 (`README_JP.md` / `README_ja.md`) を自動で選び、表の行・節ごとの code block・裸の URL の照合を足した CI の claim-check job と preflight で走る
- CI (決定論 G10): `clippy.toml` の `disallowed-methods` に `std::time::Instant::elapsed` / `std::time::SystemTime::elapsed` を追加 (`::now()` と同じく時計を読む、保持済みの 2 値の差を取る `duration_since` は対象外) `scripts/test_entropy_guard.py` に 2 entry の存在を確かめる試験、`scripts/clippy_canary_calls.tsv` に 2 行を追加 (ALICE-Analytics と byte 一致、canary は 7 行) `libalice/src/bin/alice.rs` の CLI の計時表示 4 箇所は理由付きの局所 allow G10 の記載を現状に合わせ、Python 側の白色雑音 (`alice_zip/media_generators.py` の seed 無しの `np.random.randn`) を未達の箇所として明記
- CI (G11、検査器自身の試験): clippy の禁止一覧 (G1 の超越関数、G10 の wall clock / OS 乱数) に canary を追加 `scripts/clippy_canary.py` が repo の `clippy.toml` / `rust-toolchain.toml` / `[lints]` を写した使い捨て crate に、`scripts/clippy_canary_calls.tsv` の 1 行ごとに禁止呼び出しを 1 つ置き (`f64::sin` / `f64::sin_cos` / `Instant::now` / `SystemTime::now` / `RandomState`)、行ごとにその path を名指す `disallowed_*` 診断がちょうど 1 件出ること・対照の関数には 1 件も出ないこと・各行の path が `clippy.toml` にあることを確かめる data file が 0 行、または `clippy.toml` が 0 件の時は「比較 0 件」で失敗する 試験 `scripts/test_clippy_canary.py` (cargo 無し) と共に CI の clippy job と preflight で走る 3 file は ALICE-Analytics と byte 一致
- CI (G11): `scripts/run_tests.sh` の試験 `scripts/test_run_tests.py` (ALICE-Analytics と byte 一致) を追加 test job (3 OS の bash) と preflight で走る
- CI (G1): 直接依存している `libm` の自由関数 (`libm::sin` など) は `f32::*` / `f64::*` の禁止一覧が届かず、どの clippy step も通っていた `clippy.toml` で、正確に丸められる `ceil` / `floor` / `round` / `trunc` / `sqrt` / `fma` (と f32 版) 以外の libm 0.2.16 の公開関数 112 個を path で禁止し、`libm::Libm` (同じ関数の method) を型で禁止した clippy は `use ... as` の別名・glob import・関数ポインタを通しても path で止める `scripts/libm_ban_check.py` (試験 `scripts/test_libm_ban_check.py`) が、Cargo.lock の libm の公開関数がすべて許可か禁止のどちらかにあること・禁止した path が実在すること・`libm::Libm` が禁止されていることを確かめ、同じ版の libm に依存する使い捨て crate に禁止呼び出し 4 形 (直接 / 別名 / 関数ポインタ / `Libm::<f64>`) を植えて clippy がそれぞれ 1 件ずつ止め、許可した呼び出しは止めないことを確かめる 公開関数 0 件・libm の禁止 0 件は失敗 CI の clippy job と preflight で走る
- 符号付きゼロと桁落ちの bit 一致 golden `tests/float_edge_golden.rs` (20 件、`polynomial_at` / `fit_polynomial` / `multi_sine_at` / `fourier_at` / `quantize_8bit` / `quantize_16bit` / `SignalLaw::evaluate` / 残差 RMS) と、release profile で `analytic_law_id` / `float_edge_golden` / `determinism_golden` を走らせる workflow `release-profile.yml` (ubuntu / macos、`scripts/run_tests.sh` が各 suite の 1 件以上の通過を検査、preflight の full でも走る)
- `tests/durable_writers.rs`: crate の永続形式の writer 13 個を source と件数で照合し (一覧に無い writer は fail)、それぞれが意味の識別子をどう持つかを検査する `compat/v0_10_reader`: crates.io の 0.10.0 と今の crate の間で、今の形式が 0.10 に拒否され、0.10 の形式が今の crate で `Legacy` として読めることを確かめる (CI の test job と preflight)

### Fixed

- `README_ja.md`: 英語版との差を直した 「仕組み」の式が減算 (`+`) とパラメータ「〜100バイト」のままだったのを英語版の xor と記述に合わせた code block は英語版と同じ内容にした `zlib alone` 列の名前・dtype の数・2 つの形のうちどちらが正確かの記述を英語版に揃え、冒頭の claim-test marker の段落の区切りを英語版と同じにした
- 符号の異なるゼロが混ざる入力で、`SignalLaw::fit_polynomial` の定義域の端 (lo / hi) と `quantize_8bit` / `quantize_16bit` の `min` が arm64 と x86_64 で違っていた 等しいゼロ同士の `f64::min` / `f32::min` (と max) は結果が規定されておらず、arm64 は -0 < +0 として扱い x86_64 は第 2 引数を返す 4 箇所を alice-det-math 0.5.1 の符号を決めた `min` / `max` / `min64` / `max64` に替え、`tests/minmax_sign_golden.rs` で両方の並びを固定した 68 の法則と 21 の量子化の入力で測ると、修正後は arm64 と x86_64 (debug / release) の出力が一致し、変わるのは x86_64 の混在ゼロの入力 (法則 10 件の定義域の端と law_id、係数が 0 の時の evaluate(-0.0)、量子化 3 件の min) だけで、他の入力の bit は変わらない これらの law_id は元から arch ごとに違い 1 つの値として再現できなかったので、旧い識別子を受理する表は追加しない
- 試験 (`tests/minmax_sign_golden.rs`): 入力を `std::hint::black_box` に通すようにした 入力が literal なので release build で最小値・最大値の比較が定数畳み込みされると、x86_64 の `minsd`/`maxsd` (ゼロ同士では第 2 引数を返す) を実行しないまま試験が通りうる 定義域の lo / hi と量子化の min の 3 箇所は、変更の前後とも `f64::min`/`max`/`f32::min` に戻すと x86_64 の debug と release の両方で red (aarch64 は green) 量子化の max は `max - min` の scale にしか入らず、混在ゼロでは符号が出力に届かないので戻しても全構成で green 期待値は変えていない

### Changed
- Licence: `Apache-2.0` → `LicenseRef-Proprietary`
- `.github/dependabot.yml` を削除した
- `wasm/Cargo.toml` (alice-zip-wasm) is `LicenseRef-Proprietary` too, and `scripts/license_check.py` reads every package manifest instead of a fixed list (build tools and the fuzz crate excepted)
- The PyPI workflows build and check the `alice-zip` and `libalice` packages but no longer publish them; versions already on PyPI stay as they are
- `libalice-enterprise/LICENSE` の先頭に `SPDX-License-Identifier: LicenseRef-Proprietary` を追加 (条文の内容は不変、`libalice-enterprise/Cargo.toml` の `license` と同じ識別子を条文 file 側にも明示する)
- `libalice-enterprise/` の独自の Enterprise License Agreement を廃止し、root と同じ `LICENSE-Proprietary` の写しにした
- CI: 試験 `scripts/test_public_text.py` を ALICE-SDF と同じ版にした (公開する text の検査を pin の checker で走らせ、置いた note link を tree と commit message で検出すること、pin の 3 行目 `config-sha256` が設定を承認することを確かめる、network を使う) `scripts/public_text.sh` と public-text の pin は変えないので検査の結果は変わらない `.github/shared-files-pin` を ALICE-Law の新しい版に移す
- CI: `scripts/run_tests.sh` と試験 `scripts/test_run_tests.py` を ALICE-LLM と同じ bytes にし、`.github/shared-files-pin` を ALICE-Law の新しい版に移す 名前 `lib` を library の単体試験として数えるようになったが、この repository は `lib` を名指さないので各段の試験の名前と件数は変わらない 試験 `scripts/test_public_text.py` は ALICE-SDF と同じ版にした (置いた note link を tree と commit message で検出することを pin の checker で確かめる、network を使う) `scripts/public_text.sh` と pin は変えないので検査の結果は変わらない
- `alice-law-conformance.toml`: evidence / mutant / note の行番号参照 44 件を anchor (`path#token`、ALICE-Law の roster が ref の file で 1 か所に解決するか確かめる) にした 行番号がずれて別の行を指していた参照 (clippy.toml・ci.yml・scripts/preflight.sh の範囲と列挙、src/container.rs の error と reader、src/law.rs の max64 / min64 等) は文章が名指す構文 (step 名、fn、表の entry) に直し、測定の記録に出る行番号はその行を名指す文に置き換えた 各 gate の status は変わらない
- CI: `.gitattributes` に `* text=auto eol=lf` を置き、text の file を保存も checkout も全 OS で LF にした (共有 file は ALICE-Law の表と byte で比べるので、Windows の CRLF の checkout ではずれになる) binary と既存の行はそのまま、`git add --renormalize` で commit 済みの中身は変わらない
- CI (ALICE-Law G3): `tools/no-outer-fneg-of-fma` と `scripts/fneg_fma_ir_check.py` (と試験) を ALICE-LOL と同じ新しい版にした `expected-src-fma-calls` は src/ の library code の fma 呼び出し数 (`#[test]` と test の時だけ成り立つ cfg の下の item と文、test-only の `mod y;` で読み込む file とその子は数えず、test-only の数を別に表示する) この repository の数は変わらない IR 検査は library の source に fma が無いのに IR に fma がある時を失敗にし、library の fma が 0 の時は共有の probe を target ごとに compile して判定を確かめる (0 件実行は失敗) tool は独自の workspace なので repository 全体の format 検査に入っていなかった CI と preflight に tool の format の検査を足した
- CI: clippy の canary の表の生成器 `scripts/gen_clippy_canary_calls.py` が generic 関数 (`std::thread::spawn` / `scope` / `Builder::spawn` / `spawn_scoped`) を turbofish 付きで植えるようにした (素の path では型が決まらず compile できないため) 他の行は byte 一致で、この repo の表は変わらない 共有 file の協調更新 (ALICE-Analytics などと同じ中身)
- CI: 公開する文面の検査の pin を ALICE-Law 5de698e (checker sha256 4b3d9542…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- CI: `scripts/clippy_canary.py` の説明を直した canary は repository の clippy job の flag に依らず決まった flag (`-W clippy::pedantic -D warnings`) で試す (以前は「CI の clippy job と同じ flag」と書いていたが、job によって flag は違う) 動作は変わらない (ALICE-Analytics / ALICE-DB / ALICE-Zip / ALICE-Edge と byte 一致)
- CI (G4 / G11): debug profile の golden を走らせる test step (default features / docs.rs feature set / no_std の `determinism_golden` / no_std の `container_oracle`) を `scripts/run_tests.sh` 経由にして、試験が 1 件も走らない時と、名指しした golden file (`determinism_golden` / `float_edge_golden` / `analytic_law_id` / `minmax_sign_golden`) が 1 件も走らない時に失敗させる 以前は `float_edge_golden` は release profile の job でしか 0 件検査がなく、debug の step は 0 件でも通った preflight の同じ step も同様 `tests/determinism_golden.rs` に `assert_golden` 自身の canary を 3 本追加 (crate の外で計算した 8 byte の 0 の SHA-256 を受け入れる対照、下限より短い出力で panic、異なる digest で panic) 記録済みの golden 値は変えていない
- `alice-law-conformance.toml`: G1 を fail-known から pass にした (禁止一覧は G1 の一覧をすべて含み、pending の powi も禁止、全 feature と no_std target を lint し、canary と libm の禁止が入った 変異は manifest に記載) G11 は fail-known のまま証跡・変異・note を更新した (残る未達は (c) で、ALICE-Analytics から写した 5 file の byte 一致を照合する検査が無い) README の ALICE-Law の節を作り直した
- CI: 公開する文面の検査の pin を ALICE-Law f9b9000 (checker sha256 84226ec7…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- alice-det-math を 0.5.1 に上げた `SEMANTICS_ID` が 135c407d… から 85e08846… に変わる (0.5.1 は 135c407d… を `PREVIOUS_SEMANTICS_IDS` に持つので、0.5.0 の識別子で書いた container は読める) `tests/determinism_golden.rs` の識別子と law_id の 2 つの pin を記録し直した 動いたのは識別子の入力だけで、0.5.0 の識別子を渡すと law_id と evaluate の digest は以前の 1943c258… のまま (同じ file の試験で固定)
- `tests/float_edge_golden.rs`: QR の丸めを計算したまま記録していた 2 つの pin (1e8 + 8x の傾き 8 − 2 ulp、最小二乗の直線の定数 1 − 1 ulp) に、正確な解から決めた上限 (傾きは 8 から 4 ulp 以内、定数と傾きは 2 ulp 以内、残差は最大値 3 の 2 ulp 以内) を足した bit の pin は変えていない
- CI: ci / fuzz / security-audit の concurrency を、main では commit ごとに group を分けて run を取り消さず、取り消すのは branch の古い run だけにした (以前は main への後の送信が main の run を取り消し、結果が cancelled のまま残りえた) `scripts/workflow_concurrency.py` (試験 `scripts/test_workflow_concurrency.py`、ALICE-LOL / ALICE-SDF / ALICE-DetMath と byte 一致) が branch への送信で走る全 workflow を検査し、workflow を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- CI: clippy の禁止一覧に `f32::sin_cos` / `f64::sin_cos` を足した (ALICE-Law G1 の一覧と揃える)
- CI: 上限の無かった 20 job に `timeout-minutes` を付けた (止まった step は共有 runner を GitHub の上限の 6 時間まで占有しうる) 値は直近の成功した run 12 本の所要時間の最大の 2 倍・中央値の 4 倍のうち大きい方を 5 分単位に切り上げ、cargo を使う job は最低 15 分 (cache が外れた build のため)、他は最低 10 分 既に上限のある job は変えていない `scripts/workflow_timeouts.py` (試験 `scripts/test_workflow_timeouts.py`) が全 job の上限を検査し、job を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- CI: 追跡している `Cargo.lock` (root / `libalice` / `fuzz` / `libalice-enterprise`) が manifest と合っているかを `scripts/lock_check.py` (`cargo metadata --locked`、試験 `scripts/test_lock_check.py`) で検査する CI と preflight で走り、検査した lock が 0 なら失敗する `fuzz/Cargo.lock` と `libalice-enterprise/Cargo.lock` は `alice-det-math` 0.4.0 / `alice-zip` 0.9.0 (と `alice-zip-cli` 3.0.0) を解決したままで、build のたびに作業 tree で直されていたので気付かれなかった 両方を 0.5.0 / 0.10.0 / 3.1.0 に作り直した (公開 package には入らない file)
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る
- `clippy.toml` に `std::time::Instant::now` / `std::time::SystemTime::now` / `std::collections::hash_map::RandomState::new` を `disallowed-methods` へ追加 (決定論: `src/` は OS の entropy も時計も読まない、non-workspace の `libalice/` も directory 探索で同じ検査を受ける) 同じ検査を compile 無しで行う `scripts/entropy_guard.py` (`src/` と `libalice/src/` を走査、試験 `scripts/test_entropy_guard.py`) も追加、CI (ci.yml / clippy job) と preflight で走る `libalice/src/bin/alice.rs` の CLI タイミング計測 4 箇所 (標準出力の診断のみ、library の出力には含まれない) は `#[allow(clippy::disallowed_methods, reason = "...")]` + `entropy_guard.py` の `ALLOWED` で明示許可
- CI (`scripts/entropy_guard.py`): 3 つの穴を修正 (1) `RandomState::new` だけを検査しており `RandomState::default()` (同じ OS 乱数シードのハッシャーに届く) を素通りさせていた `RandomState` の裸の語を検査する形に変更 (clippy 側の `disallowed-methods` は `Default::default` をトレイトメソッドとして実装型だけでは指せないため、`clippy.toml` に新設した `disallowed-types` の `RandomState` entry と合わせた 2 層で塞ぐ) (2) `#[cfg(test)]` module 判定のブレース深さ計算が文字列しか除去しておらず、`'{'` のような char literal や `r#"..{"#` のような raw string の埋め込み引用符がブレース深さを狂わせて test module 判定が早期終了し、直後の production code を見逃す経路があった 文字列・raw string・char literal・ブロックコメントを 1 pass で正しく除去する sanitizer に置換 (3) `<std::time::Instant>::now()` のような qualified-self 呼び出しが `Instant::now` のパターンに `>` を挟むため素通りしていた (`as Trait` を挟む形も含む) 型名と `::method` の間に `>` / `as Trait` を許容するよう正規表現を拡張 canary test 6 本追加、元の script に対して赤を確認 比較件数は変わらず 29 file / 10 pattern / 0 problems 副次: entropy guard の 2 step を `security-audit.yml` (push trigger が main のみ・paths filter に対象 3 file 無し、`ci/**` branch では一度も発火していなかった) から `ci.yml` の clippy job (push: `[main, "ci/**"]`、paths filter 無し) へ移設 `matches()` が distinct line 数で数え、1 行に 2 回の呼び出しがあっても 1 件と数える既知の限界を doc comment に明記 (clippy の compiled 層は独立に per-call-site で捕まえるため未修正)
- CI: 公開 API の互換性の検査を `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB / ALICE-Analytics / ALICE-Edge と同一) に置き換えた crates.io の最新の alice-zip を baseline にし、既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせる (ALICE-Law release-train.toml は組を宣言しない) 版が公開済みの 0.10.0 と同じ間は次の最小の版 0.10.1 として検査する 旧 job (security-audit.yml、`--release-type patch` で比較件数だけを見ていた 1 組) は外した
- CI: `semver-checks.yml` に 2 つ目の crate `alice-zip-cli` (`libalice/`) の step を足した crates.io の最新の alice-zip-cli を baseline にし、組は ALICE-Law release-train.toml の [crate.alice-zip-cli] を `.github/semver-gate-cli.toml` にそのまま写した (`cli` + `codec`、`python` は cargo check) ほかに既定・既定なしの 2 組 どの組にも入らない feature があれば失敗する 版が公開済みの 3.1.0 と同じ間は次の最小の版 3.1.1 として検査する
- CI: `scripts/semver_gate.py` が cargo-semver-checks の出力を読む parser であることを `CARGO_OUTPUT_PARSER = True` で宣言し、`scripts/test_semver_gate.py` は module を `import semver_gate` で読むようにした (色の付いた出力を試験する検査器が parser とその試験を見つけられる形 実行時は従来どおり `CARGO_TERM_COLOR=never` と escape の除去) 共有 file の中身を揃える変更で、検査の挙動は変わらない
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) が `clippy.toml` の全 entry を植えて確かめるようにした 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し (この repo は 171 件、以前は手書きの 7 件)、CI が再生成して byte で比べる canary は全 entry に行があること (植えるか、理由付きで飛ばすか) を確かめ、外部 crate の path は `Cargo.lock` の版を固定して依存に足す この repo の `libm::` の path は `Cargo.lock` が固定する libm (0.2.16) を依存に足して植える 型も禁止された inherent method (`RandomState::new`) は、その型の診断を 1 件だけ許す
- CI: `scripts/run_tests.sh` が名前ごとの最低の件数 (`float_environment=2`) を受け付けるようにし (ALICE-DB と同じ中身)、G14 の `tests/float_environment.rs` を名指す全 step を `float_environment=2` にした 2 つの試験の片方を `#[ignore]` にする・cfg で外す・名前を変えると、その step が失敗する
- CI: 公開する文面の検査の pin を ALICE-Law 779fa44 (checker sha256 e439fdd7…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- ALICE-Law G2: 浮動小数点の max / min を全部 `alice_det_math` の `max` / `min` / `max64` / `min64` (IEEE 754 maximumNumber / minimumNumber、-0.0 < +0.0) にした library は `src/law.rs` の残差の最大と ingest の幅、`src/generators/polynomial.rs` の最小二乗の `r_max` (domain の上下限は既に det-math)、加えて test・example・libalice の全箇所 `clippy.toml` で `f64::max` / `f64::min` / `f32::max` / `f32::min` を禁じ (libalice も同じ設定を読む)、canary の表に 4 行を足した 置き換えた箇所は abs() の値か `<=` の比較にしか効かないので、goldens (46 件) は前後で変わらない
- **Breaking** ALICE-Law G6: `law::SEMANTICS_ID` を `alice-det-math` の識別子の再 export から alice-zip 自身の識別子にした (`src/semantics.rs`) 生成器の golden 4 件・法則の識別子を作る tag 5 件の SHA-256・解決した `alice-det-math` の識別子を、`alice-det-math` と同じ手順 (名前の順、u32 の長さ、SHA-256) で畳み込む det-math の世代ごとの表 `ZIP_SEMANTICS_BY_DET_MATH` から引き、表に無い世代は compile が止まる 値は 65ba78c1… (det-math 0.5.1) で、crate の版では動かない 0.10 までの値 (det-math の 85e08846… と 135c407d…) は `law::PREVIOUS_SEMANTICS_IDS` に入れ、`law::check_semantics` が再現できる (VerifiedPrevious) と読む container の `law_ids_for_this_arithmetic` はこの判定を使う `law::SEMANTICS_ID` で作った法則の識別子はすべて変わる (85e08846… を渡すと 0.10 の値に戻る) 上流の識別子は `law::DET_MATH_SEMANTICS_ID`
- **Breaking** ALICE-Law G10 (Python): `AudioParams` に `seed` を足し、`AudioPattern.NOISE_WHITE` では必須にした (無いと `ValueError`) 白色雑音は信号全体で 1 つの `np.random.RandomState(seed)` から取る (numpy は RandomState の系列を版をまたいで固定している、chunk の大きさに依らない) `ALICE_AUD` の読み込みは `noise_amplitude` と `seed` を読み、seed の無い白色雑音の parameter は拒否する (以前の読み込みは振幅を読まず、毎回違う雑音になっていた) CI の Python job に scipy を入れ、これまで skip されていた音声の試験が走るようにした
- **Breaking** ALICE-Law G8 (container): `Container::new` は section 0 に critical な `FMID` section (`container::FORMAT_ID`、形式の定義の SHA-256) を書く 0.10 以前の読み手はこの container を `UnknownCritical` で拒否する 読み込みは `FormatStatus` (`Verified` / `Legacy`) を報告し、`FMID` の無い 0.10 以前の container も読める `Container::from_bytes_strict` / `ContainerView::open_strict` は `Legacy` を拒否し、別の値の `FMID` は `ContainerError::UnknownFormat`、先頭でない・critical でない・32 byte でない・2 個ある `FMID` は `Malformed` 移行: section の番号は `FMID` を 0 番と数えるので、`push` した section の番号は 1 から始まる (`LIDS` の行や `Reference::section` を数える code は 1 を足す)
- **Breaking** ALICE-Law G8 (残差容器): `compress_residual_quantized` / `_lossless` / `_xor` (と `_with`) は新しい marker 0xFB / 0xFA / 0xF9 で書き、codec byte の後に model の semantics 識別子 (32 byte、`law::SEMANTICS_ID`) を置く 復元側は記録された識別子を `law::check_semantics` で照合し、再現できない識別子は復元の前に `ResidualIdError::Mismatch` (両方の識別子を示す、`io::Error` の中) で拒否する 0.10 以前の 0xFC / 0xFD / 0xFE と版 0 は復元でき、`residual_container_status` は `Legacy` を返し、新しい `decompress_residual_*_strict` は拒否する 0.10 以前の読み手は新しい容器を読めない
- **Breaking** Python package `alice-zip` を 3.0.0 に上げた (`AudioPattern.NOISE_WHITE` は `AudioParams.seed` が必須、seed の無い白色雑音の `ALICE_AUD` は読まない) CLI crate `alice-zip-cli` (libalice) を 4.0.0 に上げた (書く残差容器は識別子を持つ新しい形式で、3.x 以前の CLI は読めない)
- **Breaking** ALICE-Law G8 (Python package): 値を再生成する形式が再生成の算術の識別子 `"semantics"` を持つ (`alice_zip/semantics.py`) 識別子は native 拡張を使う時はその `alice_zip::law::SEMANTICS_ID` (`libalice.semantics_id()`、無い拡張は使わない)、pure Python では numpy の生成器の出力の pin を畳み込んだ `PYTHON_SEMANTICS_ID` procedural の `ALICE_ZIP` payload と残差 file (`ResidualData`、版 5 から 7 = layout 2 から 4 に識別子を足したもの) は書く時に記録し、読む時に比べる `ALICE_IMG` / `ALICE_AUD` / `ALICE_VID` の読み込みは numpy の識別子と比べる 別の識別子は両方を示す `ValueError`、識別子の無い file は `LegacySemanticsWarning` を出して読み、`ALICEZip.STRICT_SEMANTICS` / `strict=True` では拒否する 3.0.0 より前の読み手は版 5 から 7 の残差 file を読まない

## [0.10.0] - 2026-10-12

### Added
- `ContainerView::law_ids_for_this_arithmetic` と `ContainerError::NotThisArithmetic`: header の semantics id を実行中の `alice-det-math` が再現できる時だけ (`check_semantics(..).is_reproducible()`) `LIDS` の行を返す、法則の識別子をこの build の bit についての主張として使う読み手 試験 `tests/container_arithmetic.rs` (0.9.0 以前の識別子・参照実装の fixture の識別子で押した container は開けるが、この読み手と `verify_law_ids` は行を比べる前に拒否する、実行中の識別子なら通る、`LawIdEntry` を返す公開関数はすべて信用するか否かを分類し、信用する読み手が `check_semantics` を呼ぶことを検査 source の走査は改行を LF にそろえ、複数行の signature も読む)
- PyPI への公開 workflow を 2 本足した (trusted publishing、token も secret も置かない、publish する job だけが `id-token: write` を持ち、environment は `pypi`) `.github/workflows/pypi.yml` は tag `py-alice-zip-v<版>` で Python package `alice-zip` (pyproject.toml) の sdist と wheel を、`.github/workflows/pypi-libalice.yml` は tag `py-libalice-v<版>` で wheel `libalice` を Linux (x86_64 / aarch64) / macOS (universal2) / Windows (x64) の各 interpreter 向けと sdist を作って出す tag の版が pyproject.toml / libalice/Cargo.toml の版と違えば失敗する (`scripts/pypi_tag_check.py`、試験 `scripts/test_pypi_tag_check.py` を CI と preflight で走らせる) どの job にも上限時間 (wheel の build は 45 分、他は 15 分) を付けた 公開前に archive ごとの版 (名前が tag の版と完全に一致)、Apache-2.0 と `LICENSE-APACHE` / `NOTICE` の同梱、`twine check --strict` を見る tag `v<版>` と `alice-zip-cli-v<版>` は crates.io の crate で、PyPI には何も出さない

### Changed
- README (英語 / 日本語): Python package の入手先を PyPI (`alice-zip`、任意の native 高速化 `libalice`) に直した (`alice-zip` 2.0.0 と `libalice` 3.0.0 は PyPI に公開済で、「PyPI には未公開」の記述が古いままだった)
- **Behavior change** `ContainerView::verify_law_ids` は行を `law_ids_for_this_arithmetic` で読み、別の算術で押した container を `recompute` を呼ぶ前に `NotThisArithmetic` で拒否する (以前は旧い識別子の行をそのまま再計算と比べた) `Container::from_bytes` / `law_ids` / `law_ids_unchecked_semantics` は従来どおり構造だけを見るので、どの算術で書いた file も読める
- **Behavior change** `alice-det-math` を 0.5 に上げた `SEMANTICS_ID` (再公開) が `d2209b30…` から `135c407d…` に変わり、それを混ぜる `SignalLaw::law_id` の値もすべて変わる 0.5.0 で出力 bit が変わったのは `sin` / `cos` / `sin_cos` の零点付近・`-0`・NaN の扱い・`powf(1, y)` 等で、この crate の golden の数値 (sinusoid / spectrum / entropy / polynomial / line law の評価) は変わらない (旧識別子を渡した `law_id` の場面が旧 digest を再現することを確かめた) 再記録した pin は `tests/determinism_golden.rs` の識別子と `law_id` の 2 つ
- CI: `scripts/fuzz_runs.py` が sanitizer の報告 (`==N==ERROR: AddressSanitizer` / LeakSanitizer の `SUMMARY: AddressSanitizer: … leaked`) を crash と読まず、`Done N runs` の行があれば ok としていた crash の pattern を `^==\d+== ?ERROR: |^SUMMARY: (?:libFuzzer|\w+Sanitizer)|panicked at ` に広げ、ASan / LSan / UBSan の行の試験を足した (ALICE-SDF / ALICE-DetMath と同じ検査器)
- CI: fuzz の time-boxed run を gate にした (`continue-on-error` を外した) crash も、1 件も入力を実行しなかった run も red になり、target ごとの実行回数・秒数・結果を job summary に出す (`scripts/fuzz_runs.py` が libFuzzer の log の `Done N runs` を読む、試験 `scripts/test_fuzz_runs.py`) crash の入力は artifact に上げる 直近 15 回の fuzz run (3 日分) で run step が失敗していたものは無く、手元の 60 秒 × 7 target も crash 0 (実行回数 45,000〜2,400,000)

## [0.9.0] - 2026-10-11

### ライセンス

- 本 release から Apache License 2.0 だけ (Rust crate `alice-zip` 0.9.0、libalice 3.0.0 の crate `alice-zip-cli` と wheel `libalice`、Python package `alice-zip` 2.0.0、bindings) `LICENSE-MIT` を削除し、`NOTICE` を足した 再配布には `LICENSE-APACHE` と `NOTICE` を添える (application の画面に出す必要は無い) crate と wheel / sdist にも両方を入れる (`libalice/` には同じ内容の写しを置く、`scripts/license_check.py` が一致を確かめる)
- それより前の版は出した時のライセンスのまま: crates.io の `alice-zip` 0.8.0 までは MIT OR Apache-2.0、crates.io の `alice-zip-cli` (libalice) 2.6.0 までと git の libalice 2.7.0 までは MIT、git から入れる Python package (PyPI には無い) の 1.0.0 までは MIT
- Apache-2.0 は AGPL-3.0 の作品に組み込めるので、AGPL-3.0 で配布している依存側 (ALICE-DB 等) はそのまま依存できる 商標は `TRADEMARK_NOTICE` のとおりでライセンスによらない
- `scripts/license_check.py` (CI と preflight): manifest と pyproject のライセンス、`license-files`、`__license__`、`NOTICE` / `LICENSE-APACHE` の有無と写しの一致、`LICENSE-MIT` が無いこと、README の表示を確かめる 比べた件数が 0 なら失敗

### 破壊的変更 (package ごと)

版: Rust crate `alice-zip` 0.8.0 → 0.9.0、Python package `alice-zip` 1.0.0 → 2.0.0、libalice (crate `alice-zip-cli` と Python の wheel `libalice`) 2.7.0 → 3.0.0 各項目の詳細は本 file の Changed (破壊的変更) / Fixed と [libalice/CHANGELOG.md](libalice/CHANGELOG.md)

- **Rust crate `alice-zip` 0.9.0**
  - 公開 API の削除や型の変更は無い (`cargo semver-checks` を 0.8.0 に対して実行し、必要な版の更新は無し) `ContainerError` は `#[non_exhaustive]` で、variant を足しただけ
  - 0.x の caret (`"0.8"`) は 0.9.0 を受け取らないので、依存する側は `alice-zip = "0.9"` に上げる
  - `lzma` feature は `crc` crate に直接依存する (`lzma-rs` を通して既に依存木にあった)
- **Python package `alice-zip` 2.0.0**
  - `decompress_residual` は保存された float32 の bit のまま float32 で返す 記録 dtype を適用するのは `reconstruct` だけ
  - ResidualData の `delta` は `base_value` が無ければ `ValueError` 書き手は `bitdelta` で書く
  - ResidualData の JSON header: `"version"` は整数の 1〜4 だけ、鍵は `method` / `shape` / `dtype` / `quant_bits` / `version`
  - procedural payload は書き手が記録した dtype で返る (以前は常に float32)
  - `compress(quantize_residual=8 / 16)` は偶数への丸め (以前は切り捨て)
  - `compute_residual` の NaN と `reconstruct` の整数の飽和は規則で決まる (以前は環境で変わった)
  - `compress(enable_lossless=True)` は元データから residual を書き、以前の結果は `decompress` が拒否する (`allow_approximate=True` で近似として読める)
  - ALICE_ZIP の読み手は書き手が出したことのない値を拒否し、lossless な payload は `original_size` と `original_hash` に照合する
  - LZMA fallback の読み手は `original_size` を超える展開・`MAX_OUTPUT_SIZE` を超える `original_size`・xz の後ろの byte・CRC-64 以外の check・64 MiB を超える辞書を拒否する
- **libalice 3.0.0** (`alice-zip-cli` crate と wheel `libalice`)
  - wheel の版は crate の版と同じになり、2.4.0 から 3.0.0 に進む (2.5〜2.7 の wheel は存在しない)
  - 詳細は [libalice/CHANGELOG.md](libalice/CHANGELOG.md) の Changed (破壊的変更): ResidualData の header を型のまま読む、header の鍵を Python にそろえる、`AliceFileHeader::from_bytes` と `.alz` header が出したことのない値を拒否する、`FormatError` / `ResidualError` の variant の追加

### Removed

- `alice-zip-enterprise` (未公開) の `security::ENCRYPTED_MAGIC` を削除 暗号化した archive はこの magic を書かず、読む側もこの値を見ていなかった (どの repo にも参照が無い) 暗号化した archive の形式を識別する値として残ると、実在しない形式を示す

### Fixed

- Python package の LZMA fallback の読み手 (`ALICEZip.decompress`) は xz を上限なしで展開していた 61 KB の payload が 400 MiB に膨らみ、header の `original_size` (4096) との不一致は展開し終えてから分かった `max_length = original_size + 1` で展開し、超えれば `ValueError`、`original_size` が `MAX_OUTPUT_SIZE` (既定 4 GiB、instance ごとに下げられる) を超えれば展開せずに `MemoryError` xz stream の後ろの byte (stream padding、2 つ目の stream、ごみ) と CRC-64 以外の check も拒否する (書き手は出さない) 辞書の大きさも `MAX_LZMA_DECODER_MEMORY` (80 MiB、辞書 64 MiB まで) で抑える (header は 4 GiB の辞書を要求できる、書き手は 8 MiB) 同じ payload の traced allocation の最大は 8 MiB の辞書を含めて 16 MiB 未満
- Python package の `decompress_residual` は none / lzma / zlib の residual を float64 に変換せず、保存された float32 の bit のまま返す float64 を経由すると signaling NaN が quiet NaN に変わっていた (ランダムな bit パターンでおよそ 1000 個に 2 個) header の `shape` と `original_len` が食い違えば `ValueError`
- Python package の ResidualData の delta は、先頭の差分を 0 にしていたため先頭の値を失い、全体がその値だけずれて戻っていた ([5.0, 5.5, 6.0, 4.0] → [0.0, 0.5, 1.0, -1.0]) 書き手は新しい method `bitdelta` で書く: 隣り合う f32 の bit パターンの差を wrapping な uint32 で持ち (先頭は先頭の bit パターン)、xz で圧縮する 整数の演算なので NaN の payload・非正規化数・±0・無限大を含めてどの値も bit 単位で戻る (ランダムな bit パターン 10,000 個と特殊値の往復の試験で確かめている) 旧 `delta` は `base_value` の記録があれば (Rust の旧 writer) 読み、無ければ `ValueError` (復元できないので元データから圧縮し直す) 差分だけが要る場合は `decompress_delta_differences` で取り出せる `decompress_residual` は記録 dtype によらず float32 で返す 記録 dtype は元データの型で、適用するのは `reconstruct` だけ (偶数への丸め、型の範囲で飽和、cast) 記録 dtype が実数の 11 種以外の header は読む時に `ValueError` 移行: `decompress_residual` の戻り値が元の dtype であることを前提にしていた呼び出し側は `reconstruct` を使う (residual を整数 dtype へ cast すると切り捨てになり、int16 で約半数の値の復元が 1 ずれる)
- ResidualData の JSON header の `"version"` は整数だけを版として受け付け、float (`2.0`) や文字列 (`"2"`) は拒否する (Python は `ValueError`、Rust は `ResidualError::InvalidHeader`) 書き手は整数で書く これまで Python は `2.0` を、Rust は `"2"` を版 2 として読み、両者の判定が分かれていた
- Python package の procedural payload の復元が、書き手が payload に記録した dtype で返る 4 つの生成器 (Perlin / Fourier / sine / polynomial) が結果を常に float32 に変換していたので、float64 の入力が float32 で返り、byte 数が header の `original_size` の半分になっていた 形式は変わらない (dtype は以前から payload の `params.dtype` にある) payload の dtype は書き手が記録しうる実数の 11 種 (`float16/32/64`、`int8〜64`、`uint8〜64`) だけを受け付け、それ以外 (complex、文字列型、未知の名前) は `ValueError` 復元した byte 数が header の `original_size` と違えば、procedural でも `ValueError`

### Added
- CI: README の比率の出どころの `compression_ratio` example を release で実行し、law + xor residual の表の全 4 行が bit 一致であることを確かめる (`scripts/example_check.py`、表が無い・行が足りない・一致しない行があれば失敗) 手元では release で約 80 秒
- CI: 追跡している Python の未定義の名前と上書きを ruff 0.14.0 (F821 / F811) で検査する (preflight も同じ、検査した file が 0 件なら失敗) 改名した試験関数の古い名前を `__main__` の block が呼んでいたのが見つかった 試験 file の独自の `__main__` runner は除き、試験関数の無かった `tests/test_quant_simple.py` は pytest の試験にした
- pytest は数値の警告 (範囲外の cast・桁あふれ)、閉じていない資源、報告できない例外も失敗にする (pyproject の filterwarnings) 写真の試験はネットワークを使わない決定的な画像にした 以前は外部の site から取得して 403 で失敗し、追跡していた `test_images/mandrill.jpg` と `tests/test_images/mandrill.jpg` は画像でなく HTML だった (両方外した) 画像の試験は `tests/test_image_shaped_data.py` で、正弦波を並べた画像で procedural の経路も確かめる
- CI: default feature を外した build で全 target に clippy をかける (preflight も同じ引数) `compression_ratio` example は `required-features = ["lzma"]` を宣言した (default を外すと compile できなかった) pytest は試験関数が判定を `return` すると失敗にする (`PytestReturnNotNoneWarning` を error に、pyproject の設定) 試験の警告 2 種 (範囲外の float を int16 に cast した入力、signaling NaN を float64 に広げる変換) を直した
- CI: Python の試験を 3.9 と 3.12 で走らせる (宣言している最小の版を試験する) `scripts/version_check.py` が libalice の版が Cargo.toml だけにあることと、`alice_zip.__version__` が pyproject の版と同じことを 3 OS で確かめる (比べた件数が 0 なら失敗)
- README / README_ja に Python の lossless モードの保証 (bit 単位の復元、residual の version 4、lzma への退避、以前の結果の扱い) を書き、`scripts/claim_check.py` は Python の試験名も照合する Python package は PyPI に未公開なので、入れ方を `pip install git+https://github.com/ext-sakamoro/ALICE-Zip` に直した (`pip install alice-zip` と書いていた)
- Python package の `ResidualCompressor.compress_original(original, generated)`: 元データと generated から、`reconstruct` で bit 単位で元に戻る residual を書く residual は元データの精度で持つ (float64 / int32 / uint32 / int64 / uint64 は float64) 元か generated が有限でない位置 (NaN・無限大) と、規則で復元できない位置は例外として元の要素をそのまま持ち、residual と一緒に圧縮する 差分では無限大や NaN の bit を運べず、inf - inf の NaN の符号のように環境で結果が変わる経路もあったため 例外か float64 の residual を持つ file は version 4 (`residual_dtype` と `exceptions`)、以前の読み手は version で拒否する `exceptions` は version 3 と 4、`residual_dtype` は version 4 でだけ読み、それ以外の version に付いていれば拒否する header は平らな object とし、同じ鍵が 2 回ある header、object の値、配列の中の配列・object は拒否する (鍵の重複を以前は後の値で読んでいた) 既知の鍵は使わない場面でも型を見る 鍵 × JSON の値の種類 × version の組 (375 行) を仕様から生成した `tests/data/residual/header_gating.txt` で両方の読み手を照合する `decompress_residual` は residual の精度 (float32 / float64) で返す libalice の `residual::compress_original` と同じ file を書く header の数の欄は整数 (base_value / min_val / scale は数) だけを受け付け、bool (Python では int の部分型) や `8.0` の形は拒否する 既知の挙動: 以前の版の読み手は、version 2 のまま exceptions の鍵と末尾の block を持つ file (どの書き手も作らない) を例外を落として読む この版の読み手は拒否する

- `container`: 複数の payload をそれぞれの SHA-256 で識別して 1 つにまとめるコンテナ 56 byte の header (非 ASCII の byte で始まる 8 byte の magic、major / minor 版、semantics id)、section の表 (tag、critical flag、offset、長さ、SHA-256)、file 全体の SHA-256 を末尾に置く `Container::id` は header と表を domain を分けて hash する 読み手は別の major 版、未知の critical section、予約 bit、隙間や余分な byte、合わない末尾や payload の hash を拒否し、未知の critical でない section は保つ `SREF` section は section を SHA-256 で参照し (無い参照は拒否)、`LIDS` section は法則の識別子を並べ、header の semantics id と、`ContainerView::verify_law_ids` で計算し直した識別子に照合する `read_any` は ALICE_ZIP の版 1.0 / 1.1 も読み、書き手が出したことのない値は拒否する `parse_legacy_alice_zip_header` は header を `LegacyHeader` で返し、`LegacyHeader::verify_original` は元データとして渡された byte を `original_size` と、記録があれば (全 0 でなければ) `original_hash` に照合する `decompress_legacy_alice_zip(bytes, limit)` (`lzma` feature) は LZMA fallback の payload (`payload_type` 0x30、Python の書き手が出す `meta_len` ‖ `{"shape":[..],"dtype":".."}` ‖ xz) を復元し、`verify_original` で照合して `LegacyArray` (shape、dtype、byte 列) を返す `original_size` が `limit` を超えれば読む前に `LegacyTooLarge` 復元の前に xz の容器 (stream 1 つ、CRC-64、辞書 64 MiB 以下の LZMA2 だけの block 1 つ、index、footer、後ろに何も無い) を自前で検査し、LZMA2 の chunk header が述べる展開後の大きさの和が `original_size` と一致することを確かめるので、数 byte で数 GB に膨らむ payload も展開せずに拒否する (61 KB が 400 MiB に膨らむ試験で allocation の最大は 1 MiB 未満、検査を外す変異では 805 MB) 展開中は復号器の buffer と返す byte 列の 2 つを持つので、最大のメモリは `original_size` (高々 `limit`) のおよそ 2 倍 書き手の形でない metadata、byte 順の接頭辞付きの dtype (`>f8` 等) と大きさの分からない dtype、shape と dtype から決まる byte 数との不一致は拒否する procedural / media / texture の payload は生成器の parameter なので復元せず `UnsupportedPayload` (Rust には生成器の復元が無い) 試験は Python の書き手が作った 5 file (uint8、2 次元 float64、0 次元、空、block padding のある 5 byte) を元の byte と比べ、容器の全 bit の反転が拒否されることを確かめる `ContainerView` は payload の hash を読む時だけ確かめる
- `tests/container_oracle.rs` (25 本): byte と識別子が独立した参照の書き手 `tests/data/container/container_ref.py` と一致する fixture の 1 bit の変更はすべて、その位置を受け持つ検査で拒否される 退化した入力は定めた error を返す
- `examples/container_roundtrip.rs`
- CI: コンテナの oracle を各 OS の `no_std` build と `wasm32-wasip1` 上の `wasmtime` で走らせる wasm の job は通った試験が 0 件なら失敗にする
- `container::LegacyHeader::original_hash_checkable` — ALICE_ZIP の payload を復元すると元データが再現されるか (`original_hash` を照合できるか) LZMA fallback (`0x30`) だけが true Python の `alice_zip.core.original_hash_checkable` と同じ規則で、両方の試験が同じ表 (`tests/container_oracle.rs` の `CHECKABLE`) を読む

### Changed (破壊的変更)
- libalice (Python の wheel) の版は `libalice/Cargo.toml` の版を使う (`libalice/pyproject.toml` は `dynamic = ["version"]`) pyproject に 2.4.0 と書いたまま Cargo.toml が 2.7.0 に進んでいたので、wheel の版は crate の版と同じになる (以前の wheel は 2.4.0 のまま、crate は 2.5.0〜2.7.0 に進んでいた 本 release で crate とともに 3.0.0 になるので、wheel の番号は 2.4.0 から 3.0.0 に進み、2.5〜2.7 の wheel は存在しない) libalice の `requires-python` は CI で試験している最小の 3.9 にした (`>=3.8` と書いていた)
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

[Unreleased]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.6.0...v0.7.0
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
