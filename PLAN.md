# rusmol 実装計画

## Context

タンパク質/DNAの立体構造を表示するコマンドラインツール「rusmol」を新規開発する。
PyMol互換のコマンドセットを持つ軽量なビューアで、構造生物学における結合ポーズ解析が主な用途。
Rust + wgpu で実装し、1秒以内のコールドスタートとシングルバイナリ配布を実現する。

**スコープ決定事項**:
- macOSのみ対応（Metal バックエンド）
- 小〜中規模分子（〜50,000原子）
- PDB形式のみ（mmCIF等は後からgemmi統合時に追加）
- 画像エクスポートは初期リリースに含めない

---

## プロジェクト構造

```
rusmol/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── app.rs                 # ApplicationHandler実装、状態管理、イベント処理
│   ├── cli.rs                 # clap によるCLI引数定義 (-c, -v, -V, -h)
│   ├── command/
│   │   ├── mod.rs             # Command / ColorSpec enum定義
│   │   ├── parser.rs          # 手書きコマンドパーサ (nom不使用)
│   │   ├── executor.rs        # コマンド実行ディスパッチ
│   │   ├── selection.rs       # 選択言語パーサ・評価エンジン
│   │   └── prompt.rs          # rustylineインタラクティブプロンプト (lsmol>)
│   ├── structure/
│   │   ├── mod.rs
│   │   ├── atom.rs            # Atom, ResidueId, Structure, SecondaryStructure
│   │   ├── pdb.rs             # PDB固定幅カラムパーサ (ATOM/HETATM/HELIX/SHEET/CONECT)
│   │   ├── bonds.rs           # CIFベース結合推定 + CONECT + ペプチド/SS/核酸結合
│   │   ├── ccd.rs             # RCSB CCD キャッシュ (~/.cache/rusmol/ccd/)
│   │   └── secondary.rs       # SsRange, assign_ss() → per-atom SecondaryStructure
│   ├── render/
│   │   ├── mod.rs
│   │   ├── state.rs           # wgpu State (pipelines, upload_scene, render)
│   │   ├── camera.rs          # Arcballカメラ (クォータニオン回転)
│   │   ├── uniform.rs         # Uniformバッファ (view_proj, light_dir, camera_pos)
│   │   ├── ball_stick.rs      # SphereInstance, CylinderInstance, icosphere, gen_cylinder
│   │   ├── ribbon.rs          # RibbonVertex, build_ribbon()
│   │   ├── surface.rs         # build_surface() — Gaussian density + Marching Cubes
│   │   ├── picker.rs          # (Phase 5)
│   │   ├── label.rs           # (Phase 5)
│   │   └── shaders/
│   │       ├── main.wgsl      # 球体シェーダ
│   │       ├── cylinder.wgsl  # シリンダーシェーダ (Rodrigues回転)
│   │       ├── ribbon.wgsl    # リボンシェーダ (両面 front_facing)
│   │       ├── surface.wgsl   # サーフェスシェーダ (勾配法線、front_facing 不使用)
│   │       ├── pick.wgsl      # (Phase 5)
│   │       └── label.wgsl     # (Phase 5)
│   ├── scene/
│   │   ├── mod.rs             # Scene, selections (indexmap で挿入順保持)
│   │   └── object.rs          # MolecularObject, RepresentationType, Representation
│   └── util/
│       ├── mod.rs
│       └── color.rs           # cpk_color, vdw_radius, ss_color, chain_color
└── tests/
    ├── pdb_parser_test.rs
    └── fixtures/
        └── 2je5.pdb           # テスト用PDB (ペニシリン結合タンパク質+L4C, 7239原子)
```

## 依存クレート

```toml
[dependencies]
wgpu        = "24"
winit       = "0.30"
pollster    = "0.4"
glam        = { version = "0.29", features = ["bytemuck"] }
bytemuck    = { version = "1", features = ["derive"] }
clap        = { version = "4", features = ["derive"] }
rustyline   = "15"
nom         = "7"          # Cargo.tomlに存在するが現在未使用
log         = "0.4"
env_logger  = "0.11"
anyhow      = "1"
crossbeam-channel = "0.5"
ureq        = "2"          # RCSB CCD ダウンロード (blocking HTTP)
glyphon     = "0.8"        # テキストレンダリング (Phase 5)
indexmap    = "2"          # Scene オブジェクトの挿入順保持

[profile.release]
opt-level = 3
lto       = true
strip     = true
```

---

## アーキテクチャ

### スレッドモデル

```
Main Thread (macOS GUI要件)        Worker Thread
┌────────────────────────┐       ┌─────────────────────┐
│ winit EventLoop        │       │ rustyline loop       │
│  ├─ WindowEvent処理    │ cmd_rx│  lsmol> コマンド入力 │
│  ├─ Camera更新         │◄──────│  → parse → Command   │
│  ├─ try_recv(cmd_rx)   │       │  ← 結果表示          │
│  ├─ Scene状態更新      │──────►│                      │
│  └─ wgpu描画           │resp_tx│                      │
└────────────────────────┘       └─────────────────────┘
```

- `crossbeam-channel` で `Command` / `CommandResponse` をやり取り
- `about_to_wait` 内で `cmd_rx.try_recv()` ポーリング → `scene_dirty` フラグ
- `scene_dirty = true` のとき次フレームで `upload_scene` → `request_redraw`

### コマンド体系

| コマンド | 書式例 | 説明 |
|---------|--------|------|
| `load` | `load path/to/file.pdb [, name]` | PDB読み込み |
| `select`/`sel` | `select mysel, chain A and resn ALA` | 選択セット作成 |
| `show` | `show ribbon [, sel]` | 表現を表示 |
| `hide` | `hide ball_stick [, sel]` | 表現を非表示 |
| `color`/`colour` | `color red [, sel]` | 色変更 |
| `enable` | `enable objname` | オブジェクト全体を表示 |
| `disable` | `disable objname` | オブジェクト全体を非表示 |
| `delete`/`del` | `delete objname` | オブジェクト削除 |
| `zoom`/`z` | `zoom [sel]` | 選択にフィット |
| `reset` | `reset` | カメラをデフォルトに戻す |
| `quit`/`q`/`exit` | `quit` | 終了 |

**表現名エイリアス**: `ball_stick`, `ball-stick`, `bs`, `sticks`, `stick`, `ball_and_stick`, `spheres` → BallAndStick / `backbone`, `trace`, `ca_trace`, `ca` → Backbone / `ribbon`, `cartoon` → Ribbon / `surface` → Surface / `lines`, `line`, `wire` → Lines

**色名**: `element`/`cpk`, `chain`/`chainbows`, `ss`/`secondary`/`secondary_structure`, および named colors: `red`, `green`, `blue`, `white`, `black`, `yellow`, `orange`, `purple`/`violet`, `magenta`, `cyan`, `grey`/`gray`, `pink`, `salmon`, `wheat`, `teal`, `marine`, `forest`, `limon`

### 選択言語

```
expr  ::= expr "or"  expr
        | expr "and" expr
        | "not" expr
        | "(" expr ")"
        | primitive

primitive ::= "all" | "*"
            | "hetatm"
            | "name"    <atom_name>
            | "resn"["ame"]  <resname>
            | "resi"["_num"|"num"]  <ranges>   e.g. 1-10+15+20-30
            | "chain"   <char>
            | "elem"["ent"]  <element>
            | <object_name>   (オブジェクト名フィルタ)
```

選択セットは `scene.selections` (HashMap) に名前付きで保存。`sele` がデフォルト名。

### レンダリング定数

| 定数 | 値 | 用途 |
|------|----|------|
| `BOND_RADIUS` | 0.15 Å | 共有結合シリンダー半径 |
| `BACKBONE_TUBE_RADIUS` | 0.30 Å | Cα trace チューブ半径 |
| `BACKBONE_JOINT_RADIUS` | 0.36 Å | Cα 位置のジョイント球半径 |
| 球体スケール (通常) | VdW × 0.32 | Ball-and-stick 原子球 |
| 球体スケール (水分子) | VdW × 0.14 | HOH/WAT/DOD の小球 |
| Icosphere 分割数 | 2 | 球体メッシュ品質 |
| シリンダーセグメント数 | 32 | 円断面の分割数 |

**ライティング**: Half-Lambert diffuse (`N·L * 0.5 + 0.5`) + Blinn-Phong specular (shininess=64, weight=0.40) + ambient=0.10。光源はカメラ空間固定 (`camera.rotation * (1,2,3).normalize()`)。背景色: 黒。

### デフォルト表示

| 対象 | 表現 | 色 |
|------|------|----|
| タンパク質 (ATOM) | Ribbon | 2次構造色 (ヘリックス=赤, シート=黄, コイル=明灰) |
| リガンド (HETATM 非水) | Ball-and-stick | CPK元素色 |
| 水分子 (HOH/WAT/DOD) | 非表示 | — |

Ribbon + BallAndStick が同時に有効な場合: BallAndStick は HETATM 非水原子とその結合のみ描画。

---

## フェーズ別実装計画

### ✅ Phase 1: 基盤 — PDB読み込み + 球体表示 + カメラ操作

- wgpu 初期化 (Metal backend)、Surface 設定、Depth buffer
- Icosphere インスタンス描画 (位置・半径・色をインスタンスバッファ)
- Arcball カメラ: 左ドラッグ=回転, 右ドラッグ=平行移動, スクロール=ズーム
- Blinn-Phong + Half-Lambert シェーダ
- PDB パーサ: ATOM/HETATM 固定幅カラムパース、元素記号推定
- CPK 元素色テーブル、VdW ラジウステーブル
- 座標原点を分子全体の重心に移動

### ✅ Phase 2: Ball-and-Stick + コマンドシステム

- **結合推定 (CIF ベース)**:
  - 標準 20 アミノ酸 + MSE/SEC/HYP/CME/MLY + SO4/PO4/GOL/EDO/ACT/ACY + 単原子イオン (CL/ZN 等) + HOH のビルトインテーブル
  - 未知残基: CONECT があればダウンロードしない; なければ RCSB CCD からダウンロード → `~/.cache/rusmol/ccd/` にキャッシュ
  - ペプチド結合: 残基番号の連続性 (`is_consecutive`、挿入コード対応) で判定; C-N 距離 1.1〜2.0Å 外なら警告
  - ジスルフィド: CYS/CYX の SG-SG < 2.30Å
  - 核酸バックボーン: O3'→P < 1.70Å
  - CONECT レコードを最後に追加 (重複チェック付き)
- **描画**:
  - シリンダーインスタンス描画 (32 セグメント、Rodrigues 回転で任意軸整合)
  - 半結合 (half-bond): 各原子色で中点まで描画
  - 水分子: 酸素球を VdW × 0.14 の小球で表示
- **コマンドシステム**:
  - 手書きパーサ (nom 不使用)
  - `select`, `show`, `hide`, `color`, `enable`, `disable`, `delete`, `zoom`, `reset`, `quit`, `load`
  - 選択言語: `name/resn/resi/chain/elem/hetatm/all` + `and/or/not` + `()`
  - `-c "cmd1; cmd2"` による起動時バッチ実行

### ✅ Phase 3: Backbone + リボン描画 + 2次構造カラーリング

- **Backbone (Cα trace)**:
  - Cα 原子をチェインごとに (seq_num, ins_code) でソート
  - 隣接 Cα 間を半結合シリンダー (BACKBONE_TUBE_RADIUS=0.30) で接続
  - 各 Cα 位置にジョイント球 (BACKBONE_JOINT_RADIUS=0.36)
- **2次構造パース**:
  - HELIX/SHEET レコードを PDB 固定幅カラムでパース
  - `SsRange { chain, start_seq, start_ins, end_seq, end_ins, ss }` のリストを構築
  - `assign_ss(atoms, ranges) → Vec<SecondaryStructure>` で per-atom SS 割り当て
  - `SecondaryStructure` enum (Coil/Helix/Sheet) を `Structure.ss` に格納
- **リボン描画**:
  - 全原子スキャンで per-chain CA/O インデックス収集 (chain_ranges 不使用: HETATM によるrange上書きバグ回避)
  - Cα 間距離 > 5Å でチェーンブレーク分割
  - Catmull-Rom スプライン (N_SUB=8 補間点/セグメント)
  - O 原子ベクトル (CA→O 方向) でリボン平面決定; β-シートの反転を一貫性チェックで修正
  - 断面形状 (N_PROF=12): ヘリックス (1.2×0.35), シート (1.6×0.28), コイル (0.22×0.22)
  - 楕円断面の外向き法線: `(cos/a, sin/b)` 正規化
  - `cull_mode: None` + `@builtin(front_facing)` で両面ライティング
- **2次構造カラーリング**:
  - `color ss` / `color secondary` コマンド
  - ヘリックス=[0.85,0.20,0.20], シート=[0.90,0.80,0.10], コイル=[0.90,0.90,0.90]
- **デフォルト表示変更**:
  - ロード時デフォルト: Ribbon (有効) + BallAndStick (有効)
  - タンパク質原子の初期色: SS 色; HETATM 原子: CPK 色

### ✅ Phase 4: 分子表面

**ゴール**: `show surface` で分子表面が表示される ✅

**実装ファイル**:
- `src/render/surface.rs`: Gaussian 密度場 + Marching Cubes による等値面抽出
- `src/render/shaders/surface.wgsl`: 勾配法線を使った両面ライティング (front_facing 不使用)
- `src/render/state.rs`: `surface_pipeline` / `surface_vb` / `surface_ib` / `surface_index_count` フィールド追加
- `src/render/mod.rs`: `pub mod surface;` 追加

**アルゴリズム仕様**:
- グリッド解像度: STEP=0.5Å、マージン: MARGIN=3.0Å
- 密度関数: ガウス関数 `exp(-r² / (2 * σ²))`、σ=1.2Å、カットオフ=5.0Å
- 等値面閾値: THRESHOLD=0.5
- 法線: 有限差分による密度勾配の負方向（外向き）
- 頂点色: 近傍原子の密度重み付き平均色
- 水分子 (HOH/WAT/DOD) は除外
- グリッドサイズ上限: 8,000,000 セル (それ以上はスキップ)
- Marching Cubes テーブル: Paul Bourke 標準テーブル (256ケース)
- サーフェスシェーダ: 勾配法線を常時使用 (front_facing で反転しない)
- パイプライン: `cull_mode: None` (両面レンダリング)
- 描画順: ribbon の後、sphere の前

**設計メモ**:
- 大規模構造 (>10,000 原子) では数秒かかる場合あり → 同期実装
- 半透明は未対応 (不透明のみ)

### ✅ Phase 5: マウスピッキング + アトムラベル

**ゴール**: 原子クリックで情報ラベルが表示される

1. `src/render/picker.rs`:
   - オフスクリーンレンダーターゲット (R32Uint) に instance_index+1 のフラット色で描画
   - クリック座標のピクセル読み取り (`map_async` + `device.poll(Wait)` 同期)
   - `sphere_instance_map: Vec<AtomRef>` で instance index → 原子情報のマッピング
   - ウィンドウリサイズ時に pick texture/depth も再作成
2. `src/render/shaders/pick.wgsl`:
   - `@builtin(instance_index)` + `@interpolate(flat)` で flat u32 カラー ID 出力
3. `src/render/label.rs`:
   - glyphon 0.8 (Cache/Viewport/TextAtlas/TextRenderer) で 2D オーバーレイ
   - `<resname> <chain>:<resseq> <name>` 形式でラベル表示
4. `src/app.rs`:
   - press/release 間の移動距離 < 5px でクリック判定（ドラッグと区別）
   - クリック時に `render.pick_at(px, py)` → `render.set_label(...)` を呼び出し
   - 背景クリック時は `render.clear_label()` でラベル消去

**検証**: 原子クリックで「ALA A:106 CA」のようなラベルが表示される

### Phase 6: 仕上げ

1. `color spectrum` (N→C 末端グラデーション)、`color b` (B-factor カラー)
2. エラーハンドリング改善
3. コールドスタート計測・最適化 (目標: 500ms 以内)
4. verbose (-v) ログ出力整備
5. テスト追加 (PDB パーサ、選択言語、コマンドパーサ)

---

## 設計上の重要判断

| 判断項目 | 決定 | 理由 |
|---------|------|------|
| クレート構成 | シングルクレート | シングルバイナリ要件に合致 |
| 線形代数 | glam | SIMD 最適化、bytemuck 対応、コンパイル高速 |
| PDB パーサ | 手書き (nom 不使用) | 固定幅カラム形式はバイト位置スライスが最適 |
| コマンドパーサ | 手書き (nom 不使用) | シンプルな文法なので nom は過剰 |
| 結合推定 | CIF ベース (RCSB CCD) | 距離ベースより正確; ビルトインテーブルで速度担保 |
| ペプチド結合 | 残基番号連続性 | 距離ベースより確実; 距離は検証・警告のみ |
| ライティング | カメラ空間固定光源 | 分子回転時に照明が変わらず自然な見た目 |
| リボン平面 | O 原子ベクトル | 化学的に意味があり β-シート一貫性補正で安定 |
| リボン chain 収集 | 全原子スキャン | chain_ranges は HETATM で上書きされるバグがある |
| 分子表面 | Gaussian surface | SES より実装容易で視覚的に十分 |
| サーフェス法線 | 密度場の勾配 (有限差分) | 三角形ワインディングに依存しないため安定 |
| サーフェスシェーダ | front_facing 不使用 | 勾配法線は常に外向きなので反転不要 |
| ピッキング | カラー ID ベース | レイキャストより実装簡単で正確 |
| テキスト描画 | glyphon 0.8 | wgpu 上で最も実用的なテキストレンダリングクレート |
| スレッド通信 | crossbeam-channel | try_recv が確実に動作 |
