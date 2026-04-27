# バージョン履歴

## v0.2.0 (2026-04-27)

### 新機能
- **PyMOL互換 show/hide**: 原子単位のビットマスク (`atom_rep_show`) による表現制御。`show ribbon, chain A` で A 鎖のみリボン表示など、選択式での部分表示が可能に
- **ポリマー分類 Method F**: SEQRES ベース (Method C) + ペプチド結合/ホスホジエステル結合連結性 (Method D) のフォールバックで MSE 等修飾アミノ酸も正確に分類
- **`light` コマンド**: 光源の仰角・方位角・強度をランタイムに変更可能
  - `light intensity 1.5` / `light elevation 45` / `light azimuth 30`
- **egui ツールバー**: ウィンドウ下部に 2 つのプリセットボタンを追加
  - 「初期表示」: リボン (タンパク質) + Ball-and-stick (リガンド) + SS カラー
  - 「Chain Surface」: チェーン別カラーの Gaussian サーフェス + リガンド Ball-and-stick

### 改善
- ライティング全シェーダ刷新: Half-Lambert バイアス強化・specular 強化でより立体感のある表示に
- サーフェスシェーダ: 高コントラスト設定 (影を暗く、ライト面を明るく)
- PDB 読み込み時のチェーン情報表示: 残基数 → 残基番号範囲 (例: `1-450`) に変更
- egui_wgpu::renderer の既知の警告を抑制

---

## v0.1.0 (2026-04-01)

### 機能
- **PDB 読み込み**: ATOM / HETATM / HELIX / SHEET / CONECT / COMPND / HETNAM / HETSYN / SEQRES レコード対応
- **表現形式**:
  - Ball-and-stick (原子球 + 結合シリンダー、半結合カラー)
  - Ribbon (Catmull-Rom スプライン、2 次構造断面形状)
  - Backbone (Cα trace チューブ)
  - Gaussian Surface (Marching Cubes、rayon 並列化)
- **カラーリング**: CPK 元素色 / 2 次構造色 / チェーン色 / スペクトル / B 因子
- **カメラ**: Arcball 回転 (左ドラッグ) / 平行移動 (右ドラッグ) / ズーム (スクロール)
- **コマンドシステム** (`rusmol>` プロンプト + `-c` バッチ実行):
  - `load` / `select` / `show` / `hide` / `color` / `enable` / `disable` / `delete`
  - `zoom` / `reset` / `bg` / `quit`
  - 選択言語: `chain` / `resn` / `resi` / `name` / `elem` / `hetatm` / `all` + `and` / `or` / `not`
- **マウスピッキング**: カラー ID ベース、Ball-and-stick は原子単位・Ribbon/Surface は残基単位
- **残基ハイライト**: クリック選択でオレンジリム照明
- **結合推定**: CIF ベース (RCSB CCD キャッシュ) + ペプチド/ジスルフィド/核酸バックボーン結合
- **プラットフォーム**: macOS (Metal バックエンド)、シングルバイナリ配布
