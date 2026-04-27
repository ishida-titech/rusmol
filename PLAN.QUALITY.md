# PLAN.QUALITY.md — 表示品質向上ロードマップ

> ブランチ: `quality`
> 現在のレンダリング: フォワードレンダリング（単一カラーアタッチメント + Depth32Float）
> 目標: フォトリアリスティックな分子表示（速度とのトレードオフを許容）

---

## 現状の制約

- G-buffer なし（MRT 未使用）
- ポストプロセスパスなし
- 透明度なし（アルファブレンディング未実装）
- シャドウマップなし
- ライティング: Half-Lambert + Blinn-Phong、カメラ空間固定光源

---

## 検討項目一覧

### 独立実装可能（依存関係なし）

| # | 手法 | 概要 | 実装コスト | 速度影響 | 優先度 |
|---|------|------|-----------|---------|--------|
| A | **MSAA × 4** | wgpu Texture の `sample_count: 4`、MSAA resolve パス追加 | 低（wgpu ネイティブサポート） | −10〜20% | ★★★ |
| B | **PyMOL スタイルリボン（β-arrow）** | β-シート端をフレア断面に変形、ヘリックス断面を楕円に | 中（ribbon.rs のプロファイル形状変更） | ほぼなし | ★★★ |
| C | **Sphere Impostor（Raycast 球）** | 球体を Billboard quad + フラグメントシェーダで光線交差計算、深度書き込み | 中（main.wgsl/ball_stick.rs の全面改修） | +5〜15%（三角形削減でほぼ同等） | ★★ |
| D | **Tone Mapping（Reinhard / ACES）** | フォワード出力値に tone-map 関数を適用 | 極低（シェーダ 1 行） | ほぼなし | ★★ |
| E | **Gamma 補正（sRGB 出力）** | `surface_format` を `Bgra8UnormSrgb` に変更、シェーダ内 pow(color, 1/2.2) 廃止 | 低 | ほぼなし | ★★★ |
| F | **Directional Shadow Map** | 光源視点でDepth テクスチャ生成 → メインパスで影 lookup | 高（シャドウパス追加、PCF フィルタ） | −15〜30% | ★ |

---

### G-buffer 導入後に実装可能

G-buffer（MRT: albedo + normal + depth）が前提となる手法群。

#### G-buffer 本体

| 手法 | 概要 | 実装コスト | 速度影響 |
|------|------|-----------|---------|
| **Deferred Rendering 基盤** | MRT パス（albedo `Rgba8Unorm` + normal `Rgba16Float` + depth `Depth32Float`）+ ライティングパス | 高（全シェーダ・state.rs 全面改修） | −5〜15%（draw call 削減で相殺） |

#### G-buffer 依存手法

| # | 手法 | 概要 | 追加実装コスト | 速度影響 | 優先度 |
|---|------|------|--------------|---------|--------|
| G | **SSAO** | 深度・法線バッファからスクリーン空間遮蔽を計算（半径 0.5〜2.0 Å 相当） | 中（SSAO パス + blur パス） | −10〜20% | ★★★ |
| H | **Screen-Space Reflections（SSR）** | 深度バッファを利用したレイマーチ反射 | 高 | −15〜25% | ★（分子表示では効果薄） |
| I | **PBR（GGX）** | albedo/roughness/metallic のマテリアルモデル | 高（マテリアルシステム追加） | −5〜10% | ★★ |
| J | **IBL（Image-Based Lighting）** | 環境マップ（HDR Cubemap）でアンビエント計算 | 高（Cubemap + 前処理が必要） | −5〜10% | ★（IBL は PBR 前提） |

---

### 透明度（OIT）

| # | 手法 | 概要 | 実装コスト | 速度影響 | 優先度 |
|---|------|------|-----------|---------|--------|
| K | **Weighted Blended OIT** | アルファ・深度重み付き平均（G-buffer 不要） | 中（追加アキュムレーションバッファ） | −10〜20% | ★★★ |
| L | **Per-Pixel Linked List** | 正確な順序依存透明度（GPU メモリ大量消費） | 極高 | −30〜50% | ★（過剰） |

透明度が実現すると: `show surface` + `show ribbon` の同時表示でサーフェスを半透明化、内部リボンを透かして見る PyMOL 的な表示が可能になる。

---

### ポストプロセス（G-buffer 不要）

| # | 手法 | 概要 | 実装コスト | 速度影響 | 優先度 |
|---|------|------|-----------|---------|--------|
| M | **Bloom** | 輝度しきい値 → 横ぼかし → 縦ぼかし → 加算合成 | 中（3 パス） | −5〜10% | ★★ |
| N | **Depth of Field（DOFF）** | 焦点距離・絞り設定でボケ表現 | 高（CoC 計算 + 可変ブラー） | −10〜20% | ★（分子表示では好みが分かれる） |
| O | **FXAA / SMAA** | シェーダによるアンチエイリアシング（MSAA との選択） | 低〜中 | −2〜5% | ★★（MSAA なしの場合の代替） |
| P | **Edge Outline（Sobel / Jump Flood）** | 法線不連続検出でシルエット輪郭線 | 中 | −5〜10% | ★★（PyMOL 風の輪郭には効果的） |

---

## 依存関係グラフ

```
独立 ─────────────────────────────────────────────────────────┐
  A: MSAA                                                       │
  B: PyMOL リボン（β-arrow）                                    │
  C: Sphere Impostor                                            │
  D: Tone Mapping                                               │
  E: Gamma 補正（sRGB）                                         │
  F: Shadow Map                                                 │
  K: Weighted Blended OIT                                       │
  M: Bloom                                                      │
  O: FXAA / SMAA                                               │
  P: Edge Outline                                               │
                                                                │
G-buffer（Deferred Rendering 基盤） ───────────────────────────┤
  └── G: SSAO                                                   │
  └── H: SSR                                                    │
  └── I: PBR（GGX）                                             │
       └── J: IBL（Image-Based Lighting）                       │
                                                               ─┘
```

---

## 推奨実装順序（コスパ重視）

### Phase Q1 — 低コスト・即効果（G-buffer 不要）

1. **E: Gamma 補正（sRGB）** — surface_format の変更のみ。色精度の基盤。
2. **D: Tone Mapping** — シェーダ 1 行追加。過露出を防ぎハイライトを自然に。
3. **A: MSAA × 4** — ジャギー解消。リボン・サーフェスのエッジが顕著に改善。
4. **B: PyMOL スタイルリボン（β-arrow）** — 化学的可読性の大幅向上。

### Phase Q2 — 中コスト・高インパクト

5. **K: Weighted Blended OIT** — サーフェス半透明化。PyMOL 的な透かし表示が実現。
6. **P: Edge Outline** — シルエット輪郭線で立体感強化。
7. **G: SSAO**（G-buffer 導入後） — 接触部の遮蔽で奥行き感を大幅向上。

### Phase Q3 — 高コスト・フォトリアリスティック

8. **C: Sphere Impostor** — Ball-and-stick の球体品質向上（深度精度も改善）。
9. **I: PBR（GGX）** — 物理ベースマテリアル。
10. **J: IBL** — PBR 前提の環境ライティング。
11. **F: Shadow Map** — 影の投影。
12. **M: Bloom** — ハイライト部の発光感。

---

## 各手法の詳細メモ

### A: MSAA
```rust
// wgpu の MultisampleState を変更するだけ
MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false }
// MSAA テクスチャをメインターゲットに、resolve_target に表示用テクスチャを指定
```
- Picker パスは MSAA 非対応のため別途 sample_count:1 のパスを維持する必要あり

### B: PyMOL スタイルリボン（β-arrow）
- ヘリックス: 断面を楕円形（幅 2.0 Å × 厚 0.5 Å）
- β-シート: ストランド方向にフレア（幅 2.0 Å → 末端 2.8 Å）、矢印チップ（三角形断面）
- コイル: 細い楕円（幅 0.8 Å × 厚 0.3 Å）
- 実装: `ribbon.rs` の `N_PROF` 断面生成部を SS に応じてスケール/変形

### E: Gamma 補正
```rust
// wgpu Surface 設定
let surface_format = TextureFormat::Bgra8UnormSrgb; // UnormSrgb に変更
// シェーダ内の手動 gamma は不要になる
```
- 現状は `Bgra8Unorm`（線形空間で描画して sRGB への変換なし）→ 暗部が潰れやすい

### G: SSAO（G-buffer 後）
- カーネルサイズ 64 サンプル、半径 2.0 Å（原子スケールに合わせる）
- blur パス（5×5 Gaussian）でノイズ除去
- ambient 項に乗算: `ambient *= ao_factor`

### K: Weighted Blended OIT
- Accumulation テクスチャ（`Rgba16Float`）+ Revealage テクスチャ（`R16Float`）を追加
- 不透明パス → 透明パス（accum/reveal に加算書き込み）→ Composite パス
- サーフェスのアルファを 0.6〜0.8 程度に設定して内部リボンを透かす

---

## 参考

- [LearnOpenGL: SSAO](https://learnopengl.com/Advanced-Lighting/SSAO)
- [Weighted Blended OIT (McGuire & Bavoil 2013)](http://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html)
- [wgpu examples: msaa-line, shadow](https://github.com/gfx-rs/wgpu/tree/trunk/examples)
- [PyMOL open-source ribbon](https://github.com/schrodinger/pymol-open-source)
