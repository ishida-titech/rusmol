# rusmol Benchmark

## 計測環境

| 項目 | 値 |
|------|----|
| 日時 | 2026-04-15 |
| マシン | Apple M1 Max |
| RAM | 32 GB |
| OS | macOS 26.2 (Sequoia) |
| Rust profile | release (opt-level=3, lto=true, strip=true) |
| バイナリサイズ | 6.5 MB |

---

## 計測方法

```
/usr/bin/time -l ./rusmol <file> -c "<commands>; quit"
```

- wall-clock (`real`) と最大 RSS を記録
- `[timing]` ログを `state.rs` の `upload_scene` および `build_surface` に埋め込み済み
- 3 回計測して中央値を採用

> **修正履歴**: 初回の計測では `-c "show surface; quit"` が `about_to_wait` の
> `cmd_rx` Quit 処理 → `scene_dirty` 処理の順序により、サーフェス計算が行われる前に
> 終了していた。`about_to_wait` の処理順を「scene_dirty → quit」に修正して再計測。

---

## コールドスタート（表面なし）

測定: `rusmol <file> -c quit`

| ファイル | atoms | upload_scene | real (s) | RSS (MB) |
|---------|-------|-------------|----------|----------|
| 1crn.pdb | 327 | 0 ms | 0.33 | 90 |
| 2je5.pdb | 7,239 | 3 ms | 0.33 | 100 |

---

## `show surface` 追加時

測定: `rusmol <file> -c "show surface; quit"`

### release ビルド (`--release`, opt-level=3, lto=true)

| ファイル | atoms | surface build | surface mesh | upload_scene 合計 | real (s) | RSS (MB) |
|---------|-------|-------------|-------------|-----------------|----------|----------|
| 1crn.pdb | 327 | 12 ms | 27,520 tris | 13 ms | 0.33 | 98 |
| 2je5.pdb | 7,239 | ~~283 ms~~ → **231 ms** (rayon) | 477,872 tris | 244 ms | 0.52 | 401 |

### debug ビルド (unoptimized+debuginfo)

| ファイル | atoms | surface build | surface mesh | upload_scene 合計 | real (s) | RSS (MB) |
|---------|-------|-------------|-------------|-----------------|----------|----------|
| 2je5.pdb | 7,239 | **4,604 ms** | 477,872 tris | 4,636 ms | 5.03 | 369 |

> debug ビルドでは Marching Cubes の inner loop (浮動小数演算・配列アクセス) が
> 最適化されず、release 比 **約 16× 低速**。ユーザー報告の「~5 秒」はこれと一致。

---

## 目標値との比較

| 目標 | release | debug | 判定 |
|------|---------|-------|------|
| コールドスタート ≤ 500 ms (表面なし) | ~330 ms | ~400 ms | ✅ |
| シングルバイナリ | 6.5 MB | — | ✅ |
| `show surface` (2je5, 7239 atoms) | **~520 ms** (rayon並列化後) | **~5,000 ms** | ⚠️ release は 500ms 境界付近、debug は論外 |

---

## ボトルネック分析

```
2je5.pdb + show surface (release) の内訳:
  GPU 初期化 / ウィンドウ作成 :  ~330 ms  (Metal shader compile 等)
  PDB パース + 結合推定        :   ~30 ms
  surface build (Marching Cubes, rayon): 231 ms
  GPU upload (VB/IB 転送)     :   ~12 ms
  ─────────────────────────────────────────
  合計                          ~655 ms+  (real: ~860 ms)
```

サーフェス計算がボトルネック:
- グリッド解像度 0.5 Å → 2je5 で **955,816 頂点 / 477,872 三角形**
- CPU シングルスレッドの Marching Cubes が支配的

### 最適化候補

| 手法 | 期待効果 | 難易度 |
|------|----------|--------|
| グリッド解像度を 0.8 Å に粗くする | ~5× 高速化、品質低下あり | 低 |
| rayon で Marching Cubes を並列化 | ~4–8× 高速化 (M1 Max 10コア) | 中 |
| `show surface` を非同期化（バックグラウンドスレッドで計算） | UX 改善（ブロックしない） | 高 |
