# GPU time per program and pass class: one steady-state run

**Setup**
- Host: RTX 5090, Wine 11.22 with vkd3d-proton GE-Proton11-14.
- Tool: the diagnostics switch `VICTORY_GPU_PASSES=1` (commits edf1bc916 and bfb22b92e).
- Timing: every draw, dispatch, indirect execution, argument clamp, copy, clear and barrier gets a GPU timestamp. The stamps are read when the submission completes, so nothing stalls.
- Figures: means over the last 150 s of a 420 s run, about 211 ms per frame.

Programs are identified by the first 16 hex digits of the BLAKE3 hash of their container.

## Totals (ms per frame)

- **GPU span of the submissions: 57.6.**
  - Timed commands: 53.9.
  - Lists not timed, and gaps between lists: 3.7.
- **Timed commands per frame:** 11,481.
- **Timestamp overhead:** ≈ 0.23 µs per command, measured; about 2.6 ms of the total.

| kind | ms/frame | count/frame | µs each |
|---|---|---|---|
| indirect dispatch | 19.31 | 924 | 20.9 |
| dispatch | 12.24 | 3,730 | 3.3 |
| argument clamp (one per indirect dispatch) | 7.33 | 924 | 7.9 |
| indirect mesh | 6.69 | 3,627 | 1.8 |
| mesh | 1.86 | 371 | 5.0 |
| argument clamp window | 1.85 | 99 | 18.7 |
| indirect draw | 1.22 | 262 | 4.7 |
| copy | 1.17 | 13 | 90 |
| draw | 1.14 | 287 | 4.0 |
| title barrier | 0.45 | 713 | 0.6 |
| other | 0.64 | ≈ 530 | — |

Indirect dispatches and their clamps take **26.6 ms, 46 % of GPU time**. That is 924 executions at ≈ 29 µs each, and no single program in this bucket is large.

### By lowering form

| form | ms/frame | commands/frame |
|---|---|---|
| compute per-lane | 13.34 | 3,006 |
| graphics per-lane | 7.25 | 3,914 |
| compute per-lane (console flow refused), waterfall | 3.63 | 229 |
| compute console flow, waterfall | 3.50 | 217 |
| compute console flow in halves | 3.20 | 317 |
| compute per-lane, waterfall | 3.19 | 320 |
| graphics pixel per-lane, waterfall | 3.18 | 358 |
| compute console flow | 1.68 | 55 |
| compute pair meetings | 0.23 | 15 |
| others / not yet worked out | ≈ 3.7 | ≈ 980 |

The BVH builder takes 1.06 ms/frame in 343 commands per frame.

## Top 25 programs

Max is the longest single command. Max values near 1 ms on programs with µs-scale means also show up for list starts and barriers, so they are system hiccups, not program cost.

The last column is host DXIL instructions per native instruction. The median over all programs is ×4.1.

| # | ms/frame | per frame | mean | max | form | program(s) | expansion |
|---|---|---|---|---|---|---|---|
| 1 | 2.15 | 1 dispatch (320×180×1 groups) | 2153 µs | 3.79 ms | console flow | 5cf00dc39042edaf | ×6.2 |
| 2 | 2.01 | 154 indirect mesh | 13.1 µs | 0.69 ms | pixel per-lane, waterfall | ef5409053e14aa82 + 94d281151ca58b4f | ×3.7 |
| 3 | 0.65 | 15 dispatch + 31 indirect | 14.1 µs | 1.08 ms | per-lane | 9146291b38ea9ce1 | — |
| 4 | 0.64 | 53 indirect mesh | 11.9 µs | 0.87 ms | pixel per-lane, waterfall | 235e56869c1ea883 + 62f761fb777fc811 | ×3.8 |
| 5 | 0.62 | 35 dispatch (1×1×1) | 17.8 µs | 1.12 ms | per-lane | f8425d3c03df62a0 | ×27 |
| 6 | 0.52 | 1 indirect dispatch | 522 µs | 1.34 ms | per-lane | bba90b2f0ff2eada | ×4.6 |
| 7 | 0.47 | 1 indirect draw | 470 µs | 0.53 ms | per-lane | 25ebc92075b62822 + 48e74d16cbfa45e3 | ×4.2 |
| 8 | 0.43 | 84 dispatch | 5.1 µs | 1.01 ms (21×1×1) | per-lane, BVH builder | 3b90165a77d95cdf | — |
| 9 | 0.40 | 14 indirect dispatch | 28.9 µs | 0.92 ms | per-lane | 3b471c122be79b3e | ×6.8 |
| 10 | 0.39 | 1 indirect dispatch | 390 µs | 1.10 ms | per-lane | 7a071f906f98311b | ×4.2 |
| 11 | 0.37 | 15 indirect dispatch | 24.8 µs | 0.99 ms | per-lane | 764ca63daec9c616 | ×7.5 |
| 12 | 0.35 | 1 dispatch (320×180×1) | 353 µs | 0.92 ms | console flow in halves | f0c91b4d2578f5bf | ×6.0 |
| 13 | 0.35 | 15 indirect dispatch | 23.5 µs | 1.09 ms | per-lane | 87bf9e3f2bf2f052 | ×5.7 |
| 14 | 0.35 | 1 indirect draw | 347 µs | 0.40 ms | per-lane | 25ebc92075b62822 + 21e71a8255a23fba | ×4.2 |
| 15 | 0.34 | 1 dispatch (320×360×1) | 339 µs | 1.34 ms | per-lane | 34fb51b8a0312e82 | ×4.4 |
| 16 | 0.32 | 5 indirect dispatch | 64.8 µs | 0.13 ms | console flow in halves | 660814153af43185 | ×4.6 |
| 17 | 0.32 | 3 indirect mesh | 106 µs | 0.62 ms | per-lane | 8321022826281144 + 659fd97066bea4fc | ×4–5 |
| 18 | 0.31 | 1 dispatch (320×180×1) | 320 µs | 0.93 ms | console flow in halves | 0a5ecfee6b79e13b | ×6.2 |
| 19 | 0.30 | 1 indirect dispatch | 316 µs | 1.12 ms | per-lane | 2dc364df5d3278b8 | ×4.1 |
| 20 | 0.30 | 1 indirect dispatch | 313 µs | 1.10 ms | per-lane | 940c2635dc1312f4 | ×3.9 |
| 21 | 0.29 | 13 mesh (299×1×1) | 23.1 µs | 0.64 ms | per-lane | f0c7efd2aceb2aaa + 7d85c9cab3c80721 | ×3.5 |
| 22 | 0.28 | 1 dispatch (57600×1×1) | 318 µs | 1.26 ms | per-lane | 66d8f8f6a74feddb | ×2.8 |
| 23 | 0.28 | 360 indirect mesh | 0.8 µs | 0.78 ms | per-lane | db875f50a166bd91 | ×2.8 |
| 24 | 0.22 | 112 dispatch (25×1×1) | 3.0 µs | 0.88 ms | per-lane | a06ef09446bd6e70 | ×20 |
| 25 | 0.22 | 1 dispatch (320×180×1) | 299 µs | 0.82 ms | per-lane | df11dcd012cd7382 | ×6.0 |

The top 25 together take ≈ 13 ms. The remaining ≈ 41 ms is spread thinly over hundreds of programs.

## Single commands of 1 ms or more

- **190fbf0ebe040169** (BVH builder, 1×1×1 groups, 5 per frame).
  - Mean 57 µs, longest **5.59 ms**.
  - Its translation has 173,158 host instructions in 4,041 blocks. One group walks the whole job.
- **5cf00dc39042edaf**: up to 3.79 ms.
- **3b90165a77d95cdf**: longest 1.0 ms on the 5090 (128 host instructions, per-lane).
- **6f6a359d1166bb27**: never reaches 1 ms on the 5090 (1.7 dispatches per frame; 4,791 host instructions, per-lane).
- Multi-second single dispatches of these two seen on a 6 GB card are therefore not arithmetic. A wait (paging or residency) is the likely cause; this is an inference that still has to be checked on that card.

## Hypotheses and candidate translation changes

1. **Window or hoist the clamps of indirect dispatches (−4 to −6 ms).**
   - Each indirect dispatch has an argument clamp pass of its own, with a barrier and a rebind. Each costs 7.9 µs, 7.3 ms/frame in total.
   - Spread dispatches cannot join a clamp window today. Change 2 removes that obstacle.
2. **Indirect dispatch through a plain DISPATCH signature (−2 ms, and it enables change 1).**
   - Spreadable compute programs dispatch indirectly through a signature that sets two root constants before the dispatch.
   - The probe below measures that at +2.3 µs per execution over a plain indirect dispatch.
   - Fix: the clamp writes the spread values where the program reads them.
   - The rest of an indirect dispatch's 20.9 µs is the program's own work and the drain behind its clamp.
3. **Native 16-bit arithmetic (−1 to −4 ms).** Packed-f16 programs expand ×20–27 when f16 is emulated in f32, against a median of ×4.
4. **Console flow only where lane reads need it (−1 to −2.5 ms).** 5cf00dc39042edaf (console flow) takes 2.15 ms. Siblings on the same grid in halves take 0.31–0.35 ms.
5. **Uniform descriptors out of waterfalls (−1 to −3 ms).** 13.9 ms/frame runs in waterfall forms.

All figures were measured under per-command timestamps, which serialise commands; untimed runs overlap more.

Indirect-dispatch probe (300 executions per variant, one group of a trivial program each, RTX 5090):

| variant | mean | median |
|---|---|---|
| direct Dispatch | 0.31 µs | 0.22 µs |
| ExecuteIndirect, plain DISPATCH signature | 0.34 µs | 0.22 µs |
| plain DISPATCH + count buffer | 0.32 µs | 0.22 µs |
| root constant + DISPATCH | 2.53 µs | 2.56 µs |
| root constant + DISPATCH + count buffer | 3.10 µs | 2.37 µs |
