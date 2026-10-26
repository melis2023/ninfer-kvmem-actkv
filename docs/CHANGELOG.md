# 淇敼涓庡熀鍑嗘€昏

> 鏈€鍚庢洿鏂帮細2026-10-26銆傛湰鏂囦欢姹囨€诲巻娆′慨琛ョ殑**鍘熷洜銆佹敼鍔ㄥ拰楠岃瘉鏁版嵁**锛涗笓棰樼粏鑺備粛鍦?> `docs/` 涓嬬殑鍘熷璁板綍閲岋紝鍏抽敭缁撹浠ユ湰鏂囦欢涓哄噯銆?
## 0. 褰撳墠鐘舵€?
| 椤?| 鍊?|
|---|---|
| 涓诲姏宸ヤ欢 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer` |
| 鐩爣妯″瀷 | IQ3_S GGUF 鍧楋紙131 涓?`gguf_iq3_s`锛?|
| 鑽夌 | q3km 鐩撮€氾細`attention/output`銆乣mlp/down` = Q4_K脳10锛沠eature/selector = Q3_K脳2 |
| 鑽夌鍚庣 | DFlash2锛宍draft 4`锛宍ngram 0`锛坅gent 鍦烘櫙鍙敤 15锛?|
| KV | `nvfp4`锛孊/R = 36,480 / 16,384锛圵=52,864锛?|
| KV 鏄惧瓨 | 1.61鈥?.67 GiB锛?26 椤碉級 |
| CUDA Graph | 寮€锛涙帰閽堝疄娴?60鈥?57 MiB锛岃鍒?256 MiB |
| 椹遍€?| 婊＄獥鍙?100% |
| 16 GB 鍗″惎鍔?free | 780鈥?55 MiB |
| 鏃ュ父 decode | 楣堥箷 74鈥?0 tok/s锛沘gent 9.8K 89 tok/s锛涗唬鐮?128K 87 tok/s |

---

## 1. 淇鍒嗙被绱㈠紩

| 绫诲埆 | 鏃ユ湡 | 鎻愪氦 | 缁撹 |
|---|---|---|---|
| KVMem 鍙洖 | 10-20 | 鈥?| ActKV LRFU 鏇夸唬 EMA锛歀ongMemEval 54/60 vs 50/60 |
| KVMem 浼犺緭 | 10-20 | 鈥?| 宸紓 restore锛?4K 163 ms 鈫?33鈥?1 ms |
| KVMem 椹遍€?| 10-26 | `c8cc858` | 閲嶉€夊洖鍒版弧绐楀彛 100%锛?1K 55 娆?鈫?0 娆?|
| KVMem 鎵撳垎 | 10-22 | 鈥?| layer-wise 宸插洖閫€锛氭參 3鈥?%锛屽彫鍥?53/60 |
| DFlash2 脳 KVMem | 10-18 | 鈥?| envelope/floor/slot 淇锛氳秺绐楁帴鍙楃巼 1.5% 鈫?42% |
| DFlash2 floor | 10-19 | 鈥?| floor 64 鈫?2048锛氳秺绐楁帴鍙楃巼 +3.7鈥?.4 鐐?|
| DFlash2 floor | 10-26 | `0d66244` | 涓夊鍏紡缁熶竴锛岄槻寰℃€т慨姝ｏ紝瀹炴祴鏃犲彉鍖?|
| DFlash2 bypass | 10-26 | `eddada7` `fbf79f2` | 鎸夋湡鏈涙敹鐩婂垽瀹氾細璇叧鑽夌 0 娆?|
| ngram 浠茶 | 10-26 | `10d1da2` | copy 鍙湪鏀剁泭 > 1.3脳 鑽夌鏃惰Е鍙?|
| CUDA Graph | 10-25 | `b4d8096` | 鎺㈤拡鏍″噯锛岀渷 700鈥?00 MiB锛屼笉鍐嶉渶瑕?`--no-cuda-graph` |
| CUDA Graph | 10-26 | `591537b` | 璇绘暟 >256 MiB 鏃跺娴嬪彇灏?|
| NVFP4 scale | 10-26 | `798988c` | Blackwell 淇濈暀鍊?`0x7F` 鏄犲皠涓?0锛堜粎缂栬瘧楠岃瘉锛?|
| 妯℃澘缂撳瓨 | 10-25 | `46a0d0d` | Jinja trim 鍚?cache boundary 涓嶅啀涓㈠け |
| 骞冲彴 | 10-22 | 鈥?| sm70/sm75 婧愮爜鏀寔锛堟湭杩愯楠岃瘉锛?|

---

## 2. KVMem 鍙洖涓庨┍閫?
### 2.1 ActKV LRFU锛?026-10-20锛?
**闂**锛欵MA 鏄姞鏉冨钩鍧囷紝鍋跺皵鍛戒腑浣嗗崟娆″垎楂樼殑鍧楁椿涓嶄箙銆?
**鏀瑰姩**锛?
```text
hit : KS_t = 位路KS_{t-1} + score
miss: KS_t = 位路KS_{t-1}
```

鏂板潡浠?0 寮€濮嬶紝棣栨鍛戒腑鏃剁偣浜€?
**楠岃瘉**锛?
| 鍦烘櫙 | EMA | ActKV LRFU |
|---|---:|---:|
| 64K prefill | 422.4 tok/s | 399.7 tok/s |
| 64K decode | 49.3 tok/s | 46.3 tok/s |
| 64K 鎺ュ彈鐜?| 51.3% | 51.3% |
| 256K NIAH 33 閽?| 33/33 | 33/33 |
| 256K 璺ㄥ伐鍏?| 14/14 | 14/14 |
| 256K 绔埌绔?| 594.8 s | 646.9 s |
| LongMemEval 54 棰?| 50/60 | **54/60** |
| Agent 220K 鍙洖 | 鈥?| **30/30** |

LongMemEval 鍒嗘。锛?28K 20/20銆?92K 18/20銆?40K 16/20銆傛渶浣冲伐浣滃尯闂?128K鈥?92K銆?
### 2.2 澧為噺 restore锛?026-10-20锛?
**闂**锛氭瘡娆￠噸閫?`truncate(0)` 鍚庡叏閲忛噸鏄犲皠锛?4K 涓€娆?163 ms銆?
**鏀瑰姩**锛?
- 閫変腑椤垫寜 id 绋冲畾鎺掑簭锛屽悓涓€閫昏緫椤典繚鎸佺墿鐞嗘Ы浣嶏紱
- 鍙鍙樺寲妲戒綅鍋?H2D锛屽彧璇诲彉鍖?host 璁板綍锛?- 鍙樺寲瓒呰繃涓€鍗婅嚜鍔ㄥ洖閫€鍏ㄩ噺銆?
**楠岃瘉锛?4K锛? 娆￠噸閫夛級**锛?
| 杞 | changed | restore 椤垫暟 | read | sync | 鎬昏€楁椂 |
|---|---:|---:|---:|---:|---:|
| 鍏ㄩ噺鍩虹嚎 | 560 | 576 | 69 ms | 94 ms | 163 ms |
| 澧為噺 | 124 | 124 | 14 ms | 20 ms | **33 ms** |
| 澧為噺 | 164 | 164 | 20 ms | 27 ms | **47 ms** |
| 澧為噺 | 181 | 181 | 21 ms | 29 ms | **51 ms** |

256K 鏃?`changed` 涓綅鏁?495/576锛岃嚜鍔ㄥ洖閫€鍏ㄩ噺锛涚鍒扮 594.8 s銆?
### 2.3 閲嶉€夊洖鍒版弧绐楀彛 100%锛?026-10-26锛宨ssue #6锛?
**闂**锛?0% 姘翠綅鍦?B > 4R 鏃讹紝鍘嬬缉鐩爣 B 鏈韩楂樹簬姘翠綅锛屾瘡涓?256-token chunk
閮借Е鍙戜竴娆＄┖杞噸閫夈€?
**鏀瑰姩**锛坄kvmem_window.cpp`锛夛細

```cpp
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;
```

**楠岃瘉锛?1K prompt锛孊/R=4.15锛?*锛?
| | 80% 姘翠綅 | 100% |
|---|---:|---:|
| reselect 娆℃暟 | 55 | **0** |
| wall | 82.2 s | **75.8 s** |
| 256K NIAH | 3/3 | 3/3 |

**浠ｄ环**锛氬钩鍧囧彲瑙佺獥鍙ｅ洖鍒?`(W+B)/2`锛?0-19 璁板綍鐨?256K 鈭?2% 涓嶅啀鏈夈€?
### 2.4 layer-wise 鎵撳垎锛?026-10-22锛屽凡鍥為€€锛?
| 鍦烘櫙 | ActKV | layer-wise | 宸紓 |
|---|---:|---:|---:|
| 64K prefill | 443.4 | 428.2 | 鈭?.4% |
| 64K decode | 46.7 | 44.5 | 鈭?.7% |
| LongMemEval | 54/60 | 53/60 | 鈭? |

缁撹锛氶澶栧眰鏉冮噸娌℃湁甯︽潵鍙洖鏀剁泭锛屽弽鑰屽鍔犲紑閿€锛屽凡鍥為€€銆?
### 2.5 鍧楀ぇ灏忓浐瀹?64

`kPagedKVPageSize` 鍚屾椂鏄?fast prompt 鍐呮牳鐨?K 鏂瑰悜 tile锛?
```cpp
static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
```

鏀规垚 32 闇€瑕侀噸鍐?MMA 杩唬鍜屽叡浜唴瀛橀绠楋紱48 涓嶆弧瓒抽〉琛?`position >> 6` 鐨?2 鐨勫箓瑕佹眰銆?**鍧楀ぇ灏忎笉鍙皟**銆?
---

## 3. DFlash2 脳 KVMem

### 3.1 envelope / floor / slot fork锛?026-10-18锛?
**鐜拌薄**锛氫笂涓嬫枃瓒呰繃绾?2048 token 鍚庯紝DFlash2 鑽夌鎺ュ彈鐜囦粠 32鈥?5% 鎺夊埌 0鈥?.5%銆?
**鏍瑰洜**锛?
1. `dflash_envelopes` 鐢?`max_context` 鐭矾锛宔nvelope 璇箟閿欒锛?2. KVMem 妫€绱㈢紳鍚堝悗锛宒raft ring 璇诲埌琚┍閫愮殑鍘嗗彶锛宖loor 鎺╃爜涓嶈冻锛?3. state fork 鍚庢Ы浣嶆湭璺熼殢锛?4. dflash 杞己灏?`prepare_window`锛岀獥鍙ｆ弧鏃?entitlement 婧㈠嚭銆?
**楠岃瘉锛坓reedy锛?*锛?
| prompt token | 淇鍓嶆帴鍙楃巼 | 淇鍚?| decode |
|---:|---:|---:|---:|
| 2,366 | 0.4% | **43.2%** | 30.9 鈫?**72.7** tok/s |
| 3,510 | 1.5% | **39.6%** | 31.3 鈫?**72.8** tok/s |
| 4,654 | 1.5% | **41.8%** | 31.8 鈫?**75.7** tok/s |
| 131K 缂濆悎绐楀彛 | 0/36 | 姝ｅ父 | 鈥?|

### 3.2 floor 淇濈暀閲?64 鈫?2048锛?026-10-19锛?
**闂**锛歚recent_tokens = 64`锛屼絾 DFlash2 鑽夌鐜閲忔槸 2048锛涙瘡娆￠噸閫夊悗鑽夌鍙墿
3.1% 鐨?conditioning 涓婁笅鏂囷紝鎺ュ彈鐜囪閽夊湪 30鈥?9%銆?
**鏀瑰姩**锛歠loor 鏀逛负 `max(recent 脳 64, masked_draft_ring_capacity())`銆?
**楠岃瘉锛?4K 瓒婄獥锛?*锛?
| 閰嶇疆 | 淇鍓?| 淇鍚?|
|---|---:|---:|
| draft 4 鎺ュ彈鐜?| 30.1% | **33.8%** |
| draft 3 鎺ュ彈鐜?| 35.4% | **40.7%** |
| t2 绱ч殢 64K锛坉raft 4锛?| 34.3% | **39.7%** |

### 3.3 floor 涓夊缁熶竴锛?026-10-26锛宍0d66244`锛?
閲嶉€夈€乹uery restore銆乧heckpoint recovery 涓夊 floor 鍏紡涓嶄竴鑷达紱鍚庝袱澶勫彧鍐?`recent`锛?浼氶殣钘?2048 绐楀彛鐨勫ぇ閮ㄥ垎銆傜粺涓€涓猴細

```cpp
max(recent * 64, masked_draft_ring_capacity())
```

**瀹炴祴锛氭病鏈夊彉鍖栥€?* 128K 鎬荤粨 prompt銆乹3km draft 4 淇鍓嶅悗閮芥槸 135/477 = 28.3%锛?decode 43.8鈥?5.8 tok/s銆傚睘浜庨槻寰℃€т竴鑷存€т慨姝ｃ€?
### 3.4 bypass 鎸夋湡鏈涙敹鐩婂垽瀹氾紙2026-10-26锛宍eddada7` `fbf79f2`锛?
**闂**锛氬師閫昏緫鈥滆繛缁?6 杞?accepted == 0 灏卞叧鑽夌鈥濓紝d4 涓嬪嵆浣?p1=0.4 姣忚疆鏈熸湜浠嶆湁
绾?1.4 token锛? 杩為浂鍦ㄥ嚑鍗佽疆鍐呭繀鐒跺嚭鐜般€傜粨鏋滄槸鎬濊€冮樁娈佃璇叧锛宒ecode 鎺夊埌绾?40 tok/s锛?杩樿绛夋渶澶?64 杞墠鎭㈠銆?
**鏀瑰姩**锛?
- 鐢?`MtpAdaptiveSignal` 鐨勬寜浣嶇疆 EMA 璁＄畻 `expected_tokens`锛?- `expected < 1.15` 杩炵画 6 杞墠 bypass锛?- 鎺㈡祴闂撮殧 64 鈫?绾?10 杞紱鎺㈤拡鏈洖鏈珛鍗抽噸鏂?bypass銆?
**楠岃瘉锛堥箞楣曪紝draft 4锛宯gram 0锛?*锛?
| | 淇鍓?| 淇鍚?|
|---|---:|---:|
| `DFLASH_BYPASS` 浜嬩欢 | 1鈥? | **0** |
| decode | 73.2 tok/s | 73.1 tok/s |
| 鎺ュ彈鐜?| 58.6% | 55.3%鹿 |

鹿 涓ゆ杈撳嚭闀垮害涓嶅悓锛?9,723 vs 14,357 token锛夛紝鎺ュ彈鐜囨湁閲囨牱鍣０銆?浠ｇ爜浠诲姟 128K锛氭帴鍙楃巼 82.1% 涓嶅彉锛宐ypass 0銆?
### 3.5 ngram copy 鎸夋敹鐩婁徊瑁侊紙2026-10-26锛宍10d1da2`锛?
**闂**锛氬彧瑕佹湁 鈮?2 token 鐨?ngram 鍖归厤灏辨暣杞?copy锛孌Flash 琚烦杩囷紝楠岃瘉瀹藉害鍥哄畾 32 鍒楋紱
鐭尮閰嶆垨杩戞湡 copy 涓€鐩翠笉鍑嗘椂锛屽弽鑰屾瘮鑽夌鎱€?
**鏀瑰姩**锛坄NgramCopySignal`锛夛細

```text
expected_copy = (accepted / drafted) 脳 match_length
copy 鏉′欢     = expected_copy > neural_expected 脳 1.3
```

- 鍓?2 杞?copy 鏃犳潯浠舵斁琛岋紝鐢ㄤ簬寤虹珛鏍锋湰锛?- 杩炵画 8 杞嫆缁濆悗寮哄埗鎺㈡祴涓€杞紱
- bypass 鏃?neural 鍩虹嚎鎸?1.0 璁＄畻锛?- 鍥?family 涓嶅彉锛歝opy 浠嶇敤 `ngram_graphs`锛屾嫆缁濆洖鍒?`dflash_graphs` / `mtp_graphs`銆?
**楠岃瘉**锛?
| 鍦烘櫙 | ngram 31 | ngram 0 | copy 杞?|
|---|---:|---:|---|
| 楣堥箷 HTML锛坉raft 4锛?| 74.7 tok/s | 72.1 tok/s | 84 杞紝753/2,604 |
| 38 宸ュ叿 agent锛?.8K锛?| 89.2 tok/s | 87.8 tok/s | 2 杞紝29/59 |

缁撹锛歯gram 31 涓嶅啀鎷栨參 DFlash锛屼袱涓満鏅兘鐣ュ揩锛沜opy 杞彧淇濈暀鐪熸鏈夋敹鐩婄殑鍖归厤銆?
---

## 4. 鍚姩涓庢樉瀛?
### 4.1 CUDA Graph 鎺㈤拡鏍″噯锛?026-10-25锛宍b4d8096`锛?
**闂**锛氭寜 topology 鐩查鐣欑害 1 GiB锛屽疄闄?Graph 鍙崰 90鈥?45 MiB锛屽鑷?KV 灏戠粰銆?闀夸笂涓嬫枃鍚姩澶辫触锛屽彧鑳?`--no-cuda-graph`銆?
**鏀瑰姩**锛氬惎鍔ㄦ椂鐢ㄦ渶灏?KV 鎺㈤拡瀹炴祴鏁村鍥撅紝瑙勫垝 `max(瀹炴祴, 256 MiB)`锛涙帰閽堝け璐ュ洖閫€鐩查鐣欙紱
`--cuda-graph-allowance-mib` 浠嶄紭鍏堬紱`--no-cuda-graph` 璺宠繃鎺㈤拡銆?
**鏁堟灉**锛?
- 鐪佺害 **700鈥?00 MiB**锛?- q3blocks 寮€鍥?KV 瀹归噺 36,352 鈫?**77,824**锛?- 寮€鍥?decode 姣斿叧鍥惧揩绾?**6%**锛?- 涓嶅啀闇€瑕?`--no-cuda-graph`銆?
### 4.2 鎺㈤拡璇绘暟澶嶆祴锛?026-10-26锛宍591537b`锛?
鍚屼竴閰嶇疆杩炵画 6 娆″惎鍔ㄥ疄娴嬶細0.0 / 83.3 / 90.0 / 117.3 / 135.6 / **288.6** MiB銆?288.6 閭ｆ鎶婅鍒掑€兼姮楂樺埌瀹归噺涔嬪锛?
```text
FATAL | requested 2403183872 bytes, but only 2353864704 bytes are available
```

鐜板湪棣栨璇绘暟 > 256 MiB 鏃跺娴嬩竴娆″彇杈冨皬鍊硷紱256 MiB 涓嬮檺涓嶅彉銆?
### 4.3 NVFP4 淇濈暀 scale `0x7F`锛?026-10-26锛宍798988c`锛?
瀛樺偍鐨?UE4M3 鏉冮噸 scale 鐢?`0x7F` 琛ㄧず 0锛岃蒋浠跺睍寮€璺緞姝ｇ‘澶勭悊锛汢lackwell 鍘熺敓
`cvt.rn.bf16x2.e4m3x2` 浼氭妸瀹冭В鎴?NaN銆傚湪 native pair 璺緞鍔狅細

```cpp
if (scale == 0x7Fu) { scale = 0; }
```

NVFP4 KV codec 鏈姩锛屼笌涓婃父涓€鑷淬€?*鏈湴鏃?NVFP4 宸ヤ欢锛屼粎缂栬瘧楠岃瘉銆?*

### 4.4 q3km 鐩撮€氳崏绋匡紙2026-10-26锛?
**鏀瑰姩**锛欴Flash2 鑽夌鐨勭嚎鎬ф潈閲嶆寜婧?GGUF 鍧楁牸寮忕洿鎺ュ姞杞斤紝涓嶅啀缁熶竴閲嶅寲鎴?Q8/NVFP4銆?q3km 宸ヤ欢鐨勫疄闄呮牸寮忥細

| 寮犻噺 | 鏍煎紡 | 鏁伴噺 |
|---|---|---:|
| `attention/output`銆乣mlp/down` | `gguf_q4_k` | 10 |
| `feature_projection`銆乣selector/hidden_projection` | `gguf_q3_k` | 2 |
| conv銆乶orm銆乧odebook | BF16 | 鍏朵綑 |

**楠岃瘉锛?28K锛宯vfp4 KV锛宯gram 0锛?*锛?
| 鑽夌 | prefill | decode | 鎺ュ彈鐜?|
|---|---:|---:|---:|
| q3km 鐩撮€?| **904.8 tok/s** | **42.8 tok/s** | 23.4% |
| 娣峰悎 NVFP4 | 844.5 tok/s | 40.7 tok/s | 23.8% |
| q4km 鐩撮€?| 904.8 tok/s | 43.6 tok/s | 24.2% |

- 鐩撮€氭瘮娣峰悎 NVFP4 prefill 蹇?6.7%銆乨ecode 蹇?4.9%锛?- q3 涓?q4 鍦ㄥ櫔澹板唴锛宷4 澶氬崰绾?0.18 GB 鈫?鎺ㄨ崘 q3km锛?- Q8 鑽夌锛?4.6 GB锛夊湪 B/R=36,480/16,384 涓嬪惎鍔ㄥけ璐ワ紙宸?184 MiB锛夛紝闇€缂╁埌 W=40,960銆?
---

## 5. 妯℃澘涓庡墠缂€缂撳瓨

### Jinja trim 鐨?cache boundary锛?026-10-25锛宍46a0d0d`锛?
`trim_blocks` / `lstrip_blocks` / `{%- -%}` 浼氭妸 source 鎶樺彔鎴?0 瀛楄妭锛岃惤鍦ㄦ姌鍙犲尯鐨?cache marker 鏃犳硶瑙ｆ瀽锛宲refix cache 鍛戒腑涓嬮檷銆?
鏀瑰姩锛歚string_part::collapsed_source_end` + `append_boundary()` + `cut_bytes(...)`锛?`TemplateBoundaryMapping` 璺熼殢鎶樺彔杈圭晫锛沗prompt_layout.cpp` 鏌ヨ `source_boundary()`銆?
鏁堟灉锛氭ā鏉胯竟鐣岄檮杩戠殑 marker 鍙互鍞竴瑙ｆ瀽銆?
---

## 6. 骞冲彴绉绘

### sm70 / sm75锛?026-10-22锛?
| 椤?| 鍐呭 |
|---|---|
| 鏂板鏂囦欢 | 31 涓紙sm70 GEMM銆乻m75 MMA銆乣pre_sm80.cuh` / `sm70_emulation.cuh`锛?|
| 鎺ュ叆淇敼 | 8 澶勶紙CMake銆乣mma.cuh`銆乣memory.cuh`銆乻ources銆乣QType`銆? 涓?codec锛?|
| GPU profile | 9 涓紙V100脳2銆乀ITAN RTX銆?060S銆?070銆?070S銆?080Ti銆丵uadro RTX 6000銆乀4锛?|
| ActKV | 鏈敼鍔?|
| 楠岃瘉 | sm120a 缂栬瘧 0 閿欒锛?4K 427.1 / 45.5 / 48.1%锛泂m75/sm70 **鏈繍琛岄獙璇?* |

---

## 7. 鎬ц兘鍩哄噯

### 7.1 128K 浠ｇ爜锛?48,774 token锛岃ˉ鍑芥暟锛?56 token 杈撳嚭锛?
| 鍚庣 | prefill | decode | 鎺ュ彈鐜?| wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km锛宒raft 4 | 826.5 | **87.1** | 82.1% (92/112) | 350.7 s |
| MTP + q3km锛宒raft 3 | 826.5 | 72.2 | **86.9%** (86/99) | 349.5 s |

### 7.2 128K 鎬荤粨锛?26,668 token锛?56 token 杈撳嚭锛?
| 鍚庣 | prefill | decode | 鎺ュ彈鐜?| wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km锛宒raft 4 | 904.8 | 45.8 | 28.3% | 288.5 s |
| MTP + q3km锛宒raft 3 | 904.8 | **47.7** | **40.8%** | 280.9 s |

### 7.3 256K锛?53,099 token锛?56 token 杈撳嚭锛屼袱涓惎鍔ㄥ櫒锛?
| 鍚姩鍣?| 鑽夌 | KV | prefill锛堢涓€閬嶏級 | decode | 鎺ュ彈鐜?| wall |
|---|---|---|---:|---:|---:|---:|
| dflash | **draft 4** | nvfp4 | **854.8** | **50.5** | **37.8%** | 597.4 s |
| mtp | draft 3 adaptive | int8 | 803.3 | 47.5 | 40.7% | 630.9 s |

DFlash2 琛屼负 2026-10-26 澶嶆祴锛坬3km銆乣ngram 0`銆佸綋鍓嶄簩杩涘埗锛夛細鎶ュ憡 prefill 427.9 tok/s
鍚殣钘?replay锛岀涓€閬嶆寜涓ゅ€嶆姌绠楋紱鎺ュ彈鐜?153/405銆侻TP 琛屼粛鏄棫鍚姩鍣ㄦ暟鎹€?
### 7.4 楣堥箷 HTML draft 鎵弿锛?2 token prompt锛宯gram 0锛?
| draft | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| decode | 68.8 | **71.4** | 70.6 | 67.8 | 64.0 |
| 鎺ュ彈鐜?| **65.7%** | 54.9% | 48.4% | 42.4% | 38.1% |

### 7.5 鐭?prompt / agent

| 鍦烘櫙 | decode | 鎺ュ彈鐜?| 璇存槑 |
|---|---:|---:|---|
| WebUI 楣堥箷锛? 宸ュ叿锛宼hinking 209锛?| 96.0 | 76.7% | 杈撳嚭 6,218 |
| harness锛?8 宸ュ叿锛宼hinking 4,826锛?| 68.2 | 51.9% | 杈撳嚭 9,030 |
| agent 9.8K锛坣gram 0锛?| 87.8 | 70.8% | 杈撳嚭 835 |
| agent 9.8K锛坣gram 31 浠茶锛?| 89.2 | 68.1% | copy 2 杞?|

### 7.6 KV 绮惧害

| | `nvfp4` | `rk8v4` |
|---|---:|---:|
| 52,864 绐楀彛 runtime | **1.62 GiB** | 1.99 GiB |
| 鍚姩 free | **835.8 MiB** | 645.9 MiB |
| 楣堥箷 decode锛坉raft 4锛?| **73.2 tok/s** | 68.6 tok/s |
| 鍚屼换鍔℃帴鍙楃巼 | **58.6%** | 51.3% |

`nvfp4` 鏄惧瓨鏈€鐪侊紝鏄?16 GB 鍗￠粯璁わ紱`rk8v4` 鐨?key 淇濇寔 INT8锛岀悊璁烘绱㈡洿绋炽€?
---

## 8. 鍙洖姹囨€?
| 娴嬭瘯 | 缁撴灉 |
|---|---|
| 256K NIAH锛?3 閽?| 33/33锛堝悓杞?/ 缁堣疆涓ユ牸 / 骞堕泦锛?|
| 256K NIAH锛岃法宸ュ叿 `send_alert` | 14/14 |
| 256K NIAH锛屽伐鍏疯皟鐢?| 80 |
| 128K NIAH锛?% / 50% / 95% 娣卞害 | 3/3 |
| 256K 涓夐拡锛?% / 50% / 95%锛?| 3/3锛坄BLUEBIRD-42` / `MAPLE-91` / `REDWOOD-77`锛?|
| LongMemEval 54 棰?| ActKV 54/60锛?28K 20/20銆?92K 18/20銆?40K 16/20锛?|
| Agent 220K | 30/30 閫愯疆宸ュ叿鍙洖锛?81 娆″伐鍏疯皟鐢?|

5% 娣卞害宸茶秴鍑?52,864 璁惧绐楀彛锛屽懡涓鏄?KVMem 鐨?Host 鍙洖鏈夋晥銆?
---

## 9. 宸茬煡闄愬埗

- **256K prefill 璺戜袱閬?*锛氱浜岄亶鏄?hidden replay锛屼笉璁?token锛屾姤鍛婇€熷害绾︿负鐬椂鐨勪竴鍗婏紱
- **DFlash2 鎺ュ彈鐜囬殢涓婁笅鏂囦笅闄?*锛坉raft 5锛夛細8.7K 43.6% 鈫?128K 36.3% 鈫?256K 33.5%锛?  draft 4 鍦?128K 鎬荤粨浠诲姟涓婃槸 28.3%锛?- **DFlash2 娌℃湁鑷€傚簲 draft 闀垮害**锛汳TP 鏈?`--adaptive-mtp`锛?- **Q8 鑽夌**鍙兘閰嶆洿灏忕獥鍙ｏ紙W=40,960锛夛紝涓嶉€傚悎 16 GB 涓诲姏閰嶇疆锛?- **鏈湴鏃?NVFP4 鐩爣宸ヤ欢**锛宍0x7F` 淇鍙仛浜嗙紪璇戦獙璇侊紱
- **涓婃父鏈Щ妞?*锛歍P銆乹uery-memory銆乧heckpoint-policy銆乵onitor-events锛?- **sm75/sm70** 鍙湁婧愮爜涓庣紪璇戯紝鏈湪鐪熸満杩愯銆?
---

## 10. 涓撻鏂囨。

| 鏂囨。 | 鍐呭 |
|---|---|
| `docs/dflash2-kvmem-fixes-20261018.md` | DFlash2 脳 KVMem 棣栨淇锛坋nvelope/floor/slot锛?|
| `docs/dflash2-kvmem-audit-20261019.md` | 浠ｇ爜瀹℃煡 F1鈥揊6 涓?floor 淇 |
| `docs/kvmem-recall-opt-20261019.md` | 鍙洖浼犺緭浼樺寲涓庡浠借褰?|
| `docs/kvmem-reselect-high-water-20261019.md` | 80% 姘翠綅涓?issue #6 鐨勫畬鏁磋繃绋?|
| `docs/kvmem-actkv-lrfu-20261020.md` | ActKV LRFU 鍏紡涓?LongMemEval |
| `docs/kvmem-incremental-restore-20261020.md` | 宸紓 restore 瀹炵幇涓?64K 鏁版嵁 |
| `docs/kvmem-layer-wise-scoring-20261022.md` | layer-wise 瀹為獙锛堝凡鍥為€€锛?|
| `docs/kvmem-sm75-sm70-port-20261022.md` | sm70/sm75 绉绘涓庡潡澶у皬缁撹 |
| `docs/cuda-graph-calibration.md` | Graph 鎺㈤拡瀹炵幇涓庡惎鍔ㄦ暟鎹?|
| `docs/dflash2-nvfp4-20261023.md` | NVFP4 鑽夌瀵煎叆璺緞 |
| `docs/dflash2-passthrough-20261026.md` | q3km 鐩撮€氥€乨raft 鎵弿銆乫loor/bypass 缁嗚妭 |
| `docs/integration-20261025.md` | 涓婃父鏁村悎璇勪及涓?128K/256K 鏁版嵁 |
| `docs/CHANGELOG-20261025.md` | 10-25 褰撳ぉ鐨勫畬鏁存棩蹇?|
