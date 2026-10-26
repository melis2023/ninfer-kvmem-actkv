# KVMem + ActKV 路 ninfer 鍚庣

鍩轰簬 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 涓?**ActKV**
锛圼arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)锛屼腑鍥界瀛︽妧鏈ぇ瀛︽彁鍑猴級锛?
- **KVMem**锛氭妸闀夸笂涓嬫枃绋€鐤忔垚鍥哄畾绐楀彛锛坄budget` 鍘嗗彶鍧?B + `gen_reserve` 宸ヤ綔鍖?R锛夛紝
  Host/纾佺洏褰掓。鍏朵綑 KV锛岀敤鍥哄畾澶у皬鐨勬敞鎰忓姏绐楀彛鎹㈠彇 200K/262K 鍙敤涓婁笅鏂囥€?- **ActKV**锛氶潰鍚戣鍔ㄧ殑妫€绱繚鎸佺瓥鐣ワ紙LRFU 鍘嗗彶鎺掑簭绛夛級锛屾彁鍗囬暱浼氳瘽鍏抽敭鍧楀彫鍥炪€?
QQ 浜ゆ祦缇わ細**1040777853**銆傛湰椤圭洰浠呬緵鐮旂┒瀛︿範锛岀ǔ瀹氫娇鐢ㄨ鍓嶅線 kvmem 鍘熶綔鑰呫€?
**鎵€鏈変慨琛ャ€佹祴閫熶笌鍙洖鏁版嵁姹囨€诲湪 [docs/CHANGELOG.md](docs/CHANGELOG.md)銆?*

## 褰撳墠鎺ㄨ崘锛?026-10-26锛?
| 椤?| 鎺ㄨ崘 |
|---|---|
| 宸ヤ欢 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer` |
| 鑽夌 | **q3km 鐩撮€?*锛圦4_K脳10 + Q3_K脳2锛宑onv/norm BF16锛?|
| 鑽夌鍚庣 | **DFlash2**锛宍draft 4`锛宍ngram 0`锛坅gent 鍙敤 15锛?|
| KV | `nvfp4`锛孊/R = 36,480 / 16,384锛圵=52,864锛?|
| CUDA Graph | **寮€**锛涙帰閽堟牎鍑?256 MiB锛屼笉闇€瑕?`--no-cuda-graph` |
| 椹遍€?| 婊＄獥鍙?100% |

```bat
--max-context 262144 --prefill-chunk 256 ^
--kvmem-budget 36480 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 --kvmem-lrfu ^
--kv-dtype nvfp4 --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 0 ^
--vision --vision-residency cpu --vision-max-merged 256 ^
--default-reasoning-effort medium --default-thinking-budget 16384
```

16 GB 鍗″惎鍔?free 绾?780鈥?55 MiB锛汢/R 鎸夋樉鍗¤皟鏁达紝涓嶈鍐嶅姞 `--no-cuda-graph`銆?
## 鎬ц兘

### 128K 浠ｇ爜锛?48,774 token锛岃ˉ鍑芥暟锛?56 token 杈撳嚭锛?
| 鍚庣 | prefill | decode | 鎺ュ彈鐜?| wall |
|---|---:|---:|---:|---:|
| **DFlash2 + q3km锛宒raft 4** | 826.5 | **87.1 tok/s** | 82.1% | 350.7 s |
| MTP + q3km锛宒raft 3 | 826.5 | 72.2 tok/s | **86.9%** | 349.5 s |

### 128K 鎬荤粨锛?26,668 token锛?
| 鍚庣 | prefill | decode | 鎺ュ彈鐜?| wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km锛宒raft 4 | 904.8 | 45.8 | 28.3% | 288.5 s |
| **MTP + q3km锛宒raft 3** | 904.8 | **47.7** | **40.8%** | 280.9 s |

### 256K锛?53,031 token锛屼袱涓惎鍔ㄥ櫒锛?
| 鍚姩鍣?| 鑽夌 | KV | prefill锛堢涓€閬嶏級 | decode | 鎺ュ彈鐜?| wall |
|---|---|---|---:|---:|---:|---:|
| dflash | **draft 4** | nvfp4 | **854.8** | **50.5** | **37.8%** | 597.4 s |
| mtp | draft 3 adaptive | int8 | 803.3 | 47.5 | 40.7% | 630.9 s |

DFlash2 琛屼负 2026-10-26 澶嶆祴锛坬3km銆乣ngram 0`銆佸綋鍓嶄簩杩涘埗锛夛紱MTP 琛屼粛鏄棫鍚姩鍣ㄦ暟鎹€?253K prompt銆佽緭鍑?256 token锛沺refill 鎶ュ憡鍊?427.9 tok/s 鍚殣钘?replay锛岀涓€閬嶆寜涓ゅ€嶆姌绠椼€?
### 楣堥箷 HTML draft 鎵弿锛?2 token prompt锛宯gram 0锛?
| draft | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| decode | 68.8 | **71.4** | 70.6 | 67.8 | 64.0 |
| 鎺ュ彈鐜?| **65.7%** | 54.9% | 48.4% | 42.4% | 38.1% |

### 鐭?prompt / agent

| 鍦烘櫙 | decode | 鎺ュ彈鐜?|
|---|---:|---:|
| WebUI 楣堥箷锛? 宸ュ叿锛宼hinking 209锛?| **96.0 tok/s** | 76.7% |
| harness锛?8 宸ュ叿锛宼hinking 4,826锛?| 68.2 tok/s | 51.9% |
| agent 9.8K锛坣gram 0锛?| 87.8 tok/s | 70.8% |
| agent 9.8K锛坣gram 31 浠茶锛?| 89.2 tok/s | 68.1% |

### KV 绮惧害

| | `nvfp4` | `rk8v4` |
|---|---:|---:|
| 52,864 绐楀彛 runtime | **1.62 GiB** | 1.99 GiB |
| 鍚姩 free | **835.8 MiB** | 645.9 MiB |
| 楣堥箷 decode锛坉raft 4锛?| **73.2 tok/s** | 68.6 tok/s |
| 鎺ュ彈鐜?| **58.6%** | 51.3% |

`nvfp4` 鏄?16 GB 鍗￠粯璁わ紱`rk8v4` 鐨?key 淇濇寔 INT8锛岀悊璁洪暱涓婁笅鏂囨绱㈡洿绋筹紝浠ｄ环鏄绾?0.37 GiB銆乨ecode 鎱㈢害 6%銆?
### 鍙洖

| 娴嬭瘯 | 缁撴灉 |
|---|---|
| 256K NIAH锛?3 閽?| **33/33**锛堝悓杞?/ 缁堣疆涓ユ牸 / 骞堕泦锛?|
| 256K 璺ㄥ伐鍏?`send_alert` | **14/14** |
| 256K 涓夐拡锛?% / 50% / 95%锛?| **3/3** |
| 128K NIAH锛?% / 50% / 95%锛?| **3/3** |
| LongMemEval 54 棰?| **54/60**锛圓ctKV锛?|
| Agent 220K | **30/30** 閫愯疆宸ュ叿鍙洖 |

## 鍏抽敭淇ˉ

瀹屾暣鍘熷洜銆佹彁浜ゅ拰楠岃瘉鏁版嵁瑙?[docs/CHANGELOG.md](docs/CHANGELOG.md)銆?
| 绫诲埆 | 缁撹 |
|---|---|
| KVMem 椹遍€?| 婊＄獥鍙?100%锛?1K 浠?55 娆￠噸閫夐檷鍒?**0 娆?*锛寃all 82.2 鈫?75.8 s |
| KVMem 鍙洖 | ActKV LRFU锛汱ongMemEval 54/60 vs EMA 50/60 |
| KVMem 浼犺緭 | 宸紓 restore锛?4K 163 ms 鈫?**33鈥?1 ms** |
| DFlash2 脳 KVMem | envelope/floor/slot 淇锛涜秺绐楁帴鍙楃巼 1.5% 鈫?**42%** |
| DFlash2 bypass | 鎸夋湡鏈涙敹鐩婂垽瀹氾紱璇叧鑽夌 **0 娆?* |
| ngram 浠茶 | copy 鍙湪鏀剁泭 > 1.3脳 鑽夌鏃惰Е鍙戯紱楣堥箷 74.7 vs 72.1 tok/s |
| CUDA Graph | 鎺㈤拡鏍″噯锛涚渷 **700鈥?00 MiB**锛屽紑鍥?+6% |
| NVFP4 | Blackwell 淇濈暀 scale `0x7F` 鈫?0锛堜粎缂栬瘧楠岃瘉锛?|
| 妯℃澘缂撳瓨 | Jinja trim 鍚?cache boundary 涓嶅啀涓㈠け |
| 骞冲彴 | sm70/sm75 婧愮爜鏀寔锛堟湭鐪熸満楠岃瘉锛?|

## 妯″瀷杞崲

杞崲宸ュ叿浠?Releases 涓嬭浇锛圥ython 3.11 + PyTorch/NumPy锛夈€傚湪 `backends/ninfer` 涓嬭繍琛岋細

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q3_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b-q3km \
  --out model.ninfer
```

- `dflash2` 婧愭敮鎸?GGUF銆丮odelOpt NVFP4銆丅F16銆乸er-row FP8锛?- 鐩撮€氳崏绋跨敤 `dflash2_gguf_blocks.py`锛堥粯璁ら厤鏂癸級锛?- NVFP4 鑽夌璧?`--override tools/convert/recipes/dflash2_import_nvfp4.py`锛?  璇﹁ [docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md)锛?- `kernel_projection` 涓?conv 蹇呴』淇濇寔 BF16锛坧repare 璺緞娌℃湁閲忓寲鍐呮牳锛夈€?
## 蹇€熷紑濮?
1. 涓嬭浇 `.ninfer` 宸ヤ欢锛屾垨鎸変笂鏂囪嚜琛岃浆鎹€?2. 鍙屽嚮 `workers/ninfer/start-kvmem-ninfer-dflash.bat`锛圖Flash2锛夋垨
   `start-kvmem-ninfer.bat`锛圡TP锛夈€傛ā鍨嬭矾寰勫拰鍙傛暟閮藉湪 bat 閲屻€?3. 鏈嶅姟鍦板潃 `http://127.0.0.1:18200`锛孫penAI 鍏煎 API銆?
## 鏂囨。

| 鏂囨。 | 鍐呭 |
|---|---|
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | **鎵€鏈変慨琛ャ€佹祴閫熴€佸彫鍥炴€昏** |
| [docs/dflash2-passthrough-20261026.md](docs/dflash2-passthrough-20261026.md) | q3km 鐩撮€氥€乨raft 鎵弿銆乥ypass/floor 缁嗚妭 |
| [docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md) | DFlash2 脳 KVMem 棣栨淇 |
| [docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md) | 椹遍€愭按浣嶄笌 issue #6 |
| [docs/kvmem-actkv-lrfu-20261020.md](docs/kvmem-actkv-lrfu-20261020.md) | ActKV LRFU 涓?LongMemEval |
| [docs/cuda-graph-calibration.md](docs/cuda-graph-calibration.md) | Graph 鎺㈤拡涓庡惎鍔ㄦ暟鎹?|
| [docs/multi-backend.md](docs/multi-backend.md) | 鍚庣鑳藉姏涓庡惎鍔ㄨ鏄?|
| [docs/multi-backend-repository.md](docs/multi-backend-repository.md) | 浠撳簱缁撴瀯涓庢瀯寤?|
