#!/usr/bin/env bash
# Host and disk experts must compute the same model: score the first 40 KB of the WikiText stream of
# eval/corpora/perplexity-1m (4,096-token windows, a 2,048-token stride, int8 KV) with each residency
# and require identical results. A difference means an expert path read wrong weights (October 9:
# host 3.995-4.047 against disk 2.5390 for Q2_0 until the prefetch fix). About three minutes on an
# RTX 3090.
#
#   MODEL=/path/flash-next.ninfer TABLE=/path/ngram-table.ninfer scripts/pods/residency_check.sh
set -Eeuo pipefail
root=/workspace/ninfer-work
cd "$root/src"
out=${NINFER_JOB_DIR:-$(mktemp -d)}
binary=${PERPLEXITY:-$root/build/apps/ninfer-perplexity}
head -c 40000 eval/corpora/perplexity-1m/data/wikitext/00.txt | iconv -f utf-8 -t utf-8 -c > "$out/slice.txt"
for residency in host disk; do
    "$binary" "$MODEL" --text "$out/slice.txt" --kv-dtype int8 --expert-residency "$residency" \
        --ngram-table "$TABLE" --output "$out/ppl-$residency" \
        > "$out/ppl-$residency.stdout.txt" 2> "$out/ppl-$residency.stderr.txt" < /dev/null
    grep -E "^overall" "$out/ppl-$residency.stdout.txt" | sed "s/^/$residency /"
done
if [ "$(grep -E '^overall' "$out/ppl-host.stdout.txt")" != "$(grep -E '^overall' "$out/ppl-disk.stdout.txt")" ]; then
    echo "RESIDENCY MISMATCH: host and disk experts scored the slice differently"
    exit 1
fi
echo "host and disk agree"
