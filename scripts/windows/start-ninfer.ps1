#Requires -Version 5.1
# Put a model with text, Vision and MTP in the package root as model.ninfer,
# or edit the model path below. Use English paths and filenames.
# Select GPUs by UUID: -Gpu 'GPU-uuid-a,GPU-uuid-b'. The first GPU holds Vision/MTP.
# Optional -StageLayers 30,34 sets the layer split; otherwise the engine plans it.
# With -Concurrency 2..8, ngram widths above 15 automatically reduce to 15; raise -RetainedSessions as needed.
# Sink defaults to 5120 tokens; recent defaults to 64 tokens (one page). -SinkTokens 0 keeps one 64-token page; -RecentTokens must be >=64.
# LRFU and GQA max are enabled by default. -NoLrfu -ScoreAggregation mean restores plain retrieval.
param(
    [string]$Gpu = $(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { '0' }),
    [uint32[]]$StageLayers = @(),
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 5120,
    [ValidateRange(64, 2147483647)][int]$RecentTokens = 64,
    [switch]$NoLrfu,
    [ValidateRange(0.0, 1.0)][double]$LrfuLambda = 0.5,
    [ValidateScript({ $_ -gt 0 -and $_ -le 1 })][double]$LrfuTopFraction = 0.9,
    [ValidateSet('mean', 'max')][string]$ScoreAggregation = 'max',
    [switch]$DryRun
)
& "$PSScriptRoot\start-backend.ps1" -Backend ninfer `
    -Model "$PSScriptRoot\..\..\model.ninfer" -Gpu $Gpu -StageLayers $StageLayers `
    -Port 18200 -Context 204800 -Concurrency 1 -Prefill 256 -MaxTokens 16384 `
    -Budget 36864 -Reserve 16384 -HostMiB 12288 -RetainedSessions 1 -KvType int8 `
    -MtpDrafts 4 -AdaptiveMtp -NgramDrafts 31 -Vision -VisionResidency cpu -VisionTokens 1024 `
    -ThinkingBudget 8192 -SinkTokens $SinkTokens -RecentTokens $RecentTokens `
    -NoLrfu:$NoLrfu -LrfuLambda $LrfuLambda -LrfuTopFraction $LrfuTopFraction `
    -ScoreAggregation $ScoreAggregation -DryRun:$DryRun
