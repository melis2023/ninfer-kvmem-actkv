param(
  [Parameter(Mandatory=$true)][ValidateSet('on','off')][string]$Mode,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][string]$Label,
  [int]$TargetTokens = 230000,
  [switch]$KeepServer
)
$ErrorActionPreference = 'Continue'
$bin  = 'I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll\bin'
$logs = 'I:\dsh\html'

# --- always start from a CLOSED server so no KV survives between arms ---
Get-Process -Name 'llama-kvmem-server' -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 4

$args = @(
  '-m','I:\model\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf',
  '--mmproj','I:\model\mmproj-Qwen3.8-27B-Q5_K-MIX.gguf','--no-mmproj-offload','--image-max-tokens','256',
  '-lm','none','--jinja','--chat-template-file','I:\templates\chat_template.jinja',
  '--host','127.0.0.1','--port','18200','-c','262144','-n','16384','-b','512','-ngl','99',
  '--kvmem-budget','36864','--kvmem-gen-reserve','16384'
)
if ($Mode -eq 'on') {
  # GQA aggregation is the folded mean and is no longer configurable: the max
  # variant was measured on the 40-needle / 256k test and regressed recall from
  # 39/40 to 28/40 (docs/actkv-section56-wiring-record.md section 5). Passing
  # --kvmem-score-agg max is now an error, so it is not passed here.
  $args += @('--kvmem-lrfu','--kvmem-lrfu-lambda','0.50','--kvmem-lrfu-top-frac','0.90')
}
$args += @(
  '-ctk','q8_0','-ctv','q8_0','--spec-type','draft-mtp','--spec-draft-n-max','3','--spec-kv-dtype','f16',
  '--enable-thinking','--reasoning-effort','medium','--reasoning-budget','8192',
  '--temp','1.0','--top-p','0.95','--top-k','20','--min-p','0.0',
  '--presence-penalty','0.0','--frequency-penalty','0.0','--repeat-penalty','1.0','--webui'
)

Start-Process -FilePath "$bin\llama-kvmem-server.exe" -ArgumentList $args `
  -RedirectStandardOutput "$logs\arm-$Mode-server.log" `
  -RedirectStandardError  "$logs\arm-$Mode-server.err.log" -WindowStyle Hidden

for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 5
  try { $r = python -c "import requests;print(requests.get('http://127.0.0.1:18200/health',timeout=5).json()['status'])" 2>$null } catch { $r = $null }
  if ($r -eq 'ok') { Write-Output "[$Mode] fresh server healthy after $((($i+1)*5))s"; break }
}

python -u "$logs\niah_dialog.py" --target-tokens $TargetTokens --out $Out --label $Label

if (-not $KeepServer) {
  Get-Process -Name 'llama-kvmem-server' -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Seconds 3
  Write-Output "[$Mode] server closed"
}
