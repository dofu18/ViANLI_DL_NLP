<#
.SYNOPSIS
    Biên dịch reports/main.tex thành PDF bằng XeLaTeX (MiKTeX), chạy offline.

.DESCRIPTION
    Script tự lo ba việc mà biên dịch tay hay quên:
      1. Gom main.tex + toàn bộ hình (reports/*.jpg|png và outputs/figures/*.png)
         vào một thư mục build riêng, nên không rác thư mục nguồn.
      2. Nếu máy thiếu font Liberation thì tự đổi sang TeX Gyre TRONG BẢN BUILD.
         reports/main.tex không bị đụng tới -> mang lên Overleaf vẫn chạy nguyên.
      3. Chạy XeLaTeX nhiều lượt cho mục lục / danh mục hình / tham chiếu ổn định,
         rồi lọc log báo đúng lỗi thật thay vì đổ cả nghìn dòng.

.PARAMETER Passes
    Số lượt biên dịch. Mặc định 3 — đủ cho mục lục và \pageref{LastPage}.

.PARAMETER Clean
    Xóa sạch thư mục build trước khi chạy. Dùng khi log cũ gây nhiễu.

.PARAMETER Open
    Mở PDF sau khi build xong.

.PARAMETER KeepLog
    Chép main.log ra cạnh PDF để soi khi cần.

.EXAMPLE
    .\build_pdf.ps1
.EXAMPLE
    .\build_pdf.ps1 -Clean -Open
#>

[CmdletBinding()]
param(
    [int]$Passes = 3,
    [switch]$Clean,
    [switch]$Open,
    [switch]$KeepLog
)

$ErrorActionPreference = 'Stop'
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# ------------------------------------------------------------------ đường dẫn
$ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ReportDir
$BuildDir  = Join-Path $ReportDir '.build'
$FigDir    = Join-Path $Root 'outputs\figures'
$SrcTex    = Join-Path $ReportDir 'main.tex'
$OutPdf    = Join-Path $ReportDir 'bao_cao_ViANLI_draft.pdf'

function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }
function Die([string]$m) { Say "`n  LỖI: $m" 'Red'; exit 1 }

if (-not (Test-Path $SrcTex)) { Die "Không thấy $SrcTex" }

# ------------------------------------------------------------ tìm xelatex.exe
$xelatex = $null
$candidates = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\xelatex.exe'),
    'C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe',
    'C:\texlive\2025\bin\windows\xelatex.exe'
)
foreach ($c in $candidates) { if (Test-Path $c) { $xelatex = $c; break } }
if (-not $xelatex) {
    $cmd = Get-Command xelatex -ErrorAction SilentlyContinue
    if ($cmd) { $xelatex = $cmd.Source }
}
if (-not $xelatex) {
    Die @"
Không tìm thấy xelatex. Cài MiKTeX rồi chạy lại:
    winget install --id MiKTeX.MiKTeX --accept-package-agreements --accept-source-agreements
"@
}
Say "  xelatex : $xelatex" 'DarkGray'

# --------------------------------------------------------------- chuẩn bị build
if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item $BuildDir -Recurse -Force
    Say "  đã xóa thư mục build cũ" 'DarkGray'
}
if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }

Copy-Item $SrcTex $BuildDir -Force
$nFig = 0
Get-ChildItem $ReportDir -File | Where-Object { $_.Extension -in '.jpg', '.png' } |
    ForEach-Object { Copy-Item $_.FullName $BuildDir -Force; $nFig++ }
if (Test-Path $FigDir) {
    Get-ChildItem $FigDir -Filter '*.png' -File |
        ForEach-Object { Copy-Item $_.FullName $BuildDir -Force; $nFig++ }
} else {
    Say "  CẢNH BÁO: chưa có $FigDir — chạy notebook 05 trước, PDF sẽ thiếu hình" 'Yellow'
}
Say "  đã chép : main.tex + $nFig hình" 'DarkGray'

# ------------------------------------------------- đổi font nếu thiếu Liberation
$hasLiberation = [bool](Get-ChildItem (Join-Path $env:WINDIR 'Fonts') -Filter 'Liberation*' `
                         -File -ErrorAction SilentlyContinue)
if (-not $hasLiberation) {
    $texPath = Join-Path $BuildDir 'main.tex'
    $tex = Get-Content $texPath -Raw -Encoding UTF8

    # Khớp bằng regex chịu được cả LF lẫn CRLF — main.tex có thể ở kiểu nào tùy
    # người sửa cuối dùng editor gì.
    $old = '(?m)^\\setmainfont\{Liberation Serif\}\r?\n' +
           '\\setsansfont\{Liberation Sans\}\r?\n' +
           '\\setmonofont\{DejaVu Sans Mono\}'
    $new = @'
% [BẢN BUILD LOCAL] máy này không có Liberation/DejaVu -> dùng TeX Gyre
% (tương thích metric với Times/Helvetica). Bản trên Overleaf giữ Liberation.
\setmainfont{texgyretermes}[
  Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
  ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[
  Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
  ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\setmonofont{texgyrecursor}[
  Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
  ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
'@

    if ($tex -match $old) {
        $tex = [regex]::Replace($tex, $old, $new.Trim().Replace('$', '$$'))
        Set-Content $texPath -Value $tex -Encoding UTF8 -NoNewline
        Say "  font    : thiếu Liberation -> đã đổi sang TeX Gyre (chỉ trong bản build)" 'Yellow'
    } else {
        Say "  font    : KHÔNG khớp khối \setmainfont — nếu build lỗi font, kiểm lại 3 dòng" 'Yellow'
    }
} else {
    Say "  font    : máy có Liberation, giữ nguyên" 'DarkGray'
}

# -------------------------------------------------------------------- biên dịch
# xelatex và miktex-dvipdfmx ghi cảnh báo ra stderr ngay cả khi chạy đúng (ví dụ
# "you have not checked for MiKTeX updates"). Windows PowerShell 5.1 bọc mỗi dòng
# stderr của native command thành ErrorRecord, làm output đầy lỗi giả. Dùng
# Start-Process với -RedirectStandardError để stderr đi thẳng vào file, không qua
# pipeline của PowerShell. Đây cũng là lý do KHÔNG dùng 2>&1 ở đây.
$outFile = Join-Path $BuildDir 'xelatex.out'
$errFile = Join-Path $BuildDir 'xelatex.err'
$texPdf  = Join-Path $BuildDir 'main.pdf'

for ($i = 1; $i -le $Passes; $i++) {
    Write-Host ("  lượt {0}/{1} ..." -f $i, $Passes) -NoNewline -ForegroundColor Gray
    $p = Start-Process -FilePath $xelatex `
        -ArgumentList '-enable-installer', '-interaction=nonstopmode', 'main.tex' `
        -WorkingDirectory $BuildDir -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile

    if (Test-Path $texPdf) { Write-Host " ok" -ForegroundColor Green }
    else { Write-Host (" thất bại (mã {0})" -f $p.ExitCode) -ForegroundColor Red }

    if ($i -eq 1 -and -not (Test-Path $texPdf)) {
        Say "`n  Không sinh được PDF ngay lượt đầu. Lỗi đầu tiên trong log:" 'Red'
        $mainLog = Join-Path $BuildDir 'main.log'
        if (Test-Path $mainLog) {
            Select-String -Path $mainLog -Pattern '^!' -Context 0, 5 |
                Select-Object -First 2 |
                ForEach-Object { Say ("    " + $_.Line) 'Red'
                                 $_.Context.PostContext | ForEach-Object { Say "    $_" 'DarkRed' } }
        }
        Die "xem đầy đủ ở $mainLog"
    }
}

$pdf = Join-Path $BuildDir 'main.pdf'
if (-not (Test-Path $pdf)) { Die "không sinh được main.pdf" }
Copy-Item $pdf $OutPdf -Force
if ($KeepLog) { Copy-Item (Join-Path $BuildDir 'main.log') (Join-Path $ReportDir 'main.log') -Force }

# ------------------------------------------------------------------ soát lại log
$log = Join-Path $BuildDir 'main.log'
# nonstopmode VẪN sinh PDF khi gặp lỗi phục hồi được (ví dụ lệnh không tồn tại),
# nên không thể lấy "có main.pdf" làm bằng chứng build sạch. Phải soi log tìm dòng
# bắt đầu bằng "!" — đó là lỗi thật của TeX.
$fatal = @(Select-String -Path $log -Pattern '^! ')
$undef = @(Select-String -Path $log -Pattern 'LaTeX Warning: (Reference|Citation).*undefined')
$over  = @(Select-String -Path $log -Pattern 'Overfull \\hbox \(([2-9][0-9]|[0-9]{3,})')
$miss  = @(Select-String -Path $log -Pattern 'LaTeX Warning: File .* not found')
$todo = (Select-String -Path $SrcTex -Pattern '\\todoText' -AllMatches |
         ForEach-Object { $_.Matches.Count } | Measure-Object -Sum).Sum

# Đếm trang: đọc từ log của XeTeX ("Output written on main.pdf (48 pages)") —
# tin cậy hơn việc đếm object trong PDF vì PDF hiện đại nén object stream.
$pages = '?'
$m = Select-String -Path $log -Pattern 'Output written on .*\((\d+) pages?' | Select-Object -Last 1
if ($m) { $pages = $m.Matches[0].Groups[1].Value }

$sw.Stop()
Say ""
Say ("  PDF     : {0}" -f $OutPdf) 'Green'
Say ("  {0} KB, {1} trang, {2:N1} giây" -f [math]::Round((Get-Item $OutPdf).Length / 1KB), $pages, $sw.Elapsed.TotalSeconds) 'Green'
Say ""
if ($fatal.Count) {
    Say ("  {0} LỖI TeX trong log — PDF vẫn sinh ra nhưng nội dung có thể sai:" -f $fatal.Count) 'Red'
    $fatal | Select-Object -First 5 | ForEach-Object { Say ("    " + $_.Line.Trim()) 'Red' }
    Say ("    chi tiết: {0}" -f $log) 'DarkGray'
}
if ($undef.Count) { Say ("  {0} tham chiếu/trích dẫn HỎNG — chạy thêm lượt hoặc kiểm \label" -f $undef.Count) 'Red' }
if ($miss.Count)  { Say ("  {0} file hình KHÔNG TÌM THẤY:" -f $miss.Count) 'Red'
                    $miss | Select-Object -First 5 | ForEach-Object { Say ("    " + $_.Line.Trim()) 'Red' } }
if ($over.Count)  { Say ("  {0} dòng tràn lề trên 20pt (chỉ là vấn đề thẩm mỹ)" -f $over.Count) 'Yellow' }
if (-not $fatal.Count -and -not $undef.Count -and -not $miss.Count -and -not $over.Count) {
    Say "  không có lỗi hay cảnh báo đáng chú ý" 'Green'
}
if ($todo) { Say ("  còn {0} chỗ \todoText phải tự điền (hiện màu đỏ trong PDF)" -f $todo) 'Yellow' }

if ($Open) { Start-Process $OutPdf }
