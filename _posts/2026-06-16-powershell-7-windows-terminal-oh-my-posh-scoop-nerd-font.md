---
title: "[Windows] PowerShell 7 설치 후 Windows Terminal 꾸미기: Scoop, Oh My Posh, Nerd Font 설정"
date: 2026-06-16 00:00:00 +0900
categories: [Tech, Windows]
tags: [windows, powershell, powershell-7, pwsh, windows-terminal, winget, scoop, oh-my-posh, nerd-font, powershell-profile, openssh, cli]
description: "Windows에서 PowerShell 7을 설치하고 Windows Terminal, Scoop, Oh My Posh, Nerd Font, OpenSSH 기본 셸 설정까지 한 번에 정리한다."
math: false
---

Windows에서 개발 환경을 맞출 때 가장 먼저 손에 닿는 도구는 결국 터미널이다. 예전에는 Git Bash, WSL, Windows PowerShell을 상황에 따라 오가며 썼지만, 요즘은 **PowerShell 7 + Windows Terminal + Scoop + Oh My Posh** 조합만 갖춰도 꽤 쾌적한 CLI 환경을 만들 수 있다.

이 글은 **“PowerShell 7 설치”**, **“Windows Terminal Oh My Posh 설정”**, **“Nerd Font 아이콘 깨짐 해결”**, **“Scoop으로 CLI 도구 설치”**, **“OpenSSH PowerShell 7 기본 셸 설정”**을 한 번에 찾는 사람을 위한 Windows 터미널 세팅 가이드다. PowerShell 7을 설치하고, 패키지 매니저와 프롬프트 테마를 설정한 뒤, `lsd`, `bat`, `zoxide`, `ripgrep` 같은 유용한 도구까지 추가하는 과정을 정리한다.

> **빠른 답변**  
> Windows에서 PowerShell 7 개발 터미널을 만들려면 `winget`으로 PowerShell 7을 설치하고, `scoop`으로 CLI 도구를 관리하며, `oh-my-posh`와 Nerd Font를 Windows Terminal 프로필에 연결하면 된다. OpenSSH에서 바로 PowerShell 7로 접속하려면 MSI 방식으로 설치한 뒤 `HKLM:\SOFTWARE\OpenSSH`의 `DefaultShell`을 `pwsh.exe` 경로로 지정한다.
{: .prompt-info }

```powershell
winget install --id Microsoft.PowerShell --source winget --installer-type wix
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
winget install JanDeDobbeleer.OhMyPosh --source winget
```

---

## 1. PowerShell 7은 왜 따로 설치할까?

Windows에는 이미 PowerShell이 들어 있다. 다만 기본으로 들어 있는 것은 보통 **Windows PowerShell 5.1**이고, 실행 파일 이름도 `powershell.exe`다. 반면 PowerShell 7은 크로스 플랫폼으로 개발되는 최신 PowerShell이며 실행 파일 이름은 **`pwsh.exe`**다.

| 구분 | Windows PowerShell | PowerShell 7 |
|:---:|:---|:---|
| 실행 파일 | `powershell.exe` | `pwsh.exe` |
| 기본 위치 | `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` | `C:\Program Files\PowerShell\7\pwsh.exe` 또는 MSIX/Store 경로 |
| 성격 | 호환성 유지를 위한 Windows 기본 셸 | 최신 PowerShell |
| 설치 방식 | Windows에 기본 포함 | 별도 설치 |
| 기존 환경 영향 | 기본 시스템 구성 | side-by-side 설치 |

> **PowerShell 7을 설치하면 기존 PowerShell이 사라질까?**  
> 아니다. PowerShell 7은 일반적으로 Windows PowerShell 5.1과 나란히 설치된다. 기존 스크립트를 바로 깨뜨리지 않고, 새 터미널 프로필에서 `pwsh`를 선택해 쓸 수 있다.
{: .prompt-question }

---

## 2. 준비물과 권한 구분

설정 전에 필요한 것은 많지 않다.

- Windows 10/11 또는 Windows Server 환경
- Windows Terminal
- App Installer 또는 `winget`
- 인터넷 연결
- 일반 사용자 PowerShell 7 탭
- 일부 단계에서만 관리자 권한 PowerShell

권한은 다음처럼 나누면 실수를 줄일 수 있다.

| 작업 | 권한 |
|:---|:---:|
| PowerShell 7 MSI 설치 | 관리자 권한 권장 |
| Scoop 설치 | 일반 사용자 권한 |
| `$PROFILE` 편집 | 일반 사용자 권한 |
| Oh My Posh 사용자 설정 | 일반 사용자 권한 |
| Nerd Font 사용자 설치 | 일반 사용자 권한 가능 |
| OpenSSH `DefaultShell` 레지스트리 설정 | 관리자 권한 필요 |
| `Restart-Service sshd` | 관리자 권한 필요 |

> **Scoop은 관리자 PowerShell에서 설치하지 않는 편이 좋다.**  
> Scoop의 기본 설치는 사용자 범위 설치다. 관리자 권한으로 설치하면 보안상 막히거나 별도 옵션이 필요할 수 있으므로, 일반 PowerShell 7 탭에서 설치하는 흐름을 추천한다.
{: .prompt-info }

---

## 3. PowerShell 7 설치하기

가장 간단한 방법은 `winget`을 사용하는 것이다. 다만 2026년 현재 Microsoft 문서 기준으로 PowerShell 7.6부터 `winget` 기본 설치는 MSIX 패키지를 사용할 수 있다. OpenSSH 기본 셸처럼 **고정된 `C:\Program Files\PowerShell\7\pwsh.exe` 경로가 필요한 경우**에는 MSI/WIX 설치 방식을 명시하는 것이 안전하다.

```powershell
winget search --id Microsoft.PowerShell --exact
winget install --id Microsoft.PowerShell --source winget --installer-type wix
```

설치가 끝나면 새 터미널을 열고 버전을 확인한다.

```powershell
pwsh --version
```

Windows Terminal을 쓰고 있다면 새 탭 메뉴에 **PowerShell** 또는 **PowerShell 7** 프로필이 추가되어 있을 수 있다. 보이지 않는다면 터미널을 완전히 종료한 뒤 다시 실행한다.

### 설치 경로 확인

PowerShell 7이 어디에 설치되었는지 확인하려면 아래 명령을 사용한다.

```powershell
Get-Command pwsh | Select-Object Source
$PSHOME
```

MSI/WIX 방식의 일반적인 설치 경로는 다음과 같다.

```text
C:\Program Files\PowerShell\7\pwsh.exe
```

### MSIX와 MSI 중 무엇을 고를까?

| 설치 방식 | 특징 | 추천 상황 |
|:---:|:---|:---|
| MSI/WIX | `C:\Program Files\PowerShell\7`처럼 예측 가능한 경로를 사용하기 쉽다. | OpenSSH 기본 셸, 서버, 고정 경로가 필요한 환경 |
| MSIX/Store | 업데이트와 앱 관리가 편하지만 `WindowsApps` 아래 경로를 사용할 수 있다. | 일반 데스크톱 사용, Store 중심 관리 |
| ZIP | 원하는 폴더에 압축 해제해 여러 버전을 나란히 둘 수 있다. | 테스트, 포터블 구성, 특정 버전 분리 |

MSIX 또는 Microsoft Store 버전은 `WindowsApps` 아래에 설치될 수 있고, 경로에 버전 문자열이 포함될 수 있다. 이 경로는 업데이트 후 바뀔 수 있으므로 OpenSSH `DefaultShell`처럼 고정 경로가 중요한 설정에는 MSI/WIX 방식이 다루기 쉽다.

---

## 4. Scoop으로 CLI 패키지 설치 준비하기

Windows에서 CLI 도구를 자주 설치한다면 Scoop을 함께 쓰는 편이 편하다. Scoop은 사용자 홈 디렉터리 아래에 도구를 설치하므로, 개발용 유틸리티를 가볍게 추가하고 지우기에 좋다.

먼저 **관리자 권한이 아닌 일반 PowerShell 7 탭**을 연다. 그 다음 현재 사용자 범위의 실행 정책을 확인한다.

```powershell
Get-ExecutionPolicy -Scope CurrentUser
```

필요하다면 현재 사용자 범위에서만 `RemoteSigned`로 바꾼다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

그 다음 Scoop을 설치한다.

```powershell
irm get.scoop.sh | iex
```

기존 축약형을 선호하면 아래처럼도 쓸 수 있다.

```powershell
iwr -useb get.scoop.sh | iex
```

설치 후 버전을 확인한다.

```powershell
scoop --version
```

기본적으로 자주 쓰는 도구를 먼저 설치해둔다. 예전에는 `sudo` 패키지를 같이 설치하는 예시가 많았지만, Windows에서 권한 상승 목적이라면 `gsudo`가 더 명확하다.

```powershell
scoop install curl gsudo vim
```

> **winget과 Scoop 중 무엇을 써야 할까?**  
> Windows 앱이나 Microsoft 공식 패키지는 `winget`, 개발용 CLI 도구는 `scoop`으로 나누면 관리가 편하다. 예를 들어 PowerShell 7과 Oh My Posh는 `winget`, `bat`이나 `ripgrep` 같은 작은 도구는 `scoop`으로 설치하는 식이다.
{: .prompt-info }

---

## 5. Oh My Posh로 프롬프트 꾸미기

터미널 프롬프트를 보기 좋게 꾸미고 Git 상태를 한눈에 보려면 Oh My Posh를 사용할 수 있다. 예전 글에서는 `Install-Module oh-my-posh` 방식이 자주 보이지만, 현재는 실행 파일을 설치하는 방식이 더 일반적이다.

```powershell
winget install JanDeDobbeleer.OhMyPosh --source winget
```

설치가 끝나면 Oh My Posh가 인식되는지 확인한다.

```powershell
oh-my-posh --version
```

Oh My Posh 테마만으로도 Git 상태 표시가 가능하다. Git 자동완성이나 기존 `posh-git` 기능이 필요하다면 선택적으로 `posh-git`을 추가한다. 설치 중 PSGallery 또는 NuGet provider 신뢰 확인 프롬프트가 나올 수 있다.

```powershell
Install-Module posh-git -Scope CurrentUser -Force
```

### `$PROFILE` 파일 열기

PowerShell은 시작할 때 `$PROFILE` 파일을 읽는다. 이 파일에 초기화 코드를 넣으면 터미널을 열 때마다 프롬프트 테마와 alias를 자동으로 적용할 수 있다.

먼저 프로필 파일을 만든다.

```powershell
New-Item -Path $PROFILE -Type File -Force
```

그 다음 메모장으로 연다.

```powershell
notepad $PROFILE
```

일반적인 PowerShell 7 프로필 경로는 다음과 비슷하다.

```text
C:\Users\<사용자명>\Documents\PowerShell\Microsoft.PowerShell_profile.ps1
```

### 테마 적용하기

예를 들어 `atomic` 테마를 쓰고 싶다면 `$PROFILE`에 아래 내용을 추가한다. `posh-git`을 설치하지 않았다면 첫 줄은 빼도 된다.

```powershell
Import-Module posh-git
oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH\atomic.omp.json" | Invoke-Expression
```

Oh My Posh만 최소 적용하려면 아래 한 줄이면 충분하다.

```powershell
oh-my-posh init pwsh | Invoke-Expression
```

테마 파일이 있는지 확인하려면 다음 명령을 사용한다.

```powershell
Get-ChildItem $env:POSH_THEMES_PATH | Where-Object Name -like '*atomic*'
```

다른 테마를 쓰고 싶다면 `$env:POSH_THEMES_PATH` 아래의 `.omp.json` 파일 이름만 바꾸면 된다.

```powershell
Get-ChildItem $env:POSH_THEMES_PATH | Select-Object Name
```

> **프롬프트 아이콘이 □ 모양으로 깨져 보인다면?**  
> Oh My Posh 테마는 Git 아이콘, 폴더 아이콘, 상태 아이콘을 많이 사용한다. 이 문자가 깨진다면 PowerShell 문제가 아니라 터미널 폰트 문제일 가능성이 높다. Nerd Font를 설치하고 Windows Terminal 프로필의 글꼴을 해당 폰트로 바꿔야 한다.
{: .prompt-question }

---

## 6. Nerd Font 설정하기

Oh My Posh를 제대로 쓰려면 Nerd Font가 필요하다. 가장 간단한 방법은 Oh My Posh의 폰트 설치 명령을 사용하는 것이다.

```powershell
oh-my-posh font install meslo
```

명령이 목록을 보여주면 `Meslo` 계열 폰트를 선택하면 된다. 수동으로 설치하고 싶다면 [Nerd Fonts 다운로드 페이지](https://www.nerdfonts.com/font-downloads) 에서 `Meslo` 또는 원하는 폰트를 내려받아 `.ttf` 파일을 설치한다. ([Meslo 바로 다운받기](https://github.com/ryanoasis/nerd-fonts/releases/download/v3.4.0/Meslo.zip))

그 다음 Windows Terminal에서 PowerShell 프로필의 폰트를 바꾼다.

1. Windows Terminal 실행
2. `설정` 열기
3. 왼쪽에서 PowerShell 7 프로필 선택
4. `모양` 또는 `Appearance` 선택
5. 글꼴을 `MesloLGM Nerd Font`, `MesloLGM Nerd Font Mono` 또는 설치한 Nerd Font 이름으로 변경
6. 저장 후 새 탭 열기

설치한 폰트 이름은 Windows Terminal의 글꼴 목록에 보이는 이름을 기준으로 고르면 된다. Meslo의 경우 환경에 따라 `MesloLGM Nerd Font`, `MesloLGM Nerd Font Mono`, `MesloLGMDZ Nerd Font Mono`처럼 보일 수 있다.

---

## 7. Windows Terminal 기본 프로필을 PowerShell 7로 바꾸기

PowerShell 7을 설치해도 Windows Terminal의 기본 탭이 자동으로 바뀌지 않을 수 있다. 기본 탭을 PowerShell 7로 고정하려면 다음 순서로 설정한다.

1. Windows Terminal 실행
2. `설정` 열기
3. `시작` 또는 `Startup` 메뉴 선택
4. `기본 프로필`에서 PowerShell 7 선택
5. 저장 후 Windows Terminal을 다시 열기

설정 JSON을 직접 편집한다면 `defaultProfile` 값이 PowerShell 7 프로필의 `guid`를 가리켜야 한다. UI에서 선택하는 방식이 실수할 가능성이 적다.

---

## 8. 같이 설치하면 좋은 CLI 도구

PowerShell 7만 설치해도 충분히 쓸 수 있지만, 몇 가지 도구를 더하면 Linux/Unix 스타일의 터미널 경험에 가까워진다.

### 8.1 Coreutils

Coreutils는 `ls`, `cat`, `cp`, `find`, `grep` 같은 기본 명령어를 Windows 환경에서도 네이티브로 사용할 수 있게 해주는 도구 모음이다. 다만 Windows용 Coreutils는 아직 preview 성격이고, PowerShell에서는 `ls`, `cat`, `find` 같은 이름이 기존 alias나 명령과 충돌할 수 있다.

```powershell
winget install Microsoft.Coreutils
```

설치 후에는 터미널을 다시 열고 `.exe`를 붙여 확인한다.

```powershell
ls.exe --version
grep.exe --version
Get-Command ls -All
```

PowerShell에서 `ls`만 입력하면 Coreutils가 아니라 `Get-ChildItem` alias가 실행될 수 있다. 실제 Coreutils 실행 파일을 확인할 때는 `ls.exe`처럼 확장자를 붙이면 확실하다.

### 8.2 lsd

`lsd`는 `ls` 결과를 아이콘과 색상으로 보기 좋게 보여주는 도구다.

```powershell
scoop install lsd
```

자주 쓴다면 `$PROFILE`에 alias를 추가할 수 있다.

```powershell
Set-Alias ll lsd
```

### 8.3 bat

`bat`은 `cat`처럼 파일 내용을 보여주지만, 문법 강조와 줄 번호가 들어가서 코드 파일을 읽기 좋다.

```powershell
scoop install bat
```

사용 예시는 다음과 같다.

```powershell
bat .\README.md
```

PowerShell에는 이미 `cat` alias가 있으므로, 바로 덮어쓰기보다 먼저 `bat` 명령 자체에 익숙해지는 것을 추천한다.

### 8.4 zoxide

`zoxide`는 자주 이동한 폴더를 기억해두고 빠르게 이동하게 해주는 도구다. 터미널에서 디렉터리를 많이 오간다면 체감이 크다.

```powershell
scoop install zoxide
```

설치 후 `$PROFILE`에 초기화 코드를 추가한다.

```powershell
Invoke-Expression (& { (zoxide init powershell | Out-String) })
```

이제 자주 가는 폴더로 이동한 기록이 쌓이면 아래처럼 이동할 수 있다.

```powershell
z githubio
```

### 8.5 ripgrep

`ripgrep`은 `grep`보다 빠르고 개발자 친화적인 검색 도구다. 코드베이스에서 특정 문자열을 찾을 때 유용하다.

```powershell
scoop install ripgrep
```

사용 예시는 다음과 같다.

```powershell
rg "PowerShell" .
```

---

## 9. 추천 `$PROFILE` 예시

위 설정을 한 번에 모으면 `$PROFILE`은 아래처럼 정리할 수 있다. `posh-git`을 설치하지 않았다면 `Import-Module posh-git` 줄은 제거한다.

```powershell
# Optional Git helper. Remove this line if posh-git is not installed.
Import-Module posh-git

# Oh My Posh theme
oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH\atomic.omp.json" | Invoke-Expression

# zoxide directory jumper
Invoke-Expression (& { (zoxide init powershell | Out-String) })

# Optional aliases
Set-Alias ll lsd
```

저장한 뒤 새 PowerShell 7 탭을 열면 적용된다. 바로 현재 세션에 적용하고 싶다면 아래 명령을 실행한다.

```powershell
. $PROFILE
```

프로필에서 오류가 나면 새 터미널을 열 때마다 같은 오류가 반복된다. 이럴 때는 `notepad $PROFILE`로 다시 열어 방금 추가한 줄을 주석 처리한 뒤 한 줄씩 다시 켜면 원인을 찾기 쉽다.

---

## 10. OpenSSH 접속 시 기본 셸을 PowerShell 7로 바꾸기

Windows에 OpenSSH Server를 켜두고 원격 접속하는 경우, 접속했을 때 기본 셸을 PowerShell 7로 바꾸고 싶을 수 있다. 이 설정은 OpenSSH **서버(sshd)** 설정이며, 클라이언트 `ssh.exe` 설정에는 적용되지 않는다. 관리자 권한 PowerShell에서 진행한다.

먼저 PowerShell 7 경로가 실제로 존재하는지 확인한다.

```powershell
$pwsh = "C:\Program Files\PowerShell\7\pwsh.exe"

if (-not (Test-Path $pwsh)) {
  throw "pwsh.exe not found. Install the MSI/WIX package or update the path."
}
```

그 다음 OpenSSH 기본 셸을 설정한다.

```powershell
New-ItemProperty `
  -Path "HKLM:\SOFTWARE\OpenSSH" `
  -Name DefaultShell `
  -Value $pwsh `
  -PropertyType String `
  -Force
```

설정 후 OpenSSH 서버를 재시작한다.

```powershell
Restart-Service sshd
```

기본 Windows PowerShell로 되돌리고 싶다면 아래 경로를 사용한다.

```powershell
$windowsPowerShell = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value $windowsPowerShell -PropertyType String -Force
Restart-Service sshd
```

> **Microsoft Store 또는 MSIX로 설치한 PowerShell 경로를 OpenSSH 기본 셸로 써도 될까?**  
> 가능은 하지만 권장하기 어렵다. MSIX/Store 앱 경로는 `C:\Program Files\WindowsApps\...` 아래에 버전이 포함될 수 있어 업데이트 후 경로가 바뀔 수 있다. OpenSSH 기본 셸처럼 고정 경로가 중요한 설정에는 `C:\Program Files\PowerShell\7\pwsh.exe` 형태가 더 관리하기 쉽다.
{: .prompt-info }

---

## 11. 자주 나는 문제와 해결 방법

### `$PROFILE` 파일이 없다고 나온다

처음에는 프로필 파일이 없을 수 있다. 아래 명령으로 만들면 된다.

```powershell
New-Item -Path $PROFILE -Type File -Force
```

### `oh-my-posh` 명령을 찾을 수 없다

설치 후에도 현재 터미널 세션의 PATH가 갱신되지 않았을 수 있다. Windows Terminal을 완전히 종료했다가 다시 실행한다. 그래도 안 되면 설치 상태를 확인한다.

```powershell
winget list OhMyPosh
Get-Command oh-my-posh
```

### 아이콘이 □ 모양으로 깨진다

Nerd Font가 설치되지 않았거나 Windows Terminal 프로필에 적용되지 않은 상태다. 폰트를 설치한 뒤 PowerShell 7 프로필의 글꼴을 Nerd Font로 바꾸고 새 탭을 연다.

### 스크립트 실행이 막힌다

현재 사용자 범위의 실행 정책을 확인한다.

```powershell
Get-ExecutionPolicy -Scope CurrentUser
```

개발 환경에서는 보통 현재 사용자 범위만 `RemoteSigned`로 바꿔도 충분하다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 테마 파일을 찾을 수 없다

Oh My Posh가 설치되었는지, 테마 경로 환경 변수가 있는지 확인한다.

```powershell
$env:POSH_THEMES_PATH
Get-ChildItem $env:POSH_THEMES_PATH
```

---

## 12. 설치 확인 체크리스트

설정을 마친 뒤 아래 명령으로 핵심 도구가 잡히는지 확인한다.

```powershell
pwsh --version
scoop --version
oh-my-posh --version
Get-Command lsd, bat, zoxide, rg
```

OpenSSH 기본 셸까지 바꿨다면 관리자 PowerShell에서 레지스트리 값을 확인한다.

```powershell
Get-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell
```

---

## 13. 전체 설치 순서 요약

처음부터 다시 정리하면 다음 순서로 진행하면 된다.

```powershell
# 1. PowerShell 7 설치: 고정 Program Files 경로가 필요하면 MSI/WIX 방식 사용
winget install --id Microsoft.PowerShell --source winget --installer-type wix

# 2. 새 터미널에서 PowerShell 7 확인
pwsh --version

# 3. 일반 사용자 PowerShell 7 탭에서 Scoop 설치
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# 4. 기본 CLI 도구 설치
scoop install curl gsudo vim lsd bat zoxide ripgrep
winget install Microsoft.Coreutils

# 5. Oh My Posh 설치
winget install JanDeDobbeleer.OhMyPosh --source winget

# 6. 선택: posh-git 설치
Install-Module posh-git -Scope CurrentUser -Force

# 7. Nerd Font 설치
oh-my-posh font install meslo

# 8. PowerShell 프로필 생성 및 편집
New-Item -Path $PROFILE -Type File -Force
notepad $PROFILE
```

그리고 `$PROFILE`에는 아래 내용을 넣는다.

```powershell
# Remove this line if posh-git is not installed.
Import-Module posh-git

oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH\atomic.omp.json" | Invoke-Expression
Invoke-Expression (& { (zoxide init powershell | Out-String) })
Set-Alias ll lsd
```

마지막으로 Nerd Font를 설치하고 Windows Terminal의 PowerShell 7 프로필 폰트로 지정하면 기본 세팅은 끝난다.

---

## FAQ

### Q. PowerShell 7과 Windows PowerShell 5.1의 차이는?

A. PowerShell 7은 `pwsh.exe`로 실행되는 최신 PowerShell이고, Windows PowerShell 5.1은 `powershell.exe`로 실행되는 Windows 기본 PowerShell이다. 둘은 나란히 설치해서 사용할 수 있다.

### Q. Windows PowerShell 5.1을 지우고 PowerShell 7만 써도 될까?

A. 지우는 방식으로 접근하지 않는 것이 좋다. Windows PowerShell 5.1은 Windows 관리 스크립트와 호환성 때문에 남겨두고, 개발용 기본 터미널을 PowerShell 7로 바꾸는 식이 안전하다.

### Q. PowerShell 7 설치 후 실행 명령은 무엇인가?

A. PowerShell 7은 `pwsh`로 실행한다. 기존 `powershell` 명령은 보통 Windows PowerShell 5.1을 가리킨다.

### Q. Oh My Posh에 Nerd Font가 꼭 필요한가?

A. 아이콘이 들어간 테마를 쓰려면 사실상 필요하다. Oh My Posh는 아이콘 문자를 출력하고, Nerd Font는 그 문자를 화면에 제대로 그려준다. 아이콘 없는 minimal 계열 테마를 쓰면 Nerd Font 의존을 줄일 수 있다.

### Q. Oh My Posh만 설치했는데 왜 아이콘이 깨질까?

A. 프롬프트 테마와 폰트는 별개다. Oh My Posh와 Nerd Font 설정을 함께 해야 한다. Nerd Font 설치 후 Windows Terminal 프로필의 글꼴까지 바꿔야 적용된다.

### Q. Scoop 없이 winget만 써도 될까?

A. 가능하다. 다만 개발용 CLI 도구를 자주 설치하고 지운다면 Scoop이 더 간단할 때가 많다. 반대로 Microsoft 공식 앱이나 Windows 앱은 winget이 자연스럽다. 둘을 같이 써도 괜찮지만 같은 도구를 양쪽에서 중복 설치하지 않는 편이 좋다.

### Q. OpenSSH 기본 셸 변경은 클라이언트에도 적용되는가?

A. 아니다. `HKLM:\SOFTWARE\OpenSSH`의 `DefaultShell`은 Windows OpenSSH Server, 즉 `sshd`에 접속했을 때의 기본 셸을 바꾸는 설정이다. 로컬에서 실행하는 OpenSSH Client 설정과는 별개다.

### Q. OpenSSH 기본 셸 변경은 꼭 해야 할까?

A. 원격 SSH 접속을 자주 하지 않는다면 필요 없다. Windows PC나 서버에 SSH로 접속했을 때 바로 PowerShell 7을 쓰고 싶은 경우에만 설정하면 된다.

---

## Reference

- [Windows에 PowerShell 설치 - Microsoft Learn](https://learn.microsoft.com/ko-kr/powershell/scripting/install/install-powershell-on-windows)
- [OpenSSH Server Configuration for Windows - Microsoft Learn](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration)
- [Oh My Posh Windows 설치](https://ohmyposh.dev/docs/installation/windows)
- [Oh My Posh Prompt 설정](https://ohmyposh.dev/docs/installation/prompt)
- [Oh My Posh Fonts 설정](https://ohmyposh.dev/docs/installation/fonts)
- [Scoop Installer](https://scoop.sh/)
- [Nerd Fonts Downloads](https://www.nerdfonts.com/font-downloads)
- [Microsoft Coreutils](https://github.com/microsoft/coreutils)
- [lsd](https://github.com/lsd-rs/lsd)
- [bat](https://github.com/sharkdp/bat)
- [zoxide](https://github.com/ajeetdsouza/zoxide)
- [ripgrep](https://github.com/BurntSushi/ripgrep)
