---
title: "[Windows] PowerShell 7 터미널 세팅: 설치, 테마, 필수 도구"
date: 2026-06-26 00:00:00 +0900
categories: [Tech, Windows]
tags: [windows, powershell, powershell-7, windows-terminal, scoop, oh-my-posh, nerd-font, cli]
description: "Windows에서 PowerShell 7을 설치하고 Scoop, Oh My Posh, Nerd Font, 자주 쓰는 CLI 도구까지 빠르게 세팅하는 실전 가이드."
math: false
---

Windows에서 개발용 터미널을 깔끔하게 쓰고 싶다면 **PowerShell 7 + Windows Terminal + Scoop + Oh My Posh + Nerd Font** 조합이면 충분하다. 이 글을 통해 실제로 설치하고 확인하는 순서만 정리한다.

> **목표**  
> 이 글을 끝까지 따라 하면 PowerShell 7을 설치하고, 프롬프트 테마를 적용하고, `lsd`, `bat`, `zoxide`, `ripgrep` 같은 터미널 도구까지 사용할 수 있다.
{: .prompt-info }

---

## 전체 순서

| 순서 | 작업 | 권한 |
|:---:|:---|:---:|
| 1 | PowerShell 7 설치 | 관리자 권한 권장 |
| 2 | Scoop 설치 | 일반 사용자 권한 |
| 3 | Oh My Posh 설치 | 일반 사용자 권한 |
| 4 | Nerd Font 적용 | 일반 사용자 권한 |
| 5 | 필수 CLI 도구 설치 | 일반 사용자 권한 |
| 선택 | OpenSSH 기본 셸 변경 | 관리자 권한 필요 |

Scoop은 일반 사용자 PowerShell에서 설치한다. 관리자 PowerShell에서 설치하면 기본 설치가 막힐 수 있다.

---

## 1. PowerShell 7 설치

먼저 Windows Terminal 또는 PowerShell을 열고 PowerShell 7 패키지를 확인한다.

```powershell
winget search --id Microsoft.PowerShell --exact
```

아래 명령어를 통해 설치한다.

```powershell
winget install --id Microsoft.PowerShell --source winget --installer-type wix
```

설치가 끝나면 새 터미널을 열고 확인한다.

```powershell
pwsh --version
Get-Command pwsh | Select-Object Source
```

`pwsh`는 PowerShell 7 실행 명령이다.

기존 Windows PowerShell은 `powershell.exe`이고, PowerShell 7은 `pwsh.exe`다.

더 편하게 사용하기 위해서, windows 터미널을 열고 상단의 설정에서 **기본 프로필** 을 **PowerShell**로 설정해준다.

> OpenSSH 기본 셸처럼 고정 경로가 필요한 설정까지 할 계획이라면 `--installer-type wix` 방식이 다루기 쉽다. 일반적으로 `C:\Program Files\PowerShell\7\pwsh.exe` 경로를 사용할 수 있다.
{: .prompt-info }

---

## 2. Scoop 설치

이 단계부터는 **관리자 권한이 아닌 일반 PowerShell 7 탭**에서 진행한다.

실행 정책을 현재 사용자 범위에서만 바꾼다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Scoop을 설치한다.

```powershell
iwr -useb get.scoop.sh | iex
```

> `Invoke-WebRequest` 또는 `iwr` 실행 중 “알려진 호스트가 없습니다” 같은 오류가 나면서 Scoop 설치가 안 된다면, 현재 PowerShell 세션에서 TLS 1.2를 먼저 지정한 뒤 설치 명령을 다시 실행해본다.
>
> ```powershell
> [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
> iwr -useb get.scoop.sh | iex
> ```
{: .prompt-info }

설치 확인:

```powershell
scoop --version
```

기본 도구를 설치한다.

```powershell
scoop install curl sudo vim
```

- `curl`: HTTP 요청 확인
- `sudo`: 관리자 권한으로 명령 실행할 때 사용
- `vim`: 터미널 편집기

---

## 3. Oh My Posh 설치

Oh My Posh는 PowerShell 프롬프트를 보기 좋게 꾸미는 도구다.

```powershell
Install-Module posh-git -Scope CurrentUser -Force
Install-Module oh-my-posh -Scope CurrentUser -Force
```

설치 확인:

```powershell
oh-my-posh --version
```

---

## 4. PowerShell 프로필에 테마 적용

PowerShell은 시작할 때 `$PROFILE` 파일을 읽는다. 이 파일에 Oh My Posh 초기화 코드를 넣으면 새 터미널을 열 때마다 테마가 자동 적용된다.

메모장으로 연다.

```powershell
notepad $PROFILE
```

파일이 없다는 메시지가 나오면 새 파일로 만들면 된다.

아래 내용을 넣고 저장한다.

```powershell
Import-Module posh-git
oh-my-posh init pwsh --config "atomic" | Invoke-Expression
```

저장 후 새 PowerShell 7 탭을 열거나 현재 세션에서 바로 적용한다.

```powershell
. $PROFILE
```

테마를 바꾸고 싶다면 `atomic`만 다른 테마 이름으로 바꾸면 된다. 직접 만든 설정 파일을 쓰고 싶다면 아래와 같이 파일 경로를 넣는다.

```powershell
Import-Module posh-git
oh-my-posh init pwsh --config "~/Documents/PowerShell/custom.omp.json" | Invoke-Expression
```


---

## 5. Nerd Font 설치와 터미널 폰트 설정

Oh My Posh 아이콘이 네모(`□`)로 깨지면 폰트 문제다. Nerd Font를 설치하고 터미널에 적용해야 한다.

PDF 메모에서는 Meslo Nerd Font를 내려받아 설치한 뒤 `MesloLGMDZ FONT Mono`를 터미널 폰트로 지정한다.

```text
https://github.com/ryanoasis/nerd-fonts/releases/download/v3.4.0/Meslo.zip
```

위 zip 파일을 받은 뒤 압축을 풀고 MesloLGMDZ 계열 폰트를 설치한다. 다른 Nerd Font를 쓰고 싶다면 아래 페이지에서 받을 수 있다.

```text
https://www.nerdfonts.com/font-downloads
```

Windows Terminal에 적용한다.

1. Windows Terminal 실행
2. `설정` 열기
3. PowerShell 7 프로필 선택
4. `모양` 선택
5. 글꼴을 `MesloLGMDZ FONT Mono`로 변경
6. 저장 후 새 탭 열기

VS Code 터미널에서도 같은 폰트를 쓰려면:

1. `Ctrl + ,`로 설정 열기
2. `terminal font` 검색
3. `Terminal › Integrated: Font Family`에 아래 값 입력

```text
MesloLGMDZ FONT Mono
```

---

## 6. 필수 CLI 도구 설치

아래 도구들은 선택이지만, 설치해두면 Windows 터미널 사용감이 많이 좋아진다.

### Coreutils

Windows에서도 `ls`, `cat`, `cp`, `find`, `grep` 같은 Unix 스타일 명령을 사용할 수 있게 해준다.

```powershell
winget install Microsoft.Coreutils
```

PowerShell에서는 `ls`가 기존 alias와 충돌할 수 있으므로 확인할 때는 `.exe`를 붙인다.

```powershell
ls.exe --version
grep.exe --version
```

### lsd

`ls` 결과를 아이콘과 색상으로 보기 좋게 보여준다.

```powershell
scoop install lsd
```

자주 쓰면 `$PROFILE`에 alias를 추가한다.

```powershell
Set-Alias ll lsd
```

### bat

`cat`보다 보기 좋은 파일 출력 도구다. 코드 하이라이트와 줄 번호가 들어간다.

```powershell
scoop install bat
```

```powershell
bat .\README.md
```

### zoxide

자주 가는 폴더를 기억해서 빠르게 이동하게 해준다.

```powershell
scoop install zoxide
```

`$PROFILE`에 추가한다.

```powershell
Invoke-Expression (& { (zoxide init powershell | Out-String) })
```

사용 예시:

```powershell
z githubio
```

### ripgrep

코드나 문서에서 문자열을 빠르게 검색한다.

```powershell
scoop install ripgrep
```

```powershell
rg "PowerShell" .
```

---

## 7. 추천 `$PROFILE` 최종 예시

위 설정을 한 번에 정리하면 `$PROFILE`은 아래 정도면 충분하다.

```powershell
Import-Module posh-git

# Prompt theme
oh-my-posh init pwsh --config "atomic" | Invoke-Expression

# Fast directory jump
Invoke-Expression (& { (zoxide init powershell | Out-String) })

# Aliases
Set-Alias ll lsd
```

너무 많이 넣기보다, 실제로 쓰는 것만 유지하는 편이 좋다.

---

## 8. 선택: OpenSSH 접속 시 PowerShell 7을 기본 셸로 쓰기

Windows에 OpenSSH Server를 켜두고 SSH로 접속한다면 기본 셸을 PowerShell 7로 바꿀 수 있다. 이 단계는 SSH 서버를 쓰는 경우에만 필요하다.

관리자 권한 PowerShell에서 실행한다.

```powershell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\PowerShell\7\pwsh.exe" -PropertyType String -Force
```

`-Value`에는 실제 PowerShell 경로를 넣는다. 기본 Windows PowerShell 경로는 아래와 같다.

```text
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
```

PowerShell 7을 일반 설치 방식으로 설치했다면 보통 아래 경로를 쓴다.

```text
C:\Program Files\PowerShell\7\pwsh.exe
```

Microsoft Store 버전을 쓴다면 `WindowsApps` 아래의 `pwsh.exe` 경로를 확인해서 넣는다.

---

## 9. 설치 확인 체크리스트

마지막으로 아래 명령이 모두 동작하는지 확인한다.

```powershell
pwsh --version
scoop --version
oh-my-posh --version
Get-Command lsd, bat, zoxide, rg
```

아이콘이 깨지면 Nerd Font 적용을 다시 확인한다. `oh-my-posh` 명령을 찾지 못하면 Windows Terminal을 완전히 종료한 뒤 다시 열어 PATH가 갱신되었는지 확인한다.

---

## 요약

- PowerShell 7은 `pwsh`로 실행한다.
- Scoop은 일반 사용자 PowerShell에서 설치한다.
- Oh My Posh는 `$PROFILE`에 초기화 코드를 넣어 적용한다.
- 아이콘이 깨지면 Nerd Font를 Windows Terminal과 VS Code에 설정한다.
- `lsd`, `bat`, `zoxide`, `ripgrep`만 설치해도 터미널 사용성이 크게 좋아진다.

---

## Reference

- [Windows에 PowerShell 설치 - Microsoft Learn](https://learn.microsoft.com/ko-kr/powershell/scripting/install/install-powershell-on-windows)
- [Scoop Installer](https://github.com/ScoopInstaller/Install)
- [Oh My Posh Windows 설치](https://ohmyposh.dev/docs/installation/windows)
- [Oh My Posh Prompt 설정](https://ohmyposh.dev/docs/installation/prompt)
- [Oh My Posh Fonts 설정](https://ohmyposh.dev/docs/installation/fonts)
- [OpenSSH Server Configuration for Windows - Microsoft Learn](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration)
