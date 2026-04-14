---
title: "[Linux] 비밀번호 없는 SSH 접속: Ubuntu 24.04에서 키 인증과 ssh.service 설정까지"
date: 2026-04-13 00:00:00 +0900
categories: [Tech, Linux]
tags: [ssh, remote, linux, ubuntu, ubuntu_24_04, security, public_key, systemd, server, infra]
description: "SSH 키 인증의 원리부터 Ubuntu 24.04의 ssh.socket/ssh.service 차이, 안전한 접속 설정까지 한 번에 정리한다."
math: false
---

> 서버를 만지기 시작하면 결국 가장 자주 쓰게 되는 문이 하나 있다. 바로 **SSH(Secure Shell)** 다. 처음에는 비밀번호만 알아도 접속이 되니 편해 보이지만, 서버를 오래 운영할수록 “편한 설정”이 아니라 “예측 가능한 설정”이 더 중요해진다. 이 글에서는 **SSH 키 인증**으로 접속을 안전하게 바꾸는 과정부터, Ubuntu 24.04에서 헷갈리기 쉬운 **`ssh.socket`과 `ssh.service`의 차이**, 그리고 실제 운영에서 덜 흔들리는 설정 흐름까지 한 번에 정리해본다.

---

## 1. 왜 아직도 SSH를 제대로 설정해야 할까?

개인 서버든, 클라우드 VM이든, 라즈베리파이든 서버 운영의 출발점은 결국 원격 접속이다. 그런데 많은 초보 설정은 여기서 멈춘다.

* 비밀번호 로그인 켜둔 채 그대로 사용한다.
* `openssh-server`만 설치하고 실제 서비스 동작 방식은 확인하지 않는다.
* 키 인증이 된 뒤에도 `PasswordAuthentication`을 끄지 않는다.

문제는 이 상태가 “일단 되니까 괜찮아 보이는” 상태라는 점이다. 하지만 외부에 노출된 서버는 생각보다 빨리 스캔당하고, SSH 포트에는 자동화된 로그인 시도가 계속 들어온다. 그래서 SSH는 단순한 접속 도구가 아니라, **서버 운영의 첫 번째 보안 경계선**으로 이해하는 편이 맞다.

> **비밀번호 로그인은 왜 위험할까?**  
> 비밀번호는 사람이 기억할 수 있는 길이와 형태를 가져야 하므로 결국 공격 가능한 표적이 된다. 반면 공개키 기반 인증은 길고 복잡한 키 쌍을 사용하므로 무차별 대입 공격에 훨씬 강하다.
{: .prompt-question }

| 항목 | 비밀번호 로그인 | SSH 키 로그인 |
|:---:|:---|:---|
| 편의성 | 처음엔 쉽다 | 처음 한 번만 설정하면 편하다 |
| 보안성 | 추측·재사용 위험이 크다 | 개인키를 잘 보관하면 훨씬 강하다 |
| 자동화 | 불편하다 | 배포/스크립트/원격 작업에 유리하다 |
| 운영 안정성 | 사람 실수에 취약하다 | 설정만 잡히면 일관성이 높다 |

![ssh overview](/assets/img/2026-04-13/01-ssh-overview.png)
_로컬 PC에서 원격 서버로 SSH 접속이 이뤄지는 전체 흐름_

---

## 2. SSH 키 인증은 어떻게 동작할까?

SSH 키 인증은 **공개키(Public Key)** 와 **개인키(Private Key)** 한 쌍으로 동작한다.

* **공개키**는 서버에 둔다.
* **개인키**는 내 컴퓨터에 안전하게 보관한다.
* 접속 시 서버는 “이 사용자가 진짜 개인키를 갖고 있는가?”를 확인한다.

즉, 서버 쪽에 자물쇠를 걸어두고, 내 컴퓨터가 열쇠를 들고 들어가는 구조라고 보면 된다. 이때 중요한 것은 **공개키는 노출되어도 괜찮지만, 개인키는 절대 노출되면 안 된다**는 점이다.

> **핵심은 ‘비밀번호를 서버에 맞추는 방식’이 아니라, ‘내가 가진 열쇠를 증명하는 방식’으로 바뀐다는 것**이다.
{: .prompt-info }

![public key and private key diagram](/assets/img/2026-04-13/02-key-diagram.png)
_공개키는 서버에, 개인키는 로컬 PC에 두는 SSH 키 인증 구조_

---

## 3. 1단계: 내 컴퓨터에서 SSH 키 만들기

요즘 가장 무난한 선택은 `ed25519` 알고리즘이다. 키 길이 대비 보안성이 좋고, 속도도 빠르며, 현대적인 기본 선택지에 가깝다.

```bash
ssh-keygen -t ed25519 -C "my-server-key"
```
> -C 옵션으로 key 이름을 지정해줄 수 있다. 안쓰면 현재 사용자이름으로 자동 생성 된다.

![ssh-keygen start and passphrase prompt](/assets/img/2026-04-13/ssh_keygen-start&passphrase-prompt.png)
_`ssh-keygen` -f 옵션으로 저장 경로를 지정할 수 있고, passphrase를 입력하는 화면_


명령을 실행하면 보통 아래 순서로 질문이 나온다.

1. 키 파일을 어디에 저장할지
2. 암호문(passphrase)을 걸지

기본값을 그대로 쓰면 일반적으로 다음 경로가 생성된다.

* 개인키: `~/.ssh/id_ed25519`
* 공개키: `~/.ssh/id_ed25519.pub`

![ssh directory files](/assets/img/2026-04-13/ssh-directory-files.png)
_`~/.ssh` 디렉터리에 `id_ed25519`와 `id_ed25519.pub`가 생성된 모습_

공개키 내용은 아래처럼 확인할 수 있다.

```bash
cat ~/.ssh/id_ed25519.pub
```

> **passphrase는 꼭 걸어야 할까?**  
> 개인 노트북처럼 물리적으로 들고 다니는 장비라면 가능하면 설정하는 편이 좋다. 다만 자동화 서버나 학습용 환경에서는 편의성을 위해 생략하기도 한다. 중요한 것은 “생략해도 되는가”가 아니라 “왜 생략했는가를 알고 선택하는가”다.
{: .prompt-question }

개인키 파일은 권한 관리도 중요하다.

```bash
chmod 600 ~/.ssh/id_ed25519
chmod 700 ~/.ssh
```
>>>> 권한 설정 하는 이유

---

## 4. 2단계: 서버에 공개키 등록하기

키를 만들었다면 이제 서버가 이 키를 신뢰하도록 공개키를 올려야 한다. 가장 쉬운 방법은 `ssh-copy-id`다.

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<host>
```

![ssh-copy-id](/assets/img/2026-04-13/ssh-copy-id.png)
![ssh-copy-id success](/assets/img/2026-04-13/ssh-copy-id-success.png)
_공개키 등록이 완료되어 이후 키 기반 로그인 준비가 끝난 상태_

이 명령은 서버에 접속한 뒤, 알아서 공개키를 원격 서버의 `~/.ssh/authorized_keys` 파일에 넣어준다. 처음 한 번은 비밀번호를 입력해야 할 수 있지만, 성공하면 이후부터는 개인키 기반 로그인이 가능해진다.

만약 `ssh-copy-id`를 쓸 수 없는 환경이라면 서버에서 직접 수동으로도 넣어줄 수 있다.

#### **공개키 예시**
```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGxVQ9iZF80HnXEkvo1S4yP6tzZK4mtxyH1RBpunvSZZ my-server-key
```

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

그리고 로컬에서 확인한 공개키 문자열 한 줄을 그대로 붙여넣으면 된다.

> **서버 쪽 권한이 왜 중요할까?**  
> SSH는 보안을 위해 `.ssh` 디렉터리나 `authorized_keys`의 권한이 너무 느슨하면 키를 무시할 수 있다. “키는 맞는데 로그인 안 된다”는 문제의 상당수가 사실은 권한 문제다.
{: .prompt-question }

등록 후에는 바로 테스트한다.

```bash
ssh <user>@<host>
```

이 단계에서는 아직 비밀번호 로그인을 끄지 말고, **키 로그인이 실제로 되는지 먼저 확인**하는 것이 안전하다.


---

## 5. 3단계: `sshd_config`에서 비밀번호 로그인을 끄기 전에 볼 것들

키 로그인이 성공했다면 이제야 서버 문을 제대로 잠글 차례다. 설정 파일은 보통 아래 경로에 있다.

```bash
sudo nano /etc/ssh/sshd_config
```

우선 가장 중요한 항목은 다음과 같다.

```text
PubkeyAuthentication yes
PasswordAuthentication no
PermitRootLogin no
```

의미는 각각 이렇다.

* `PubkeyAuthentication yes`: 공개키 로그인 허용
* `PasswordAuthentication no`: 비밀번호 로그인 금지
* `PermitRootLogin no`: root 직접 로그인 금지

여기서 꼭 기억해야 할 운영 순서가 있다.

1. 키 로그인 성공 확인
2. 다른 터미널 하나를 더 열어 재접속 테스트
3. 그다음 `PasswordAuthentication no` 적용

이 순서를 지키는 이유는 간단하다. **설정을 잘못 건드렸을 때 내 접속 문까지 같이 잠가버리면 복구가 번거롭기 때문**이다.

> **운영에서는 "설정을 바꾸는 것"보다 "언제 바꿔도 복구 가능하게 바꾸는 것"이 더 중요하다.**
{: .prompt-info }

![sshd config full](/assets/img/2026-04-13/sshd_config-full.png)
_`/etc/ssh/sshd_config`에서 주요 SSH 설정을 점검하는 화면_

![sshd config password and pubkey](/assets/img/2026-04-13/sshd-config-password-pubkey.png)
_`PasswordAuthentication`과 `PubkeyAuthentication`을 조정한 모습_

설정을 저장했다면 아래처럼 SSH 서비스를 재시작해 반영한다.

```bash
sudo systemctl restart ssh
```

---

## 6. Ubuntu 24.04에서 특히 헷갈리는 지점: `ssh.socket` vs `ssh.service`

이 글에서 가장 실전적인 포인트는 여기다. 예전에는 SSH 데몬이 그냥 항상 떠 있는 `ssh.service` 방식으로 이해하면 됐다. 그런데 Ubuntu 24.04 환경에서는 `ssh.socket`이 보이는 경우가 있어 처음 보면 꽤 헷갈린다.

두 방식의 차이는 간단히 말해 이렇다.

| 항목 | `ssh.service` | `ssh.socket` |
|:---:|:---|:---|
| 동작 방식 | SSH 데몬이 항상 실행됨 | 소켓이 먼저 대기하다가 접속 시 서비스 호출 |
| 장점 | 직관적이고 관리가 쉽다 | 유휴 자원을 조금 더 아낄 수 있다 |
| 단점 | 항상 프로세스가 떠 있다 | 포트 변경·동작 파악이 더 헷갈릴 수 있다 |
| 추천 상황 | 개인 서버, 학습 환경, 소규모 운영 | 대규모 시스템 최적화가 중요한 경우 |

소규모 서버를 직접 운영하는 입장에서는 보통 `ssh.service`가 더 다루기 쉽다. 왜냐하면 설정 파일을 바꾸고, 재시작하고, 상태를 확인하는 흐름이 훨씬 직관적이기 때문이다.

```bash
sudo systemctl status ssh
sudo systemctl status ssh.socket
```

만약 `socket` 기반으로 동작하고 있고, 이를 익숙한 `service` 방식으로 바꾸고 싶다면 보통 아래처럼 처리한다.

```bash
sudo systemctl disable --now ssh.socket
sudo systemctl enable --now ssh.service
```

변경 후에는 꼭 상태를 다시 확인한다.

```bash
sudo systemctl status ssh.service
ss -tlnp | grep ssh
```

> **왜 `ssh.socket`이 헷갈릴까?**  
> SSH 포트는 `sshd_config`만 바꾸면 끝날 것 같지만, 실제로는 어떤 유닛이 포트를 잡고 있는지까지 봐야 하기 때문이다. 즉 “설정 파일 하나”의 문제가 아니라 “systemd가 어떻게 SSH를 기동하는가”의 문제다.
{: .prompt-question }

![systemctl status ssh](/assets/img/2026-04-13/after-setting-systemctl-status-ssh.png)
_설정 반영 후 `systemctl status ssh`로 서비스 상태를 확인한 화면_

![systemctl status ssh.socket](/assets/img/2026-04-13/after-setting-ssh.socket-status.png)
_`ssh.socket` 상태를 함께 확인해 어떤 유닛이 실제로 동작 중인지 비교한 화면_

![ss -tlnp output](/assets/img/2026-04-13/port-listening-ss-tlnp.png)
_`ss -tlnp`로 SSH가 어떤 포트에서 대기 중인지 확인한 결과_

---

## 7. 포트 변경까지 한다면 무엇을 같이 봐야 할까?

보안을 위해 SSH 포트를 기본값 22에서 다른 값으로 바꾸는 경우도 많다. 다만 이것은 “숨기기”에 가깝지 “본질적인 보안 강화”는 아니다. 핵심은 여전히 **키 인증**과 **비밀번호 로그인 차단**이다.

포트를 바꾼다면 최소한 아래 3가지를 함께 확인해야 한다.

1. `/etc/ssh/sshd_config`의 `Port` 값
2. 방화벽(UFW, 보안 그룹 등) 허용 여부
3. 실제 기동 방식이 `ssh.service`인지 `ssh.socket`인지

위에서 계속 본 것처럼 필자는 포트를 1014로 설정했는데 2222로 설정하는 예시를 들겠다.
위 설정을 다음과 같이 설정할 수 있다.

```text
Port 2222
PubkeyAuthentication yes
PasswordAuthentication no
```

그리고 접속은 -p 옵션으로 포트를 지정해서 접속한다.

```bash
ssh -p 2222 <user>@<host>
```

방화벽을 함께 쓴다면 예를 들어 이런 작업도 필요하다.

```bash
sudo ufw allow 2222/tcp
sudo ufw reload
```

> **포트 변경은 ‘부가 방어막’이지 ‘주 방어막’이 아니다.**  
> 키 인증을 안 해둔 상태에서 포트만 바꾸는 것은 문을 튼튼하게 잠그는 대신 초인종 위치만 바꿔놓는 것에 가깝다.
{: .prompt-info }

![sshd port config](/assets/img/2026-04-13/sshd-config-port.png)
_`sshd_config`에서 SSH 포트를 변경한 설정 화면_

---

## 8. 매번 긴 명령을 치기 싫다면 `~/.ssh/config`를 쓰자

SSH를 한두 번만 쓰는 것이 아니라면 로컬 설정 파일을 만드는 편이 훨씬 편하다.

```text
Host myserver
    HostName 103.11.113.10
    User ubuntu
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
```

이제부터는 이렇게 접속하면 된다.

```bash
ssh myserver
```

이 방식이 좋은 이유는 단순히 타이핑이 줄어서가 아니다. 접속 대상이 늘어날수록 “어느 서버에 어떤 포트와 어떤 키를 쓰는지”를 사람 머리로 기억하기 어려워지는데, `~/.ssh/config`는 그 정보를 선언적으로 정리해준다.

> **여러 서버를 운영할수록 `~/.ssh/config`는 편의 기능이 아니라 실수 방지 장치가 된다.**
{: .prompt-info }

---

## 9. 실전에서 자주 막히는 문제들

설정을 따라 했는데도 SSH 키 로그인이 안 되는 경우가 있다. 이때는 대부분 아래 범위에서 원인이 나온다.

### 9-1. 권한 문제

* `~/.ssh` 권한이 너무 넓다 (`chmod 700 ~/.ssh`)
* `authorized_keys` 권한이 너무 넓다 (`chmod 600 ~/.ssh/authorized_keys`)
* 홈 디렉터리 권한이 과도하게 열려 있다

### 9-2. 서비스 재기동·적용 문제

* 설정 파일을 수정했지만 서비스 재시작을 안 했다 (`sudo systemctl restart ssh`)
* 실제로는 `ssh.socket`이 포트를 잡고 있어서 기대와 다르게 동작한다

### 9-3. 클라이언트 키 선택 문제

* 접속하려는 키와 서버에 등록한 공개키가 서로 다르다
* `~/.ssh/config`에서 잘못된 `IdentityFile`을 지정했다

로그를 확인하면 생각보다 빨리 감이 잡힌다.

```bash
sudo journalctl -u ssh.service -n 50 --no-pager
```

또는 자세한 접속 로그를 보고 싶다면 클라이언트 쪽에서 다음처럼 디버그 옵션을 켠다.

```bash
ssh -vvv <user>@<host>
```

> **문제가 생겼을 때는 ‘다시 설정한다’보다 ‘로그를 본다’가 먼저다.**  
> SSH는 비교적 정직한 도구라서, 실패 원인의 상당수를 로그에서 직접 드러낸다.
{: .prompt-info }

---

## 10. 마무리: 좋은 SSH 설정은 화려하지 않고, 예측 가능하다

SSH 보안의 핵심은 생각보다 단순하다.

1. **`ed25519` 키를 만든다.**
2. **서버에 공개키를 등록한다.**
3. **키 로그인을 확인한 뒤 비밀번호 로그인을 끈다.**
4. **Ubuntu 24.04에서는 `ssh.socket`과 `ssh.service` 중 실제 동작 방식을 확인한다.**
5. **반복 접속 환경이라면 `~/.ssh/config`까지 정리한다.**

결국 좋은 운영은 기능이 많은 상태가 아니라, **문제가 생겼을 때 어디를 봐야 하는지가 분명한 상태**에 가깝다. 그런 의미에서 개인 서버나 학습용 환경에서는 자원 절약보다도 **가시성**과 **직관성**이 더 큰 가치가 된다. SSH는 단순한 접속 명령이 아니라, 운영 철학이 그대로 드러나는 출입문이다.
