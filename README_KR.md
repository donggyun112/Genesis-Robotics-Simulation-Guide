# M2 Mac Genesis 로봇 시뮬레이션 환경 구축 가이드

## 1. Homebrew 설치

Homebrew는 macOS용 패키지 관리자입니다. 터미널에서 다음 명령어를 실행하세요:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

설치 후 환경 초기화:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 2. Miniforge 설치

Apple Silicon 환경에 최적화된 Conda 배포판을 설치합니다:

```bash
brew install miniforge
```

셸 초기화 (zsh 기준):

```bash
conda init zsh
source ~/.zshrc
```

## 3. Python 가상 환경 설정

Python 3.10 기반의 가상 환경을 생성하고 활성화합니다:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

## 4. FFmpeg 설치

비디오 처리를 위한 FFmpeg 설치:

```bash
brew install ffmpeg
```

## 5. Genesis 설치

가상 환경에 Genesis 패키지를 설치:

```bash
pip install genesis-world
```

## 설치 확인

Genesis가 정상적으로 설치되었는지 확인하려면 Python 인터프리터에서 다음을 실행해보세요:

```python
import genesis
print(genesis.__version__)
```

## 주의사항

* 모든 명령어는 반드시 순서대로 실행해야 합니다
* 가상 환경 이름(myenv)은 원하는 이름으로 변경 가능합니다
* 설치 과정에서 오류가 발생하면 각 단계의 로그를 확인하세요

## 기본 로봇팔 학습 예제 코드

* 올라와있는 example.py를 테스트하실 수 있습니다 로봇팔 강화학습 훈련 코드를 실행시킬 수 있습니다.

