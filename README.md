# Physical AI Daily Digest Bot

Physical AI, 로보틱스, 시뮬레이션 관련 RSS 피드를 수집하여 Slack 채널로 일일 요약을 전송하는 봇입니다.

## 기능

- 7개 RSS 피드에서 기사 자동 수집
  - Korea: 로봇신문, 전자신문 AI
  - NVIDIA: Newsroom, Omniverse Blog
  - Robotics: The Robot Report, IEEE Spectrum
  - Research: arXiv cs.RO
- LLM 기반 한국어 번역 및 요약
- 카테고리별 그룹화 (NVIDIA, Robotics, Research, Industry, Korea)
- Slack Block Kit 형식의 리치 메시지
- 중복 기사 방지
- GitHub Actions 기반 서버리스 운영

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/physicalai-news-bot.git
cd physicalai-news-bot

# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 실제 값 입력
```

## 환경 변수

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL | O |
| `LLM_API_KEY` | LLM API 키 (config.yaml provider에 따라) | O |
| `GOOGLE_API_KEY` | Google Gemini API 키 | - |
| `OPENAI_API_KEY` | OpenAI API 키 | - |
| `ANTHROPIC_API_KEY` | Anthropic API 키 | - |

## 실행

```bash
# 로컬 실행
python daily_digest.py

# 또는 환경 변수와 함께
SLACK_WEBHOOK_URL="https://hooks.slack.com/..." LLM_API_KEY="..." python daily_digest.py
```

## Slack Webhook 설정

1. [Slack API](https://api.slack.com/apps)에서 새 앱 생성
2. "Incoming Webhooks" 활성화
3. "Add New Webhook to Workspace" 클릭
4. 채널 선택 후 Webhook URL 복사
5. `.env` 파일에 `SLACK_WEBHOOK_URL` 설정

## GitHub Actions 설정

1. 저장소 Settings > Secrets and variables > Actions
2. 다음 시크릿 추가:
   - `SLACK_WEBHOOK_URL`: Slack Webhook URL
   - `LLM_API_KEY`: LLM API 키

자동으로 매일 한국시간 오전 9시에 실행됩니다.

## 설정 파일

`config.yaml`에서 다음을 설정할 수 있습니다:

- LLM 프로바이더 및 모델
- RSS 피드 목록
- 카테고리 분류 규칙
- Slack 메시지 설정
- 스케줄 설정

## 라이선스

MIT License
