#!/usr/bin/env python3
"""
daily_digest.py - Physical AI Daily Digest 메인 실행 파일

Physical AI 관련 RSS 피드를 수집하여 Slack 채널로 일일 요약을 전송합니다.
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

from llm_client import create_llm_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dotenv():
    """프로젝트 루트의 .env 파일에서 환경 변수 로드"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if key not in os.environ:
                    os.environ[key] = value


# 앱 시작 시 로드
load_dotenv()


def strip_html(html_content: str) -> str:
    """HTML 태그를 제거하고 순수 텍스트만 반환"""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return text
    except Exception:
        return html_content


class PhysicalAIDailyDigest:
    """Physical AI Daily Digest 봇 메인 클래스"""

    def __init__(self, webhook_url: str, config_path: str = "config.yaml"):
        """
        Args:
            webhook_url: Slack 웹훅 URL
            config_path: 설정 파일 경로
        """
        self.webhook_url = webhook_url
        self.config = self._load_config(config_path)
        self.feeds = self.config.get("feeds", {})
        self.llm_client = create_llm_client(self.config.get("llm", {}))

        # 프롬프트 로드
        self.system_prompt = self._load_prompt("prompts/system.txt")
        self.batch_prompt_template = self._load_prompt("prompts/translate_summarize_batch.txt")

        # 배치 처리 설정
        llm_config = self.config.get("llm", {})
        self.batch_size = llm_config.get("batch_size", 10)

        # 상태 파일 경로
        self.state_file = Path(__file__).parent / "sent_articles.json"

        # Slack 설정
        slack_config = self.config.get("slack", {})
        self.max_articles_per_category = slack_config.get("max_articles_per_category", 5)

        # 스케줄 설정
        schedule_config = self.config.get("schedule", {})
        self.lookback_hours = schedule_config.get("lookback_hours", 24)

        # 카테고리 분류 설정
        cat_config = self.config.get("categorization", {})
        self.categorization_enabled = cat_config.get("enabled", False)
        self.categories = cat_config.get("categories", {})

    def _load_config(self, path: str) -> Dict:
        """YAML 설정 파일 로드"""
        config_path = Path(__file__).parent / path
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML 파싱 오류: {e}")
            return {}

    def _load_prompt(self, path: str) -> Optional[str]:
        """프롬프트 파일 로드"""
        prompt_path = Path(__file__).parent / path
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return None

    def _categorize_article(self, article: Dict, processed: Dict) -> str:
        """기사를 카테고리로 분류"""
        if not self.categorization_enabled:
            return "general"

        # LLM이 제안한 카테고리 힌트 확인
        category_hint = processed.get("category_hint", "")
        if category_hint and category_hint in self.categories:
            return category_hint

        # 키워드 기반 분류
        search_text = " ".join([
            article.get("title", ""),
            article.get("description", ""),
            processed.get("translated_title", ""),
            processed.get("summary", ""),
        ]).lower()

        for cat_key, cat_info in self.categories.items():
            if cat_key == "general":
                continue
            keywords = cat_info.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in search_text:
                    return cat_key

        return "general"

    def _group_by_category(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """기사들을 카테고리별로 그룹화"""
        grouped = {}
        for item in articles:
            category = item.get("category", "general")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(item)
        return grouped

    def _load_state(self) -> Dict:
        """이전 상태 로드"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    today = datetime.now().strftime("%Y-%m-%d")
                    if state.get("date") != today:
                        return {"date": today, "sent_today": []}
                    return state
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"상태 파일 로드 실패: {e}")

        return {"date": datetime.now().strftime("%Y-%m-%d"), "sent_today": []}

    def _save_state(self, state: Dict):
        """상태 저장"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"상태 파일 저장 실패: {e}")

    def _generate_article_id(self, source: str, url: str) -> str:
        """기사 고유 ID 생성"""
        return hashlib.md5(f"{source}:{url}".encode()).hexdigest()

    def _parse_published_date(self, entry: Any) -> Optional[datetime]:
        """RSS 엔트리에서 발행일 추출"""
        date_fields = ["published_parsed", "updated_parsed", "created_parsed"]

        for field in date_fields:
            parsed = getattr(entry, field, None)
            if parsed:
                try:
                    return datetime(*parsed[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue

        return None

    def fetch_feeds(self) -> List[Dict]:
        """모든 RSS 피드에서 기사 수집"""
        articles = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        state = self._load_state()
        sent_ids = set(state.get("sent_today", []))

        for source_name, feed_url in self.feeds.items():
            logger.info(f"피드 수집 중: {source_name}")
            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo and feed.bozo_exception:
                    logger.warning(f"피드 파싱 경고 ({source_name}): {feed.bozo_exception}")

                for entry in feed.entries[:10]:
                    try:
                        url = entry.get("link", "")
                        article_id = self._generate_article_id(source_name, url)

                        if article_id in sent_ids:
                            continue

                        pub_date = self._parse_published_date(entry)
                        if pub_date and pub_date < cutoff_time:
                            continue

                        description = ""
                        if hasattr(entry, "summary"):
                            description = entry.summary
                        elif hasattr(entry, "description"):
                            description = entry.description

                        articles.append({
                            "id": article_id,
                            "title": entry.get("title", "제목 없음"),
                            "link": url,
                            "source": source_name,
                            "description": description,
                            "published": pub_date.isoformat() if pub_date else None,
                        })

                    except Exception as e:
                        logger.warning(f"엔트리 처리 오류 ({source_name}): {e}")
                        continue

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"피드 수집 실패 ({source_name}): {e}")
                continue

        logger.info(f"총 {len(articles)}개 기사 수집됨")
        return articles

    def _prepare_article_for_batch(self, article: Dict) -> Dict:
        """배치 처리를 위해 기사 정보를 준비"""
        return {
            "title": article["title"],
            "description": strip_html(article.get("description", "") or "")[:500],
            "source": article["source"],
        }

    def _create_fallback_result(self, article: Dict) -> Dict:
        """LLM 실패 시 폴백 결과 생성"""
        clean_desc = strip_html(article.get("description", "") or "")
        return {
            "translated_title": article["title"],
            "summary": clean_desc[:300],
            "category_hint": None,
        }

    def translate_and_summarize_batch(self, articles: List[Dict]) -> List[Dict]:
        """여러 기사를 배치로 번역 및 요약"""
        if not articles:
            return []

        if not self.llm_client:
            return [self._create_fallback_result(article) for article in articles]

        if not self.batch_prompt_template:
            logger.warning("배치 프롬프트 템플릿이 없습니다.")
            return [self._create_fallback_result(article) for article in articles]

        # 배치용 기사 정보 준비
        articles_data = []
        for idx, article in enumerate(articles):
            prepared = self._prepare_article_for_batch(article)
            prepared["index"] = idx
            articles_data.append(prepared)

        articles_json = json.dumps(articles_data, ensure_ascii=False, indent=2)
        prompt = self.batch_prompt_template.format(articles_json=articles_json)

        try:
            logger.info(f"배치 처리 중: {len(articles)}개 기사")
            response = self.llm_client.generate(prompt=prompt, system_prompt=self.system_prompt)

            # JSON 추출
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:])
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()

            results = json.loads(json_str)

            if not isinstance(results, list):
                logger.warning("배치 결과가 리스트가 아닙니다.")
                return [self._create_fallback_result(article) for article in articles]

            # article_index 기준으로 정렬
            sorted_results = [None] * len(articles)
            for result in results:
                idx = result.get("article_index", -1)
                if 0 <= idx < len(articles):
                    sorted_results[idx] = result

            for idx, result in enumerate(sorted_results):
                if result is None:
                    logger.warning(f"기사 {idx} 결과 누락, 폴백 처리")
                    sorted_results[idx] = self._create_fallback_result(articles[idx])

            logger.info(f"배치 처리 완료: {len(articles)}개 기사")
            return sorted_results

        except json.JSONDecodeError as e:
            logger.warning(f"배치 JSON 파싱 실패: {e}")
        except Exception as e:
            logger.warning(f"배치 처리 실패: {e}")

        return [self._create_fallback_result(article) for article in articles]

    def create_slack_block(self, article: Dict, processed: Dict) -> Dict:
        """Slack Block 생성"""
        title = processed.get("translated_title", article["title"])[:150]
        summary = processed.get("summary", "")[:500]
        link = article["link"]
        source = article["source"]

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*<{link}|{title}>*\n{summary}\n_출처: {source}_"
            }
        }

    def _send_webhook(self, payload: Dict) -> bool:
        """Slack 웹훅으로 페이로드 전송"""
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. {retry_after}초 후 재시도...")
                time.sleep(retry_after)
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )

            response.raise_for_status()
            return True

        except requests.RequestException as e:
            logger.error(f"Slack 전송 실패: {e}")
            return False

    def send_to_slack(self, processed_articles: List[Dict]) -> bool:
        """Slack으로 메시지 전송"""
        if not processed_articles:
            logger.info("전송할 기사가 없습니다.")
            return True

        # 헤더
        today = datetime.now().strftime("%Y년 %m월 %d일")
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🤖 Physical AI Daily Digest - {today}",
                    "emoji": True
                }
            },
            {"type": "divider"}
        ]

        if self.categorization_enabled:
            # 카테고리별 그룹화
            grouped = self._group_by_category(processed_articles)
            category_order = ["nvidia", "robotics", "airesearch", "research", "industry", "korea", "general"]

            for cat_key in category_order:
                if cat_key not in grouped:
                    continue

                articles = grouped[cat_key][:self.max_articles_per_category]
                if not articles:
                    continue

                cat_info = self.categories.get(cat_key, {})
                cat_name = cat_info.get("name", f"📰 {cat_key}")

                # 카테고리 헤더
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{cat_name}* ({len(articles)}개)"
                    }
                })

                # 기사들
                for item in articles:
                    blocks.append(item["block"])

                blocks.append({"type": "divider"})
        else:
            # 카테고리 없이 전송
            for item in processed_articles[:self.max_articles_per_category * 5]:
                blocks.append(item["block"])

        # Slack은 블록 50개 제한
        if len(blocks) > 50:
            blocks = blocks[:49]
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_...더 많은 기사가 생략되었습니다._"}
            })

        payload = {"blocks": blocks}

        if not self._send_webhook(payload):
            return False

        logger.info(f"Slack 전송 성공: {len(processed_articles)}개 기사")
        return True

    def run(self):
        """메인 실행 로직"""
        logger.info("Physical AI Daily Digest 시작")

        # 1. 피드 수집
        articles = self.fetch_feeds()
        if not articles:
            logger.info("새로운 기사가 없습니다.")
            return

        # 2. 번역 및 요약 (배치 처리)
        processed_articles = []
        state = self._load_state()
        sent_ids = set(state.get("sent_today", []))

        rate_limit_delay = self.config.get("llm", {}).get("rate_limit_delay", 1)

        for batch_start in range(0, len(articles), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(articles))
            batch_articles = articles[batch_start:batch_end]

            logger.info(f"배치 처리 중: {batch_start + 1}-{batch_end}/{len(articles)}")

            try:
                batch_results = self.translate_and_summarize_batch(batch_articles)

                for article, processed in zip(batch_articles, batch_results):
                    block = self.create_slack_block(article, processed)
                    category = self._categorize_article(article, processed)

                    processed_articles.append({
                        "article": article,
                        "processed": processed,
                        "block": block,
                        "category": category,
                    })
                    sent_ids.add(article["id"])

                if batch_end < len(articles) and self.llm_client:
                    logger.info(f"Rate limit 대기: {rate_limit_delay}초")
                    time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"배치 처리 실패: {e}")
                for article in batch_articles:
                    processed = self._create_fallback_result(article)
                    block = self.create_slack_block(article, processed)
                    category = self._categorize_article(article, processed)
                    processed_articles.append({
                        "article": article,
                        "processed": processed,
                        "block": block,
                        "category": category,
                    })
                    sent_ids.add(article["id"])

        # 3. Slack 전송
        if processed_articles:
            success = self.send_to_slack(processed_articles)

            if success:
                state["sent_today"] = list(sent_ids)
                self._save_state(state)
                logger.info(f"총 {len(processed_articles)}개 기사 전송 완료")
            else:
                logger.error("Slack 전송 실패")
        else:
            logger.info("처리된 기사가 없습니다.")

        logger.info("Physical AI Daily Digest 완료")


def main():
    """메인 함수"""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.error("SLACK_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    bot = PhysicalAIDailyDigest(webhook_url=webhook_url)
    bot.run()


if __name__ == "__main__":
    main()
