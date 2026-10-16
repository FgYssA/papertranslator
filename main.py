import os
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
import requests
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Metric, RunReportRequest
from google.oauth2 import service_account

# 環境変数の取得
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
GA4_PROPERTY_ID = os.environ.get("GA4_PROPERTY_ID", "").strip()
GA4_CREDENTIALS_JSON = os.environ.get("GA4_CREDENTIALS_JSON", "").strip()

# 設定値（モデルは gemini-3.8-flash に固定）
GEMINI_MODEL = "gemini-3.8-flash"
DATA_FILE = "docs/data.json"
ARXIV_QUERY = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.CV"
MAX_NEW_PAPERS = 2  # 毎朝2件の論文を処理


def load_existing_data():
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_latest_arxiv_papers(existing_urls, limit=2):
    url = f"http://export.arxiv.org/api/query?search_query={ARXIV_QUERY}&sortBy=submittedDate&sortOrder=descending&max_results=15"
    req = urllib.request.urlopen(url, timeout=30)
    xml_data = req.read()
    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        paper_url = entry.find("atom:id", ns).text.strip()
        if paper_url in existing_urls:
            continue
        title = " ".join(entry.find("atom:title", ns).text.strip().split())
        abstract = " ".join(entry.find("atom:summary", ns).text.strip().split())
        published = entry.find("atom:published", ns).text.strip()[:10]

        papers.append({
            "url": paper_url,
            "original_title": title,
            "abstract": abstract,
            "published_date": published
        })
        if len(papers) >= limit:
            break
    return papers


def fetch_jina_content(url):
    try:
        jina_url = f"https://r.jina.ai/{url}"
        res = requests.get(jina_url, timeout=20)
        if res.status_code == 200 and len(res.text) > 200:
            return res.text[:8000]
    except Exception as e:
        print(f"Jina Reader 取得スキップ: {e}")
    return None


def analyze_with_gemini(paper, jina_content):
    source_text = jina_content if jina_content else paper["abstract"]

    prompt = f"""
以下の海外最新論文の情報を分析し、日本の個人開発者・エンジニア・リサーチャー向けに「実務やマイクロSaaS開発にどう転用できるか」という視点で構造化してください。

【論文タイトル】: {paper["original_title"]}
【公開日】: {paper["published_date"]}
【内容】:
{source_text}
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "ja_title": {"type": "STRING", "description": "読者の目を引く分かりやすい日本語タイトル（40文字以内）"},
            "tags": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "関連する技術・応用分野タグ3個"},
            "problem": {"type": "STRING", "description": "これまで何が課題・限界だったのか（2〜3文で簡潔に）"},
            "mechanism": {"type": "STRING", "description": "どのように解決したか、コアとなる仕組みの非専門家向け解説（3〜4文）"},
            "key_advantage": {"type": "STRING", "description": "既存の常識や従来手法と比べた最大のブレイクスルー・数値的優位性"},
            "saas_ideas": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "idea_name": {"type": "STRING"},
                        "target_user": {"type": "STRING"},
                        "implementation": {"type": "STRING"},
                        "monetization": {"type": "STRING"}
                    },
                    "required": ["idea_name", "target_user", "implementation", "monetization"]
                }
            },
            "search_keyword": {"type": "STRING", "description": "Amazonで関連技術書を検索するためのキーワード"}
        },
        "required": ["ja_title", "tags", "problem", "mechanism", "key_advantage", "saas_ideas", "search_keyword"]
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "temperature": 0.4
        }
    }

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    res = requests.post(endpoint, json=payload, timeout=60)
    res.raise_for_status()

    result_json = res.json()
    raw_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_text)


def send_paper_to_discord(item):
    if not DISCORD_WEBHOOK_URL:
        return

    ideas_text = ""
    for i, idea in enumerate(item.get("saas_ideas", []), 1):
        ideas_text += (
            f"**💡 案{i}: {idea['idea_name']}**\n"
            f"・対象: {idea['target_user']}\n"
            f"・実装: {idea['implementation']}\n"
            f"・収益化: {idea['monetization']}\n\n"
        )

    embed = {
        "title": f"🌱 {item['ja_title']}",
        "url": item["url"],
        "color": 0x2563EB,
        "fields": [
            {"name": "■ 従来の課題", "value": item["problem"], "inline": False},
            {"name": "■ コア技術の仕組み", "value": item["mechanism"], "inline": False},
            {"name": "■ ブレイクスルー", "value": item["key_advantage"], "inline": False},
            {"name": "🚀 マイクロSaaS応用アイデア", "value": ideas_text.strip() or "なし", "inline": False},
        ],
        "footer": {
            "text": f"公開日: {item['published_date']} | タグ: {', '.join(item.get('tags', []))}"
        }
    }

    requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=30)


def fetch_ga4_yesterday_stats():
    """GA4から昨日の訪問者数・新規ユーザー数・PV数を取得する"""
    if not GA4_PROPERTY_ID or not GA4_CREDENTIALS_JSON:
        print("GA4の環境変数が未設定のため、アナリティクス取得をスキップします。")
        return None

    try:
        creds_info = json.loads(GA4_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/analytics.readonly"]
        )
        client = BetaAnalyticsDataClient(credentials=credentials)

        request = RunReportRequest(
            property=f"properties/{GA4_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="yesterday", end_date="yesterday")],
            metrics=[
                Metric(name="activeUsers"),
                Metric(name="newUsers"),
                Metric(name="screenPageViews"),
            ],
        )
        response = client.run_report(request)

        if response.rows:
            row = response.rows[0]
            return {
                "active_users": row.metric_values[0].value,
                "new_users": row.metric_values[1].value,
                "page_views": row.metric_values[2].value,
            }
        else:
            return {"active_users": "0", "new_users": "0", "page_views": "0"}
    except Exception as e:
        print(f"GA4データ取得エラー: {e}")
        return None


def send_analytics_to_discord(stats, total_archive_count):
    """2記事の送信後に、昨日のアクセス解析レポートをDiscordへ送る"""
    if not DISCORD_WEBHOOK_URL or stats is None:
        return

    embed = {
        "title": "📊 TechSeed Archive 昨日のアクセス日報",
        "description": "Google Analytics (GA4) による昨日1日分のサイト訪問レポートです。",
        "color": 0x10B981,
        "fields": [
            {
                "name": "👥 昨日の訪問者数",
                "value": f"**{stats['active_users']}** 人（うち新規: {stats['new_users']} 人）",
                "inline": True
            },
            {
                "name": "👀 昨日のページ閲覧数 (PV)",
                "value": f"**{stats['page_views']}** 回",
                "inline": True
            },
            {
                "name": "📚 現在のアーカイブ総数",
                "value": f"全 **{total_archive_count}** 記事",
                "inline": True
            }
        ]
    }
    requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=30)


def main():
    existing_data = load_existing_data()
    existing_urls = {item["url"] for item in existing_data}

    new_papers = fetch_latest_arxiv_papers(existing_urls, limit=MAX_NEW_PAPERS)
    print(f"新規取得対象: {len(new_papers)} 件")

    for paper in new_papers:
        print(f"処理中: {paper['original_title']}")
        jina_content = fetch_jina_content(paper["url"])
        analysis = analyze_with_gemini(paper, jina_content)

        record = {
            "url": paper["url"],
            "original_title": paper["original_title"],
            "published_date": paper["published_date"],
            **analysis
        }

        existing_data.insert(0, record)
        send_paper_to_discord(record)
        time.sleep(2)

    if new_papers:
        save_data(existing_data)

    # 記事2件の送信が終わったあとに、昨日のGA4アクセス解析をDiscordに通知
    print("GA4から昨日のアクセスデータを取得中...")
    stats = fetch_ga4_yesterday_stats()
    send_analytics_to_discord(stats, len(existing_data))
    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()
