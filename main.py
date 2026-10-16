import os
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
import requests

# 環境変数の取得
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
# ご自身の環境に合わせてモデル名を指定（デフォルトは gemini-2.5-flash 等でも動作可能）
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DATA_FILE_PATH = "docs/data.json"
MAX_ITEMS_PER_RUN = 2  # 無料枠のレート制限を考慮し、1回の実行あたり2件ずつ蓄積

# 取得対象（例：arXivのAI応用・統計応用・コンピュータと社会）
# クエリを変えるだけで任意の分野（医療画像、金融、ソフトウェア工学など）に特化可能
ARXIV_QUERY = "cat:cs.AI+OR+cat:stat.AP+OR+cat:cs.CY"
ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query={}&sortBy=submittedDate&sortOrder=descending&max_results=10".format(ARXIV_QUERY)


def load_existing_data():
    if not os.path.exists(DATA_FILE_PATH):
        os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
        return []
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data):
    with open(DATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_latest_papers(existing_ids):
    """標準ライブラリのみでarXiv APIから未処理の最新論文を取得"""
    req = urllib.request.Request(ARXIV_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.strip()
        if paper_id in existing_ids:
            continue

        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text.strip()[:10]

        papers.append({
            "id": paper_id,
            "url": paper_id,
            "original_title": title,
            "abstract": summary,
            "published_date": published
        })
        if len(papers) >= MAX_ITEMS_PER_RUN:
            break

    return papers


def extract_with_jina(url):
    """Jina Reader APIを使ってWebページのテキストをリッチに抽出"""
    jina_url = "https://r.jina.ai/{}".format(url)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(jina_url, headers=headers, timeout=30)
        if res.status_code == 200 and len(res.text) > 200:
            return res.text[:8000]  # トークン節約のため先頭8000文字に制限
    except Exception as e:
        print("Jina Reader fallback due to: {}".format(e))
    return None


def analyze_with_gemini(paper, jina_content):
    """requestsを使ったREST API直接呼び出し＋JSONスキーマ強制出力"""
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
        GEMINI_MODEL, GEMINI_API_KEY
    )

    source_text = jina_content if jina_content else paper["abstract"]

    prompt = """
以下の海外最新論文の情報を分析し、日本の個人開発者・エンジニア・リサーチャー向けに「実務やマイクロSaaS開発にどう転用できるか」という視点で構造化してください。

【論文タイトル】: {title}
【公開日】: {date}
【内容】:
{content}
""".format(title=paper["original_title"], date=paper["published_date"], content=source_text)

    # Gemini JSON Schema 定義
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "ja_title": {
                "type": "STRING",
                "description": "読者の目を引く分かりやすい日本語タイトル（40文字以内）"
            },
            "tags": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "関連する技術・応用分野タグ3個（例：医療AI, 統計モデル, 自然言語処理, 画像解析 など）"
            },
            "problem": {
                "type": "STRING",
                "description": "これまで何が課題・限界だったのか（2〜3文で簡潔に）"
            },
            "mechanism": {
                "type": "STRING",
                "description": "どのように解決したか、コアとなる仕組みの非専門家向け解説（3〜4文）"
            },
            "key_advantage": {
                "type": "STRING",
                "description": "既存の常識や従来手法と比べた最大のブレイクスルー・数値的優位性"
            },
            "saas_ideas": {
                "type": "ARRAY",
                "description": "この知見・アルゴリズムを個人開発やマイクロSaaS、業務自動化ツールに転用する具体案2つ",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "idea_name": {"type": "STRING", "description": "プロダクト・ツールの名称案"},
                        "target_user": {"type": "STRING", "description": "誰のどんな悩みを解決するか"},
                        "implementation": {"type": "STRING", "description": "PythonやAPIを使ってどう最小実装（MVP）するか"},
                        "monetization": {"type": "STRING", "description": "具体的なマネタイズ方法（月額課金、B2B提携、特化アフィリエイト等）"}
                    },
                    "required": ["idea_name", "target_user", "implementation", "monetization"]
                }
            },
            "search_keyword": {
                "type": "STRING",
                "description": "このテーマを学ぶのにおすすめな日本語の技術書・専門書をAmazonで検索するためのキーワード（例：ベイズ統計 機械学習 Python）"
            }
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

    res = requests.post(endpoint, json=payload, timeout=60)
    res.raise_for_status()
    result_json = res.json()

    raw_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_text)


def send_discord_notification(item):
    """Discordへnote/Zennの下書きとしてそのまま使えるリッチフォーマットで通知"""
    if not DISCORD_WEBHOOK_URL:
        return

    ideas_text = ""
    for idx, idea in enumerate(item["saas_ideas"], 1):
        ideas_text += (
            "**💡 応用案{}: {}**\n"
            "・対象: {}\n"
            "・実装: {}\n"
            "・収益化: {}\n\n"
        ).format(idx, idea["idea_name"], idea["target_user"], idea["implementation"], idea["monetization"])

    embed = {
        "title": "📄 " + item["ja_title"],
        "url": item["url"],
        "color": 3447003,
        "fields": [
            {"name": "🏷️ タグ", "value": " / ".join(item["tags"]), "inline": False},
            {"name": "❓ 従来の課題", "value": item["problem"], "inline": False},
            {"name": "⚙️ コア技術の仕組み", "value": item["mechanism"], "inline": False},
            {"name": "🚀 従来手法との違い", "value": item["key_advantage"], "inline": False},
            {"name": "💰 個人開発・マイクロSaaS転用アイデア", "value": ideas_text.strip(), "inline": False}
        ],
        "footer": {"text": "的原題: {} | 公開: {}".format(item["original_title"][:60], item["published_date"])}
    }

    requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=15)


def main():
    if not GEMINI_API_KEY:
        raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")

    existing_data = load_existing_data()
    existing_ids = {item["id"] for item in existing_data}

    print("1. 最新論文の取得中...")
    new_papers = fetch_latest_papers(existing_ids)
    if not new_papers:
        print("新規の未処理論文はありませんでした。")
        return

    for paper in new_papers:
        print("処理中: {}".format(paper["original_title"]))
        jina_text = extract_with_jina(paper["url"])
        structured = analyze_with_gemini(paper, jina_text)

        record = {
            "id": paper["id"],
            "url": paper["url"],
            "original_title": paper["original_title"],
            "published_date": paper["published_date"],
            "processed_at": datetime.now().strftime("%Y-%m-%d"),
            **structured
        }

        # 最新データが先頭に来るように挿入
        existing_data.insert(0, record)
        save_data(existing_data)

        send_discord_notification(record)
        time.sleep(5)  # APIレートリミット対策

    print("完了: {}件のデータを追加しました。".format(len(new_papers)))


if __name__ == "__main__":
    main()
