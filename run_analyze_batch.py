import csv
import json
import sys
import time

import requests

from app.preprocessing import clean_ticket_message, is_incomplete_input

API_URL = "http://localhost:8000/analyze"
INPUT_FILE = "tickets-training-20.json"
OUTPUT_FILE = "results.csv"
TIMEOUT = 600

CSV_FIELDS = [
    "conversationNo", "customerName", "last_message", "cleaned_message",
    "sentiment", "tone", "urgency", "sentiment_score",
    "query_type", "churn_risk",
    "prompt_tokens", "completion_tokens",
    "response_time_s", "status", "error",
]


def load_tickets(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_last_customer_message(ticket: dict) -> str | None:
    texts = [
        msg.get("text", "").strip()
        for msg in ticket.get("messages", [])
        if msg.get("senderType") == "customer" and msg.get("text", "").strip()
    ]
    return texts[-1] if texts else None


def call_analyze(message: str) -> dict:
    resp = requests.post(API_URL, json={"message": message}, timeout=TIMEOUT)
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise requests.HTTPError(
            f"HTTP {resp.status_code}: {str(detail)[:300]}",
            response=resp,
        )
    return resp.json()


def write_row(writer, row: dict):
    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main():
    tickets = load_tickets(INPUT_FILE)
    total = len(tickets)

    csv_file = open(OUTPUT_FILE, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    try:
        for i, ticket in enumerate(tickets, 1):
            conv_no = ticket.get("conversationNo", "?")
            customer = ticket.get("customerName", "?")
            print(f"[{i}/{total}] {conv_no} — {customer}", end=" ... ", flush=True)

            last_msg = get_last_customer_message(ticket)
            if not last_msg:
                print("SKIP (no customer messages)")
                write_row(writer, {
                    "conversationNo": conv_no,
                    "customerName": customer,
                    "status": "skipped",
                    "error": "no customer messages",
                })
                csv_file.flush()
                continue

            cleaned = clean_ticket_message(last_msg)

            if is_incomplete_input(cleaned):
                print(f"SKIP (incomplete input) — {repr(cleaned[:80])}")
                write_row(writer, {
                    "conversationNo": conv_no,
                    "customerName": customer,
                    "last_message": last_msg[:200],
                    "cleaned_message": cleaned[:200],
                    "status": "error",
                    "error": "incomplete input — only metadata found after cleaning",
                })
                csv_file.flush()
                continue

            t0 = time.perf_counter()
            try:
                data = call_analyze(cleaned)
                elapsed = time.perf_counter() - t0
                result = data.get("result", {})
                write_row(writer, {
                    "conversationNo": conv_no,
                    "customerName": customer,
                    "last_message": last_msg[:200],
                    "cleaned_message": cleaned[:200],
                    "sentiment": result.get("sentiment"),
                    "tone": result.get("tone"),
                    "urgency": result.get("urgency"),
                    "sentiment_score": result.get("sentiment_score"),
                    "query_type": result.get("query_type"),
                    "churn_risk": result.get("churn_risk"),
                    "prompt_tokens": data.get("prompt_tokens"),
                    "completion_tokens": data.get("completion_tokens"),
                    "response_time_s": round(elapsed, 2),
                    "status": "success",
                    "error": "",
                })
                print(f"OK ({elapsed:.1f}s) — {result.get('sentiment')} / {result.get('urgency')}")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                write_row(writer, {
                    "conversationNo": conv_no,
                    "customerName": customer,
                    "last_message": last_msg[:200],
                    "cleaned_message": cleaned[:200],
                    "response_time_s": round(elapsed, 2),
                    "status": "error",
                    "error": str(e),
                })
                print(f"ERROR ({elapsed:.1f}s) — {e}")

            csv_file.flush()
    finally:
        csv_file.close()

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
