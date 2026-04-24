import json
from pathlib import Path
from typing import List, Dict

import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def get_test_prompts() -> List[str]:
    return [
        "Объясни простыми словами, что такое MLP (my little pony).",
        "Напиши 3 идеи стартапа для темщиков.",
        "Составь короткий план изучения Python за 2 недели.",
        "Переведи на английский: 'Сегодня отличная погода для прогулки'.",
        "Придумай слоган для кофейни в одно предложение.",
        "Какие есть способы уменьшить стресс перед экзаменом?",
        "Напиши мини-историю про то как я сдал диплом.",
        "Сгенерируй список из 5 полезных привычек для продуктивности перед дипломом.",
        "Чем заняться после отчисления из университета?",
        "Предложи 3 темы для поста в техническом блоге про AI.",
    ]


def send_prompt(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def run_inference(prompts: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for prompt in prompts:
        output = send_prompt(prompt)
        rows.append({"prompt": prompt, "output": output})
    return rows


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build_table_lines(rows: List[Dict[str, str]]) -> List[str]:
    """Build markdown table lines for inference results."""
    lines = ["| Запрос | Ответ |", "|---|---|"]
    for row in rows:
        prompt = markdown_escape(row["prompt"])
        output = markdown_escape(row["output"])
        lines.append(f"| {prompt} | {output} |")
    return lines

def save_readme(rows: List[Dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Лабораторная работа 2: NLP",
        "",
        "## Цель",
        (
            "Провести инференс LLM `Qwen2.5:0.5b` через HTTP API Ollama, "
            "отправить 10 запросов из Python-скрипта и зафиксировать результаты."
        ),
        "",
        "## Использованный стек",
        "- Ollama",
        "- Модель `qwen2.5:0.5b`",
        "- Python 3 + `requests`",
        "",
        "## Что реализовано",
        "1. Поднят сервер Ollama (`ollama serve`) на `http://127.0.0.1:11434`.",
        "2. Скачана модель `qwen2.5:0.5b`.",
        "3. Написан скрипт `main.py`, который отправляет 10 запросов в `POST /api/generate`.",
        "4. Результаты сохраняются в `README.md`.",
        "",
        "## Инструкция по запуску",
        "```bash",
        "pip install -r requirements.txt",
        "ollama serve",
        "ollama pull qwen2.5:0.5b",
        "python3 main.py",
        "```",
        "",
        "## Инференс",
        "",
    ]
    lines.extend(build_table_lines(rows))
    lines.extend(["", "Выполнил Клименко В.М. из М8О-403Б-22"])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    prompts = get_test_prompts()
    rows = run_inference(prompts)
    save_readme(rows, Path("README.md"))


if __name__ == "__main__":
    main()
