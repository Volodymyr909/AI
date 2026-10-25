"""Стрімінговий запит до OpenAI-сумісного сервера з вимірюванням TTFT і tokens/s.

Відповідь «друкується» в терміналі в міру генерації, а наприкінці скрипт показує:
TTFT, швидкість генерації, повну затримку й кількість токенів.
Працює з будь-яким OpenAI-сумісним сервером: Ollama, llama-server, LM Studio, vLLM.
Перший запит після запуску сервера включає завантаження моделі в памʼять, тож TTFT буде завищений —
запустіть скрипт двічі.

Встановлення:
    pip install openai

Приклади:
    python stream_demo.py --model qwen3:1.7b --reasoning-effort none
    python stream_demo.py --model qwen3:1.7b --reasoning-effort none --prompt "Explain in two sentences why the sky is blue."
    python stream_demo.py --base-url http://localhost:8080/v1 --model local
    python stream_demo.py --model qwen3:1.7b --reasoning-effort none --temperature 1.2 --top-p 0.9 --seed 42
"""

import argparse
import time

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Стрімінговий запит з вимірюванням TTFT і tokens/s.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="адреса API, разом із /v1")
    parser.add_argument("--model", required=True, help="назва моделі на сервері")
    parser.add_argument("--api-key", default="local", help="ключ API; локальні сервери його зазвичай не перевіряють")
    parser.add_argument("--prompt", default="Поясни в трьох реченнях, що таке квантизація моделі.")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"],
                        help='для моделей з міркуванням (напр. qwen3 в Ollama) передайте "none"')
    # Параметри семплінгу; якщо не задані, сервер бере значення за замовчуванням моделі
    parser.add_argument("--temperature", type=float, help="випадковість вибору токенів; 0 — завжди найімовірніший")
    parser.add_argument("--top-p", type=float, help="вибирати лише з найімовірніших токенів із сумарною ймовірністю top_p")
    parser.add_argument("--seed", type=int, help="фіксує випадковість, щоб відповідь повторювалась")
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    options = {"reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
               "top_p": args.top_p, "seed": args.seed}
    extra = {name: value for name, value in options.items() if value is not None}

    start = time.perf_counter()
    first = usage = None
    reasoning = False
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Ollama надсилає міркування в полі reasoning, llama-server і vLLM — у reasoning_content
            if (delta.model_extra or {}).get("reasoning") or (delta.model_extra or {}).get("reasoning_content"):
                reasoning = True
            if delta.content:
                if first is None:
                    first = time.perf_counter()
                print(delta.content, end="", flush=True)
        if chunk.usage:
            usage = chunk.usage
    end = time.perf_counter()
    print("\n")

    if first is None:
        print("Модель не повернула жодного токена відповіді — якщо вона міркує, запустіть з --reasoning-effort none.")
        return
    tokens = usage.completion_tokens if usage else None
    print(f"TTFT:             {first - start:.2f} с")
    if tokens:
        print(f"tokens/s:         {(tokens - 1) / (end - first):.1f}")
        print(f"токенів:          {tokens}")
    print(f"повна затримка:   {end - start:.2f} с")
    if reasoning:
        print("УВАГА: модель міркувала перед відповіддю, тож TTFT включає час міркування, "
              "а tokens/s завищений. Для коректних вимірювань додайте --reasoning-effort none.")


if __name__ == "__main__":
    main()
