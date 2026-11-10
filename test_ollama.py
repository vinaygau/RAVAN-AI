import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"


def ask_ollama(question):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Ravan AI, a driving safety assistant. "
                    "Explain driving risks in simple English. "
                    "Give short and clear safety recommendations."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


if __name__ == "__main__":
    question = (
        "The car ahead is 12 meters away. "
        "My speed is 60 km/h and the front car is moving at 30 km/h. "
        "Explain the risk and suggest what I should do."
    )

    try:
        answer = ask_ollama(question)
        print("\nRavan AI Response:\n")
        print(answer)

    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Check whether Ollama is running.")

    except requests.exceptions.Timeout:
        print("Ollama took too long to respond.")

    except requests.exceptions.RequestException as error:
        print(f"Request failed: {error}")
