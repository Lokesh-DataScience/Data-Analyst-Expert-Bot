from chatbot_evaluator import evaluate_chatbot

if __name__ == "__main__":
    results = evaluate_chatbot("dataset.json")

    print("\n📊 Final Chatbot Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")