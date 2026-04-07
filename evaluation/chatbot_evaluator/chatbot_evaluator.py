import json
from groq_judge import judge_chatbot
from bot_client import run_bot_chat

def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate_chatbot(dataset_path):
    dataset = load_dataset(dataset_path)

    all_scores = []

    for sample in dataset:
        question = sample["question"]
        reference = sample.get("expected_answer")

        response = run_bot_chat(question)

        scores = judge_chatbot(question, response, reference)

        print(f"\nQ: {question}")
        print(f"Response: {response}")
        print(f"Scores: {scores}")

        all_scores.append(scores)

    return aggregate(all_scores)


def aggregate(results):
    agg = {k: 0 for k in results[0]}

    for r in results:
        for k in r:
            agg[k] += r[k]

    for k in agg:
        agg[k] /= len(results)

    return agg