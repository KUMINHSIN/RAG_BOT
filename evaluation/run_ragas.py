from __future__ import annotations

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from app.rag import ask_question


def run_eval(csv_path: str = "evaluation/sample_eval_set.csv") -> None:
    df = pd.read_csv(csv_path)

    answers = []
    contexts = []
    for q in df["question"].tolist():
        result = ask_question(q)
        answers.append(result["answer"])
        contexts.append([s["content"] for s in result["sources"]])

    ds = Dataset.from_dict(
        {
            "question": df["question"].tolist(),
            "answer": answers,
            "contexts": contexts,
            "ground_truth": df["ground_truth"].tolist(),
        }
    )

    score = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    print(score)


if __name__ == "__main__":
    run_eval()
