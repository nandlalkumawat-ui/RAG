"""Run a fixed set of test questions and measure how well the RAG system does.

    python evaluate.py

For each question we know which pages of book_1.pdf hold the answer (from the
book's table of contents). That lets us compute RETRIEVAL PRECISION
automatically:

    precision = (retrieved chunks that come from the right pages) / top_k

Answer relevance (1-5) and citation accuracy are judged by a human reading the
output. See EVALUATION.md for the scores.

Results are written to evaluation_results.md.
"""

import time

from rag_app.rag_pipeline import RAGPipeline
from rag_app.settings import load_config

# (question, first page, last page) -- page range of the section that answers it.
# None = the answer is NOT in the book; the system should say it cannot find it.
TEST_QUESTIONS = [
    ("What is the difference between breadth-first search and depth-first search?", 8, 11),
    ("What makes a heuristic admissible?", 12, 16),
    ("How does alpha-beta pruning work?", 47, 53),
    ("What is the Bellman equation?", 64, 75),
    ("What is the trade-off between exploration and exploitation?", 78, 89),
    ("How does particle filtering work?", 140, 142),
    ("What is d-separation in a Bayes net?", 100, 111),
    ("Who won the FIFA World Cup in 2018?", None, None),
]


def main():
    rag = RAGPipeline(load_config())
    lines = ["# Evaluation results\n"]
    precisions = []

    for question, first_page, last_page in TEST_QUESTIONS:
        start = time.time()
        result = rag.ask(question)
        seconds = time.time() - start

        lines.append(f"## {question}\n")
        if first_page is not None:
            correct = sum(first_page <= s["page"] <= last_page for s in result["sources"])
            precision = correct / len(result["sources"])
            precisions.append(precision)
            lines.append(f"Expected pages {first_page}-{last_page} | "
                         f"retrieval precision: {correct}/{len(result['sources'])} = {precision:.2f}\n")
        else:
            lines.append("Out-of-scope question (answer is not in the book)\n")

        lines.append(f"**Answer** ({seconds:.0f} s):\n\n{result['answer']}\n")
        lines.append("**Sources:**\n")
        for s in result["sources"]:
            preview = " ".join(s["text"].split())[:200]
            rerank = f", rerank {s['rerank_score']}" if "rerank_score" in s else ""
            lines.append(f"- [{s['source_number']}] {s['citation']} "
                         f"(similarity {s['similarity']}{rerank}): {preview}...")
        lines.append("")
        print(f"done: {question}")

    average = sum(precisions) / len(precisions)
    reranking = "on" if rag.reranker else "off"
    lines.insert(1, f"Average retrieval precision@{rag.default_top_k()} (reranking {reranking}): "
                    f"**{average:.2f}**\n")

    with open("evaluation_results.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Average retrieval precision: {average:.2f}  ->  see evaluation_results.md")


if __name__ == "__main__":
    main()
