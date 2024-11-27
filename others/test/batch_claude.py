import os
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor
from typing import List
from common_functions import get_api_key


class BatchProcessor:
    def __init__(self, api_key: str, model="claude-3-haiku-20240307", batch_size: int = 10):
        self.client = Anthropic(api_key=api_key)
        self.batch_size = batch_size
        self.model = model

    def send_single_message(self, question: str):
        try:
            message = self.client.messages.create(
                max_tokens=1024,
                messages=[{"role": "user", "content": question}],
                model=self.model,
            )
            return {"question": question, "response": message.content[0].text, "status": "success"}
        except Exception as e:
            return {"question": question, "response": str(e), "status": "error"}

    def process_batch(self, questions: List[str]):
        results = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            results = list(executor.map(self.send_single_message, questions))
        return results


if __name__ == "__main__":
    questions = [
        """Can you analyze the cost difference between Claude API vs subscription if:
        - Both cost $20/month and using claude-3-sonnet-20240229 model
        - Each API question uses approximately 5000 tokens (input + output combined)""",
        "Please calculate the maximum number of questions possible with the API option(if i use claude-3-sonnet-20240229 model)",
        "What are the main advantages of using the Batches API?",
        "How can I optimize my token usage?"
    ]

    processor = BatchProcessor(
        api_key=get_api_key(),
        batch_size=5,
    )

    results = processor.process_batch(questions)

    print("\nBatch Processing Results:")
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}:")
        print(f"Status: {result['status']}")
        print(f"Q: {result['question']}")
        print(f"A: {result['response']}")
        # print(f"A: {result['response'][:200]}...")  # Print first 200 chars of response
        print("-" * 80)

    # total_questions = len(questions)
    # estimated_tokens = total_questions * 5000

    # regular_cost = (estimated_tokens / 1_000_000) * ((3 + 15) / 2)
    # discounted_cost = regular_cost * 0.5

    # print(f"\nBatch Processing Summary:")
    # print(f"Total questions processed: {total_questions}")
    # print(f"Estimated total tokens: {estimated_tokens:,}")
    # print(f"Estimated regular cost: ${regular_cost:.2f}")
    # print(f"Estimated cost with batch discount: ${discounted_cost:.2f}")
