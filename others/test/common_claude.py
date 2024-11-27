from anthropic import Anthropic
from common_functions import get_api_key


def calculate_questions_per_month(monthly_budget=20, tokens_per_question=5000):
    input_cost_per_million = 3
    output_cost_per_million = 15

    input_tokens = tokens_per_question / 2
    output_tokens = tokens_per_question / 2

    cost_per_question = (
        (input_tokens / 1_000_000) * input_cost_per_million +
        (output_tokens / 1_000_000) * output_cost_per_million
    )

    questions_per_month = monthly_budget / cost_per_question

    return int(questions_per_month)


client = Anthropic(api_key=get_api_key(),)

question = """
Can you analyze the difference between using Claude API vs subscription (chat) if:
- Both cost $20/month and using claude-3-sonnet-20240229 model
- Each API question uses approximately 5000 tokens (input + output combined)
Please calculate the maximum number of questions possible with the API option.
"""

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": question
        }
    ],
    model="claude-3-haiku-20240307",
)

num_questions = calculate_questions_per_month()

print(f"Analysis Results:")
print(f"Maximum API questions per month with $20 budget: {num_questions}")
print("\nClaude's detailed response:")
print(message.content)
