from anthropic import Anthropic
from common_functions import get_api_key

client = Anthropic(api_key=get_api_key(),)

# Create the analysis message
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
# print(message.content[0].text)
# print("-" * 80)


def format_response(message):
    text = message.content[0].text

    lines = text.split('\n')
    formatted_text = ''

    for line in lines:
        if line.strip().startswith('-'):
            formatted_text += '  ' + line + '\n'
        else:
            formatted_text += line + '\n'

    return formatted_text


# print("\nFormatted Response:")
# print(format_response(message))
# print("-" * 80)


def parse_and_format_response(message):
    text = message.content[0].text
    sections = text.split('\n\n')

    formatted_output = ""
    for section in sections:
        if ':' in section.split('\n')[0]:
            header, *content = section.split('\n')
            formatted_output += f"\033[1m{header}\033[0m\n"
            formatted_output += '\n'.join(content) + '\n'
        else:
            formatted_output += section + '\n'
        formatted_output += '\n'

    return formatted_output


print("\nEnhanced Formatting:")
print(parse_and_format_response(message))
