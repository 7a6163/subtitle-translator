import os
import re
import requests
import argparse
import time
import spacy
from pathlib import Path

def read_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    blocks = content.strip().split('\n\n')
    srt_data = []

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text = ' '.join(lines[2:])
            srt_data.append({
                'index': index,
                'timestamp': timestamp,
                'text': text
            })

    return srt_data

def combine_continuous_subtitles(srt_data):
    """Use spaCy for intelligent subtitle merging"""
    # Load English model
    nlp = spacy.load('en_core_web_sm')

    combined_data = []
    i = 0

    while i < len(srt_data):
        current = srt_data[i].copy()
        current_text = current['text'].strip()

        # Check if there is a next subtitle
        if i + 1 < len(srt_data):
            next_text = srt_data[i + 1]['text'].strip()

            # Use spaCy to analyze current and next text
            current_doc = nlp(current_text)
            next_doc = nlp(next_text)

            should_combine = False

            if len(current_doc) > 0 and len(next_doc) > 0:
                last_token = current_doc[-1]
                first_token = next_doc[0]

                # Determine merge conditions
                should_combine = any([
                    # Condition 1: Current sentence ends with preposition or particle
                    last_token.pos_ in ['ADP', 'PART'],

                    # Condition 2: Current sentence ends with determiner or article
                    last_token.pos_ == 'DET',

                    # Condition 3: Current sentence ends with specific conjunctions
                    last_token.pos_ == 'CCONJ' and last_token.text.lower() not in ['and', 'or', 'but'],

                    # Condition 4: Current sentence is the beginning of a clause but incomplete
                    (last_token.dep_ in ['mark', 'prep'] and
                     not any(t.dep_ == 'ROOT' for t in current_doc)),

                    # Condition 5: Next sentence starts with present or past participle
                    first_token.tag_ in ['VBG', 'VBN'] and not current_text.endswith('.'),

                    # Condition 6: Current sentence has no complete predicate verb
                    not any(t.dep_ == 'ROOT' and t.pos_ == 'VERB' for t in current_doc),

                    # Condition 7: Current text has no ending punctuation and next segment starts with lowercase
                    (not current_text[-1] in '.!?,"' and next_text[0].islower()),

                    # Condition 8: Check for split phrases
                    any(t.dep_ in ['pobj', 'dobj', 'attr'] and t.head.i == len(current_doc)-1
                        for t in next_doc),

                    # Condition 9: Check for incomplete verb phrases
                    (last_token.pos_ == 'VERB' and
                     any(t.dep_ == 'aux' for t in current_doc) and
                     not any(t.dep_ in ['dobj', 'attr', 'prep'] for t in current_doc))
                ])

                # Special cases: Avoid incorrect merging
                should_not_combine = any([
                    # Avoid merging independent parallel clauses
                    first_token.text.lower() in ['and', 'or', 'but'] and
                    last_token.text.endswith('.'),

                    # Avoid merging obvious new sentences
                    next_text[0].isupper() and current_text.endswith('.'),

                    # Avoid merging quoted dialogue
                    current_text.endswith('"') and next_text.startswith('"'),

                    # Avoid merging complete independent sentences
                    current_text.endswith('.') and
                    any(t.dep_ == 'ROOT' for t in current_doc) and
                    any(t.dep_ == 'ROOT' for t in next_doc)
                ])

                if should_combine and not should_not_combine:
                    # Merge text
                    current['text'] = f"{current_text} {next_text}"
                    # Update timestamp
                    current_time = current['timestamp']
                    next_time = srt_data[i + 1]['timestamp']
                    current['timestamp'] = f"{current_time.split(' --> ')[0]} --> {next_time.split(' --> ')[1]}"
                    i += 1  # Skip next subtitle

        combined_data.append(current)
        i += 1

    # Renumber indices
    for idx, item in enumerate(combined_data, 1):
        item['index'] = str(idx)

    return combined_data

def translate_text(api_key, text, model, system_prompt, temperature, max_retries=5, initial_delay=1):
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': text
                    }
                ],
                'temperature': temperature
            }

            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:  # Rate limit error
                if attempt < max_retries - 1:
                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = delay
                        delay *= 2  # If no Retry-After, use exponential backoff

                    print(f"Rate limit reached, waiting {wait_time} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Maximum retries ({max_retries}) reached, skipping this text")
                    return text
            else:
                print(f"API request failed: {response.status_code}")
                print(f"Error message: {response.text}")
                if attempt < max_retries - 1:
                    print(f"Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                return text

        except Exception as e:
            print(f"Error during translation: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= 2
                continue
            return text

    return text

def write_srt(srt_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, block in enumerate(srt_data):
            if i > 0:
                file.write('\n')
            file.write(f"{block['index']}\n")
            file.write(f"{block['timestamp']}\n")
            file.write(f"{block['text']}\n")

def main():
    # Default system prompt
    default_prompt = """You are a professional subtitle translator.
Follow these translation rules:
1. Translate from English to Traditional Chinese
2. Keep the translation natural, conversational and fluent
3. Maintain the original tone, style and emotional expression
4. Keep informal language informal, using common daily expressions
5. Preserve any special terms or proper nouns
6. Make sure the translation length is suitable for subtitles
7. Consider the context and relationship between characters
8. Adapt cultural references appropriately
9. Use appropriate Chinese spoken language patterns
10. Don't translate word-by-word, focus on conveying the meaning
11. Don't include the original English text
12. Don't add any explanations or notes

For dialogue:
- Use natural conversation patterns
- Match the speaker's personality and speaking style
- Consider the emotional context
- Use appropriate Chinese colloquialisms
- Maintain character relationships in the choice of words

Just provide the direct Chinese translation."""

    # Set command-line arguments
    parser = argparse.ArgumentParser(description='Translate SRT subtitle file')
    parser.add_argument('--input', '-i', required=True, help='Input SRT file path')
    parser.add_argument('--output', '-o', required=True, help='Output SRT file path')
    parser.add_argument('--api-key', '-k', required=True, help='x.ai API key')
    parser.add_argument('--model', '-m',
                      default='grok-beta',
                      choices=['grok-beta'],
                      help='Model to use (default: grok-beta)')
    parser.add_argument('--prompt', '-p',
                      default=default_prompt,
                      help='Custom system prompt')
    parser.add_argument('--prompt-file', '-pf',
                      help='Read system prompt from file')
    parser.add_argument('--temperature', '-t',
                      type=float,
                      default=0.7,
                      help='Model temperature value (0.0~1.0, default: 0.7)')
    parser.add_argument('--delay', '-d',
                      type=float,
                      default=1.0,
                      help='Delay between translations in seconds (default: 1.0)')
    parser.add_argument('--max-retries', '-r',
                      type=int,
                      default=5,
                      help='Maximum retries (default: 5)')
    parser.add_argument('--initial-delay', '-id',
                      type=float,
                      default=1.0,
                      help='Initial retry delay in seconds (default: 1.0)')

    args = parser.parse_args()

    # If prompt file is provided, read from file
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            return
    else:
        system_prompt = args.prompt

    # Check temperature value range
    if not 0 <= args.temperature <= 1:
        print("Error: temperature must be between 0.0 and 1.0")
        return

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' does not exist")
        return

    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read SRT file
    print(f"Reading file: {args.input}")
    print(f"Using model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Maximum retries: {args.max_retries}")
    print(f"Initial retry delay: {args.initial_delay} seconds")
    srt_data = read_srt(args.input)

    # Merge subtitles first
    print("Merging continuous subtitles...")
    combined_srt_data = combine_continuous_subtitles(srt_data)

    # Translate merged subtitles
    total_blocks = len(combined_srt_data)
    for i, block in enumerate(combined_srt_data, 1):
        print(f"Translating: [{i}/{total_blocks}] {block['text']}")
        translated_text = translate_text(
            args.api_key,
            block['text'],
            args.model,
            system_prompt,
            args.temperature,
            max_retries=args.max_retries,
            initial_delay=args.initial_delay
        )
        block['text'] = translated_text

        # Add delay between translations
        if i < total_blocks:
            time.sleep(args.delay)

    # Write translated SRT file
    write_srt(combined_srt_data, args.output)
    print(f"Translation complete! Saved to {args.output}")

if __name__ == "__main__":
    main()
