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
    """使用 spaCy 進行智能字幕合併"""
    # 載入英文模型
    nlp = spacy.load('en_core_web_sm')

    combined_data = []
    i = 0

    while i < len(srt_data):
        current = srt_data[i].copy()
        current_text = current['text'].strip()

        # 檢查是否有下一個字幕
        if i + 1 < len(srt_data):
            next_text = srt_data[i + 1]['text'].strip()

            # 使用 spaCy 分析當前和下一個文本
            current_doc = nlp(current_text)
            next_doc = nlp(next_text)

            should_combine = False

            if len(current_doc) > 0 and len(next_doc) > 0:
                last_token = current_doc[-1]
                first_token = next_doc[0]

                # 判斷合併條件
                should_combine = any([
                    # 條件1: 當前句子以介系詞或助詞結尾
                    last_token.pos_ in ['ADP', 'PART'],

                    # 條件2: 當前句子以限定詞或冠詞結尾
                    last_token.pos_ == 'DET',

                    # 條件3: 當前句子以特定連接詞結尾
                    last_token.pos_ == 'CCONJ' and last_token.text.lower() not in ['and', 'or', 'but'],

                    # 條件4: 當前句子是從句的開始但未完成
                    (last_token.dep_ in ['mark', 'prep'] and
                     not any(t.dep_ == 'ROOT' for t in current_doc)),

                    # 條件5: 下一句開頭是現在分詞或過去分詞
                    first_token.tag_ in ['VBG', 'VBN'] and not current_text.endswith('.'),

                    # 條件6: 當前句子沒有完整的謂語動詞
                    not any(t.dep_ == 'ROOT' and t.pos_ == 'VERB' for t in current_doc),

                    # 條件7: 當前文本沒有結束標點符號且下一段以小寫開頭
                    (not current_text[-1] in '.!?,"' and next_text[0].islower()),

                    # 條件8: 檢查是否為分開的片語
                    any(t.dep_ in ['pobj', 'dobj', 'attr'] and t.head.i == len(current_doc)-1
                        for t in next_doc),

                    # 條件9: 檢查是否為不完整的動詞片語
                    (last_token.pos_ == 'VERB' and
                     any(t.dep_ == 'aux' for t in current_doc) and
                     not any(t.dep_ in ['dobj', 'attr', 'prep'] for t in current_doc))
                ])

                # 特殊情況：避免錯誤合併
                should_not_combine = any([
                    # 避免合併獨立的並列句
                    first_token.text.lower() in ['and', 'or', 'but'] and
                    last_token.text.endswith('.'),

                    # 避免合併明顯的新句子
                    next_text[0].isupper() and current_text.endswith('.'),

                    # 避免合併引號內的對話
                    current_text.endswith('"') and next_text.startswith('"'),

                    # 避免合併完整的獨立句子
                    current_text.endswith('.') and
                    any(t.dep_ == 'ROOT' for t in current_doc) and
                    any(t.dep_ == 'ROOT' for t in next_doc)
                ])

                if should_combine and not should_not_combine:
                    # 合併文本
                    current['text'] = f"{current_text} {next_text}"
                    # 更新時間戳
                    current_time = current['timestamp']
                    next_time = srt_data[i + 1]['timestamp']
                    current['timestamp'] = f"{current_time.split(' --> ')[0]} --> {next_time.split(' --> ')[1]}"
                    i += 1  # 跳過下一個字幕

        combined_data.append(current)
        i += 1

    # 重新編號
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
                    # 檢查是否有 Retry-After 標頭
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = delay
                        delay *= 2  # 如果沒有 Retry-After，則使用指數退避

                    print(f"遇到速率限制，等待 {wait_time} 秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"已達最大重試次數 ({max_retries})，跳過此文本")
                    return text
            else:
                print(f"API 請求失敗: {response.status_code}")
                print(f"錯誤訊息: {response.text}")
                if attempt < max_retries - 1:
                    print(f"等待 {delay} 秒後重試...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                return text

        except Exception as e:
            print(f"翻譯時發生錯誤: {e}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒後重試...")
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
    # 預設的 system prompt
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

    # 設定命令列參數
    parser = argparse.ArgumentParser(description='翻譯SRT字幕檔')
    parser.add_argument('--input', '-i', required=True, help='輸入的SRT檔案路徑')
    parser.add_argument('--output', '-o', required=True, help='輸出的SRT檔案路徑')
    parser.add_argument('--api-key', '-k', required=True, help='x.ai API 金鑰')
    parser.add_argument('--model', '-m',
                      default='grok-beta',
                      choices=['grok-beta'],
                      help='使用的模型 (預設: grok-beta)')
    parser.add_argument('--prompt', '-p',
                      default=default_prompt,
                      help='自定義 system prompt')
    parser.add_argument('--prompt-file', '-pf',
                      help='從檔案讀取 system prompt')
    parser.add_argument('--temperature', '-t',
                      type=float,
                      default=0.7,
                      help='模型溫度值 (0.0~1.0，預設: 0.7)')
    parser.add_argument('--delay', '-d',
                      type=float,
                      default=1.0,
                      help='每次翻譯之間的延遲秒數 (預設: 1.0)')
    parser.add_argument('--max-retries', '-r',
                      type=int,
                      default=5,
                      help='最大重試次數 (預設: 5)')
    parser.add_argument('--initial-delay', '-id',
                      type=float,
                      default=1.0,
                      help='初始重試等待時間 (預設: 1.0)')

    args = parser.parse_args()

    # 如果提供了 prompt 檔案，從檔案讀取
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"讀取 prompt 檔案時發生錯誤: {e}")
            return
    else:
        system_prompt = args.prompt

    # 檢查溫度值範圍
    if not 0 <= args.temperature <= 1:
        print("錯誤：temperature 必須在 0.0 到 1.0 之間")
        return

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input):
        print(f"錯誤: 輸入檔案 '{args.input}' 不存在")
        return

    # 檢查輸出目錄是否存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取 SRT 檔案
    print(f"正在讀取檔案: {args.input}")
    print(f"使用模型: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"最大重試次數: {args.max_retries}")
    print(f"初始重試等待時間: {args.initial_delay}秒")
    srt_data = read_srt(args.input)

    # 先合併需要合併的字幕
    print("正在智能合併連續字幕...")
    combined_srt_data = combine_continuous_subtitles(srt_data)

    # 翻譯合併後的字幕
    total_blocks = len(combined_srt_data)
    for i, block in enumerate(combined_srt_data, 1):
        print(f"正在翻譯: [{i}/{total_blocks}] {block['text']}")
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

        # 在每次翻譯之間添加延遲
        if i < total_blocks:
            time.sleep(args.delay)

    # 寫入翻譯後的 SRT 檔案
    write_srt(combined_srt_data, args.output)
    print(f"翻譯完成！已保存至 {args.output}")

if __name__ == "__main__":
    main()
