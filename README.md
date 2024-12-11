# Subtitle Translator

A Python tool for intelligently translating SRT subtitle files while maintaining proper sentence structure and timing.

## Features

- Reads and processes SRT subtitle files
- Intelligently combines continuous subtitles using NLP (Natural Language Processing)
- Translates text using x.ai API
- Maintains original subtitle timing and formatting
- Smart sentence combination based on linguistic analysis
- Handles rate limiting and retries automatically

## Prerequisites

- Python 3.x
- Required Python packages:
  - spacy
  - requests
  - argparse

You'll also need to download the English language model for spaCy:
```bash
python -m spacy download en_core_web_sm
```

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install spacy requests
```

## Usage

Run the script with the following command:

```bash
python translate_srt.py --input <input_srt_file> --output <output_srt_file> --api-key <your_x.ai_api_key>
```

### Optional Arguments

- `--model`: Specify the x.ai model to use for translation (default: available model)
- `--system-prompt`: Custom system prompt for translation
- `--temperature`: Set the temperature for translation (controls creativity)

## How it Works

1. **SRT Parsing**: The tool reads and parses the input SRT file, maintaining subtitle timing and index information.

2. **Intelligent Subtitle Combination**: Uses spaCy NLP to analyze and combine subtitles based on linguistic patterns:
   - Combines incomplete phrases
   - Handles prepositions and articles at line breaks
   - Maintains proper sentence structure
   - Considers grammatical dependencies

3. **Translation**: Utilizes x.ai API for high-quality translation with:
   - Automatic retry mechanism
   - Rate limit handling
   - Error recovery

4. **Output Generation**: Creates a new SRT file with translated content while preserving original timing and formatting.

## Error Handling

The script includes robust error handling:
- Automatic retries for API failures
- Exponential backoff for rate limiting
- Graceful fallback to original text if translation fails

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!
