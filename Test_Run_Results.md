# LM Studio MCP Test Run Results

This report records a test of the `lmstudio-rs-mcp` server, the available LM
Studio models, and two local models' ability to generate a Fibonacci script.

## 1. Connecting to LM Studio

### Initial attempt

The server was not registered with Claude Code, although a release binary was
available at `target/release/lmstudio-rs-mcp`. LM Studio was running at
`127.0.0.1:1234`, but requests to `GET /v1/models` failed with
`invalid_api_key`. No relevant API token environment variable was set.

The server was subsequently registered as a local stdio MCP server. The command
below uses a placeholder rather than the token that appeared in the original
transcript:

```sh
claude mcp add lmstudio \
  -e LMSTUDIO_API_TOKEN="<REDACTED>" \
  -- /Users/byron/dev/lmstudio-rs-mcp/target/release/lmstudio-rs-mcp
```

Claude Code requires a restart to load newly registered MCP servers. The server
was therefore called directly over stdio for this test. The token was stored in
plain text in `~/.claude.json` and appeared in the original transcript; it has
been redacted from this report. Rotate the token in LM Studio before reusing or
sharing the original transcript.

### Available models

The model library contained 81 models. None were loaded at the time of the
initial listing.

| Family | Models |
| --- | --- |
| Qwen 3.x chat and coder | Qwen3.8 27B (several variants), Qwen3.6 27B and 35B-A3B, Qwen3.5 35B-A3B and 9B, Qwen3 Coder Next (80B), Qwen3 Coder 30B, Qwen3 32B, Qwen3 14B, Qwen3 4B, Qwen3 VL 8B and 4B |
| Gemma | Gemma 4 31B, 31B QAT, 26B-A4B (MLX and QAT), 12B QAT and E4B; Gemma 3 12B |
| Mistral | Magistral Small 2509, Ministral 3 14B Reasoning, Devstral Small 2507, Codestral 22B, Microsoft NextCoder 32B |
| Other chat models | GPT-OSS 20B, GLM 4.7 Flash, GLM 4.6v Flash, Nemotron 3 Nano and Nano Omni, Phi 4 and Phi 4 Reasoning Plus, Granite 3.2 8B and 4 H Tiny, Lfm2 24B-A2B, DeepSeek R1 distills |
| Community fine-tunes | Several, mostly Qwen-based and "uncensored" variants |
| Non-text models | Embedding: Nomic Embed v1.5, Laya. Image: Qwen Image Edit, Z Image Turbo, Stable Diffusion 3.5. Speech and OCR: Qwen3 TTS, VibeVoice ASR, Chandra OCR 2 |

Most chat models support tool use, and many support vision. The largest model by
file size was Qwen3 Coder Next at approximately 48 GB.

## 2. First Fibonacci attempt: ambiguous prompt

The initial request was:

> Load Qwen3.8 (choose a specific model) and ask it, as a subagent, to create a
> Python script to generate the Fibonacci series from 0 to 100. Display the code
> and review it for correctness.

The phrase "from 0 to 100" was ambiguous: it could mean Fibonacci indices 0
through 100, or Fibonacci values no greater than 100.

### Qwen3.8 27B

The selected model was `qwen/qwen3.8-27b`, an 8-bit MLX build. It wrote the
script in three turns, including a re-read to verify the result.

```python
"""Generate and print the Fibonacci series for n = 0..100 (inclusive)."""


def fibonacci_series(limit: int = 100) -> list[int]:
    """Return a list of Fibonacci numbers F(0) through F(limit).

    The series starts 0, 1, 1, 2, 3, 5, ...
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    series = []
    a, b = 0, 1
    for _ in range(limit + 1):
        series.append(a)
        a, b = b, a + b
    return series


def main() -> None:
    for value in fibonacci_series(100):
        print(value)


if __name__ == "__main__":
    main()
```

**Review**

- The script correctly prints 101 terms, from `F(0)` through `F(100)`.
- Its final value is `354224848179261915075`, the correct value of `F(100)`.
- The iterative update, integer arithmetic, negative-limit check, type hints,
  and `main()` guard are correct.
- It did not match the intended interpretation of Fibonacci *values* up to
  100. That sequence contains 12 values and ends at 89.
- A tool log reported a 140-character write, while the resulting file was 573
  bytes. The report considered this likely a truncated log summary; the script
  itself was complete.

### Gemma 4 31B

Qwen was unloaded and `google/gemma-4-31b` was loaded using a Q4_K_M GGUF
build. Loading took 58 seconds. Gemma received the same prompt, finished in two
turns, and made the same index-versus-value choice.

```python
def generate_fibonacci(n):
    """Generates Fibonacci numbers from 0 up to index n."""
    fib_sequence = []
    a, b = 0, 1
    for i in range(n + 1):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence


def main():
    # Generate and print Fibonacci series for values from 0 to 100 inclusive
    limit = 100
    series = generate_fibonacci(limit)
    for index, value in enumerate(series):
        print(f"{index}: {value}")


if __name__ == "__main__":
    main()
```

**Review**

- The script correctly prints 101 terms, from `F(0)` through `F(100)`.
- Its output labels each term with its index, ending with
  `100: 354224848179261915075`.
- As with Qwen, it does not produce only Fibonacci values less than or equal to
  100.
- The comment in `main()` says "for values from 0 to 100 inclusive", but the
  function parameter is used as an index. The comment contradicts the code and
  docstring.

### Comparison

| Aspect | Qwen3.8 27B | Gemma 4 31B |
| --- | --- | --- |
| Turns | 3 (wrote, then re-read to verify) | 2 (wrote, no verification) |
| Output | Bare numbers, one per line | `index: value` lines |
| Extras | Type hints, docstrings, negative-limit `ValueError` | Minimal, no validation |
| Style | Type annotations and more documentation | Simpler, but an inline comment contradicts the code |

Both models interpreted the ambiguous prompt as indices 0 through 100 and did
not mention the ambiguity. Qwen's script was more robust and it checked its own
work. Gemma's output was more directly readable. Neither was clearly better on
correctness for this ambiguous request.

## 3. Fibonacci attempt: clarified prompt

The task was made precise by distinguishing values from indices and specifying
the expected values. The requested sequence was:

```text
0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
```

Gemma was already loaded, so it was tested first. Qwen was then loaded and given
the same precise prompt. Both models produced scripts that printed exactly the
12 expected values.

### Gemma 4 31B

Gemma completed the task in three turns, including a re-read.

```python
def fibonacci_up_to(max_value: int) -> list[int]:
    fibs = []
    a, b = 0, 1
    while a <= max_value:
        fibs.append(a)
        a, b = b, a + b
    return fibs
```

**Review**

- The loop is idiomatic and correct.
- A negative cap returns `[]`; a cap of 0 returns `[0]`; and a cap of 1
  returns `[0, 1, 1]`.
- This was the smaller and cleaner of the two implementations.

### Qwen3.8 27B

Qwen completed the task in three turns, including a re-read. Its implementation
special-cased `[0]` and `[0, 1]`, then used `while True` with a `break`.

**Review**

- The implementation was correct for caps of -1, 0, 1, and 100.
- The special cases were more verbose than necessary because a single
  `while a <= max_value` loop handles those cases.
- Its docstring and report were more thorough. The report traced the result for
  a cap of 100 without running the script.

### Results

Both scripts printed the same 12 values:

```text
0
1
1
2
3
5
8
13
21
34
55
89
```

The clarified value-versus-index distinction and explicit expected output
resolved the only reported problem. Gemma re-read its file in this round, as
Qwen had done in the first round. Gemma's implementation was simpler, while
Qwen's was more defensive and documented. The task was too small to distinguish
the models on correctness.

At the end of the test, both models were still loaded in LM Studio.
