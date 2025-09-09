# Object Labelling Toolkit (YOLO-ready)

This repository contains two primary utilities for preparing and labelling image datasets for YOLO-style object detection:

- dataset_cleanup.py — Deduplicate, de-near-duplicate, and optionally downsample a dataset (with automatic backups).
- label_with_llm.py — Auto-label images by asking a multimodal LLM (Google Gemini by default). Produces YOLO .txt label files (class cx cy w h, normalized).

Optional: You can review labels visually with tools like Label Studio (Quickstart: https://labelstud.io/guide/quick_start), but Label Studio is not required to run these scripts.

---

## Prerequisites

- Python 3.10+ recommended

### Install required packages

For convenience, install all dependencies at once:
```bash
pip install -r requirements.txt
```

<details>
<summary>Or install packages individually</summary>

- For dataset cleanup (images only):
  ```bash
  pip install pillow
  ```
- For auto-labelling with Gemini:
  ```bash
  pip install pillow google-generativeai
  ```
- Optional (review labels in a UI):
  ```bash
  pip install label-studio
  ```
</details>

---

## Dataset layout expected

Both scripts assume a conventional YOLO-style split under a dataset root:

```
<dataset_root>/
  train/
    images/ ... .jpg|.png ...
    labels/ ... .txt (YOLO format)
  valid/
    images/
    labels/
```

- label_with_llm.py can write labels alongside your images (default) or into a specified directory.
- dataset_cleanup.py maps an image path to its label by replacing the path segment "images" with "labels" and the extension with .txt.

---

## API key — where to put it

label_with_llm.py reads the GEMINI_API_KEY environment variable.

Examples below use placeholders like {{GEMINI_API_KEY}} — replace them with your actual key.

<details>
<summary>API Key Setup Examples</summary>

### fish (current shell only)
```bash
set -x GEMINI_API_KEY {{GEMINI_API_KEY}}
```

### fish (persist across sessions)
Create ~/.config/fish/conf.d/gemini.fish with:
```bash
set -Ux GEMINI_API_KEY {{GEMINI_API_KEY}}
```
Note: -U stores a universal variable that persists; use -x to export into the environment.

### bash/zsh (current shell only)
```bash
export GEMINI_API_KEY={{GEMINI_API_KEY}}
```

### bash/zsh (persist)
Add to ~/.bashrc or ~/.zshrc:
```bash
export GEMINI_API_KEY={{GEMINI_API_KEY}}
```
</details>

---

## Configure classes and prompt (run this first)

Run the config helper to define your class list and any extra prompt instructions. It writes labeling_config.json.

```bash
python configure_labeling.py
```

- Classes are defined in order (IDs assigned 0..N-1).
- You can optionally add a short description per class.
- You can add extra prompt notes (instructions appended to the base prompt).
- Re-run this script any time to change classes or the prompt, or edit labeling_config.json directly.

---

## label_with_llm.py — what it does

- Loads configuration from labeling_config.json (classes + prompt notes).
- Builds a strict prompt listing your classes and labelling rules, then appends your extra notes.
- Sends the image(s) and prompt to Gemini (using GEMINI_API_KEY).
- First asks for a JSON object with a list of boxes; if empty, falls back to plain YOLO line output.
- Writes YOLO labels as <class> <cx> <cy> <w> <h> (normalized to [0,1]) per image.
- Optionally saves the raw JSON response for debugging.

### Usage examples

<details>
<summary>label_with_llm.py Usage Examples</summary>

- Label a single image, writing a foo.txt next to foo.jpg:
```bash
python label_with_llm.py --images path/to/foo.jpg --config labeling_config.json
```

- Label an entire directory (recursively), writing labels next to images:
```bash
python label_with_llm.py --images path/to/images_dir --config labeling_config.json
```

- Write labels to a separate directory and save raw JSON responses:
```bash
python label_with_llm.py \
  --images path/to/images_dir \
  --out-dir path/to/labels_out \
  --save-raw path/to/raw_json \
  --config labeling_config.json
```

- Overwrite existing labels:
```bash
python label_with_llm.py --images path/to/images_dir --overwrite --config labeling_config.json
```

- Choose a different Gemini model ID:
```bash
python label_with_llm.py --images path/to/images_dir --model gemini-1.5-flash --config labeling_config.json
```
</details>

Notes:
- The script requires GEMINI_API_KEY. If missing, it exits with an error.
- Fallback: If the JSON path returns no boxes, the script asks for line-only YOLO output and parses it.

---

## dataset_cleanup.py — what it does

- Scans one or more subsets (default: train and valid) under a dataset root.
- Detects exact duplicates via SHA-256 and near duplicates via perceptual difference hash (dHash) with Hamming distance.
- Optionally performs class-balanced downsampling per subset to reach target counts.
- Produces a dry-run plan (JSON summary) by default.
- When applied, moves removed images (and their labels, when present) into a timestamped backup folder under the dataset root, preserving a mirror of images/ and labels/.

### Key stages
1) Exact duplicates: cluster by SHA-256 and keep the best file (preferring valid, then higher resolution).
2) Near duplicates: bucket by dHash prefix and union clusters with Hamming distance <= threshold (default 5), keep best and mark others for removal.
3) Downsampling: compute the most frequent YOLO class per image, then sample per-class to hit the requested target count per subset.

### Usage examples

<details>
<summary>dataset_cleanup.py Usage Examples</summary>

- Dry run with summary only:
```bash
python dataset_cleanup.py --root /path/to/dataset
```

- Specify subsets, a tighter near-duplicate threshold, and write the full plan to JSON:
```bash
python dataset_cleanup.py \
  --root /path/to/dataset \
  --subsets train valid \
  --dhash-threshold 4 \
  --output-json cleanup_plan.json
```

- Cross-split deduplication (avoid train/valid leakage):
```bash
python dataset_cleanup.py --root /path/to/dataset --cross-split-dedup
```

- Downsample to targets (class-balanced) and apply with backup:
```bash
python dataset_cleanup.py \
  --root /path/to/dataset \
  --train-target 5000 \
  --valid-target 1000 \
  --apply
```

- Apply and store backup in a custom location:
```bash
python dataset_cleanup.py --root /path/to/dataset --apply --backup-dir /path/to/backup
```
</details>

Notes:
- On apply, removed files are moved (not deleted) to the backup folder. A copy of the full plan is written to backup_root/cleanup_plan.json.
- Only images with typical extensions (.jpg, .png, .bmp, .tif, .tiff, .webp) are considered.

---

## Switching from Gemini to other providers (DeepSeek Janus, OpenAI ChatGPT/GPT-4o, xAI Grok)

label_with_llm.py is currently wired to Google Gemini via google-generativeai. To use a different multimodal model, swap the model client and the request functions that send the prompt+image and parse a JSON response.

Recommended approach: create a thin adapter layer that provides the same outputs as request_boxes_json(...) and request_lines_fallback(...), but calls your provider of choice.

High-level steps:
1) Install the provider SDK or use their OpenAI-compatible HTTP API.
2) Export the provider’s API key as an environment variable.
3) Add provider-specific functions that:
   - Send the prompt + image
   - Ask for JSON-only output matching {"boxes": [{"class_id": int, "center_x": float, "center_y": float, "width": float, "height": float}]}
   - Return a Python list of dicts for boxes, or return raw YOLO lines in a fallback path.
4) Swap the calls based on a flag or create a separate script (e.g., label_with_openai.py, label_with_deepseek.py, label_with_grok.py).

<details>
  <summary>OpenAI adapter example (Responses API style)</summary>

  Install:
  ```bash
  pip install openai pillow
  ```

  Environment:
  - OPENAI_API_KEY={{OPENAI_API_KEY}}

  Example function:
  ```python
  import base64
  import json
  from openai import OpenAI

  client = OpenAI()

  def openai_boxes_from_image(prompt: str, image_path: str, model: str = "gpt-4o-mini"):
      with open(image_path, "rb") as f:
          b64 = base64.b64encode(f.read()).decode()
      resp = client.responses.create(
          model=model,
          input=[{
              "role": "user",
              "content": [
                  {"type": "input_text", "text": prompt},
                  {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
              ],
          }],
          response_format={
              "type": "json_schema",
              "json_schema": {
                  "name": "boxes_schema",
                  "schema": {
                      "type": "object",
                      "properties": {
                          "boxes": {
                              "type": "array",
                              "items": {
                                  "type": "object",
                                  "properties": {
                                      "class_id": {"type": "integer"},
                                      "center_x": {"type": "number"},
                                      "center_y": {"type": "number"},
                                      "width": {"type": "number"},
                                      "height": {"type": "number"},
                                  },
                                  "required": ["class_id", "center_x", "center_y", "width", "height"],
                              },
                          }
                      },
                      "required": ["boxes"],
                  },
              },
          },
      )
      data = json.loads(resp.output[0].content[0].text)
      return data.get("boxes", [])
  ```

  If your OpenAI-compatible provider uses a different base URL, initialize the client with base_url, for example:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])  # DeepSeek Janus (example)
  # or
  client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])  # xAI Grok (example)
  ```
</details>

Tip: Always test a single image with --save-raw to verify the returned JSON before batch labelling.

---

## Troubleshooting

<details>
<summary>Common Issues and Solutions</summary>

### API and Authentication Issues
- **Missing key**: Ensure GEMINI_API_KEY is set for Gemini, or the appropriate key for your chosen provider.
  ```bash
  echo $GEMINI_API_KEY  # Should show your key (redacted)
  ```
- **Invalid API key**: Verify your key is active and has the correct permissions in the Google AI Studio console.
- **Quota exceeded**: Check your API usage limits and billing status.

### File and Path Issues
- **No images found**: Confirm your path is correct and files have a supported extension (.jpg, .png, .bmp, .tif, .tiff, .webp).
  ```bash
  find /path/to/images -name "*.jpg" -o -name "*.png" | head -5  # Check if images exist
  ```
- **Permission errors**: Ensure you have read/write permissions for the target directories.
- **Path with spaces**: Use quotes around paths containing spaces: `"path/with spaces/images"`

### Labeling Issues
- **Empty labels**: Try a different model, or inspect the raw JSON via --save-raw to see what the model returned.
- **Inconsistent class IDs**: Verify your labeling_config.json has the correct class definitions and order.
- **Malformed YOLO format**: Check that coordinates are normalized (0-1 range) and format is `class_id cx cy w h`.

### Performance and Network Issues
- **Rate limits or network errors**: Add retries/sleep, or reduce batch size (run in smaller folders).
- **Slow processing**: Consider using gemini-1.5-flash instead of gemini-1.5-pro for faster responses.
- **Memory issues**: Process images in smaller batches if you encounter out-of-memory errors.

### Dataset Cleanup Issues
- **Backup directory conflicts**: Ensure the backup directory path doesn't already exist or use --backup-dir with a unique path.
- **Insufficient disk space**: Check available space before running cleanup, especially with large datasets.
- **Cross-split deduplication taking too long**: Consider processing subsets separately or increasing --dhash-threshold for faster processing.

</details>

---

## Contributing

<details>
<summary>How to Contribute</summary>

Contributions are welcome! Here's how you can help:

### Reporting Issues
- Use the issue tracker to report bugs or request features
- Include relevant error messages, command-line arguments, and system information

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Follow the existing code style and add comments where helpful
5. Update the README if you're adding new features or changing usage
6. Submit a pull request with a clear description of your changes

### Adding New Model Providers
We welcome adapters for additional multimodal AI providers:
- Follow the pattern established in the OpenAI adapter example
- Ensure consistent JSON output format for bounding boxes
- Test with various image types and class configurations
- Document any provider-specific requirements or limitations

### Testing
Before submitting contributions:
- Test with different image formats and sizes
- Verify YOLO label format compatibility
- Test both single images and batch processing
- Check edge cases (empty images, no objects found, etc.)

</details>

---

## Notes
- The class ID mapping is configured via labeling_config.json (generated by configure_labeling.py).
- YOLO labels are normalized; ensure your training framework expects normalized coordinates.
- Backups from dataset_cleanup.py let you restore removed files; nothing is permanently deleted by that script.

