# Train a tiny GPT to write Chinese poetry, without knowing Chinese

**TangShiGPT-Course · a hands-on lab in English**

Imagine these marks came from another planet. You could notice repeated symbols, count which ones follow each other, and build a little machine that predicts the next mark. Then you could ask a translator what the machine just wrote.

Here the marks are real Chinese characters and the training material is an electronic collection of Tang poetry. The “alien script” is a thought experiment for the learner, not a description of Chinese, a living language with a long literary tradition. You do not have to read, write, or type it. Copy the sample strings you need. The instructions are in English; the model's training text and output remain Chinese.

**What you'll make:** a locally trained character-level GPT, a saved checkpoint, and an English reading of a generated poem. You can read and translate the example on this page without installing anything. For the coding labs you need a little Python, a terminal, Python 3.10+, and an internet connection for the initial download. No GPU, Chinese keyboard, AI account, or paid API is needed for the local labs.

Choose a route: **just curious?** Do [the translation exercise](#1-read-a-result-before-installing-anything) on this page. **Want to build?** Work through all five sections. The supplied checkpoint download is only for section 2. You can skip it and train from the repository's bundled corpus in sections 3–4.

<img src="assets/english-course-poster.jpg" alt="TangShi GPT poster: Train a tiny GPT. Read across time. No Chinese required. The character 月 appears inside a glass lens." width="380">

## 1. Read a result before installing anything

This is sample 4 of 4 from an actual run of the **supplied** checkpoint. It did not come from our 200-step student model.

> 江上送故人不知，  
> 楊花流水東西飛。  
> 故人何處逢春草，  
> 目送黃雲不可歸。

| Chinese | Close English reading | What to check |
| --- | --- | --- |
| 江上送故人不知 | Seeing an old friend off by the river; I do not know… | 送 is “see off”; 故人 is “old friend.” Who doesn't know, and what they don't know, are not stated. |
| 楊花流水東西飛 | Willow catkins, flowing water, flying east and west. | 楊花 suggests drifting tree fluff/catkins. The catkins are probably what is flying. The compressed line does not spell out how they relate to the water. |
| 故人何處逢春草 | Where will my old friend come across spring grass? | 何處 asks “where”; 春草 is spring grass. “My” and the future tense are English choices. |
| 目送黃雲不可歸 | I watch yellow clouds go; no return is possible. | 目送 means to watch something go; 黃雲 is yellow clouds. The line leaves unclear who or what cannot return. |

A readable rendering, with uncertainty left in:

> By the river, I see an old friend off. I do not know…  
> Willow catkins fly this way and that by the running water.  
> Where might my old friend meet the spring grass?  
> I watch the yellow clouds go. There is no returning.

The added “I” and “by” make the English read naturally; they are interpretation. There is some continuity beyond the shape: 故人 (“old friend”) recurs in lines 1 and 3, and the first line's 送 (“see off”) echoes 目送 (“watch go”) in line 4. The farewell phrase itself was supplied by the prompt; the later repetition and imagery were generated. This supports a reading of a farewell, moving water, spring growth, and an uncertain return. It doesn't tell you the friend died, where they were going, or what happened next. This is generated practice material, not a historical poem or a claim about Li Bai.

**Try it:** Give the four original lines to an AI translator you already use, or work just with the table. Before accepting a fluent version, ask:

> Translate these four generated Chinese lines into English, one line at a time. Give a close gloss and a readable version. Quote the Chinese characters that support each concrete image or action. Mark inferred subjects, tense, and ambiguous words. Say if a line is unclear. Do not reconstruct a story, identify an author, or add scenery. Keep the line order.

**Quick check:** If your translation adds a funeral, ask which characters justify it. “It feels like a farewell” is an interpretation; “a funeral” is an unsupported event. A fluent translation can conceal trouble in the original. Use the gloss as something to check, not as a quality score.

If you don't have an AI translator, look for 故人 in two lines and find its gloss in the table. Then compare the words “catkins” and “funeral”: which one has something concrete in the source to point to?

## 2. Run the supplied model

Download the [v1.0.0 learning ZIP](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/TangShiGPT-Course.zip) (about 129 MiB), extract it, and open a terminal where `requirements.txt` and `tangshi/` are both visible. The ZIP predates this English edition; keep this page open for the lessons. If you prefer not to download the big checkpoint, [download the tested source ZIP](https://github.com/OakcoderX/TangShiGPT-Course/archive/6a5b74b44673ef3645825829bb6837ddd4844525.zip), set up Python below, and start at section 3. To run the supplied model from a source checkout, [download the supplied checkpoint](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/more_2400_li.safetensors) and put it in `weights/` next to `vocab.json`. [Weight files and checksums](../weights/README.md).

**macOS / Linux**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, replace the leading `python` in each command below with `.\.venv\Scripts\python.exe`. If the Chinese appears garbled, try a UTF-8 terminal and a font that supports Chinese. If Python cannot find `tangshi`, check that you are in the extracted project directory, not its parent. If dependency installation fails, use [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for your platform, then install the remaining requirements in the same environment. The examples below use CPU.

```sh
python -m tangshi generate --device cpu --weights weights/more_2400_li.safetensors --form seven --prompt "江上送故人" --seed 42 --temperature 0.65 --count 4
```

The supplied prompt means roughly “see an old friend off by/on the river.” The prompt is given to the model; the continuation is sampled. The poem above was output **4 of 4**. Exact text can differ across machines and library builds. Check that you get four samples, not an exact verbal match. If copying the prompt fails, remove `--prompt "江上送故人"` to generate from the beginning. English text does not substitute for the Chinese prompt in this model's vocabulary.

**Experiment:** Compare with an unconstrained run using the same checkpoint, seed and prompt:

```sh
python -m tangshi generate --device cpu --weights weights/more_2400_li.safetensors --form natural --prompt "江上送故人" --seed 42 --temperature 0.65 --max-chars 60 --count 1
```

Natural mode stops at a learned end token or the length limit. This run returned `江上送故人，相思離遠客。白雲送明月，相憶不能去。`: four five-character lines by the shape check, **without** the seven-character rule. A single natural sample is not evidence that it always does this. The five- and seven-character modes enforce four lines and the punctuation pattern. Four neat lines don't prove the model learned rhyme, strict prosody, coherence, or a poet’s style.

**Spot the echo:** Find 送 in `江上送故人` (“see an old friend off by the river”) and again in `白雲送明月`. We supplied the five characters in the first line; the model sampled its comma and every character of the third line. With 白雲 (“white clouds”) and 明月 (“bright moon”), one reading is “White clouds see the bright moon off.” You can check the image and the echo without knowing Chinese. This doesn’t tell us whether the model intended the echo or borrowed a phrase from the corpus.

**Compare like with like:** Use sample 1 from each run (the first seven-character sample is in the appendix), or set `--count 1` in both commands. The translated opener was sample 4. `--seed` initializes the sampler once per command, so different sample positions have different draw histories. Holding the checkpoint, prompt, seed, temperature and device fixed compares the modes as a bundle: their candidate rules, punctuation and stopping differ.

## 3. Count which character comes next

Imagine a toy archive containing `月明`, `月明`, and `月下`. The symbol after 月 is 明 twice and 下 once: 2/3 and 1/3. 月 often means moon, but counting doesn't need that meaning. This course treats characters as tokens; it does not assume they work like letters in English.

```sh
python scripts/prepare_data.py
python scripts/next_char.py --data runs/lesson-data/tangshi_clean.txt --char 月
```

**Windows PowerShell: `UnicodeEncodeError`?** If Python reports that `charmap` cannot encode the Chinese output, run this in the same PowerShell window, then retry the command that failed:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

This sets Python’s [standard-stream encoding](https://docs.python.org/3.13/using/cmdline.html#envvar-PYTHONIOENCODING) for commands started from that window. It does not change the dataset. If the command runs but shows boxes instead of characters, try a terminal font that supports Chinese.

The first command writes a clean copy to `runs/lesson-data/`, leaving the bundled source data untouched. Look for `JSON 记录 57607` (57,607 records) and `清洗后 56937` (56,937 cleaned lines). The second displays counts and percentages among the observed **successors of 月**, including punctuation and possibly the end marker. It is not counting shares of the entire collection. The recorded run saw the comma after 月 1,997 times (16.67%); the full stop was the most common successor of 山 (7.02%).

**Try it:** Change `--char 月` to `--char 山` (mountain). Do the most frequent followers change? Can that tell you the meaning of the full poem? This baseline uses one character at a time. GPT uses several earlier tokens as context to predict the next one, samples a character, appends it, and repeats.

## 4. Train and reload your own model

```sh
python -m tangshi train --device cpu --seed 42 --data runs/lesson-data/tangshi_clean.txt --limit 2000 --context 48 --width 64 --layers 2 --steps 200 --output runs/english-student.pt
python -m tangshi generate --device cpu --weights runs/english-student.pt --form five --seed 42 --count 2
```

This uses up to 2,000 examples to train a two-layer, width-64 model with up to 48 tokens in context. In the recorded CPU run the CLI used 1,800 for training and 200 for validation; sampled training loss moved from 8.2187 to 6.0483 and the printed validation sample loss was 6.0618. These are sanity checks on the prediction task, not grades for poetry. The second command runs in a new process and loads your saved checkpoint. Training refuses to overwrite an existing checkpoint. For a repeat run, choose a new name and use it in both commands.

One real short-trained output was `門雲一裏又，幾訣不人已。紅侵多同幾，不在青寨居。`. It has the requested shape and individual symbols you can recognize, but phrases like `幾訣不人已` are unclear. You can pick out 門 (gate), 雲 (cloud), and 人 (person), but the four lines do not support a stable reading. Try feeding this sample to the same translator. If it gives you a complete scene or a clear plot, ask it to show the characters that support each claim.

## 5. Check form, meaning and guesswork separately

Keep your first four supplied-checkpoint samples and first two student samples, including the awkward ones. Make a small scorecard:

| Check | What earns a yes? |
| --- | --- |
| Shape | Four lines, with five or seven characters each, excluding punctuation. |
| Images | Two concrete things or actions can be traced to exact characters. |
| Connection | The lines plausibly share a scene or idea; the ending does not require an invented event. |
| Translation restraint | Inferred subjects, events, and tense are marked or left open. |

Record yes, no, or uncertain and keep the number of samples visible. The two runs use different checkpoints and different settings. They are a demonstration, not a controlled comparison. For a cleaner comparison, first match the prompt, form, temperature and number of samples across both models; use no prompt if the small vocabulary does not support it. Record the seed and settings, and then change just one of them. If possible, hide which model generated which sample before judging clarity. The upstream experiments generated Tang-form drafts but did **not** establish reliable Li Bai imitation.

**Finish:** Save a command you ran, the original output, your English reading, and one thing you couldn't verify. If you didn't train or generate locally, label your result “translation-only.” Share only non-private samples, with the versions and exact command. For the optional Chinese companion, open `docs/index.html` in a browser after extracting the [learning ZIP](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/TangShiGPT-Course.zip). You don't need it to complete this course.

## Reproduction and credits

<details>
<summary>All six outputs in sampling order</summary>

```text
Supplied checkpoint, seven-character mode (four consecutive samples):
江上送故人不還，三千子陵山下間。日斜風起花當樓，一行相送到天邊。
江上送故人不知，明日暮雲愁又時。我今華髮明月促，君去宋家留我辭。
江上送故人不見，正當秋涼月下時。楚山斷色空自見，楚水寒聲更不知。
江上送故人不知，楊花流水東西飛。故人何處逢春草，目送黃雲不可歸。

New checkpoint, five-character mode (two consecutive samples):
門雲一裏又，幾訣不人已。紅侵多同幾，不在青寨居。
轉別不深須，樓與危馬人。只風家入漢，歸淚忙月一。
```

</details>

Format check in this run: supplied **4/4** and student **2/2** met their requested shape. Those numbers measure a rule the program enforced. They do not measure meaning, and we did not score all six for coherence.

The command paths were run from both the checked-out source and an extracted v1.0.0 release ZIP on Linux x86-64 CPU with Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 1.26.4 and safetensors 0.5.3, using [this tested source revision](https://github.com/OakcoderX/TangShiGPT-Course/tree/6a5b74b44673ef3645825829bb6837ddd4844525). Supplied checkpoint SHA-256: `340b33beeeabc215968f3f12fa74f14ebfd41668e95b01de3c9ee8d2aa995c30`. The English readings are teaching glosses, not published translations. The installation paths for macOS and Windows were not separately run for this edition.

Inspired by [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/); the project uses its own PyTorch implementation and is not an official Karpathy course. The electronic [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) collection retains its [MIT notice](../data/LICENSE.chinese-poetry). It is not a critical edition. This project's code, course and weights are available to download and study; no additional open-source license is supplied for them. [Source and release notes](来源与发布.md).

## Listed on

<a href="https://aisotools.com/tool/tangshigpt-course" rel="nofollow"><img src="https://aisotools.com/badges/aiso-featured-basic.svg" alt="TangShiGPT-Course listed on AISO Tools" width="176"></a>
<a href="https://www.ainavdir.com/tool/tangshigpt-course" rel="nofollow"><img src="https://www.ainavdir.com/badges/listed-on-ainavdir.svg" alt="TangShiGPT-Course listed on AINavDir" width="176"></a>

**Learning and project directories:** [AI Learn Nav](https://ailearnnav.com/en/resource/tangshigpt-course) · [Indie CN](https://dev.summhub.com/en/p/tangshigpt-course-%E4%BB%8E-microgpt-%E5%88%B0%E5%94%90%E8%AF%97-gpt) · [DevTrends](https://devtrends.site/en/projects/oakcoderx/tangshigpt-course/)

These links document listings, not endorsements. Some directory entries still describe only the original Chinese course.
