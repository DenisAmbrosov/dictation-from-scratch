# Dictation From Scratch

How AI dictation apps like [Wispr Flow](https://wisprflow.ai) and Superwhisper work, built from scratch one stage at a time. Every dictation app runs the same steps between pressing a key and seeing your text: trigger, capture, voice activity detection, speech to text, cleanup, inject, learn, plus the context that feeds them. Each notebook builds one stage and times it.

The yardstick is the latency budget Wispr Flow has published: **full transcription plus LLM formatting within 700ms of you finishing speaking**, at p99 rather than p50. Roughly 200ms ASR, 200ms LLM, 200ms global networking, 100ms slack.

## The pipeline

```
 1 trigger → 2 capture → 3 VAD → 4 ASR → 5 cleanup → 6 inject → 7 learn
                                   │       ↑            │
                                   │  [8 context]       │
                                   └───────┴────────────┘
                                   7 teaches 8, 8 improves 4

 stage 4 is really two:  [language ID] → [transcribe]
     context = the app you are in · custom dictionary · your history · voice embedding
```

Seven stages in the chain, plus context feeding stages 4 and 5, plus a loop back from 7 that turns the chain into a cycle.

Most write-ups about dictation apps only cover stage 4. Stages 1, 2 and 6 are operating system plumbing, stages 7 and 8 are where the actual product moat lives, and the reason a naive clone feels bad is almost never the model.

**Language identification is drawn separately because leaving it out cost a live run.** With no language pinned, English spoken with a Slavic accent came back as fluent Ukrainian. It is a hard gate: once the language is chosen, every token comes from that vocabulary and nothing downstream recovers.

## Notebooks

Each one is runnable, explains the theory inline, and times everything.

| Notebook | Stage | Covers |
|---|---|---|
| `01-audio.ipynb` | 2 capture | What audio physically is. Sample rates, Nyquist, clipping, waveform, spectrum, spectrogram |
| `02-vad.ipynb` | 3 VAD | Silero VAD. Why VAD and endpointing are different problems, and why an energy threshold fails |
| `03-asr.ipynb` | 4 ASR | Whisper via Groq. Encoder-decoder vs transducer, silence hallucination, VAD-trimming, why the network dominates |
| `04-cleanup.ipynb` | 5 LLM cleanup | Qwen via Groq. Turning what you said into what you meant to type |
| `05-app.ipynb` | 1 trigger, 6 inject, 7 learn | Live microphone, streaming ASR, global hotkey, pasting into any app, and a dictionary that grows by itself, timed against 700ms |

## The headline number

**About a second from releasing the key to text on screen, against Wispr Flow's published 700ms.** Run `05-app.ipynb` §5.4 and it prints this for every press, stage by stage.

## Findings

Measured during development (October 2026) on earlier, longer versions of these notebooks. Latency depends on your network and hardware, so expect your numbers to differ; the five notebooks here are the versions shown in the video.

**On the recogniser**

- **Whisper invents text from silence.** Three seconds of digital zeros reliably returns `" Thank you."`, learned from subtitle tracks in its training data. A transducer cannot do this, because "nothing" is a symbol in its output vocabulary and Whisper has no way to say it.
- **Whisper silently deletes your filler words; ElevenLabs Scribe keeps them.** Stage 4 is quietly doing part of stage 5's job, uncontrollably, and you cannot reason about a pipeline stage without knowing what the previous one already did to your data.
- **Language identification is a hidden stage inside ASR, and it is a hard gate.** Left to auto-detect, English spoken with a Slavic accent came back as fluent Ukrainian. Not partly wrong: entirely wrong, and confident.
- **The cleanup pass then hid it.** The raw transcript had a truncated word and a trailing fragment, the visible signals that something went wrong. The cleaned one was a tidy sentence. **A stage built to make output look confident makes wrong output look confident too, and the better it gets the more dangerous this is.**

**On the cleanup LLM, which is where the real problem is**

- **A prompt requests a constraint, it cannot enforce one.** Four live failures, one root cause: the strict prompt *invented* a clause the speaker never said, *laundered* a total upstream failure, *answered* a spoken question instead of transcribing it, and *summarised* a 40-word dictation down to 40% and inverted its meaning.
- **Your own voice is a prompt injection and it needs no attacker.** `system=rules, user=transcript` is the exact shape of a normal chat request, so the instruction-tuned prior beats the system prompt. Speak a question and you get an answer typed into your document.
- **Our own context leaked into the output.** One line of app context in the system prompt came out as words in the typed text. Wispr puts the text around your cursor and on your screen in that same channel.

**On latency**

- **The network was the latency all along.** Three independent measurements converge on ~200ms of fixed overhead, matching the ~200ms networking line in Wispr's own published budget. Fitting two cleanup runs gives ~4ms per output token on top of ~209ms you pay before a single token exists.
- **Output tokens are latency.** A stricter prompt halved both, 453ms to 224ms, with no model or infrastructure change.
- **Streaming is not faster, it is constant.** On a 10-second clip streaming *lost*, 380ms against 292ms batch, because a trimmed upload is only ~144KB and there is nothing to overlap. But 2.7s of speech took 897ms and 21.9s took 841ms. Batch latency is a function of how long you spoke; streaming's is not.
- **Silero VAD is free.** 0.105ms per 32ms frame in the live path, about 300x real time. VAD is never where the latency is.
- **Push-to-talk is a latency optimisation, not a UI choice.** Key-up means the user endpointed for you, deleting the 500ms silence wait that is the largest single item in the live timing table. **~450ms recovered by a keyboard**, same models, same network.

**On the parts nobody writes about**

- **Stage 6, injection, costs 3.8ms of real work.** Save the clipboard, set it, post ⌘V, restore. Everything else is a hardcoded sleep, which exists because `CGEventPost` is asynchronous and macOS never tells you when another process read the pasteboard. Every dictation app on macOS carries that magic number.
- **A dictation app of this kind can never ship on the Mac App Store.** The Accessibility permission it requires is incompatible with the sandbox the store requires. A platform restriction created the direct-download business model of an entire product category.
- **The self-teaching dictionary is forty lines, not a company.** Read the injected text back through the Accessibility API, diff it, keep corrections that are not in `/usr/share/dict/words`. Only the *style* half of the learning loop needs a user population.
- **The hard part was never the AI.** Five things happen before a single audio sample exists, all local, all inside ~30ms, and the model is one row in the middle.

### One finding that did not replicate, kept on purpose

An early run suggested trimming silence before upload cut the worst case by 706ms and collapsed latency spread 27x. **On re-run it did not hold**: the median improved by 35ms and the worst case got 31ms *worse*. Halving an upload cannot make anything slower, so this is 5-sample noise, not a regression.

It is left here because it makes the same point one layer deeper, and because it is the fourth time in this project that a single sample pointed the wrong way. **A repo about tail latency should not quote an n=5 result as a finding**, and the correction is more useful than the claim was.

## Setup

```bash
uv sync
cp .env.example .env    # then add your own keys
```

Needs `GROQ_API_KEY` (batch ASR and the cleanup LLM) and `ELEVENLABS_API_KEY` (streaming ASR). `.env` is gitignored.

Python is pinned to 3.12 rather than 3.14, because torch and the audio ML wheels lag new releases.

## Why two ASR providers

**Groq** is batch only. No websocket, so it can never demonstrate how real dictation works, however fast the chip is. It is the fastest way to understand the pipeline.

**ElevenLabs** Scribe v2 Realtime has a websocket, publishes p95 and p99, and supports keyterm prompting for up to 1,000 terms, which is Wispr's custom-dictionary feature exposed as an API parameter. It is how you understand the product.

Not better, different.

## License

MIT
