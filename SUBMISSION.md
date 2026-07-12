# Hackathon Submission Text

Copy these into the submission form.

## Submission Title

Style-Aware Vision Caption Agent

## Short Description

A Dockerized agent that watches short video clips and generates four stylistically distinct captions — formal, sarcastic, humorous tech, and humorous non-tech — using a two-step vision pipeline for accuracy and style guardrails.

## Long Description

This project is a Dockerized submission for the AMD Developer Hackathon Track 2: Video Captioning. Given a list of short video URLs, the agent downloads each clip, extracts a small set of scene-aware keyframes, and produces four captions per video: formal, sarcastic, humorous_tech, and humorous_non_tech.

The pipeline is designed around accuracy and style control. First, a vision model writes a dense structured brief from the sampled frames. The same vision model then verifies and corrects that brief to remove unsupported or invented details. A text-only caption model then generates the four required styles sequentially, with each style aware of the previous captions so the outputs do not sound repetitive. Built-in keyword guardrails detect weak style matches and trigger a single retry.

The agent uses only Fireworks AI endpoints, keeps API keys configurable via environment variables, and is packaged as a linux/amd64 Docker image. It handles failures gracefully: every task gets an entry in results.json, even if a download or model call fails. Optional local Whisper transcription is included but disabled by default to keep runtime fast and predictable.

## Categories

Event Tracks → Video Captioning

## Technologies Used

- Fireworks AI (MiniMax M3, Kimi K2P6)
- Python 3.11
- Docker / GitHub Actions
- FFmpeg
- faster-whisper
- Pydantic
- OpenAI-compatible client SDK

## Form Fields

### GitHub Repository

```
https://github.com/pjmorales1123/amd-track2-agent-v2
```

### Demo Application Platform

Select **None** (or **N/A** if available). This is a headless Docker agent, not a web demo.

### Demo Application URL

```
N/A
```

If the form requires a URL, use the Docker Hub page:

```
https://hub.docker.com/r/pjmorales04/amd-track2-agent-v2
```

### Docker Image

```
pjmorales04/amd-track2-agent-v2:latest
```
