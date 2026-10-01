# Opus 5.5 × ElevenLabs Voice

A mobile-friendly realtime voice assistant using Anthropic Claude Opus 5.5, ElevenLabs Scribe realtime STT, and ElevenLabs streaming TTS.

## Railway

Required secrets:

- `ANTHROPIC_API_KEY`
- `ELEVENLABS_API_KEY`

Optional:

- `ELEVENLABS_VOICE_ID`

The app listens on Railway's `PORT` and exposes `/health`.
