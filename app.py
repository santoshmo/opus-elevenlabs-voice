import asyncio, base64, json, os
from contextlib import suppress
from urllib.parse import urlencode, quote

import httpx, uvicorn, websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
AKEY=os.getenv("ANTHROPIC_API_KEY","")
EKEY=os.getenv("ELEVENLABS_API_KEY","")
MODEL=os.getenv("ANTHROPIC_MODEL","claude-opus-5-5")
EFFORT=os.getenv("ANTHROPIC_EFFORT","low")
VOICE=os.getenv("ELEVENLABS_VOICE_ID","")
TTS_MODEL=os.getenv("ELEVENLABS_TTS_MODEL","eleven_multilingual_v2")
OUT=os.getenv("ELEVENLABS_OUTPUT_FORMAT","mp3_44100_128")
STABILITY=float(os.getenv("ELEVENLABS_STABILITY","0.30"))
SIMILARITY=float(os.getenv("ELEVENLABS_SIMILARITY","0.80"))
STYLE=float(os.getenv("ELEVENLABS_STYLE","0.55"))
SPEED=float(os.getenv("ELEVENLABS_SPEED","1.0"))
BOOST=os.getenv("ELEVENLABS_SPEAKER_BOOST","true").lower() in {"1","true","yes","on"}
SYSTEM=os.getenv("SYSTEM_PROMPT","You are a natural spoken conversational assistant. Lead with the answer. Keep normal responses concise and conversational. Avoid markdown unless asked.")
app=FastAPI(title="Opus + ElevenLabs Voice")
HERE=os.path.dirname(__file__)
app.mount("/static",StaticFiles(directory=os.path.join(HERE,"static")),name="static")

@app.get("/")
async def root(): return FileResponse(os.path.join(HERE,"static","index.html"))

@app.get("/health")
async def health():
    return {"ok":bool(AKEY and EKEY),"anthropic_model":MODEL,"tts_model":TTS_MODEL,"tts_output_format":OUT}

@app.get("/api/voices")
async def voices():
    if not EKEY: return JSONResponse({"error":"ELEVENLABS_API_KEY missing"},500)
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.get("https://api.elevenlabs.io/v2/voices",headers={"xi-api-key":EKEY},params={"page_size":100,"include_total_count":"false"})
        if r.status_code>=400: return JSONResponse({"error":r.text},r.status_code)
        vs=[{"voice_id":v.get("voice_id"),"name":v.get("name") or v.get("voice_id")} for v in r.json().get("voices",[])]
        return {"voices":vs,"default_voice_id":VOICE}

class Session:
    def __init__(self,browser,voice,effort):
        self.b=browser; self.voice=voice; self.effort=effort; self.hist=[]; self.task=None; self.turn=0; self.current=None; self.closed=False
        self.http=httpx.AsyncClient(timeout=httpx.Timeout(connect=20,read=None,write=30,pool=30))
        self.lock=asyncio.Lock()
    async def send(self,x):
        if self.closed:return
        async with self.lock:
            try: await self.b.send_json(x)
            except Exception:self.closed=True
    async def interrupt(self):
        old=self.current
        if self.task and not self.task.done():
            self.task.cancel()
            with suppress(asyncio.CancelledError,Exception): await self.task
        self.task=None; self.current=None
        if old: await self.send({"type":"interrupt","turn_id":old})
    async def user(self,text):
        text=(text or "").strip()
        if not text:return
        await self.interrupt(); self.hist.append({"role":"user","content":text}); await self.send({"type":"user_final","text":text})
        self.turn+=1; tid=f"assistant-{self.turn}"; self.current=tid
        self.task=asyncio.create_task(self.reply(tid))
    async def audio_reader(self,ws,tid):
        chunks=bytes_=0
        async for raw in ws:
            d=json.loads(raw)
            if d.get("error") or d.get("message_type")=="error": raise RuntimeError(str(d))
            a=d.get("audio")
            if a:
                chunks+=1
                with suppress(Exception): bytes_+=len(base64.b64decode(a))
                if chunks==1: await self.send({"type":"audio_state","turn_id":tid,"state":"receiving","message":"Receiving ElevenLabs audio"})
                await self.send({"type":"audio","turn_id":tid,"audio":a,"format":OUT,"chunk_index":chunks,"bytes_total":bytes_})
            if d.get("isFinal") or d.get("is_final"):break
        await self.send({"type":"audio_state","turn_id":tid,"state":"received","message":f"Audio received · {chunks} chunks · {bytes_//1024} KB"})
        return chunks,bytes_
    async def reply(self,tid):
        full=""; tws=None; reader=None
        try:
            await self.send({"type":"assistant_start","turn_id":tid})
            await self.send({"type":"audio_state","turn_id":tid,"state":"connecting","message":"Connecting to ElevenLabs"})
            q=urlencode({"model_id":TTS_MODEL,"output_format":OUT,"inactivity_timeout":60})
            tws=await websockets.connect("wss://api.elevenlabs.io/v1/text-to-speech/"+quote(self.voice,safe="")+"/stream-input?"+q,max_size=8*1024*1024)
            await tws.send(json.dumps({"text":" ","xi_api_key":EKEY,"voice_settings":{"stability":STABILITY,"similarity_boost":SIMILARITY,"style":STYLE,"speed":SPEED,"use_speaker_boost":BOOST},"generation_config":{"chunk_length_schedule":[50,90,140,220]}}))
            reader=asyncio.create_task(self.audio_reader(tws,tid))
            headers={"x-api-key":AKEY,"anthropic-version":"2023-06-01","content-type":"application/json"}
            payload={"model":MODEL,"max_tokens":2048,"system":SYSTEM,"messages":self.hist,"stream":True,"output_config":{"effort":self.effort}}
            async with self.http.stream("POST","https://api.anthropic.com/v1/messages",headers=headers,json=payload) as r:
                if r.status_code>=400: raise RuntimeError(f"Anthropic {r.status_code}: {(await r.aread()).decode(errors='replace')[:700]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):continue
                    raw=line[5:].strip()
                    if not raw:continue
                    e=json.loads(raw); d=e.get("delta") or {}
                    if e.get("type")!="content_block_delta" or d.get("type")!="text_delta":continue
                    t=d.get("text") or ""
                    if not t:continue
                    full+=t; await self.send({"type":"assistant_delta","turn_id":tid,"text":t}); await tws.send(json.dumps({"text":t}))
            await tws.send(json.dumps({"text":" ","flush":True})); await tws.send(json.dumps({"text":""}))
            chunks,bytes_=await reader
            if full.strip():self.hist.append({"role":"assistant","content":full.strip()})
            await self.send({"type":"assistant_end","turn_id":tid,"text":full.strip(),"audio_chunks":chunks,"audio_bytes":bytes_})
        except asyncio.CancelledError:
            if full.strip():self.hist.append({"role":"assistant","content":full.strip()})
            if reader and not reader.done():reader.cancel()
            raise
        except Exception as e:
            await self.send({"type":"error","turn_id":tid,"message":f"{type(e).__name__}: {e}"})
            await self.send({"type":"audio_state","turn_id":tid,"state":"error","message":str(e)})
        finally:
            if tws:
                with suppress(Exception):await tws.close()
    async def close(self):
        self.closed=True
        if self.task and not self.task.done():self.task.cancel()
        await self.http.aclose()

async def stt_read(ws,s):
    async for raw in ws:
        d=json.loads(raw); k=d.get("message_type")
        if k=="session_started": await s.send({"type":"ready"})
        elif k=="partial_transcript":
            t=(d.get("text") or "").strip(); await s.send({"type":"user_partial","text":t})
            if t and s.current: await s.interrupt()
        elif k in {"committed_transcript","edited_transcript"}:
            t=(d.get("edited_text") or d.get("text") or "").strip()
            if t: await s.user(t)
        elif k in {"error","rate_limited"}: await s.send({"type":"error","message":d.get("error") or str(d)})

@app.websocket("/ws")
async def socket(b:WebSocket):
    await b.accept()
    if not AKEY or not EKEY:
        await b.send_json({"type":"error","message":"Server API keys are not configured."}); await b.close(code=1011); return
    voice=b.query_params.get("voice_id") or VOICE; effort=b.query_params.get("effort") or EFFORT
    if not voice: await b.send_json({"type":"error","message":"Choose a voice first."}); await b.close(code=1008); return
    s=Session(b,voice,effort)
    params={"model_id":"scribe_v2_realtime","audio_format":"pcm_16000","language_code":"en","commit_strategy":"vad","vad_silence_threshold_secs":"0.55","vad_threshold":"0.4","min_speech_duration_ms":"120","min_silence_duration_ms":"120","filter_background_audio":"true","keepalive_interval_ms":"3000"}
    try:
        async with websockets.connect("wss://api.elevenlabs.io/v1/speech-to-text/realtime?"+urlencode(params),additional_headers={"xi-api-key":EKEY},max_size=8*1024*1024) as stt:
            rr=asyncio.create_task(stt_read(stt,s))
            try:
                while True:
                    m=await b.receive_json(); k=m.get("type")
                    if k=="audio" and m.get("audio"): await stt.send(json.dumps({"message_type":"input_audio_chunk","audio_base_64":m["audio"]}))
                    elif k=="text": await s.user(m.get("text",""))
                    elif k=="interrupt": await s.interrupt()
                    elif k=="ping": await s.send({"type":"pong"})
                    elif k=="clear": await s.interrupt(); s.hist.clear(); await s.send({"type":"cleared"})
            finally:
                rr.cancel()
                with suppress(asyncio.CancelledError,Exception):await rr
    except WebSocketDisconnect:pass
    except Exception as e: await s.send({"type":"error","message":f"Voice session failed: {e}"})
    finally: await s.close()

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=int(os.getenv("PORT","8000")))
