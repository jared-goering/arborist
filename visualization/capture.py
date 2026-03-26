#!/usr/bin/env python3
"""Capture animation frames via CDP and stitch with ffmpeg."""
import subprocess, base64, os, json, socket

FRAMES_DIR = "/tmp/arborist_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# Clean old frames
for f in os.listdir(FRAMES_DIR):
    if f.endswith('.png'):
        os.unlink(os.path.join(FRAMES_DIR, f))

# Animation params
FPS = 30
SPEED = 1.5
MAX_LINEAR = 100
TREE_COUNT = 23
TOTAL_ANIM_FRAMES = int(max(MAX_LINEAR, TREE_COUNT * 3) / SPEED) + 5
INTRO_HOLD = FPS * 1
END_HOLD = FPS * 3
TOTAL_FRAMES = INTRO_HOLD + TOTAL_ANIM_FRAMES + END_HOLD

print(f"Rendering {TOTAL_FRAMES} frames")

# Simple CDP via raw websocket (no external deps)
import http.client
import hashlib
import struct
import ssl

# Use simple HTTP to get CDP info, then use subprocess with node for WS
# Actually, let's just use curl + Chrome's HTTP endpoints

# Alternative: use Chrome DevTools Protocol via HTTP
TARGET_ID = "A27B47780D01BEDAAC87C538E3C30216"
CDP_HOST = "127.0.0.1"
CDP_PORT = 18800

def cdp_request(method, params=None):
    """Make CDP request via HTTP (won't work for most methods, need WS)"""
    pass

# Let's use node.js for the CDP connection since it's available
node_script = f"""
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const WS_URL = 'ws://127.0.0.1:18800/devtools/page/{TARGET_ID}';
const FRAMES_DIR = '{FRAMES_DIR}';
const FPS = {FPS};
const SPEED = {SPEED};
const TOTAL_ANIM = {TOTAL_ANIM_FRAMES};
const INTRO = {INTRO_HOLD};
const HOLD = {END_HOLD};

let msgId = 1;
const pending = new Map();

const ws = new WebSocket(WS_URL);

ws.on('open', async () => {{
  console.log('Connected to Chrome');
  await run();
}});

ws.on('message', (data) => {{
  const msg = JSON.parse(data.toString());
  if (msg.id && pending.has(msg.id)) {{
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  }}
}});

function send(method, params) {{
  return new Promise((resolve) => {{
    const id = msgId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({{ id, method, params }}));
  }});
}}

async function evaluate(expr) {{
  const r = await send('Runtime.evaluate', {{ expression: expr, returnByValue: true }});
  return r?.result?.result?.value;
}}

async function screenshot() {{
  const r = await send('Page.captureScreenshot', {{
    format: 'png',
    clip: {{ x: 0, y: 0, width: 1920, height: 1080, scale: 1 }}
  }});
  return Buffer.from(r.result.data, 'base64');
}}

async function run() {{
  let frame = 0;

  // Intro
  for (let i = 0; i < INTRO; i++) {{
    await evaluate('renderFrame(0)');
    const png = await screenshot();
    fs.writeFileSync(path.join(FRAMES_DIR, \`frame_${{String(frame).padStart(5, '0')}}.png\`), png);
    frame++;
    if (i % FPS === 0) console.log(\`Intro ${{i}}/${{INTRO}}\`);
  }}

  // Animation
  for (let f = 0; f < TOTAL_ANIM; f++) {{
    const v = f * SPEED;
    await evaluate(\`renderFrame(${{v}})\`);
    const png = await screenshot();
    fs.writeFileSync(path.join(FRAMES_DIR, \`frame_${{String(frame).padStart(5, '0')}}.png\`), png);
    frame++;
    if (f % 10 === 0) console.log(\`Anim ${{f}}/${{TOTAL_ANIM}}\`);
  }}

  // Hold
  const finalVal = TOTAL_ANIM * SPEED;
  for (let i = 0; i < HOLD; i++) {{
    await evaluate(\`renderFrame(${{finalVal}})\`);
    const png = await screenshot();
    fs.writeFileSync(path.join(FRAMES_DIR, \`frame_${{String(frame).padStart(5, '0')}}.png\`), png);
    frame++;
    if (i % FPS === 0) console.log(\`Hold ${{i}}/${{HOLD}}\`);
  }}

  console.log(\`Done: ${{frame}} frames\`);
  ws.close();
  process.exit(0);
}}
"""

# Write and run node script
node_path = os.path.join(FRAMES_DIR, "capture.js")
with open(node_path, 'w') as f:
    f.write(node_script)

print("Starting frame capture via Node.js...")
result = subprocess.run(["node", node_path], capture_output=True, text=True, timeout=300)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
if result.returncode != 0:
    print(f"Node exited with code {result.returncode}")
    exit(1)

# Stitch with ffmpeg
output = "/Users/jared/Projects/arborist/visualization/arborist-animation.mp4"
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(FRAMES_DIR, "frame_%05d.png"),
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-vf", "scale=1920:1080",
    output
]
print(f"Stitching with ffmpeg...")
subprocess.run(cmd, check=True)
print(f"\\n✅ Done: {output}")
print(f"Size: {os.path.getsize(output) / 1024 / 1024:.1f} MB")
