#!/usr/bin/env python3
"""Transform dark-theme slides to a clean white/light theme."""
import os, re, glob

replacements = [
    # --- Backgrounds ---
    (r'background:\s*#080808', 'background: #ffffff'),
    (r'background:\s*#070d09', 'background: #f8faf8'),
    (r'background:\s*#050a08', 'background: #f4f8f5'),
    (r'background:\s*#0a0a0a', 'background: #ffffff'),
    (r'background:\s*#000', 'background: #ffffff'),
    (r'background: linear-gradient\(135deg, #030a05[^)]+\)', 'background: linear-gradient(135deg, #f0f8f2 0%, #e8f5ec 50%, #f0f8f2 100%)'),
    (r'background: linear-gradient\(135deg, #0a0a0a[^)]+\)', 'background: linear-gradient(135deg, #f8fff9 0%, #edf7f0 40%, #f0faf3 70%, #e8f5ec 100%)'),
    (r'background: linear-gradient\(160deg, #0d1f12[^)]+\)', 'background: linear-gradient(160deg, #e8f5ec 0%, #f8fff9 100%)'),

    # --- Grid/dot patterns: make them light green ---
    (r'rgba\(0,180,80,0\.0[234]\)', 'rgba(0,150,60,0.06)'),
    (r'rgba\(0,200,80,0\.025\)', 'rgba(0,150,60,0.07)'),
    (r'rgba\(0,200,80,0\.03\)', 'rgba(0,150,60,0.07)'),
    (r'rgba\(255,255,255,0\.02\)', 'rgba(0,0,0,0.04)'),
    (r'rgba\(255,255,255,0\.025\)', 'rgba(0,0,0,0.04)'),

    # --- Text: white -> dark ---
    (r'color:\s*#ffffff', 'color: #0d1f12'),
    (r'color:\s*white', 'color: #0d1f12'),
    (r'color: rgba\(255,255,255,0\.9\)', 'color: rgba(0,20,10,0.85)'),
    (r'color: rgba\(255,255,255,0\.85\)', 'color: rgba(0,20,10,0.8)'),
    (r'color: rgba\(255,255,255,0\.8\)', 'color: rgba(0,20,10,0.75)'),
    (r'color: rgba\(255,255,255,0\.75\)', 'color: rgba(0,20,10,0.7)'),
    (r'color: rgba\(255,255,255,0\.7\)', 'color: rgba(0,20,10,0.65)'),
    (r'color: rgba\(255,255,255,0\.65\)', 'color: rgba(0,20,10,0.6)'),
    (r'color: rgba\(255,255,255,0\.6\)', 'color: rgba(0,20,10,0.55)'),
    (r'color: rgba\(255,255,255,0\.55\)', 'color: rgba(0,20,10,0.5)'),
    (r'color: rgba\(255,255,255,0\.5\)', 'color: rgba(0,20,10,0.5)'),
    (r'color: rgba\(255,255,255,0\.45\)', 'color: rgba(0,20,10,0.45)'),
    (r'color: rgba\(255,255,255,0\.4\)', 'color: rgba(0,20,10,0.45)'),
    (r'color: rgba\(255,255,255,0\.35\)', 'color: rgba(0,20,10,0.4)'),
    (r'color: rgba\(255,255,255,0\.3\)', 'color: rgba(0,20,10,0.4)'),
    (r'color: rgba\(255,255,255,0\.25\)', 'color: rgba(0,20,10,0.35)'),
    (r'color: rgba\(255,255,255,0\.2\)', 'color: rgba(0,20,10,0.3)'),

    # --- Cards/boxes: dark glass -> white with border ---
    (r'background: rgba\(255,255,255,0\.03\)', 'background: rgba(0,0,0,0.02)'),
    (r'background: rgba\(255,255,255,0\.04\)', 'background: rgba(0,0,0,0.02)'),
    (r'background: rgba\(255,255,255,0\.06\)', 'background: rgba(0,0,0,0.04)'),
    (r'background: rgba\(255,255,255,0\.08\)', 'background: rgba(0,0,0,0.04)'),

    # --- Borders: white faint -> dark faint ---
    (r'border: 1px solid rgba\(255,255,255,0\.07\)', 'border: 1px solid rgba(0,0,0,0.1)'),
    (r'border: 1px solid rgba\(255,255,255,0\.08\)', 'border: 1px solid rgba(0,0,0,0.1)'),
    (r'border: 1px solid rgba\(255,255,255,0\.06\)', 'border: 1px solid rgba(0,0,0,0.08)'),
    (r'border: 1px solid rgba\(255,255,255,0\.1\)', 'border: 1px solid rgba(0,0,0,0.1)'),
    (r'border: 1px solid rgba\(255,255,255,0\.12\)', 'border: 1px solid rgba(0,0,0,0.12)'),
    (r'border: 1px solid rgba\(255,255,255,0\.18\)', 'border: 1px solid rgba(0,0,0,0.15)'),
    (r'border: 1px solid rgba\(255,255,255,0\.2\)', 'border: 1px solid rgba(0,0,0,0.15)'),

    # --- Vertical divider lines ---
    (r'rgba\(255,255,255,0\.08\)', 'rgba(0,0,0,0.08)'),
    (r'rgba\(255,255,255,0\.1\)', 'rgba(0,0,0,0.08)'),

    # --- Radial glows: keep but lighten ---
    (r'rgba\(0,180,60,0\.18\)', 'rgba(0,160,60,0.12)'),
    (r'rgba\(0,160,60,0\.08\)', 'rgba(0,160,60,0.08)'),
    (r'rgba\(0,200,80,0\.08\)', 'rgba(0,180,60,0.08)'),
    (r'rgba\(0,200,80,0\.07\)', 'rgba(0,180,60,0.07)'),
    (r'rgba\(0,120,255,0\.07\)', 'rgba(0,80,200,0.05)'),
    (r'rgba\(0,100,200,0\.06\)', 'rgba(0,80,180,0.05)'),

    # --- Green card backgrounds ---
    (r'background: linear-gradient\(135deg, rgba\(0,180,60,0\.12\)', 'background: linear-gradient(135deg, rgba(0,180,60,0.08)'),
    (r'background: linear-gradient\(145deg, rgba\(0,180,60,0\.14\)', 'background: linear-gradient(145deg, rgba(0,180,60,0.1)'),
    (r'background: rgba\(0,200,80,0\.1\)', 'background: rgba(0,180,60,0.08)'),
    (r'background: rgba\(0,200,80,0\.12\)', 'background: rgba(0,180,60,0.1)'),
    (r'background: rgba\(0,200,80,0\.15\)', 'background: rgba(0,180,60,0.12)'),

    # --- Frosted panel backgrounds (left/right split) ---
    (r"background: linear-gradient\(160deg, #0d1f12 0%, #0a0a0a 100%\)", 'background: linear-gradient(160deg, #e8f5ec 0%, #f8fff9 100%)'),
    (r"background: rgba\(0,200,80,0\.04\)", 'background: rgba(0,160,60,0.06)'),
    (r"border-right: 1px solid rgba\(255,255,255,0\.06\)", 'border-right: 1px solid rgba(0,0,0,0.08)'),

    # --- Bar backgrounds ---
    (r'background: rgba\(255,255,255,0\.05\)', 'background: rgba(0,0,0,0.07)'),
    (r'background: rgba\(255,255,255,0\.06\)', 'background: rgba(0,0,0,0.07)'),
    (r'background: rgba\(255,255,255,0\.07\)', 'background: rgba(0,0,0,0.08)'),

    # --- thead background ---
    (r'background: rgba\(0,200,80,0\.1\)', 'background: rgba(0,160,60,0.1)'),

    # --- tbody even row ---
    (r'background: rgba\(255,255,255,0\.015\)', 'background: rgba(0,0,0,0.02)'),

    # --- tbody border ---
    (r'border-bottom: 1px solid rgba\(255,255,255,0\.05\)', 'border-bottom: 1px solid rgba(0,0,0,0.06)'),

    # --- body bg ---
    (r"background: #000", 'background: #fff'),
]

files = sorted(glob.glob('/Users/I558118/Documents/Projects/Whisper-Ng/slides-light/slide*.html'))
for fpath in files:
    with open(fpath) as f:
        content = f.read()
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)
    with open(fpath, 'w') as f:
        f.write(content)
    print(f"Themed: {os.path.basename(fpath)}")

print("All done.")
