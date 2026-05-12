#!/usr/bin/env python3
"""Draw CaoshuReader Cursive Character Recognition Pipeline Architecture"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(28, 34))
ax.set_xlim(0, 28)
ax.set_ylim(0, 34)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E3F2FD',
    'yolo': '#FFF3E0',
    'frozen': '#E8F5E9',
    'train': '#FFEBEE',
    'embed': '#F3E5F5',
    'match': '#E0F7FA',
    'output': '#FFFDE7',
    'text': '#1A237E',
    'arrow': '#424242',
    'highlight': '#D32F2F',
}

def draw_module(ax, x, y, w, h, title, desc, color, title_color='#1A237E', title_size=26):
    """Draw a module box with title inside, description on the right"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.4",
                         facecolor=color, edgecolor='#333333', linewidth=2.5)
    ax.add_patch(box)
    
    # Title inside box (large, bold, centered)
    ax.text(x, y, title, ha='center', va='center', fontsize=title_size,
            color=title_color, weight='bold', wrap=True)
    
    # Description on the right (2 sizes smaller)
    desc_size = title_size - 2
    desc_x = x + w/2 + 0.5
    desc_lines = desc.split('\n')
    line_height = desc_size * 0.038
    start_y = y + (len(desc_lines) - 1) * line_height / 2
    for i, line in enumerate(desc_lines):
        ax.text(desc_x, start_y - i * line_height, line, 
                ha='left', va='center', fontsize=desc_size,
                color='#444444', style='italic')

def draw_arrow(ax, x1, y1, x2, y2, color='#424242', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

# ========== Title ==========
ax.text(14, 33.3, 'CaoshuReader Cursive Character Recognition Pipeline', 
        ha='center', va='center', fontsize=30, weight='bold', color='#1A237E')
ax.text(14, 32.5, 'Decoupled Architecture based on InternVL PerceiverResampler', 
        ha='center', va='center', fontsize=18, color='#555555')

# ========== Layer 1: Input ==========
draw_module(ax, 10, 31, 8.0, 1.8, 
            'Input: Calligraphy Image',
            'Whole-page scanned artwork\n(e.g. 2.jpg, 1344 x H pixels)',
            colors['input'], title_size=26)

# ========== Layer 2: YOLO Detection ==========
draw_arrow(ax, 10, 30.1, 10, 29.2)
draw_module(ax, 10, 28.3, 8.0, 1.8,
            'YOLO v10 Detection',
            'Detect individual characters\nBounding boxes (x, y, w, h)\nWeight: best.pt (62MB) | FROZEN',
            colors['yolo'], title_size=26)

# ========== Layer 3: Reading Order Sorting ==========
draw_arrow(ax, 10, 27.4, 10, 26.5)
draw_module(ax, 10, 25.6, 8.0, 1.8,
            'Reading Order Sorting',
            'Arrange boxes top-to-bottom\nleft-to-right\nModel: OrderFormer (26MB) | FROZEN',
            colors['frozen'], title_size=26)

# ========== Layer 4: Crop Single Characters ==========
draw_arrow(ax, 10, 24.7, 10, 23.8)
draw_module(ax, 10, 22.9, 8.0, 1.8,
            'Crop Character Patches',
            'Extract each character\nResize → 448 x 448 (no intermediate crop)',
            colors['input'], title_size=26)

# ========== Loop indicator ==========
draw_arrow(ax, 10, 22.0, 10, 21.1)
ax.text(10, 20.3, 'For Each Character Patch', ha='center', va='center', 
        fontsize=17, color=colors['highlight'], weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor=colors['highlight'], linewidth=2.5))

# ========== Layer 5: ViT Feature Extraction ==========
draw_arrow(ax, 10, 19.5, 10, 18.6)
draw_module(ax, 10, 17.7, 8.0, 1.8,
            'InternViT Encoder',
            'Extract visual features\n448 x 448 input\nWeight: vit_model.pt (580M) | FROZEN',
            colors['frozen'], title_size=26)

# ========== Layer 6: MLP1 Projection ==========
draw_arrow(ax, 10, 16.8, 10, 15.9)
draw_module(ax, 10, 15.0, 8.0, 1.8,
            'MLP1 Projection',
            'Project to LLM space\n4096 dimensions\nWeight: mlp1.pth (65M) | FROZEN',
            colors['frozen'], title_size=26)

# ========== Layer 7: PerceiverResampler (Core) ==========
draw_arrow(ax, 10, 14.1, 10, 13.2)
draw_module(ax, 10, 11.8, 9.0, 2.2,
            'Perceiver Resampler',
            'ONLY TRAINABLE MODULE\nCompress visual features to 3 fixed queries\n4 layers x 3 queries x 4096D\nWeight: caoshu_best.pt (3.2GB)',
            colors['train'], title_color=colors['highlight'], title_size=28)

# ========== Side branch: Precomputed Character Embeddings ==========
# Place it further right to avoid overlap
embed_box = FancyBboxPatch((20.5, 11.5), 4.0, 1.8,
                           boxstyle="round,pad=0.02,rounding_size=0.3",
                           facecolor=colors['embed'], edgecolor='#333333', linewidth=2)
ax.add_patch(embed_box)
ax.text(22.5, 12.6, 'Char Token Embedding', ha='center', va='center', fontsize=16,
        color='#1A237E', weight='bold')
ax.text(22.5, 11.9, 'Precompute 8,398 chars', ha='center', va='center', fontsize=14, color='#444444')
ax.text(22.5, 11.3, 'token_embedding.pth (724M)', ha='center', va='center', fontsize=13, color='#666666')
ax.text(22.5, 10.7, 'FROZEN', ha='center', va='center', fontsize=13, color='#2E7D32', weight='bold')
draw_arrow(ax, 22.5, 11.5, 19.0, 11.8, color='#666666')

# ========== Layer 8: Cosine Similarity Matching ==========
draw_arrow(ax, 10, 10.7, 10, 9.8)
draw_module(ax, 10, 9.0, 8.5, 1.8,
            'Cosine Matching',
            'Cosine similarity between\nprediction and all char embeddings\nLoss: 1 - cosine_sim',
            colors['match'], title_size=26)

# Arrow from embedding to matching
draw_arrow(ax, 22.5, 10.7, 18.5, 9.5, color='#666666')

# ========== Layer 9: Top-K Output ==========
draw_arrow(ax, 10, 8.1, 10, 7.2)
draw_module(ax, 10, 6.3, 8.0, 1.8,
            'Top-K Output',
            'Return Top-1 / Top-3 / Top-5\ncandidate characters with confidence',
            colors['output'], title_size=26)

# ========== Left annotations: Freeze/Train status ==========
ax.text(3, 17.7, '[FROZEN]', ha='center', va='center', fontsize=17, 
        color='#2E7D32', weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E7D32', linewidth=2))
ax.text(3, 15.0, '[FROZEN]', ha='center', va='center', fontsize=17, 
        color='#2E7D32', weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E7D32', linewidth=2))
ax.text(3, 11.8, '[TRAIN]', ha='center', va='center', fontsize=17, 
        color=colors['highlight'], weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['highlight'], linewidth=2))
ax.text(3, 9.0, '[MATCH]', ha='center', va='center', fontsize=17, 
        color='#006064', weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#006064', linewidth=2))

# ========== Right info box: Key Metrics ==========
# Move it lower to avoid overlap
info_box = FancyBboxPatch((20.5, 3.0), 4.2, 5.5,
                          boxstyle="round,pad=0.02,rounding_size=0.4",
                          facecolor='#FAFAFA', edgecolor='#999999', linewidth=2.5)
ax.add_patch(info_box)
ax.text(22.6, 8.2, 'Key Metrics', ha='center', va='center', fontsize=18, weight='bold', color='#333')
ax.text(22.6, 7.3, 'Top-1: 49.35%', ha='center', va='center', fontsize=17, color=colors['highlight'], weight='bold')
ax.text(22.6, 6.4, 'Top-5: 71.52%', ha='center', va='center', fontsize=17, color='#2E7D32', weight='bold')
ax.text(22.6, 5.5, 'Best Loss: 0.1751', ha='center', va='center', fontsize=15, color='#555')
ax.text(22.6, 4.6, 'Val Set: 61,181', ha='center', va='center', fontsize=15, color='#555')
ax.text(22.6, 3.7, 'Train: 974,113', ha='center', va='center', fontsize=15, color='#555')
ax.text(22.6, 2.8, 'Chars: 8,398', ha='center', va='center', fontsize=15, color='#555')

# ========== Bottom ==========
ax.text(14, 0.4, 'CaoshuReader v2 | backup/pipeline-v1 | 2026-05-13', 
        ha='center', va='center', fontsize=13, color='#888')

plt.tight_layout()
plt.savefig('/caoshu/caoshu.jpg', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Saved to /caoshu/caoshu.jpg")
