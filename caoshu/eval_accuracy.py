"""
用 best-train checkpoint 对所有 example 图片做全字识别，与原文对比，计算正确率。
"""
import os, sys, glob, torch, torch.nn.functional as F, cv2
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = '/workspace/CalliReader'
sys.path.insert(0, PROJECT_ROOT)

from models.model import (load_vision_model, load_mlp1, load_perceiver_resampler,
                           load_normed_tok_embeddings, load_tokenizer)
from config.configu import DOWNSAMPLE_RATIO
from caoshu.dataset import CaoshuDataset
from caoshu.train import get_visual_embed
from ultralytics import YOLO

# ── 原文 ground truth（去掉标点、空格、题款，只保留正文汉字）─────────────────
GROUND_TRUTH = {
    '2':    '雲龍遠飛駕天馬自行空',
    '2_1':  '雲龍遠飛駕天馬自行空',
    '2_2':  '雲龍遠飛駕天馬自行空',
    '2_3':  '雲龍遠飛駕天馬自行空',
    '2_4':  '雲龍遠飛駕天馬自行空',
    '6':    '国破山河在城春草木深感时花溅泪恨别鸟惊心烽火连三月家书抵万金白头搔更短浑欲不胜簪',
    '7':    '素心愛雲水此日東南行笑解塵纓處滄浪無限清',
    '8':    '千山草木如雲暗陸地波瀾接海平蜀搖書',
    '9':    '吾嘗好奇古來草聖無不知豈不知右軍與獻之雖有壯麗之骨恨無狂逸之姿',
    '10':   '清風明月誰與共高山流水少之音薄冰',
    '11':   '海內存知己天涯若比鄰',
    '12':   '春風杏花雨柳色今又深',
    '13':   '三合也紙墨相發四合也偶然欲書五合也心遷體留一乖也意違勢屈二乖也風燥日炎三乖也紙墨不',
    '14':   '天涯落乎猶眾星之列河漢同自然之妙有非力運之能成信可謂智巧兼優心手雙暢翰不虛動下必有由',
    '20':   '雁门关外不嫌远五十三驿是皇州浮云一百八盘萦为问西风几时来不教黄叶自惊回五十三驿天山雪八月一旬心登严',
    '21':   '山谷谪黔南亦有竹枝二篇云撑崖拄谷蝮蛇愁入菁扳天猿掉头',
    '22_1': '',   # 无原文，跳过
    '0':    '',   # 无原文，跳过
}

# ── 配置 ───────────────────────────────────────────────────────────────
DATA_ROOT    = '/root/sj-tmp/datasets/CursiveChineseCalligraphyDataset/Cursive_Chinese_Calligraphy_Dataset'
CKPT         = '/root/sj-tmp/checkpoints/CaoshuReader/caoshu_best.pt'
YOLO_PATH    = '/workspace/CalliReader/params/best.pt'
EXAMPLES_DIR = '/workspace/CalliReader/examples'
CONF         = 0.25

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── 加载模型 ────────────────────────────────────────────────────────────
print("加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit  = load_vision_model(location='cuda').eval()
mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO).eval()
for p in list(vit.parameters()) + list(mlp1.parameters()):
    p.requires_grad = False

tok_embeddings, _ = load_normed_tok_embeddings(load_checkboard=True, location='cpu')
tok_embeddings = tok_embeddings.to(device).to(torch.bfloat16).eval()
tokenizer = load_tokenizer()

dataset = CaoshuDataset(DATA_ROOT, 'Validation', transform=None)
idx2char = dataset.idx2char
num_classes = len(idx2char)
char2idx = {v: k for k, v in idx2char.items()}

print(f"字符类别数: {num_classes}")
print("预计算字符 embeddings...")
all_chars = [idx2char[i] for i in range(num_classes)]
all_token_ids = tokenizer(all_chars, return_tensors='pt', add_special_tokens=False,
                          padding=True, truncation=True, max_length=4).input_ids[:, 0].to(device)
with torch.no_grad():
    all_embeds_norm = F.normalize(tok_embeddings(all_token_ids), dim=-1)

resampler = load_perceiver_resampler(path=None, num_layers=4)
resampler = resampler.to(device).to(torch.bfloat16).eval()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
resampler.load_state_dict(sd, strict=True)
print(f"已加载 best-train checkpoint: step={ckpt.get('step')}, train_loss={ckpt.get('loss'):.4f}")

yolo = YOLO(YOLO_PATH)
print("模型加载完成\n")

# ── 推理 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def recognize_top1(crop_pil):
    img_t = transform(crop_pil.convert('RGB')).unsqueeze(0).to(device).to(torch.bfloat16)
    feats = get_visual_embed(img_t, vit, mlp1)
    pred  = resampler(feats)
    scores = torch.matmul(F.normalize(pred, dim=-1)[0], all_embeds_norm.t()).sum(dim=0)
    return idx2char[scores.argmax().item()]

def yolo_crops_ordered(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return [], img_bgr
    results = yolo(img_bgr, conf=CONF, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        boxes.append((cy, cx, x1, y1, x2, y2))
    boxes.sort()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    crops = []
    for cy, cx, x1, y1, x2, y2 in boxes:
        pad = 4
        x1c, y1c = max(0,x1-pad), max(0,y1-pad)
        x2c, y2c = min(w,x2+pad), min(h,y2+pad)
        crop = img_rgb[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            crops.append(Image.fromarray(crop))
    return crops, len(boxes)

# ── NED（归一化编辑距离）────────────────────────────────────────────────
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n+1):
            dp[j] = prev[j-1] if s1[i-1]==s2[j-1] else 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]

def ned(pred, gt):
    if not gt:
        return 0.0
    ed = edit_distance(pred, gt)
    return 1.0 - ed / max(len(pred), len(gt))

# ── 主循环 ──────────────────────────────────────────────────────────────
jpg_files = sorted(glob.glob(os.path.join(EXAMPLES_DIR, '*.jpg')))

total_gt_chars  = 0
total_correct   = 0
total_predicted = 0

print("="*70)
print(f"{'图片':<8} {'检测字数':>6} {'原文字数':>6} {'正确数':>6} {'字符准确率':>10} {'NED':>8}")
print("="*70)

detail_rows = []

for jpg_path in jpg_files:
    stem = os.path.basename(jpg_path).replace('.jpg','').replace('.JPG','')
    gt_text = GROUND_TRUTH.get(stem, None)

    crops, n_det = yolo_crops_ordered(jpg_path)

    if not crops:
        print(f"{stem:<8} {'YOLO无检测':>30}")
        continue

    # 识别全部字
    pred_chars = [recognize_top1(c) for c in crops]
    pred_text  = ''.join(pred_chars)

    if not gt_text:
        print(f"{stem:<8} {len(crops):>6} {'—':>6} {'—':>6} {'（无原文）':>10} {'—':>8}")
        detail_rows.append((stem, pred_text, gt_text or '', pred_chars, [], 0, 0))
        continue

    gt_chars = list(gt_text)
    n_gt   = len(gt_chars)
    n_pred = len(pred_chars)

    # 字符级准确率：按位置对齐（min长度）
    n_cmp   = min(n_pred, n_gt)
    correct = sum(p == g for p, g in zip(pred_chars[:n_cmp], gt_chars[:n_cmp]))

    char_acc = correct / n_gt * 100  # 以GT为分母
    ned_score = ned(pred_text, gt_text) * 100

    total_gt_chars  += n_gt
    total_correct   += correct
    total_predicted += n_pred

    print(f"{stem:<8} {n_pred:>6} {n_gt:>6} {correct:>6} {char_acc:>9.1f}% {ned_score:>7.1f}%")
    detail_rows.append((stem, pred_text, gt_text, pred_chars, gt_chars, correct, n_gt))

# ── 总体统计 ─────────────────────────────────────────────────────────────
print("="*70)
overall_acc = total_correct / total_gt_chars * 100 if total_gt_chars else 0
print(f"{'总计':<8} {total_predicted:>6} {total_gt_chars:>6} {total_correct:>6} {overall_acc:>9.1f}%")
print()

# ── 逐图详细对比 ──────────────────────────────────────────────────────────
for stem, pred_text, gt_text, pred_chars, gt_chars, correct, n_gt in detail_rows:
    if not gt_text:
        print(f"\n[{stem}] 预测（无原文对比）: {pred_text}")
        continue

    print(f"\n[{stem}]")
    print(f"  原文({n_gt}字): {gt_text}")
    print(f"  预测({len(pred_chars)}字): {pred_text}")

    # 逐字对比标注
    n_cmp = min(len(pred_chars), len(gt_chars))
    diff_line = []
    for i, (p, g) in enumerate(zip(pred_chars[:n_cmp], gt_chars[:n_cmp])):
        if p == g:
            diff_line.append(f"✓{p}")
        else:
            diff_line.append(f"✗{p}[{g}]")
    # 超出部分
    for p in pred_chars[n_cmp:]:
        diff_line.append(f"+{p}")
    for g in gt_chars[n_cmp:]:
        diff_line.append(f"缺[{g}]")
    print(f"  对比: {'  '.join(diff_line)}")
    print(f"  字符准确率: {correct}/{n_gt} = {correct/n_gt*100:.1f}%  NED: {ned(pred_text,gt_text)*100:.1f}%")
