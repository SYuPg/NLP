import os
import random
import re
import tarfile
import jieba
import sentencepiece as spm
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib

# ---------- 1. 配置文件名和路径 ----------
archive_name = "opus-100-corpus-en-zh-v1.0.tar.gz"
extracted_dir = "opus-100-corpus-en-zh-v1.0"

if not os.path.exists(extracted_dir):
    if not os.path.exists(archive_name):
        print(f"错误：找不到 {archive_name}，请确认文件已放在当前目录。")
        exit(1)
    print("正在解压本地语料文件...")
    with tarfile.open(archive_name, "r:gz") as tar:
        tar.extractall()
    print("解压完成。")

en_file = os.path.join(extracted_dir, "opus-100-corpus", "v1.0", "supervised", "en-zh", "opus.en-zh-train.en")
zh_file = os.path.join(extracted_dir, "opus-100-corpus", "v1.0", "supervised", "en-zh", "opus.en-zh-train.zh")

with open(en_file, "r", encoding="utf-8") as f:
    en_sents = [line.strip() for line in f if line.strip()]
with open(zh_file, "r", encoding="utf-8") as f:
    zh_sents = [line.strip() for line in f if line.strip()]

min_len = min(len(en_sents), len(zh_sents))
en_sents = en_sents[:min_len]
zh_sents = zh_sents[:min_len]

NUM_SENTS = 50000
en_sents = en_sents[:NUM_SENTS]
zh_sents = zh_sents[:NUM_SENTS]

mixed_corpus = en_sents + zh_sents
random.seed(42)
random.shuffle(mixed_corpus)
print(f"混合语料总句数：{len(mixed_corpus)} （中文约 {len(zh_sents)}，英文约 {len(en_sents)}）")

# 数据清洗
def clean_english(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9.,!?;:\-\'\"() ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_chinese(text):
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]', '', text)
    return text

cleaned_corpus = []
for sent in mixed_corpus:
    if any('\u4e00' <= c <= '\u9fff' for c in sent):
        cleaned_corpus.append(clean_chinese(sent))
    else:
        cleaned_corpus.append(clean_english(sent))

cleaned_corpus = [s for s in cleaned_corpus if len(s) >= 2]
print(f"清洗后句子数：{len(cleaned_corpus)}")

# ---------- 2. 三种分词方案 ----------
def tokenize_scheme_A(sentence):
    tokens = []
    i = 0
    while i < len(sentence):
        c = sentence[i]
        if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf':
            tokens.append(c)
            i += 1
        elif c.isalpha():
            start = i
            while i < len(sentence) and sentence[i].isalpha():
                i += 1
            tokens.append(sentence[start:i].lower())
        else:
            i += 1
    return tokens

def tokenize_scheme_B(sentence):
    tokens = []
    parts = re.split(r'([a-zA-Z]+)', sentence)
    for part in parts:
        if not part:
            continue
        if re.match(r'^[a-zA-Z]+$', part):
            tokens.append(part.lower())
        else:
            tokens.extend(list(jieba.cut(part)))
    return tokens

bpe_input_file = "mixed_for_bpe.txt"
with open(bpe_input_file, "w", encoding="utf-8") as f:
    for sent in cleaned_corpus:
        f.write(sent + "\n")

bpe_model_prefix = "mixed_bpe"
spm.SentencePieceTrainer.train(
    input=bpe_input_file,
    model_prefix=bpe_model_prefix,
    vocab_size=8000,
    character_coverage=0.9999,
    model_type='bpe',
    input_sentence_size=500000,
    shuffle_input_sentence=True,
)

sp = spm.SentencePieceProcessor(model_file=f"{bpe_model_prefix}.model")

def tokenize_scheme_C(sentence):
    return sp.encode_as_pieces(sentence)

corpus_A, corpus_B, corpus_C = [], [], []
total = len(cleaned_corpus)
for i, sent in enumerate(cleaned_corpus):
    if i % 5000 == 0:
        print(f"分词进度：{i}/{total}")
    corpus_A.append(tokenize_scheme_A(sent))
    corpus_B.append(tokenize_scheme_B(sent))
    corpus_C.append(tokenize_scheme_C(sent))

# 保存分词结果（可选）
for fname, corp in zip(["corpus_A.txt", "corpus_B.txt", "corpus_C.txt"],
                       [corpus_A, corpus_B, corpus_C]):
    with open(fname, "w", encoding="utf-8") as f:
        for tokens in corp:
            f.write(" ".join(tokens) + "\n")
print("三种分词方案处理完成。")

# ---------- 3. 训练 Word2Vec ----------
vector_size = 300
window = 5
min_count = 3
sg = 1
negative = 5
epochs = 10
workers = 4

print("训练模型 A ...")
model_A = Word2Vec(sentences=corpus_A, vector_size=vector_size, window=window,
                   min_count=min_count, sg=sg, negative=negative, epochs=epochs,
                   workers=workers, seed=42)
model_A.save("model_A.bin")

print("训练模型 B ...")
model_B = Word2Vec(sentences=corpus_B, vector_size=vector_size, window=window,
                   min_count=min_count, sg=sg, negative=negative, epochs=epochs,
                   workers=workers, seed=42)
model_B.save("model_B.bin")

print("训练模型 C ...")
model_C = Word2Vec(sentences=corpus_C, vector_size=vector_size, window=window,
                   min_count=min_count, sg=sg, negative=negative, epochs=epochs,
                   workers=workers, seed=42)
model_C.save("model_C.bin")

print("所有模型训练完毕！")

# 通用工具：获取词向量（支持子词回退）
def get_vector(model, word, sp_model=None):
    if word in model.wv:
        return model.wv[word]
    if sp_model is not None:
        subwords = sp_model.encode_as_pieces(word)
    else:
        subwords = list(word)
    vecs = [model.wv[sw] for sw in subwords if sw in model.wv]
    if vecs:
        return sum(vecs) / len(vecs)
    return None

# ---------- 4. 原有简单评估（最近邻、t-SNE、Spearman） ----------
test_words = {
    "高频": ["中国", "china", "经济", "development", "美国"],
    "低频": ["琥珀", "serendipity", "异或", "astronaut", "涟漪"],
    "错拼/OOV": ["computre", "苹狗", "infomration"]
}

for category, words in test_words.items():
    print(f"\n===== {category}词 =====")
    for w in words:
        for model_name, model, sp_model in zip(
            ["A (字+词)", "B (词+词)", "C (共享BPE)"],
            [model_A, model_B, model_C],
            [None, None, sp]
        ):
            vec = get_vector(model, w, sp_model)
            if vec is None:
                print(f"[{model_name}] '{w}' 无法获得向量（OOV且无回退）")
            else:
                sims = model.wv.most_similar([vec], topn=5)
                neighbors = [f"{nei}({s:.3f})" for nei, s in sims]
                print(f"[{model_name}] '{w}' 近邻: {', '.join(neighbors)}")

# t-SNE 可视化（修复缩进）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

viz_words = [
    "中国", "china", "美国", "america", "经济", "economy",
    "发展", "development", "和平", "peace", "法律", "law",
    "computre", "infomration", "苹狗",
    "银行", "bank", "市场", "market"
]

def prepare_viz_data(model, sp_model=None):
    vectors, labels = [], []
    for w in viz_words:
        vec = get_vector(model, w, sp_model)
        if vec is not None:
            vectors.append(vec)
            labels.append(w)
    return np.array(vectors), labels

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Scheme A: char + word", "Scheme B: word + word", "Scheme C: shared BPE"]
for ax, model, sp_model, title in zip(
    axes, [model_A, model_B, model_C], [None, None, sp], titles
):
    vecs, labs = prepare_viz_data(model, sp_model)
    if len(vecs) < 2:
        ax.set_title(f"{title}\n(数据不足)")
        continue
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vecs)-1))
    vec_2d = tsne.fit_transform(vecs)
    ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c='steelblue')
    for i, lab in enumerate(labs):
        ax.annotate(lab, (vec_2d[i, 0], vec_2d[i, 1]), fontsize=9)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("tsne_comparison.png", dpi=150)
plt.show()
print("t-SNE 图已保存为 tsne_comparison.png")

# Spearman 评估
zh_test_pairs = [
    ("中国", "美国", 7.5), ("经济", "金融", 8.2), ("法律", "律师", 7.8),
    ("发展", "增长", 8.0), ("和平", "战争", 2.0), ("计算机", "电脑", 9.0),
    ("医生", "医院", 8.5), ("学校", "学生", 8.3), ("猫", "狗", 7.0),
    ("汽车", "火车", 7.2), ("总统", "领导", 7.6), ("环境", "自然", 8.4),
    ("音乐", "歌曲", 8.8), ("电影", "戏剧", 7.4), ("语言", "文字", 8.1),
]
en_test_pairs = [
    ("china", "america", 7.5), ("economy", "finance", 8.0), ("law", "lawyer", 7.8),
    ("growth", "development", 8.5), ("peace", "war", 1.5), ("computer", "laptop", 8.8),
    ("doctor", "hospital", 8.2), ("school", "student", 8.3), ("cat", "dog", 7.2),
    ("car", "train", 7.0), ("president", "leader", 7.7), ("environment", "nature", 8.4),
    ("music", "song", 8.9), ("movie", "film", 9.0), ("language", "text", 7.8),
]

def evaluate_spearman(model, test_pairs, sp_model=None):
    human, cos = [], []
    for w1, w2, score in test_pairs:
        v1 = get_vector(model, w1, sp_model)
        v2 = get_vector(model, w2, sp_model)
        if v1 is None or v2 is None:
            continue
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        human.append(score)
        cos.append(cos_sim)
    if len(human) < 3:
        return None
    rho, p = spearmanr(human, cos)
    return rho

print("\n===== 词相似度 Spearman 评估 =====")
for model_name, model, sp_model in zip(
    ["A (字+词)", "B (词+词)", "C (共享BPE)"],
    [model_A, model_B, model_C],
    [None, None, sp]
):
    rho_zh = evaluate_spearman(model, zh_test_pairs, sp_model)
    rho_en = evaluate_spearman(model, en_test_pairs, sp_model)
    if rho_zh is not None and rho_en is not None:
        avg_rho = (rho_zh + rho_en) / 2
        print(f"{model_name}: 中文 Spearman = {rho_zh:.3f},  英文 Spearman = {rho_en:.3f},  平均 = {avg_rho:.3f}")
    else:
        print(f"{model_name}: 无法计算（有效词对不足）")

# ============================================================
# 扩展分析 1：语义边界保持性
# ============================================================
print("\n" + "="*60)
print("扩展分析 1：语义边界保持性")
print("="*60)

# 选择一组中英文复合词/固定搭配，这些词在BPE或jieba中可能被切分
target_words = {
    "中文": [
        "火车", "火山", "电话", "电脑", "手机",
        "图书馆", "自行车", "人工智能", "机器学习"
    ],
    "英文": [
        "firefighter", "sunflower", "notebook", "airport",
        "basketball", "football", "homework", "smartphone",
        "machinelearning", "deeplearning"
    ]
}

# 为每个目标词准备语义相关词和无关词（用于计算边界清晰度）
# 为了通用性，我们使用模型自身的 top‑N 邻居作为相关词，随机词作为无关词
def semantic_boundary_score(model, word, sp_model, topn=5, random_samples=50):
    """返回语义边界保持度：相关词平均相似度 - 无关词平均相似度"""
    vec = get_vector(model, word, sp_model)
    if vec is None:
        return None
    # 相关词：模型给出的最相似词（排除自身）
    try:
        similar = model.wv.most_similar([vec], topn=topn+1)  # 第一个可能是自身
        related_vecs = [model.wv[w] for w, _ in similar if w != word][:topn]
    except KeyError:
        return None
    if not related_vecs:
        return None
    # 随机挑选无关词（来自词表，且不是相关词）
    related_words = set(w for w, _ in similar)
    all_words = list(model.wv.index_to_key)
    random.shuffle(all_words)
    unrelated_words = [w for w in all_words if w not in related_words and w != word][:random_samples]
    unrelated_vecs = [model.wv[w] for w in unrelated_words]
    # 计算平均余弦相似度
    def mean_cos_sim(target, vecs):
        return np.mean([np.dot(target, v)/(np.linalg.norm(target)*np.linalg.norm(v)) for v in vecs])
    rel_sim = mean_cos_sim(vec, related_vecs)
    unr_sim = mean_cos_sim(vec, unrelated_vecs)
    return rel_sim - unr_sim   # 差值越大，边界越清晰

# 计算三种方案在每个目标词上的边界分数
results_boundary = {"Scheme A": [], "Scheme B": [], "Scheme C": []}
models = [model_A, model_B, model_C]
sp_models = [None, None, sp]
scheme_names = ["Scheme A", "Scheme B", "Scheme C"]

for lang, words in target_words.items():
    print(f"\n{lang}语义边界分数 (相关相似 - 无关相似):")
    print(f"{'Word':<20} {'A (字+词)':<15} {'B (词+词)':<15} {'C (BPE)':<15}")
    for word in words:
        scores = []
        for model, sp_m in zip(models, sp_models):
            score = semantic_boundary_score(model, word, sp_m)
            scores.append(score)
        if any(s is None for s in scores):
            print(f"{word:<20} 至少一个模型无法计算")
            continue
        for i, scheme in enumerate(scheme_names):
            results_boundary[scheme].append(scores[i])
        print(f"{word:<20} {scores[0]:<15.4f} {scores[1]:<15.4f} {scores[2]:<15.4f}")

# 汇总平均
print("\n平均语义边界分数:")
for scheme in scheme_names:
    avg = np.mean(results_boundary[scheme]) if results_boundary[scheme] else float('nan')
    print(f"{scheme}: {avg:.4f}")

# ============================================================
# 扩展分析 2：形态构词一致性
# ============================================================
print("\n" + "="*60)
print("扩展分析 2：形态构词一致性")
print("="*60)

# 构建中英文词族（同词根/同语素）
zh_families = [
    ["游泳", "泳池", "泳装", "蛙泳", "蝶泳"],
    ["学生", "学校", "学习", "学问", "学院"],
    ["火车", "汽车", "自行车", "马车", "电车"],
    ["电脑", "电话", "电视", "电影", "电信"],
    ["医生", "医院", "医学", "医药", "医治"]
]

en_families = [
    ["run", "runs", "running", "runner", "ran"],        # ran 可能在词表中不存在
    ["happy", "unhappy", "happiness", "happily"],
    ["play", "player", "playful", "playing", "played"],
    ["compute", "computer", "computing", "computation"],
    ["act", "action", "active", "actor", "activity"]
]

def family_consistency(model, family, sp_model):
    """计算词族内所有词对的平均余弦相似度"""
    vectors = []
    for w in family:
        v = get_vector(model, w, sp_model)
        if v is not None:
            vectors.append(v)
    if len(vectors) < 2:
        return None
    sims = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            cos = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            sims.append(cos)
    return np.mean(sims)

# 评估三种方案
print("\n中文词族内部平均余弦相似度:")
print(f"{'词族':<30} {'A (字+词)':<15} {'B (词+词)':<15} {'C (BPE)':<15}")
zh_results = {"Scheme A": [], "Scheme B": [], "Scheme C": []}
for family in zh_families:
    scores = []
    for model, sp_m in zip(models, sp_models):
        scores.append(family_consistency(model, family, sp_m))
    if any(s is None for s in scores):
        print(f"{str(family):<30} 至少一个模型无法计算")
        continue
    for i, scheme in enumerate(scheme_names):
        zh_results[scheme].append(scores[i])
    print(f"{str(family):<30} {scores[0]:<15.4f} {scores[1]:<15.4f} {scores[2]:<15.4f}")

print("\n英文词族内部平均余弦相似度:")
print(f"{'词族':<30} {'A (字+词)':<15} {'B (词+词)':<15} {'C (BPE)':<15}")
en_results = {"Scheme A": [], "Scheme B": [], "Scheme C": []}
for family in en_families:
    scores = []
    for model, sp_m in zip(models, sp_models):
        scores.append(family_consistency(model, family, sp_m))
    if any(s is None for s in scores):
        print(f"{str(family):<30} 至少一个模型无法计算")
        continue
    for i, scheme in enumerate(scheme_names):
        en_results[scheme].append(scores[i])
    print(f"{str(family):<30} {scores[0]:<15.4f} {scores[1]:<15.4f} {scores[2]:<15.4f}")

# 汇总平均值
print("\n平均词族一致性（中英文合并）:")
for scheme in scheme_names:
    combined = zh_results[scheme] + en_results[scheme]
    avg = np.mean(combined) if combined else float('nan')
    print(f"{scheme}: {avg:.4f}")