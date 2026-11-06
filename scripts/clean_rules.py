"""Historical cleaning rules retained to reproduce the training corpus."""
import re

def clean_poem(text):
    """
    清洗单首诗歌，去除所有非诗歌内容
    
    清洗规则：
    1. 去除圆括号内的注释 （...）
    2. 去除书名号引用 《...》
    3. 去除方括号内容 [...] 和 〖...〗
    4. 去除"卷X"、"按："等标注关键词所在的句段
    5. 去除多余的标点符号连续
    6. 去除过短的结果（< 10字，可能是残句）
    """
    if not text or not text.strip():
        return ""
    
    original = text
    
    # ── 第一步：去除所有括号内的注释 ──
    # 处理嵌套括号：（...（...）...）
    while True:
        # 中文圆括号 （...）
        cleaned = re.sub(r'（[^（）]*）', '', text)
        # 英文圆括号 (...)  
        cleaned = re.sub(r'\([^()]*\)', '', cleaned)
        # 方括号：单字的 [X] 保留X（这是编者补字），多字的去除
        cleaned = re.sub(r'\[(\w)\]', r'\1', cleaned)   # [煙] → 煙
        cleaned = re.sub(r'\[[^\[\]]{2,}\]', '', cleaned)  # [多字注释] → 去除
        # 尖括号 〖...〗
        cleaned = re.sub(r'〖[^〖〗]*〗', '', cleaned)
        if cleaned == text:
            break
        text = cleaned
    
    # ── 第二步：去除书名号及其内容 ──
    # 《全唐诗》 → 去掉
    text = re.sub(r'《[^《》]*》', '', text)
    
    # ── 第三步：去除残留的标注关键词 ──
    # 常见的标注用语
    noise_patterns = [
        r'按：.*',              # "按：..."
        r'見[同前].*',          # "見同前..." 
        r'以上.*均見.*',        # "以上X首均見..."
        r'一作「[^」]*」',      # "一作「某」"
        r'又作「[^」]*」',      # "又作「某」"
        r'卷[一二三四五六七八九十百千\d]+',  # "卷三十八"
        r'收此詩.*',            # "收此诗..."
        r'今重錄.*',            # "今重錄..."
        r'原署.*',              # "原署..."
    ]
    for pat in noise_patterns:
        text = re.sub(pat, '', text)
    
    # ── 第四步：清理标点 ──
    # 去除连续的标点符号（如 。。。 → 。）
    text = re.sub(r'[，。！？、；：]{2,}', '。', text)
    # 去除开头的标点
    text = re.sub(r'^[，。！？、；：\s]+', '', text)
    # 去除结尾多余标点
    text = re.sub(r'[，、；：\s]+$', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', '', text)
    
    # ── 第五步：质量过滤 ──
    # 去除标点后的纯文字长度
    pure_text = re.sub(r'[，。！？、；：「」『』]', '', text)
    
    # 过短的文本可能是残句或标注碎片
    if len(pure_text) < 10:
        return ""
    
    # 如果剩余文本中超过30%是数字/拉丁字母，可能是标注残留
    non_cjk = len(re.findall(r'[0-9a-zA-Z]', text))
    if non_cjk > len(text) * 0.2:
        return ""
    
    return text


