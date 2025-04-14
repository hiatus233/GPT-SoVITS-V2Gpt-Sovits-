import gradio as gr
import re
import requests
from pydub import AudioSegment
import os
import uuid
from datetime import datetime
import json

# API 基础 URL
API_BASE_URL = 'https://api.deepseek.com'  # 请确认此URL是否正确

# 从环境变量中读取 API Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")

# 定义说话人与模型权重的映射（可根据需要扩展）
speaker_models = {
    "卡芙卡": {
        "gpt_weights": r"E:\GPT-SoVITS-v2\GPT_weights_v2\卡芙卡-e10.ckpt",
        "sovits_weights": r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\卡芙卡_e10_s570.pth"
    },
    "屠夫": {
        "gpt_weights": r"E:\GPT-SoVITS-v2\GPT_weights_v2\屠夫-e15.ckpt",
        "sovits_weights": r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\屠夫_e8_s208.pth"
    },
    "丹恒": {
        "gpt_weights": r"E:\GPT-SoVITS-v2\GPT_weights_v2\丹恒-e10.ckpt",
        "sovits_weights": r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\丹恒_e10_s1210.pth"
    },
    "克拉拉": {
        "gpt_weights": r"E:\GPT-SoVITS-v2\GPT_weights_v2\克拉拉-e10.ckpt",
        "sovits_weights": r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\克拉拉_e10_s630.pth"
    },
    "景元": {
        "gpt_weights": r"E:\GPT-SoVITS-v2\GPT_weights_v2\景元-e10.ckpt",
        "sovits_weights": r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\景元_e10_s900.pth"
    }
    # 如果未来有更多说话人有模型，继续添加
}

# 定义情感列表
EMOTIONS = ["生气", "厌恶", "恐惧", "开心", "中立_neutral", "其他", "难过", "吃惊"]

def load_reference_audios(directory=r'E:\GPT-SoVITS-v2\参考音频'):
    """
    加载所有参考音频文件，按说话人和情感分类。
    目录结构示例：
    参考音频/
    ├── 丹恒/
    │   ├── 中立_neutral/
    │   │   ├── audio1.wav
    │   │   └── audio2.wav
    │   └── 开心_happy/
    │       └── ...
    └── 其他说话人/
    """
    reference_audios = {}  # {speaker: {full_emotion: [list of audios]}}
    if not os.path.exists(directory):
        print(f"参考音频目录不存在: {directory}")
        return reference_audios

    for speaker in os.listdir(directory):
        speaker_path = os.path.join(directory, speaker)
        if os.path.isdir(speaker_path):
            reference_audios[speaker] = {}
            for emotion_folder in os.listdir(speaker_path):
                emotion_path = os.path.join(speaker_path, emotion_folder)
                if os.path.isdir(emotion_path):
                    # 目录名格式为 "中立_neutral"
                    match = re.match(r'([^_]+)_(.+)', emotion_folder)
                    if match:
                        full_emotion = emotion_folder  # '中立_neutral'
                        # 列出该情感文件夹下的所有音频文件
                        audio_files = [
                            f for f in os.listdir(emotion_path)
                            if f.lower().endswith(('.wav', '.mp3'))
                        ]
                        reference_audios[speaker][full_emotion] = audio_files
    print(f"参考音频加载完成: {json.dumps(reference_audios, ensure_ascii=False)}")  # 调试输出
    return reference_audios

def set_gpt_weights(weights_path):
    """
    调用 /set_gpt_weights 端点设置 GPT 模型权重。
    """
    try:
        response = requests.get(f"{API_BASE_URL}/set_gpt_weights", params={"weights_path": weights_path})
        if response.status_code == 200:
            return "GPT 模型权重设置成功。"
        else:
            return f"设置 GPT 模型权重失败: {response.json().get('error', {}).get('message', '')}"
    except Exception as e:
        return f"请求 /set_gpt_weights 端点失败: {e}"

def set_sovits_weights(weights_path):
    """
    调用 /set_sovits_weights 端点设置 SoVITS 模型权重。
    """
    try:
        response = requests.get(f"{API_BASE_URL}/set_sovits_weights", params={"weights_path": weights_path})
        if response.status_code == 200:
            return "SoVITS 模型权重设置成功。"
        else:
            return f"设置 SoVITS 模型权重失败: {response.json().get('error', {}).get('message', '')}"
    except Exception as e:
        return f"请求 /set_sovits_weights 端点失败: {e}"

def parse_annotated_script(file_path):
    """
    解析标注好的文本文件，提取角色、情感和台词。
    假设文本格式为：
    [角色_情感]：“台词”
    或
    [角色]：“台词” (默认情感为中立_neutral)
    """
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配 [角色_情感]：“台词” 或 [角色]：“台词”
    pattern = r'\[(.*?)\]：“(.*?)”'
    matches = re.findall(pattern, content, re.DOTALL)

    for role_with_emotion, line in matches:
        # 如果角色中包含情感，用第一个 '_' 分割
        if '_' in role_with_emotion:
            parts = role_with_emotion.split('_', 2)
            if len(parts) >= 3:
                # 支持情感中包含下划线的情况
                role = parts[0]
                full_emotion = '_'.join(parts[1:])  # 例如 '中立_neutral'
            else:
                role, full_emotion = role_with_emotion.split('_', 1)
        else:
            role = role_with_emotion
            full_emotion = '中立_neutral'  # 默认情感文件夹名称
        # 将 '中立' 映射为 '中立_neutral'
        if full_emotion == '中立':
            full_emotion = '中立_neutral'
        dialogues.append({
            'role': role.strip(),
            'emotion': full_emotion.strip(),
            'text': line.strip()
        })
    return dialogues

def parse_and_get_roles(file):
    """
    仅解析文本文件，返回角色列表和提示信息。
    """
    if file is None:
        return [], "未选择标注文本文件。"

    file_path = file.name  # 获取文件路径
    dialogues = parse_annotated_script(file_path)
    if not dialogues:
        return [], "未能解析标注文本文件，请检查文件格式。"

    # 提取角色列表
    roles = sorted(set(dialogue['role'] for dialogue in dialogues))

    return roles, None

def deepseek_chat(message, output_file_path):
    """
    调用 DeepSeek LLM 对文本进行标注，并保存结果到输出文件。
    """
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": """
请分析以下小说文本，识别并标注出其中的角色对话、内心独白、旁白以及动作描述。请按照以下标准进行标注：

### 标注规则：

1. **角色对话**：
   - 使用「」符号括起来的是角色的对话内容。
   - 在每个对话前标注角色名称和情感状态，格式为：[角色名_情感]：“对话内容”。
   - 情感分类仅限于以下八种：生气、厌恶、恐惧、开心、中立_neutral、其他、难过、吃惊。
   - 角色说话时的动作描述（如“出声招呼”）应放在旁白部分标注。

2. **内心独白**：
   - 角色的内心独白，例如“想”、“心中揣度”以及类似表达（如：“他想”，“她暗自想着”），应归类为角色的对话并标注相应的情感。
   - 格式：如 [角色名_情感]：“角色的内心独白”。

3. **旁白**：
   - 未使用「」符号括起来的叙述性文本，应标注为旁白，情感状态默认为“中立”。
   - 包含对环境、人物动作的描述。
   - 格式：[旁白_中立]：“描述性文本”。

4. **特殊句式标注**：
   - **第一种形式：提示语在前，冒号后跟对话内容（带引号）。**
     - **错误示范**：  
       陈二狗看到了他，笑着挥手招呼道，“大明星，你怎么这么闲，电视剧不用拍了吗。”  
     - **正确示范**：  
       陈二狗看到了他，笑着挥手招呼道：“大明星，你怎么这么闲？电视剧不用拍了吗？”

   - **第二种形式：提示语在中间，前后双引号之间的提示语后用逗号。**
     - **错误示范**：  
       “啊？”江寒吃惊出声，以为自己听错了。“老先生，您，您说什么？”
     - **正确示范**：  
       “啊？”江寒吃惊出声，以为自己听错了，“老先生，您，您说什么？”

   - **第三种形式：提示语在后，说话的内容用引号，提示语后面用句号。**
     - **正确示范**：  
       王母的脸上开始抽搐。“搬给他。”她说。

   - **第四种形式：没有提示语，说话的内容用引号。**
     - **正确示范**：  
       “不是。是一家杂志社的记者。”

   - **第五种形式：动作描述后跟对话内容**
     - **示例**：
       笹垣看了看写着‘烤乌贼饼四十元’的牌子，付了钱。老板娘亲切地说：“多谢。”然后拿起报纸，坐回椅子。
     - **标注**：
       [旁白_中立]：“笹垣看了看写着‘烤乌贼饼四十元’的牌子，付了钱。老板娘亲切地说，”
       [老板娘_开心]：“多谢。”
       [旁白_中立]：“然后拿起报纸，坐回椅子。”

5. **情感标注举例**：
   - **角色直接对话**：
     - [笹垣_开心]：“老板娘，给我烤一片。”
   - **旁白描述动作**：
     - [旁白_中立]：“笹垣出声招呼。”
   - **角色内心独白**：
     - [笹垣_其他]：“笹垣心中暗想，今天的天气真热。”
   - **角色疑问或惊讶**：
     - [笹垣_惊讶]：“小孩？大楼里怎么会有小孩？”

6. **注意事项**：
   - 标注情感时，应根据角色对话的语气、行为或情境来判断。
   - 旁白描述应包括对角色动作、环境变化等的描写，确保不遗漏。
   - 确保所有「」符号括起来的对话内容都被正确标注，不遗漏或错误分类。

7. **示例输入与输出**：
   - **输入**：
     ```
     出了近铁布施站，沿着铁路径直向西。已经十月了，天气仍闷热难当，地面也很干燥。每当卡车疾驰而过，扬起的尘土极可能会让人又皱眉又揉眼睛。

     「笹垣润三_中性」我今天终于可以休息了。
     「旁白」笹垣润三的脚步说不上轻快。他今天本不必出勤。很久没休假了，还以为今天可以悠游地看点书。为了今天，他特地留着松本清张的新书没看。
     「笹垣润三_开心」真不错，终于可以放松一下了。

     笹垣看了看写着‘烤乌贼饼四十元’的牌子，付了钱。老板娘亲切地说：“多谢。”然后拿起报纸，坐回椅子。
     ```

   - **输出**：
     ```
     [旁白_中立]：“出了近铁布施站，沿着铁路径直向西。已经十月了，天气仍闷热难当，地面也很干燥。每当卡车疾驰而过，扬起的尘土极可能会让人又皱眉又揉眼睛。”
     [笹垣润三_中性]：“我今天终于可以休息了。”
     [旁白_中立]：“笹垣润三的脚步说不上轻快。他今天本不必出勤。很久没休假了，还以为今天可以悠游地看点书。为了今天，他特地留着松本清张的新书没看。”
     [笹垣润三_开心]：“真不错，终于可以放松一下了。”
     [旁白_中立]：“笹垣看了看写着‘烤乌贼饼四十元’的牌子，付了钱。老板娘亲切地说，”
     [老板娘_开心]：“多谢。”
     [旁白_中立]：“然后拿起报纸，坐回椅子。”
     ```
    """},
            {"role": "user", "content": message},
        ],

    }
    try:
        response = requests.post(f"{API_BASE_URL}/v1/chat/completions", headers=headers, json=data,stream=True)
        if response.status_code == 200:
            result = response.json()
            annotated_text = result.get('choices')[0].get('message').get('content')
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(annotated_text)
            return annotated_text
        else:
            error_info = response.json()
            return f"标注失败: Error code: {response.status_code} - {error_info}"
    except Exception as e:
        return f"标注失败: {e}"

def assign_reference_audio(role_name, emotion, selected_audio, role_to_audio_state):
    """
    为指定角色和情感分配一个参考音频。
    """
    print(f"Assigning reference audio: role_name={role_name}, emotion={emotion}, selected_audio={selected_audio}")
    print(f"Current role_to_audio_state: {role_to_audio_state}")
    print(f"type(role_to_audio_state): {type(role_to_audio_state)}")

    if not role_name:
        return role_to_audio_state, "请提供角色名称。"

    if not emotion:
        return role_to_audio_state, "请提供情感。"

    if not selected_audio:
        return role_to_audio_state, "请选择参考音频。"

    # 检查角色名称是否存在于 reference_audios
    if role_name not in reference_audios:
        return role_to_audio_state, f"角色名称 '{role_name}' 不存在于参考音频库中。"

    # 检查所选情感是否有对应的参考音频
    if emotion not in reference_audios[role_name]:
        return role_to_audio_state, f"情感 '{emotion}' 在角色 '{role_name}' 中不存在对应的参考音频。"

    # 检查所选音频是否在对应的情感列表中
    if selected_audio not in reference_audios[role_name][emotion]:
        return role_to_audio_state, f"所选参考音频 '{selected_audio}' 不存在于角色 '{role_name}' 的情感 '{emotion}' 中。"

    # 更新映射
    if role_name not in role_to_audio_state:
        role_to_audio_state[role_name] = {}
    role_to_audio_state[role_name][emotion] = selected_audio

    print(f"已为角色 '{role_name}' 的情感 '{emotion}' 分配参考音频: {selected_audio}")

    return role_to_audio_state, f"已为角色 '{role_name}' 的情感 '{emotion}' 分配参考音频: {selected_audio}"

def generate_audio_for_line_srt(text, ref_audio_path, fragment_interval, speaker, line_number, emotion):
    """
    调用 TTS API 的 /srt 端点生成单句音频并获取音频和 SRT 文件的 URL。
    """
    tts_url = f"{API_BASE_URL}/srt"  # 请确认此端点是否正确

    # 构建完整的参考音频路径
    ref_audio_full_path = os.path.join(r'E:\GPT-SoVITS-v2\参考音频', speaker, emotion, ref_audio_path)

    # 添加调试输出
    print(f"生成音频的参考路径: {ref_audio_full_path}")

    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": ref_audio_full_path,
        "prompt_lang": "zh",
        "prompt_text": "",  # 可以根据需要填写
        "text_split_method": "cut1",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": fragment_interval,  # 保持为秒
        "seed": -1,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }

    try:
        response = requests.post(tts_url, json=payload)
        if response.status_code == 200:
            resp_json = response.json()
            return resp_json  # 包含 "code", "srt", "audio" URL
        else:
            print(f"TTS 生成失败: {response.json()}")
            return response.json()  # 返回详细的错误信息
    except Exception as e:
        print(f"TTS API 请求失败: {e}")
        return {"message": f"TTS API 请求失败: {e}"}

def download_audio(audio_url, speaker, line_number):
    """
    下载音频文件并保存到对应说话人名称的文件夹，文件名为 说话人_台词编号.wav
    """
    try:
        response = requests.get(audio_url)
        if response.status_code == 200:
            audio_content = response.content
            # 构建合理的文件名，避免非法字符
            safe_speaker = re.sub(r'[\\/*?:"<>|]', "_", speaker)
            # 创建对应说话人的文件夹
            output_dir = os.path.join(r"音频输出", safe_speaker)
            os.makedirs(output_dir, exist_ok=True)
            # 文件名为 说话人_台词编号.wav
            output_filename = os.path.join(
                output_dir,
                f"{safe_speaker}_{line_number}.wav"
            )
            with open(output_filename, "wb") as f:
                f.write(audio_content)
            print(f"音频已保存到: {output_filename}")  # 调试输出
            return output_filename
        else:
            print(f"下载音频失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"下载音频时出错: {e}")
        return None

def concatenate_audios(audio_file_paths, pause_duration_ms):
    """
    合并多个音频文件为一个完整的音频，并在音频之间插入静音段。
    """
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=pause_duration_ms)  # 创建静音段
    for i, file_path in enumerate(audio_file_paths):
        audio_segment = AudioSegment.from_file(file_path, format='wav')
        combined += audio_segment
        # 在每个音频段之间插入静音，除最后一个音频段外
        if i < len(audio_file_paths) - 1:
            combined += silence
    output_dir = r"音频输出"  # 指定保存目录
    os.makedirs(output_dir, exist_ok=True)
    # 使用日期和时间戳生成唯一且有意义的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(
        output_dir,
        f"projectname_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
    )
    combined.export(output_filename, format="wav")
    print(f"合并后的音频已保存到: {output_filename}")  # 调试输出
    return output_filename

def generate_full_audio_srt(dialogues, role_to_audio, pause_duration_ms, role_speaker_mapping,
                            selected_reference_audios):
    """
    根据解析后的对话列表，逐行生成音频，下载音频文件，并合并为一个完整的音频文件。
    selected_reference_audios: dict, 形如 {role: {emotion: ref_audio_filename, ...}, ...}
    """
    audio_file_paths = []
    errors = []
    line_number = 1  # 初始化台词编号
    for dialogue in dialogues:
        role = dialogue['role']
        text = dialogue['text']
        emotion = dialogue['emotion']  # 获取情感信息

        # 使用用户选择的配音者
        speaker = role_speaker_mapping.get(role)
        if not speaker:
            error_msg = f"未为角色 '{role}' 选择配音者。"
            print(error_msg)
            errors.append(error_msg)
            line_number += 1
            continue

        # 获取用户选择的参考音频
        ref_audio_filename = selected_reference_audios.get(role, {}).get(emotion)
        if not ref_audio_filename:
            error_msg = f"未为角色 '{role}' 的情感 '{emotion}' 配置参考音频。"
            print(error_msg)
            errors.append(error_msg)
            line_number += 1
            continue

        # 将 pause_duration 从毫秒转换为秒，确保 fragment_interval 单位正确
        fragment_interval_sec = pause_duration_ms / 1000.0

        # 设置模型权重
        if speaker in speaker_models:
            gpt_path = speaker_models[speaker]['gpt_weights']
            sovits_path = speaker_models[speaker]['sovits_weights']
            gpt_message = set_gpt_weights(gpt_path)
            sovits_message = set_sovits_weights(sovits_path)
            print(f"{speaker} - {gpt_message}")
            print(f"{speaker} - {sovits_message}")
        else:
            print(f"配音者 '{speaker}' 没有关联的模型权重，使用默认参考音频。")

        # 生成音频，传递 emotion 参数（如果后端需要）
        resp = generate_audio_for_line_srt(text, ref_audio_filename, fragment_interval_sec, speaker, line_number,
                                           emotion)
        if resp and resp.get("code") == "200":
            audio_url = resp.get("audio")
            srt_url = resp.get("srt")
            audio_file = download_audio(audio_url, speaker, line_number)
            if audio_file:
                audio_file_paths.append(audio_file)
            else:
                error_msg = f"下载音频失败，角色: {role}, 文本: {text}"
                print(error_msg)
                errors.append(error_msg)
        else:
            # 提取后端返回的错误信息
            error_detail = resp.get("Exception") or resp.get("message", "未知错误")
            error_msg = f"TTS 生成失败，角色: {role}, 文本: {text}, 错误: {error_detail}"
            print(error_msg)
            errors.append(error_msg)

        line_number += 1  # 增加台词编号

    if audio_file_paths:
        combined_audio = concatenate_audios(audio_file_paths, pause_duration_ms)
        if errors:
            return combined_audio, '\n'.join(errors)
        return combined_audio, "音频生成完成。"
    else:
        return None, '\n'.join(errors)

def handle_generate_audio(pause_duration, role_to_audio_state, annotated_file, *speaker_assignments):
    """
    生成音频文件。
    参数：
        - pause_duration: 句间停顿时长（毫秒）
        - role_to_audio_state: 当前参考音频映射（字典）
        - annotated_file: 标注后的文本文件
        - speaker_assignments: list, 配音者选择列表
    """
    print(f"开始生成音频 - 标注文件: {annotated_file}, 停顿时长: {pause_duration}, 配音者: {speaker_assignments}")
    print(f"role_to_audio_state: {role_to_audio_state}")
    print(f"type(role_to_audio_state): {type(role_to_audio_state)}")

    selected_file = annotated_file

    if not selected_file:
        return None, "未选择标注后的文本文件。"

    # 解析文本文件
    dialogues = parse_annotated_script(selected_file.name)
    if not dialogues:
        return None, "未能解析标注后的文本文件，请检查文件格式。"

    # 获取角色列表
    roles = sorted(set(dialogue['role'] for dialogue in dialogues))

    if len(roles) > 10:
        return None, f"超过最大支持的角色数量 (10)。"

    # 构建角色与配音者的映射
    role_speaker_mapping = {}
    for i, role in enumerate(roles):
        if i < len(speaker_assignments):
            speaker = speaker_assignments[i]
            if speaker:
                role_speaker_mapping[role] = speaker
            else:
                return None, f"未为角色 '{role}' 选择配音者。"
        else:
            return None, f"未为角色 '{role}' 选择配音者。"
    print(role_speaker_mapping)

    # 构建用户选择的参考音频映射
    selected_reference_audios = {}
    for role, speaker in role_speaker_mapping.items():
        selected_reference_audios[role] = {}
        for dialogue in dialogues:
            if dialogue['role'] == role:
                emotion = dialogue['emotion']
                # 获取参考音频文件名
                ref_audio_filename = role_to_audio_state.get(speaker, {}).get(emotion)
                if ref_audio_filename:
                    selected_reference_audios[role][emotion] = ref_audio_filename
                else:
                    return None, f"未为角色 '{role}' 的情感 '{emotion}' 配置参考音频。"

    # 生成音频
    combined_audio_path, audio_messages = generate_full_audio_srt(
        dialogues,
        role_to_audio_state,
        pause_duration,
        role_speaker_mapping,
        selected_reference_audios
    )

    if combined_audio_path:
        return combined_audio_path, audio_messages
    return None, audio_messages

def handle_annotation(file_path):
    if not file_path:
        return "未选择文件。", ""
    try:
        message = read_text_from_file(file_path)
        output_file_path = f"annotated_{uuid.uuid4().hex[:8]}.txt"
        annotated_text = deepseek_chat(message, output_file_path)
        if "余额不足" in annotated_text:
            return "标注失败: 余额不足。请充值或联系支持团队。", ""
        return f"标注完成，输出文件: {output_file_path}", annotated_text
    except Exception as e:
        return f"标注失败: {e}", ""

# 重新标注
def handle_reannotation(text):
    if not text:
        return "未输入文本。", ""
    try:
        output_file_path = f"annotated_{uuid.uuid4().hex[:8]}_reannotate.txt"
        annotated_text = deepseek_chat(text, output_file_path)
        if "余额不足" in annotated_text:
            return "标注失败: 余额不足。请充值或联系支持团队。", ""
        return f"重新标注完成，输出文件: {output_file_path}", annotated_text
    except Exception as e:
        return f"重新标注失败: {e}", ""

def read_text_from_file(file_path):
    """
    从文件路径读取文本内容。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def play_reference_audio(role, emotion, audio_file):
    """
    播放所选的参考音频文件。
    """
    if not role or not emotion or not audio_file:
        return None
    audio_path = os.path.join(r'E:\GPT-SoVITS-v2\参考音频', role, emotion, audio_file)
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return None
    return audio_path

# 加载参考音频文件
reference_audios = load_reference_audios()

# 获取所有可用的角色名称
available_speakers = list(reference_audios.keys())

def update_audio(role,emotion):
    if role in reference_audios:
        audio_list = reference_audios[role][emotion]
        selected_audio = audio_list[0] if audio_list else None
        return [
            gr.update(choices=audio_list,value=selected_audio)
        ]
    else:
        return [
            gr.update(choice=[],value=None)
        ]


# 新增：定义同时更新情感和参考音频的函数
def update_emotions_and_audio(role):
    """
    当角色变化时，同时更新情感和参考音频下拉菜单。
    """
    if role in reference_audios:
        emotions = list(reference_audios[role].keys())
        new_emotion = emotions[0] if emotions else None
        if new_emotion:
            # 获取该情感对应的音频列表
            audio_list = reference_audios[role][new_emotion]
            selected_audio = audio_list[0] if audio_list else None
            return [
                gr.update(choices=emotions, value=new_emotion),  # 更新情感下拉菜单
                gr.update(choices=audio_list, value=selected_audio)  # 更新参考音频下拉菜单
            ]
        else:
            return [
                gr.update(choices=emotions, value=None),
                gr.update(choices=[], value=None)  # 更新参考音频为空
            ]
    return [
        gr.update(choices=[], value=None),  # 如果角色不在字典中，清空情感下拉菜单
        gr.update(choices=[], value=None)   # 清空参考音频下拉菜单
    ]


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 多角色语音合成器")
    gr.Markdown(
        """
        **使用说明：**
        - 在“文本标注”标签页中：
            - 上传原始文本文件。
            - 点击“标注文本”按钮，系统将调用大型语言模型对文本进行标注。
            - 查看并手动修改标注后的文本，或点击“重新标注”按钮让系统重新标注。
        - 在“配置参考音频”标签页中：
            - 选择角色名称（说话人名称，例如：丹恒）。
            - 选择情感（例如：中立_neutral）。
            - 从下拉菜单中选择对应的参考音频。选择后，音频播放器将自动播放所选音频。
            - 点击“分配参考音频”按钮，将选择的参考音频分配给指定的角色和情感。
        - 在“音频生成”标签页中：
            - 上传标注后的文本文件。
            - 点击“解析文本并获取角色”按钮，系统将解析文本并提取所有角色。
            - 为每个角色选择对应的配音者。
            - 点击“生成音频”按钮，系统将根据您的选择生成合成语音，并合并为一个完整的音频文件。
            - 生成的音频文件将显示在下方，可直接播放或下载。
        - 在“模型权重设置”标签页中：
            - 输入 GPT 或 SoVITS 模型权重路径，并点击相应按钮以动态切换模型权重。
        """
    )

    # “文本标注”标签页
    with gr.Tab("文本标注"):
        with gr.Column():
            gr.Markdown("### 上传原始文本文件并进行标注")
            original_text_file = gr.File(label="选择原始文本文件", file_types=[".txt"], type="filepath")
            annotate_btn = gr.Button("标注文本")
            reannotate_btn = gr.Button("重新标注")
            annotated_text_output = gr.Textbox(label="标注后的文本（可手动修改）", lines=20)
            annotate_status = gr.Textbox(label="状态信息", lines=2, interactive=False)

        # 标注文本按钮点击事件
        annotate_btn.click(
            fn=handle_annotation,
            inputs=[original_text_file],
            outputs=[annotate_status, annotated_text_output]
        )

        # 重新标注按钮点击事件
        reannotate_btn.click(
            fn=handle_reannotation,
            inputs=[annotated_text_output],
            outputs=[annotate_status, annotated_text_output]
        )

    # “配置参考音频”标签页
    with gr.Tab("配置参考音频"):
        with gr.Row():
            with gr.Column():
                role_name_config = gr.Dropdown(
                    label="角色名称",
                    choices=["请选择角色名称"] + available_speakers,  # 添加默认选项
                    value="请选择角色名称"  # 设置默认值
                )
                emotion_config = gr.Dropdown(
                    label="情感",
                    choices=[],  # 初始为空，动态更新
                    value=None
                )
                selected_audio_config = gr.Dropdown(
                    label="选择参考音频",
                    choices=[],  # 初始为空，动态更新
                    value=None
                )
                assign_btn = gr.Button("分配参考音频")
            with gr.Column():
                assign_output = gr.Textbox(label="分配结果", lines=2)
                # 新增音频播放组件
                play_output = gr.Audio(label="参考音频预览")

        # 修改：角色变化时同时更新情感和参考音频
        role_name_config.change(
            fn=update_emotions_and_audio,
            inputs=[role_name_config],
            outputs=[emotion_config, selected_audio_config]  # 返回情感和参考音频更新
        )

        # 情感变化时，更新参考音频列表
        emotion_config.change(
            fn=lambda emotion, role: gr.update(choices=reference_audios.get(role, {}).get(emotion, [])),
            inputs=[emotion_config, role_name_config],
            outputs=[selected_audio_config]
        )


        # 定义播放参考音频的函数
        def play_reference_audio(role, emotion, audio_file):
            """
            播放所选的参考音频文件。
            """
            if not role or not emotion or not audio_file:
                return None
            audio_path = os.path.join(r'E:\GPT-SoVITS-v2\参考音频', role, emotion, audio_file)
            if not os.path.exists(audio_path):
                print(f"音频文件不存在: {audio_path}")
                return None
            return audio_path

        # 当参考音频选择变化时，自动播放音频
        selected_audio_config.change(
            fn=play_reference_audio,
            inputs=[role_name_config, emotion_config, selected_audio_config],
            outputs=[play_output]
        )

        # 使用 gr.State 来保存 role_to_audio 映射
        role_to_audio_state = gr.State(value={})  # {speaker: {emotion: audio_filename}}

        # 点击“分配参考音频”按钮
        assign_btn.click(
            fn=lambda role, emotion, audio, mapping: assign_reference_audio(role, emotion, audio, mapping),
            inputs=[role_name_config, emotion_config, selected_audio_config, role_to_audio_state],
            outputs=[role_to_audio_state, assign_output]
        )

    # “音频生成”标签页
    with gr.Tab("音频生成"):
        with gr.Row():
            with gr.Column():
                # 仅保留“选择标注后的文本文件”
                annotated_text_file_input = gr.File(
                    label="选择标注后的文本文件",
                    file_types=[".txt"],
                    type="filepath"
                )
                pause_input = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    step=100,
                    label="句间停顿时长（毫秒）",
                    value=500  # 默认500毫秒
                )
                parse_btn = gr.Button("解析文本并获取角色")
                parse_output = gr.Textbox(label="解析结果", lines=2)
                # 使用 gr.State 来保存解析后的角色列表
                parsed_roles = gr.State(value=[])
                # 创建角色选择下拉菜单（动态生成）
                speaker_dropdowns = []
                MAX_ROLES = 10  # 预定义最多支持的角色数量
                for i in range(MAX_ROLES):
                    dropdown = gr.Dropdown(
                        label=f"角色 {i + 1} 分配配音者",
                        choices=list(speaker_models.keys()),
                        value=list(speaker_models.keys())[0] if speaker_models else None,  # 设置默认值
                        visible=False  # 初始时隐藏
                    )
                    speaker_dropdowns.append(dropdown)
                generate_btn = gr.Button("生成音频")
            with gr.Column():
                output_audio = gr.Audio(label="生成的音频文件", type="filepath")
                output_messages = gr.Textbox(label="提示信息", lines=10)

        # 解析文本并获取角色
        def handle_parse_roles(file):
            roles, message = parse_and_get_roles(file)
            if message:
                return message, [], []  # message, empty speaker_dropdowns
            else:
                return "角色解析成功，请为每个角色选择配音者。", roles, roles  # message, roles list

        parse_btn.click(
            fn=lambda file: handle_parse_roles(file),
            inputs=[annotated_text_file_input],
            outputs=[parse_output, parsed_roles, parsed_roles]
        )

        # 设置角色下拉菜单的配音者选择
        def set_speaker_dropdowns(message, roles, _):
            updates = [gr.update(value=message)]
            for i in range(MAX_ROLES):
                if i < len(roles):
                    updates.append(gr.update(
                        label=f"角色 '{roles[i]}' 分配配音者",
                        choices=list(speaker_models.keys()),
                        value=list(speaker_models.keys())[0],
                        visible=True
                    ))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        parse_btn.click(
            fn=set_speaker_dropdowns,
            inputs=[parse_output, parsed_roles, parsed_roles],
            outputs=[output_messages] + speaker_dropdowns
        )

        # 生成音频按钮点击事件
        generate_btn.click(
            fn=lambda pause, mapping, annotated_file, *speakers: handle_generate_audio(
                pause, mapping, annotated_file, *speakers
            ),
            inputs=[pause_input, role_to_audio_state, annotated_text_file_input] + speaker_dropdowns,
            outputs=[output_audio, output_messages]
        )

    # “模型权重设置”标签页
    with gr.Tab("模型权重设置"):
        with gr.Row():
            with gr.Column():
                set_gpt_weights_input = gr.Textbox(
                    label="GPT 模型权重路径",
                    placeholder=r"E:\GPT-SoVITS-v2\GPT_weights_v2\model.pth",
                    lines=1
                )
                set_sovits_weights_input = gr.Textbox(
                    label="SoVITS 模型权重路径",
                    placeholder=r"E:\GPT-SoVITS-v2\SoVITS_weights_v2\model.pth",
                    lines=1
                )
                set_gpt_btn = gr.Button("设置 GPT 权重")
                set_sovits_btn = gr.Button("设置 SoVITS 权重")
            with gr.Column():
                set_weights_output = gr.Textbox(label="权重设置结果", lines=2)

        # 点击“设置 GPT 权重”按钮
        set_gpt_btn.click(
            fn=set_gpt_weights,
            inputs=[set_gpt_weights_input],
            outputs=[set_weights_output]
        )

        # 点击“设置 SoVITS 权重”按钮
        set_sovits_btn.click(
            fn=set_sovits_weights,
            inputs=[set_sovits_weights_input],
            outputs=[set_weights_output]
        )

    demo.launch(share=False)
