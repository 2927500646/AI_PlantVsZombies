from openai import OpenAI
import numpy as np
import pyautogui
import cv2
import time
import json


# 轮询间隔
SLEEP_TIME = 7


# 游戏区域坐标（假设游戏窗口在屏幕左上角，需根据实际调整）
GAME_REGION = {
    "sun_display": (285, 110, 370, 150),    # 阳光显示区域 (x1, y1, x2, y2)
    "zombie_panel": (400, 18, 1240, 142),  # 底部僵尸选择面板 (x1, y1, x2, y2)
    "grid_start": (300, 150),              # 草坪左上角第一个格子的坐标 (x, y)
    "cell_width": 145,                    # 每个格子的宽度（像素）
    "cell_height": 175,                  # 每个格子的高度（像素）
    "rows": 5,
    "cols": 9,
    "first_brain" : (268, 236),
    "brain_dist" : 175
}

ZOMBIE_NAMES = [
    "小鬼僵尸", "路障僵尸", "铁通僵尸", "舞王僵尸", "矿工僵尸",
    "飞贼僵尸", "梯子僵尸", "橄榄球僵尸", "撑杆僵尸"
]

ZOMBIE_COSTS = {
    "小鬼僵尸" : 50, "路障僵尸" : 75, "铁通僵尸" : 125,
    "舞王僵尸" : 350, "矿工僵尸" : 125, "飞贼僵尸" : 125,
    "梯子僵尸" : 150, "橄榄球僵尸" : 175, "撑杆僵尸" : 75
}

ZOMBIE_TEMPLATES = {}  # 全局变量
def load_zombie_templates():
    print("[INFO]: 加载僵尸图片模板")
    for z_name in ZOMBIE_NAMES:
        template = imread_chinese(f"./pic/zombies/{z_name}.png", 0)
        if template is not None:
            ZOMBIE_TEMPLATES[z_name] = template
        else:
            print(f"[WARN] 无法加载僵尸模板: {z_name}.png")

PLANT_NAMES = [
    "保护伞", "磁力菇", "大喷菇", "大嘴花", "地刺", "寒冰射手",
    "火焰树桩", "坚果", "三重射手", "双向射手", "土豆地雷", "豌豆射手",
    "倭瓜", "向日葵", "小喷菇", "杨桃", "玉米投手"
]

PLANT_TEMPLATES = {}  # 全局变量
def load_plant_templates():
    print("[INFO]: 加载植物图片模板")
    for p_name in PLANT_NAMES:
        template = imread_chinese(f"./pic/plants/{p_name}.png", cv2.IMREAD_COLOR)
        if template is not None:
            PLANT_TEMPLATES[p_name] = template
        else:
            print(f"[WARN] 无法加载植物模板: {p_name}.png")

# 在程序启动时预加载数字模板
DIGIT_TEMPLATES = {}  # 全局
def load_digit_templates():
    print("[INFO]: 加载数字图片模板")
    for i in range(10):
        template = imread_chinese(f"./pic/digits/{i}.png", cv2.IMREAD_GRAYSCALE)
        if template is not None:
            DIGIT_TEMPLATES[str(i)] = template
        else:
            print(f"[WARN] 无法加载数字模板: {i}.png")


BRAIN_TEMPLATE = None
def load_brain_template():
    global BRAIN_TEMPLATE
    BRAIN_TEMPLATE = imread_chinese("./pic/brain.png", cv2.IMREAD_GRAYSCALE)
    if BRAIN_TEMPLATE is not None:
        print("[INFO] 大脑模板加载成功")
    else:
        print("[WARN] 大脑模板加载失败")


def imread_chinese(path, flags=cv2.IMREAD_GRAYSCALE):
    """支持中文路径的图片读取函数"""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception as e:
        print(f"读取文件失败 [{path}]: {e}")
        return None

def find(name):
    # 截图，并保存到路径
    pyautogui.screenshot().save("./pic/screenshot.png")

    # 从路径读取刚刚截好的图，存入变量
    img = cv2.imread("./pic/screenshot.png")

    # 读取僵尸模板图片
    img_template = imread_chinese(f"./pic/zombies/{name}.png", cv2.IMREAD_COLOR)

    if img_template is None:
        print(f"[ERROR] 无法加载模板 {name}")
        return (0, 0)

    # 读取模板的长和宽
    h, w, channel = img_template.shape

    # 进行模板匹配
    res = cv2.matchTemplate(img, img_template, cv2.TM_SQDIFF_NORMED)

    # 解析出匹配区域的左上角坐标
    upper_left = cv2.minMaxLoc(res)[2]

    # 计算匹配区域右下角的坐标
    lower_right = (upper_left[0] + w, upper_left[1] + h)

    # 计算中心区域的坐标并且返回
    avg = ((upper_left[0] + lower_right[0]) // 2, (upper_left[1]+lower_right[1]) / 2)

    # 返回中心点坐标
    return avg

def click(var_avg):
    pyautogui.click(var_avg[0], var_avg[1], button='left')
    time.sleep(0.1)

def findAndClick(name):
    avg = find(name)
    print(f'[INFO]: 正在点击 {name}')
    click(avg)

def get_ai_decision(state):
    print("[INFO]: 正在寻求AI决策")
    """
    本函数用AI进行放置僵尸最佳方案决策，
    传入本关可选僵尸集合，
    大模型思考，
    返回JSON行动指令，
    再进行一一执行

    :param state: 游戏状态描述
    :return: JSON格式，代表行动指令
    """

    system_prompt = """
你是一个《植物大战僵尸》游戏中的“僵尸指挥官”。你的任务是决定在“小游戏”\
模式的“无尽僵尸”关卡中，何时以及在哪一路放出僵尸，以尽可能小的阳光消耗\
以获得尽可能多的胜利。

**游戏规则：**
- 游戏场地有5条横向的“路”。
- 你可以选择放置该关卡提供的僵尸。
- 你的目标是吃掉5路位于后排的脑子。
- 你需要根据当前的游戏局势来做出最优决策。
- 只有第六列及以后能放置僵尸(飞贼僵尸完全相反)。
- 僵尸攻击向日葵可以获得阳光，放置僵尸消耗阳光。
- 充分利用你的阳光，请不要在已夺取大脑的行浪费阳光。
- 磁力菇在4格范围内会夺取僵尸的金属物品(铁桶，铁镐，铁头盔，铁梯子)。
- 梯子僵尸可以抵挡豌豆类植物射击。

**输出格式：**
你必须**只**输出一个JSON数组，不要包含任何其他解释或文字。

示例:
[{"row":1, "col":6, "zombie_name":"普通僵尸", "delay":0}, {"row":3, "col":7, "zombie_name":"路障僵尸", "delay":3}]

一个JSON代表一只僵尸
示例JSON格式如下：
{
    "row": 1, 
    "col": 6,
    "zombie_name": "普通僵尸"
    "delay": 3
}
 
- "row": 整数，取值范围 1 到 5，代表你要在哪一行放僵尸。
- "col": 整数，取值范围 5 到 9，代表你要在哪一列放僵尸
- "zombie_name": 字符串，代表可选僵尸类型名（从游戏状态里获取）。
- "delay" 整数（秒），表示相对于从上一个操作开始的等待时间。

特别的，如果暂时不想放僵尸，返回空数组 []。
    """

    user_prompt = f"当前游戏状态如下:\n{state}\n请根据以上信息做出决策。"

    try:
        res = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
            # ,reasoning_effort = "high",
        )
        print("[INFO]: 提示词已发送")
        ai_msg = res.choices[0].message.content
        print(f"[ AI ]: {ai_msg}")
        commands = json.loads(ai_msg)
        return commands

    except Exception as e:
        print("[ERROR]: 未能获得AI决策")
        print(f"[ERROR]: 错误信息:{e}")
        # 返回空指令数组
        return []

def get_zombies(img):
    x1, y1, x2, y2 = GAME_REGION["zombie_panel"]
    panel_roi = img[y1:y2, x1:x2]
    panel_gray = cv2.cvtColor(panel_roi, cv2.COLOR_BGR2GRAY)

    available = []  # 存储 "名称(代价)" 字符串
    for z_name in ZOMBIE_NAMES:
        template = imread_chinese(f"./pic/zombies/{z_name}.png", 0)
        if template is None:
            continue
        result = cv2.matchTemplate(panel_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.73:   # 匹配度阈值
            cost = ZOMBIE_COSTS.get(z_name, "未知")
            available.append(f"{z_name}({cost})")

    return ", ".join(available) if available else "无可用僵尸"


def get_sun(img):
    x1, y1, x2, y2 = GAME_REGION["sun_display"]
    sun_roi = img[y1:y2, x1:x2]
    sun_gray = cv2.cvtColor(sun_roi, cv2.COLOR_BGR2GRAY)

    # 二值化（用于轮廓提取）
    _, thresh = cv2.threshold(sun_gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集候选数字区域（在原灰度图上）
    digit_rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 3 and h > 5 and w < 60 and h < 70:
            digit_rois.append((x, y, w, h))

    if not digit_rois:
        print("[WARN] 未找到任何数字轮廓")
        return 0

    digit_rois.sort(key=lambda rect: rect[0])  # 从左到右排序

    digits = []
    for (x, y, w, h) in digit_rois:
        # 从原始灰度图中截取数字区域（而不是二值图）
        digit_roi = sun_gray[y:y+h, x:x+w]

        best_match = None
        best_score = 0.45  # 阈值

        for d, template in DIGIT_TEMPLATES.items():
            # 将模板缩放到与 digit_roi 相同尺寸（保持宽高比）
            # 先确定缩放比例，使模板恰好能完全覆盖 digit_roi（或略微缩小）
            # 这里我们直接 resize 到完全一致，忽略原始比例（数字一般是等比例）
            # 更稳健：保持宽高比，填充到相同尺寸
            t_h, t_w = template.shape
            # 如果模板尺寸与 digit_roi 不同，直接 resize（可能会轻微变形，但数字变形不大）
            if t_h != h or t_w != w:
                tmpl = cv2.resize(template, (w, h))
            else:
                tmpl = template

            # 匹配（模板与数字区域尺寸相同）
            result = cv2.matchTemplate(digit_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_match = d

        if best_match is not None:
            digits.append(best_match)
        else:
            # 调试：打印该数字区域的最高分
            print(f"[WARN] 数字区域 ({w}x{h}) 未能匹配，最高分 {best_score:.2f}")

    if digits:
        result = int(''.join(digits))
        print(f"[INFO]: 识别阳光: {result}")
        return result
    else:
        print("[WARN] 未能识别任何数字")
        return 0


def get_plants_layout(img):
    start_x, start_y = GAME_REGION["grid_start"]
    cell_w = GAME_REGION["cell_width"]
    cell_h = GAME_REGION["cell_height"]
    rows = GAME_REGION["rows"]
    cols = GAME_REGION["cols"]

    padding = 10  # 可调参数
    h_img, w_img = img.shape[:2]

    layout_str = ""
    for row in range(1, rows + 1):
        layout_str += f"第{row}行："
        for col in range(1, cols + 1):
            # 计算原始格子边界
            x1 = start_x + (col - 1) * cell_w
            y1 = start_y + (row - 1) * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # 扩大区域，并限制在图像边界内
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w_img, x2 + padding)
            y2_pad = min(h_img, y2 + padding)

            # 裁剪扩大后的区域（保持彩色）
            cell_roi = img[y1_pad:y2_pad, x1_pad:x2_pad]

            # 接下来匹配逻辑不变，但注意模板尺寸必须小于 cell_roi，由于我们之前已将模板缩放到固定尺寸（如100px），而 cell_roi 尺寸约为 (150+20)x(150+20)，模板尺寸更小，没问题。

            plant_found = "空地"
            best_score = 0.6

            for plant_name, template in PLANT_TEMPLATES.items():
                # 模板尺寸已经确保小于格子，也小于扩大后的区域
                if template.shape[0] > cell_roi.shape[0] or template.shape[1] > cell_roi.shape[1]:
                    # 这种情况理论上不会发生，如果发生则跳过
                    continue
                result = cv2.matchTemplate(cell_roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    plant_found = plant_name

            layout_str += f"C{col}:{plant_found}  "

            # 在匹配循环后，输出调试信息
            print(f"[DEBUG] ({row},{col}) 最佳匹配: {plant_found} 分数: {best_score:.2f}")
        layout_str += "\n"

    return layout_str


def get_brain(img):
    """
    检测每一行大脑是否已被夺取
    :param img: OpenCV BGR 图像
    :return: 列表，长度为5，True表示已夺取，False表示未夺取
    """
    if BRAIN_TEMPLATE is None:
        return [False] * 5  # 若未加载模板，默认全部未夺取

    x_first, y_first = GAME_REGION["first_brain"]
    dist = GAME_REGION["brain_dist"]
    brain_status = []

    # 将图像转为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for row in range(5):
        # 计算该行大脑图标区域 (假设图标尺寸固定，可能需要调整)
        # 这里假设大脑图标宽高约为 30x30，你可以根据实际调整
        x = x_first
        y = y_first + row * dist
        # 截取该区域
        roi = gray[y:y+40, x:x+40]  # 适当扩大范围

        # 模板匹配
        result = cv2.matchTemplate(roi, BRAIN_TEMPLATE, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 如果匹配度低于阈值（例如0.6），认为大脑已被夺取
        if max_val < 0.6:
            brain_status.append(True)   # 已夺取
        else:
            brain_status.append(False)  # 未夺取

    print(f"[DEBUG]: 大脑夺取状态{brain_status}")
    return brain_status


def get_state():
    """
    返回一个字符串，用于向大模型描述游戏状态
    :return: 字符串
    """
    screenshot = pyautogui.screenshot()

    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    state = ""
    state += f"你有: {get_sun(img_cv)}阳光\n"# 阳光数量
    state += f"你可以选择的僵尸以及代价: {get_zombies(img_cv)}\n"# 僵尸可选集
    state += f"植物的布局情况:\n{get_plants_layout(img_cv)}\n"# 植物布局情况

    brain_status = get_brain(img_cv)
    state += "\n大脑夺取情况：\n"
    for i, taken in enumerate(brain_status, start=1):
        status = "已夺取" if taken else "未夺取"
        state += f"第{i}行大脑：{status}\n"

    return state

def lay_zombie(row, col, zombie_name, delay):
    # 起点
    x, y = GAME_REGION["grid_start"]

    # 长宽
    height = GAME_REGION["cell_height"]
    width = GAME_REGION["cell_width"]

    # 延迟指定秒数
    time.sleep(delay)

    # 找到指定僵尸的卡片，并点击选择
    findAndClick(zombie_name)

    # 找到指定的位置，并点击放置
    nx, ny = x+(col-0.5)*width, y+(row-0.5)*height
    print(nx, ny)
    pyautogui.click(nx, ny)


def execute(commands):
    print("[INFO]: 执行放置命令")
    for idx, op in enumerate(commands):
        row = op.get("row")
        col = op.get("col")
        zombie_name = op.get("zombie_name")
        delay = op.get("delay", 0)

        if row is None or col is None or zombie_name is None:
            print(f"操作 {idx + 1} 缺少必要字段，跳过")
            continue

        print(f"[INFO]: {delay} 秒后在 ({row}, {col}) 放置 {zombie_name}")

        # 放置僵尸
        lay_zombie(row, col, zombie_name, delay)


def main():
    while True:
        time.sleep(SLEEP_TIME)

        # 获取当局状态，发送给AI进行决策
        commands = get_ai_decision(get_state())  # 已经是 list，不要再 json.loads
        if commands:
            # 执行命令
            execute(commands)
        else:
            print("[INFO]: AI返回空指令，等待下一轮")


if __name__ == "__main__":
    # apikey写这里
    with open("apikey.txt") as file:
        apikey = file.read()

    # 选定大模型
    client = OpenAI(
        api_key=apikey,
        base_url="https://api.deepseek.com"
    )

    # 资源预加载
    load_zombie_templates()
    load_plant_templates()
    load_digit_templates()
    load_brain_template()

    # 执行主函数逻辑
    main()
