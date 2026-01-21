LOG_SEPARATOR = '=' * 15


class DictKey:
    """一些 key """
    # 经典的
    ID = 'id'
    TASK = 'task'
    IMAGE_PATH = 'image_path'
    PAGE_ID = 'page_id'
    SCORE = 'score'
    INFO = 'info'  # 一些额外的信息
    DATA = 'data'

    # 节点属性
    CLASS = 'class'
    TEXT = 'text'
    NODE_ID = 'node_id'
    BOUNDS = 'bounds'
    RESOURCE_ID = 'resource_id'
    IS_VISITED = 'isVisited'
    SHOULD_VISIT = 'shouldVisit'
    CLICKABLE = 'clickable'
    SOM_TAG = 'SoM_tag'  # SoM 标签，绘制的角标文字
    CONTENT_DESC = 'content_desc'

    # UTG 相关的
    FINISHED = 'finished'  # 页面是否探索完毕
    NODES = 'nodes'
    SUMMARY = 'summary'
    DESCRIPTION = 'description'
    FUNCTION = 'function'
    PACKAGE = 'package'
    ACTIVITY = 'activity'
    FRAGMENT = 'fragment'
    NODE_COUNT = 'node_count'
    TYPE = 'type'
    LABEL = 'label'

    EDGES = 'edges'
    EDGE_ID = 'edge_id'
    ACTION = 'action'
    FROM = 'from'
    TO = 'to'

    # 测试用例相关
    CASE_START_PAGE_ID = 'case_start_page_id'
    CASE_ID = 'case_id'
    CASE_LEN = 'case_len'  # case 长度
    CASE_STEPS = 'case_steps'
    CASE_SUMMARY = 'case_summary'
    CASE_SENTENCE = 'case_sentence'
    STEP_ID = 'step_id'  # 步骤id
    STEP_NUM = 'step_num'  # 实际执行步骤数

    # 推理相关
    CONTENT = 'content'
    RESPONSE = 'response'
    USAGE = 'usage'
    RESPONSE_CONTENT = 'response_content'
    PROMPT = 'prompt'
    PROMPT_TOKENS = 'prompt_tokens'
    COMPLETION_TOKENS = 'completion_tokens'
    COST = 'cost'
    REASON = 'reason'  # 用于记录为什么选择这个操作
    TIME = 'time'  # 通用时间key

    # 实验要用的
    CASE_TIME = 'case_time'  # 单纯case执行时间
    TOTAL_TIME = 'total_time'  # 一个case的总时间
    AVERAGE_STEP_TIME = 'average_step_time'  # 平均每一步执行的时间
    PRE_EDGE_IDS = 'pre_edge_ids'  # 前置边的id
    ORACLE = 'oracle'
    CORRECT_RATIO = 'correct_ratio'  # 正确比例

    # 统计所有数据
    STEP_CHAT_DATA = 'step_chat_data'  # 每一步所有的聊天数据




class FilePath:
    # 写入的文件名
    TEMP_DIR = 'temp'
    DATA_DIR = 'data'
    TASKS_DIR = 'tasks'
    

class ACTION:
    """交互类型"""
    BACK = 'back'
    CLICK = 'click'
    LONG_CLICK = 'long_click'
    INPUT = 'input'
    SCROLL = 'scroll'
    DRAG = 'drag'  # 拖拽
    STOP = 'stop'  # 任务结束
    

class APPInfo:
    INTENT_MEITUAN = "com.sankuai.meituan/com.meituan.android.pt.homepage.activity.MainActivity"  # 美团
    INTENT_XHS = "com.xingin.xhs/com.xingin.xhs.index.v2.IndexActivityV2"  # 小红书
    INTENT_DIANPING = "com.dianping.v1/com.dianping.v1.NovaMainActivity"  # 大众点评
    INTENT_BROCCOLI = "com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity"  # 食谱
    INTENT_NOTE = "com.miui.notes/com.miui.notes.ui.NotesListActivity"  # 小米笔记
    INTENT_NEWS = "com.google.android.apps.magazines/com.google.apps.dots.android.app.activity.CurrentsStartActivity" # Google News
    INTENT_ANTENNA_POD = "de.danoeh.antennapod/de.danoeh.antennapod.activity.MainActivity"
    INTENT_ANKI = "com.ichi2.anki/com.ichi2.anki.DeckPicker"
    INTENT_AMAZE = "com.amaze.filemanager/com.amaze.filemanager.ui.activities.MainActivity"
    INTENT_MEMO = "org.liberty.android.fantastischmemo/org.liberty.android.fantastischmemo.ui.AnyMemo"
    INTENT_BRAIN = "com.mhss.app.mybrain/.presentation.main.MainActivity"


class STATE:
    """一些返回状态"""
    # 错误
    ERROR_OTHER_APP = -1  # 不在指定 app 中

    # 信息
    INFO_SKIP_NODE = 10  # 跳过节点
    INFO_MAX_DEPTH = 11  # 到达最大深度

    # 正常
    OK = 0  # 无事发生，继续运行
    SUCCESS = 1


class EXPLORE_MODE:
    """探索模式"""
    TRAVERSAL = "traversal"  # 遍历模式
    DIRECTED = "directed"  # 定向模式


class COLOR:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'  # 用于恢复默认颜色
