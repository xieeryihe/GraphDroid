from utils.logger import logger
from utils.macro import DictKey, ACTION, COLOR
from utils.draw import *
from core.controller import get_center, AndroidController


class Executor:
    def __init__(self, task_dir, controller: AndroidController):
        self.task_dir = task_dir
        self.controller = controller

    def execute(self, info):
        """一个 info 的格式如下:
        {
            "action": "click",
            "bounds": [0, 0, 100, 100],
            "image_path": "path/to/image.png"
        }
        """
        action = info[DictKey.ACTION]
        logger.log(f">>> Executing action '{action}'...", color=COLOR.YELLOW)
        src_image_path = info.get(DictKey.IMAGE_PATH, None)
        dst_image_path = src_image_path.replace(".png", f"_{action}.png")
        bounds = info.get(DictKey.BOUNDS, None)
        if action == ACTION.CLICK:
            center_x, center_y = get_center(info)
            self.controller.click(center_x, center_y)
        elif action == ACTION.BACK:
            self.controller.back()  # 无需绘图
        elif action == ACTION.LONG_CLICK:
            center_x, center_y = get_center(info)
            self.controller.long_click(center_x, center_y)
        elif action == ACTION.SCROLL:
            # 滑动半屏
            direction = info.get(DictKey.DIRECTION, None)
            width, height = self.controller.width, self.controller.height
            if direction == "up":
                start_point = (width // 2, height * 3 // 4)
                end_point = (width // 2, height // 4)
            elif direction == "down":
                start_point = (width // 2, height // 4)
                end_point = (width // 2, height * 3 // 4)
            elif direction == "left":
                start_point = (width * 3 // 4, height // 2)
                end_point = (width // 4, height // 2)
            elif direction == "right":
                start_point = (width // 4, height // 2)
                end_point = (width * 3 // 4, height // 2)
            else:
                raise ValueError(f"Invalid direction: {direction}. Expected one of ['up', 'down', 'left', 'right'].")
            self.controller.scroll(start_point, end_point)
        elif action == ACTION.INPUT:
            text = info.get(DictKey.TEXT, None)
            self.controller.input(text)
        draw_rect(src_image_path, dst_image_path, bounds)
