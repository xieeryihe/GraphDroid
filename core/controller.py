import os
import re
import subprocess
import time
from xml.dom import minidom, Node
from utils.macro import *
from utils.utils import get_package
import platform

# 判断当前操作系统
current_os = platform.system()

def get_device():
    adb_command = "adb devices"
    result = execute_adb(adb_command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        # 只返回第一个设备名
        for device in devices:
            if device.endswith("device"):
                device_name = device.split("\t")[0]
                return device_name
    raise Exception("No device found")


def get_bounds(node):
    # 返回 [x1,y1,x2,y2] 形式的 bounds
    # node是dict时，需要解析bounds，如果是列表，直接解析
    try:
        if isinstance(node, dict):
            bounds = node[DictKey.BOUNDS]
        elif isinstance(node, list):
            bounds = node
        else:
            raise TypeError("node must be a dict or a list")
        # 解析 bounds
        if len(bounds) == 2:  # [[776,213][929,299]] 的 xml 形式
            bounds = [int(bounds[0][0]), int(bounds[0][1]), int(bounds[1][0]), int(bounds[1][1])]
        elif len(bounds) == 4:  # [x1,y1,x2,y2] 形式
            bounds = [int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])]
        else:
            bounds = None
        return bounds
    except Exception as e:
        print(e)
    finally:
        return bounds


def get_center(node):
    try :
        bounds = get_bounds(node)
        x1, y1, x2, y2 = bounds
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)
    except Exception as e:
        print(e)
    return (None, None)


# =============== adb 相关的操作 =================


def execute_adb(adb_command):
    # print(adb_command)
    result = subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    time.sleep(3)
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Command execution failed: {adb_command}")
    print(result.stderr, "red")
    return "ERROR"


# =============== Controller 类 =================

class AndroidController:
    def __init__(self, device=None):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.width, self.height = self.get_device_size()

    def get_device_size(self):
        adb_command = f"adb -s {self.device} shell wm size"
        result = execute_adb(adb_command)
        if result != "ERROR":
            if result.find('Override size') != -1:
                pattern = r".*?Override size: (\d+)x(\d+)"
                match = re.search(pattern, result)
                print(match)
                return int(match.group(1)), int(match.group(2))
            return map(int, result.split(": ")[1].split("x"))
        else:
            raise Exception("get device size failed")

    def is_app_installed(self, package_name):
        adb_command = f"adb -s {self.device} shell pm list packages"
        result = execute_adb(adb_command)
        if result != "ERROR":
            # pm list packages 返回的格式是 package:com.example.app
            packages = result.split('\n')
            for package in packages:
                if package_name in package:
                    return True
            return False
        return False

    def install_app(self, apk_path):
        # 通过apk安装
        if not os.path.exists(apk_path):
            print(f"APK file not found: {apk_path}")
            return False
        
        adb_command = f"adb -s {self.device} install -r {apk_path}"
        result = execute_adb(adb_command)
        
        if result != "ERROR" and "Success" in result:
            print(f"App installed successfully from {apk_path}")
            return True
        else:
            print(f"Failed to install app from {apk_path}")
            print(f"Error: {result}")
            return False

    def get_screenshot(self, prefix, save_dir):
        prefix = str(prefix)
        cap_command = f"adb -s {self.device} shell screencap -p /sdcard/temp.png"
        pull_command = f"adb -s {self.device} pull /sdcard/temp.png {os.path.join(save_dir, prefix + '.png')}"
        result = execute_adb(cap_command)
        if result != "ERROR":
            result = execute_adb(pull_command)
            if result != "ERROR":
                return os.path.join(save_dir, prefix + ".png")
            return result
        return result

    def get_xml(self, prefix, save_dir):
        prefix = str(prefix)
        dump_command = f"adb -s {self.device} shell uiautomator dump /sdcard/temp.xml"
        pull_command = f"adb -s {self.device} pull /sdcard/temp.xml {os.path.join(save_dir, prefix + '.xml')}"
        result = execute_adb(dump_command)
        if result != "ERROR":
            result = execute_adb(pull_command)
            if result != "ERROR":
                xml_path = os.path.join(save_dir, prefix + ".xml")
                if os.path.getsize(xml_path) > 0:  # 文件内容可能为空
                    # 重新读取 xml，以缩进美观写入
                    with open(xml_path, 'rb') as file:
                        xml_content = file.read()
                    dom = minidom.parseString(xml_content)
                    pretty_xml = dom.toprettyxml(indent="  ")
                    with open(xml_path, 'wb') as file:
                        file.write(bytes(pretty_xml, 'utf-8'))
                return xml_path
        return result
    
    def get_u2_xml(self, prefix, save_dir):
        import uiautomator2 as u2
        u2_device = u2.connect_usb(self.device)
        xml_data = u2_device.dump_hierarchy()
        xml_path = os.path.join(save_dir, str(prefix) + ".xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_data)
        return xml_path

    def get_current_intent(self):
        if current_os != 'Windows':
            adb_command = f"adb -s {self.device} shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'"
        else:
            # adb_command = f'adb -s {self.device} shell dumpsys window | findstr "mCurrentFocus"'
            # adb_command = f"adb -s {self.device} shell \"dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'\""  # miui 可能会导致通知消息在栈顶，影响结果
            adb_command = f'adb -s {self.device} shell dumpsys activity | findstr "mResumedActivity"'  # 获取前台APP
        result = execute_adb(adb_command)
        if result != "ERROR":
            match = re.search(r'[\w\.]+/[\w\.]+', result)
            if match:
                intent = match.group(0)
                result = intent
        return result
    
    def getcurfrag(self):
        curfrag = ""
        adb_command = f"adb -s {self.device} shell dumpsys activity top"
        result = execute_adb(adb_command)
        split = result.split('\n')
        fragflag = 0
        flag = 0
        for line in range(0, len(split)):
            try:
                if "Local FragmentActivity " in split[line]:
                    fragflag = 1
                if fragflag == 1 and "Added Fragments:" in split[line]:
                    flag = 1
                if flag == 1 and fragflag == 1 and "#0:" in split[line]:
                    try:
                        name = split[line].split('#' + str(0) + ": ")[1].split("{")[0]
                        id = split[line].split('id=')[1].split("}")[0]
                        curfrag = f'Fragment(id:{id} name:{name})'
                        break
                    except:
                        pass
            except:
                pass
        return curfrag
    
    def start_app(self, intent):
        print(f"start app {intent}")
        adb_command = f"adb -s {self.device} shell am start -n {intent}"
        result = execute_adb(adb_command)
        time.sleep(5)
        if result != "ERROR":
            return True
        # 通过intent启动失败，尝试通过包名启动
        package_name = get_package(intent)
        adb_command_package = f"adb -s {self.device} shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
        result = execute_adb(adb_command_package)  # 此时再出现异常则抛出到外层，不在本函数处理
        time.sleep(5)
        if result != "ERROR":
            return True
        print(f"start app {intent} failed")
        return False

    def kill_app(self, intent):
        package_name = get_package(intent)
        command = f"adb -s {self.device} shell am force-stop {package_name}"
        result = execute_adb(command)
        if result != "ERROR":
            print(f"kill app {package_name}")
            return True
        return False

    def back(self):
        adb_command = f"adb -s {self.device} shell input keyevent KEYCODE_BACK"
        ret = execute_adb(adb_command)
        return ret
    
    def home(self):
        adb_command = f"adb -s {self.device} shell input keyevent KEYCODE_HOME"
        ret = execute_adb(adb_command)
        return ret

    def click(self, x, y):
        return self.tap(x, y)
    
    def tap(self, x, y):
        adb_command = f"adb -s {self.device} shell input tap {x} {y}"
        ret = execute_adb(adb_command)
        return ret

    def backspace(self):
        adb_command = f"adb -s {self.device} shell input keyevent KEYCODE_DEL"
        subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    def input(self, input_str):
        input_str = input_str.replace(" ", "%s")
        input_str = input_str.replace("'", "")
        adb_command = f"adb -s {self.device} shell input text '{input_str}'"
        adb_command = f"adb -s {self.device} shell am broadcast -a ADB_INPUT_TEXT --es msg '{input_str}'"  # 需要启动 adb keyboard
        ret = execute_adb(adb_command)
        return ret

    def text(self, input_str, x, y):
        input_str = input_str.replace(" ", "%s")
        input_str = input_str.replace("'", "")
        self.tap(x, y)
        for i in range(10):
            self.backspace()
        # adb_command = f"adb -s {self.device} shell am broadcast -a ADB_INPUT_TEXT --es msg '{input_str}'"
        adb_command = f"adb -s {self.device} shell input text '{input_str}'"
        ret = execute_adb(adb_command)
        return ret

    def long_click(self, x, y, duration=1000):
        adb_command = f"adb -s {self.device} shell input swipe {x} {y} {x} {y} {duration}"
        ret = execute_adb(adb_command)
        return ret
    
    def scroll(self, start_point, end_point, duration=400):
        start_x, start_y = start_point
        end_x, end_y = end_point
        adb_command = f"adb -s {self.device} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}"
        ret = execute_adb(adb_command)
        return ret

    def swipe_precise(self, start, end, duration=400):
        start_x, start_y = start
        end_x, end_y = end
        adb_command = f"adb -s {self.device} shell input swipe {start_x} {start_x} {end_x} {end_y} {duration}"
        ret = execute_adb(adb_command)
        return ret
    
    
