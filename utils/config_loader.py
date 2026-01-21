import json
from pathlib import Path


class ConfigLoader:
    # 配置文件加载器，单例模式
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"config.json格式错误: {e}")
    
    def get(self, key, default=None):
        # key: 配置键名，支持点号访问嵌套值，如 'database.host'
        # default: 键不存在时的默认值
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_all(self):
        return self._config
    
    def __getitem__(self, key):
        """支持字典式访问: config['apks_dir']"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"配置键不存在: {key}")
        return value


# 创建全局配置实例
config = ConfigLoader()
