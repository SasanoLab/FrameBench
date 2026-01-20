import os
import json
import yaml
from typing import List, Dict, Optional, Tuple

class DatasetManager:
    """データセットファイルの管理を行うクラス"""
    
    def __init__(self, config_file: str = "src/dataset_config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {self.config_file}")
            return self._get_default_config()
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            "datasets": {
                "manual_datasets": []
            }
        }
    
    def find_qa_datasets(self) -> List[Dict[str, str]]:
        """configで手動指定されたQAデータセットファイルの一覧を返す"""
        datasets = []
        
        # 手動で指定されたデータセットのみを追加
        manual_datasets = self.config.get("datasets", {}).get("manual_datasets", [])
        for dataset_config in manual_datasets:
            if dataset_config.get("enabled", True):
                path = dataset_config["path"]
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    if self._is_qa_dataset(path):
                        datasets.append({
                            "name": dataset_config["name"],
                            "path": path,
                            "relative_path": os.path.relpath(path, "data") if path.startswith("data/") else path,
                            "size": os.path.getsize(path),
                            "line_count": self._count_lines(path),
                            "description": dataset_config.get("description", ""),
                            "source": "manual"
                        })
        
        # パスでソート
        return sorted(datasets, key=lambda x: x["relative_path"])
    
    def _is_qa_dataset(self, file_path: str) -> bool:
        """ファイルがQAデータセットかどうかを判定"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 最初の数行を読んでQAデータかどうか判定
                for i, line in enumerate(f):
                    if i >= 3:  # 最初の3行をチェック
                        break
                    try:
                        data = json.loads(line.strip())
                        # QAデータの特徴的なキーをチェック
                        if any(key in data for key in ["question", "result", "sentence_pair", "lex_unit_name"]):
                            return True
                    except json.JSONDecodeError:
                        continue
            return False
        except Exception:
            return False
    
    def _count_lines(self, file_path: str) -> int:
        """ファイルの行数をカウント"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def add_manual_dataset(self, name: str, path: str, description: str = "", enabled: bool = True) -> bool:
        """手動データセットを追加"""
        try:
            # 設定を再読み込み
            self.config = self._load_config()
            
            # 新しいデータセットを追加
            new_dataset = {
                "name": name,
                "path": path,
                "description": description,
                "enabled": enabled
            }
            
            if "datasets" not in self.config:
                self.config["datasets"] = {}
            if "manual_datasets" not in self.config["datasets"]:
                self.config["datasets"]["manual_datasets"] = []
            
            self.config["datasets"]["manual_datasets"].append(new_dataset)
            
            # 設定ファイルに保存
            self._save_config()
            return True
        except Exception as e:
            print(f"データセットの追加に失敗: {e}")
            return False
    
    def remove_manual_dataset(self, path: str) -> bool:
        """手動データセットを削除"""
        try:
            # 設定を再読み込み
            self.config = self._load_config()
            
            manual_datasets = self.config.get("datasets", {}).get("manual_datasets", [])
            self.config["datasets"]["manual_datasets"] = [
                d for d in manual_datasets if d.get("path") != path
            ]
            
            # 設定ファイルに保存
            self._save_config()
            return True
        except Exception as e:
            print(f"データセットの削除に失敗: {e}")
            return False
    
    def _save_config(self):
        """設定ファイルに保存"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            print(f"設定ファイルの保存に失敗: {e}")
    
    def get_config_info(self) -> Dict:
        """設定情報を取得"""
        return {
            "config_file": self.config_file,
            "manual_datasets_count": len(self.config.get("datasets", {}).get("manual_datasets", []))
        }
    
    def get_dataset_info(self, dataset_path: str) -> Optional[Dict]:
        """指定されたデータセットの詳細情報を取得"""
        if not os.path.exists(dataset_path):
            return None
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    sample_data = json.loads(first_line)
                    return {
                        "path": dataset_path,
                        "size": os.path.getsize(dataset_path),
                        "line_count": self._count_lines(dataset_path),
                        "sample_data": sample_data,
                        "keys": list(sample_data.keys()) if isinstance(sample_data, dict) else []
                    }
        except Exception:
            pass
        
        return None
    
    def validate_dataset(self, dataset_path: str) -> Tuple[bool, str]:
        """データセットファイルの妥当性を検証"""
        if not os.path.exists(dataset_path):
            return False, "ファイルが存在しません"
        
        if os.path.getsize(dataset_path) == 0:
            return False, "ファイルが空です"
        
        try:
            valid_lines = 0
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        # 基本的なQAデータの構造をチェック
                        if not isinstance(data, dict):
                            return False, f"行 {line_num}: JSONオブジェクトではありません"
                        
                        # 必須フィールドのチェック（柔軟に）
                        if not any(key in data for key in ["question", "result", "sentence_pair"]):
                            return False, f"行 {line_num}: QAデータの必須フィールドが見つかりません"
                        
                        valid_lines += 1
                        
                    except json.JSONDecodeError as e:
                        return False, f"行 {line_num}: JSON解析エラー - {str(e)}"
            
            if valid_lines == 0:
                return False, "有効なデータ行が見つかりません"
            
            return True, f"検証成功: {valid_lines}行の有効なデータ"
            
        except Exception as e:
            return False, f"ファイル読み込みエラー: {str(e)}"

# グローバルインスタンス
dataset_manager = DatasetManager()
