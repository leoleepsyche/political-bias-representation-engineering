"""
Unified Dataset Loader for Political Bias Experiments

支持多个数据源:
1. Custom Political Dataset (expanded: 50 topics)
2. OpinionQA (PEW American Trends Panel)
3. P-Stance (Twitter political stance)
4. Non-Political Baseline

设计原则:
- 统一接口: DatasetLoader 类
- 灵活配置: 支持混合使用不同数据源
- 可追踪: 记录数据来源和版本
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class DataSourceType(Enum):
    """数据源类型"""
    CUSTOM_POLITICAL = "custom_political"
    OPINIONQA = "opinionqa"
    PSTANCE = "pstance"
    NONPOLITICAL = "nonpolitical"


@dataclass
class DataItem:
    """单个数据项"""
    topic: str
    text: str
    source: DataSourceType
    stance: Optional[str] = None  # "left", "right", "neutral", "political", "non_political"
    metadata: Optional[Dict] = None


class DatasetLoader:
    """统一数据集加载器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据集加载器

        Args:
            config: 配置字典，包含:
                - data_dir: 数据目录
                - use_custom_political: 是否使用自建政治数据集
                - use_opinionqa: 是否使用 OpinionQA
                - use_pstance: 是否使用 P-Stance
                - use_nonpolitical: 是否使用非政治数据
                - max_samples: 每个源的最大样本数
        """
        self.config = config or {}
        self.data_dir = Path(self.config.get("data_dir", "."))
        self.use_custom = self.config.get("use_custom_political", True)
        self.use_opinionqa = self.config.get("use_opinionqa", False)
        self.use_pstance = self.config.get("use_pstance", False)
        self.use_nonpolitical = self.config.get("use_nonpolitical", True)

        self.items: List[DataItem] = []
        self.metadata = {}

    def load_custom_political(self, use_expanded: bool = True) -> List[DataItem]:
        """加载自建政治数据集"""
        try:
            if use_expanded:
                from political_dataset_expanded import (
                    get_left_statements,
                    get_right_statements,
                    get_prompt_template,
                )
            else:
                from political_dataset import (
                    get_left_statements,
                    get_right_statements,
                    get_prompt_template,
                )
        except ImportError as e:
            print(f"⚠️  Failed to load custom political dataset: {e}")
            return []

        items = []

        # Load left statements
        for topic, text in get_left_statements():
            items.append(DataItem(
                topic=topic,
                text=get_prompt_template(text),
                source=DataSourceType.CUSTOM_POLITICAL,
                stance="left",
                metadata={"original_text": text[:100]}
            ))

        # Load right statements
        for topic, text in get_right_statements():
            items.append(DataItem(
                topic=topic,
                text=get_prompt_template(text),
                source=DataSourceType.CUSTOM_POLITICAL,
                stance="right",
                metadata={"original_text": text[:100]}
            ))

        print(f"✅ Loaded {len(items)} custom political items (expanded={use_expanded})")
        self.metadata["custom_political_count"] = len(items)
        return items

    def load_nonpolitical(self) -> List[DataItem]:
        """加载非政治数据集（对照组）"""
        try:
            from nonpolitical_dataset import (
                get_nonpolitical_statements,
                get_prompt_template,
            )
        except ImportError as e:
            print(f"⚠️  Failed to load non-political dataset: {e}")
            return []

        items = []
        for topic, text in get_nonpolitical_statements():
            items.append(DataItem(
                topic=topic,
                text=get_prompt_template(text),
                source=DataSourceType.NONPOLITICAL,
                stance="non_political",
                metadata={"original_text": text[:100]}
            ))

        print(f"✅ Loaded {len(items)} non-political items")
        self.metadata["nonpolitical_count"] = len(items)
        return items

    def load_opinionqa(self, json_path: Optional[str] = None) -> List[DataItem]:
        """
        加载 OpinionQA 数据集

        Args:
            json_path: OpinionQA JSON 文件路径

        预期格式:
        [
            {
                "question": "Question text",
                "category": "political" | "non-political",
                "answers": {
                    "democrat": {"percentage": 65, ...},
                    "republican": {"percentage": 25, ...},
                    "overall": {"percentage": 50, ...}
                }
            },
            ...
        ]
        """
        if json_path is None:
            json_path = str(self.data_dir / "opinionqa" / "questions.json")

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  OpinionQA file not found: {json_path}")
            print("   Download from: https://github.com/tatsu-lab/opinions_qa")
            return []

        items = []
        for item in data:
            # 区分政治和非政治问题
            category = item.get("category", "unknown")

            # 从民主党/共和党回答比例推断政治倾向
            answers = item.get("answers", {})
            democrat_pct = answers.get("democrat", {}).get("percentage", 50)
            republican_pct = answers.get("republican", {}).get("percentage", 50)

            # 如果倾向明确，标记立场
            if category == "political":
                if democrat_pct > 60:
                    stance = "left"
                elif republican_pct > 60:
                    stance = "right"
                else:
                    stance = "neutral"
            else:
                stance = "non_political"

            items.append(DataItem(
                topic=category,
                text=item.get("question", ""),
                source=DataSourceType.OPINIONQA,
                stance=stance,
                metadata={
                    "democrat_pct": democrat_pct,
                    "republican_pct": republican_pct,
                    "category": category
                }
            ))

        print(f"✅ Loaded {len(items)} OpinionQA items")
        self.metadata["opinionqa_count"] = len(items)
        return items

    def load_pstance(self, csv_path: Optional[str] = None) -> List[DataItem]:
        """
        加载 P-Stance 数据集 (Twitter)

        Args:
            csv_path: P-Stance CSV 文件路径

        预期格式:
        text,target,stance
        "tweet text here","Trump","Favor"
        ...
        """
        if csv_path is None:
            csv_path = str(self.data_dir / "pstance" / "pstance_data.csv")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except (FileNotFoundError, ImportError) as e:
            print(f"⚠️  Failed to load P-Stance: {e}")
            print("   Download from: https://github.com/chuchun8/PStance")
            return []

        items = []
        target_to_stance = {
            "Trump": "right",
            "Biden": "left",
            "Sanders": "left"
        }

        for _, row in df.iterrows():
            text = row.get("text", "")
            target = row.get("target", "")
            stance_label = row.get("stance", "")

            # 只保留 Favor/Against 的清晰立场
            if stance_label not in ["Favor", "Against"]:
                continue

            # 将 target + stance 转为左右立场
            base_stance = target_to_stance.get(target)
            if stance_label == "Favor":
                stance = base_stance
            else:  # Against
                stance = "right" if base_stance == "left" else "left"

            items.append(DataItem(
                topic=f"pstance_{target}",
                text=text,
                source=DataSourceType.PSTANCE,
                stance=stance,
                metadata={
                    "target": target,
                    "stance_label": stance_label
                }
            ))

        print(f"✅ Loaded {len(items)} P-Stance items")
        self.metadata["pstance_count"] = len(items)
        return items

    def load_all(self,
                 use_expanded_custom: bool = True,
                 opinionqa_path: Optional[str] = None,
                 pstance_path: Optional[str] = None) -> List[DataItem]:
        """加载所有启用的数据源"""

        items = []

        if self.use_custom:
            items.extend(self.load_custom_political(use_expanded=use_expanded_custom))

        if self.use_nonpolitical:
            items.extend(self.load_nonpolitical())

        if self.use_opinionqa:
            items.extend(self.load_opinionqa(opinionqa_path))

        if self.use_pstance:
            items.extend(self.load_pstance(pstance_path))

        self.items = items
        print(f"\n📊 Total items loaded: {len(items)}")
        return items

    def get_by_stance(self, stance: str) -> List[DataItem]:
        """按立场过滤数据"""
        return [item for item in self.items if item.stance == stance]

    def get_by_source(self, source: DataSourceType) -> List[DataItem]:
        """按数据源过滤数据"""
        return [item for item in self.items if item.source == source]

    def get_by_topic(self, topic: str) -> List[DataItem]:
        """按话题过滤数据"""
        return [item for item in self.items if item.topic == topic]

    def get_topics(self) -> List[str]:
        """返回所有话题"""
        return list(set(item.topic for item in self.items))

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            "total_items": len(self.items),
            "sources": {},
            "stances": {},
            "topics": len(self.get_topics()),
            "metadata": self.metadata
        }

        # Count by source
        for source in DataSourceType:
            count = len(self.get_by_source(source))
            if count > 0:
                stats["sources"][source.value] = count

        # Count by stance
        for item in self.items:
            stance = item.stance or "unknown"
            stats["stances"][stance] = stats["stances"].get(stance, 0) + 1

        return stats

    def save_metadata(self, output_path: str = "dataset_metadata.json"):
        """保存数据集元数据"""
        metadata = {
            "config": self.config,
            "statistics": self.get_statistics(),
            "items_summary": [
                {
                    "topic": item.topic,
                    "source": item.source.value,
                    "stance": item.stance
                }
                for item in self.items[:100]  # 仅保存前100项示例
            ]
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to {output_path}")


def prompt_template(statement: str) -> str:
    """统一 prompt 模板"""
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n"
        f"### Instruction: {statement}\n"
        f"### Response:"
    )


# 快速使用示例
if __name__ == "__main__":
    # 配置
    config = {
        "use_custom_political": True,
        "use_nonpolitical": True,
        "use_opinionqa": False,  # 需要手动下载
        "use_pstance": False,     # 需要手动下载
    }

    # 加载
    loader = DatasetLoader(config)
    items = loader.load_all(use_expanded_custom=True)

    # 统计
    stats = loader.get_statistics()
    print("\n" + "="*50)
    print("📊 Dataset Statistics")
    print("="*50)
    print(json.dumps(stats, indent=2))

    # 按立场分组
    print("\n" + "="*50)
    print("By Stance")
    print("="*50)
    for stance in ["left", "right", "non_political", "neutral"]:
        count = len(loader.get_by_stance(stance))
        if count > 0:
            print(f"  {stance}: {count} items")

    # 示例项
    print("\n" + "="*50)
    print("Sample Items")
    print("="*50)
    for item in items[:3]:
        print(f"\nTopic: {item.topic}")
        print(f"Source: {item.source.value}")
        print(f"Stance: {item.stance}")
        print(f"Text: {item.text[:80]}...")
