#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rules_v2.py — 统一定义扰动系统的决策逻辑。

支持四大类扰动系统：
independent: A和P独立判断（如 A_none + P_animate） 
dualP: P双控系统（如 P同时是 [+animate, +definite] 才加）
global: A/P统一判断（如 A的animacy比P小就加）
debug: 用于测试或特殊自定义

标签体系采用新版本：
    animacy: ["inanimate", "animate"]
    nptype: ["common", "pronoun"]
    definiteness: ["indef", "definite"]
"""

from typing import Dict

# ------------------------------
# 层级定义（新体系）
# ------------------------------

ANIMACY_HIER = ["inanimate", "animate"]
NPTYPE_HIER = ["common", "pronoun"]
DEFINITENESS_HIER = ["indef", "definite"]

HIERARCHY = {
    "animacy": ANIMACY_HIER,
    "nptype": NPTYPE_HIER,
    "definiteness": DEFINITENESS_HIER,
}

# ------------------------------
# 工具函数
# ------------------------------

def rank(label: str, hierarchy: list[str]) -> int:
    try:
        return hierarchy.index(label)
    except ValueError:
        return -1


def compare_labels(a_label: str, p_label: str, hierarchy: list[str]) -> int:
    """返回 a_label 相对 p_label 的层级关系。
    -1 = A < P
     0 = A == P
     1 = A > P
    """
    a_r = rank(a_label, hierarchy)
    p_r = rank(p_label, hierarchy)
    if a_r < 0 or p_r < 0:
        return 0
    if a_r < p_r:
        return -1
    elif a_r > p_r:
        return 1
    else:
        return 0


# ------------------------------
# 1️⃣ Independent 系统
# ------------------------------

def independent_rule(A_labels: Dict[str, str], P_labels: Dict[str, str],
                     A_mode: str, P_mode: str, inverse: bool = False) -> Dict[str, bool]:
    """
    A 和 P 各自独立判断。
    例如：
        A_mode = 'animate', P_mode = 'none'
        表示 A 不满足 animacy=animate 时就加标记。
    """
    res = {"A_mark": False, "P_mark": False}

    # ----- A -----
    if A_mode != "none":
        hier = _hierarchy_of(A_mode)
        cutoff = A_mode
        label = A_labels.get(hier)
        if label is not None:
            res["A_mark"] = (label != cutoff) if not inverse else (label == cutoff)

    # ----- P -----
    if P_mode != "none":
        hier = _hierarchy_of(P_mode)
        cutoff = P_mode
        label = P_labels.get(hier)
        if label is not None:
            res["P_mark"] = (label == cutoff) if not inverse else (label != cutoff)

    return res


# ------------------------------
# 2️⃣ DualP 系统
# ------------------------------

def dualP_rule(P_labels: Dict[str, str],
               combo: str = "and", inverse: bool = False) -> bool:
    """
    双控系统（P 自身两个维度控制）
    combo: "and" / "or"
    inverse: 是否反向（不加）
    """
    is_anim = P_labels.get("animacy") == "animate"
    is_def  = P_labels.get("definiteness") == "definite"

    if combo == "and":
        cond = is_anim and is_def
    elif combo == "or":
        cond = is_anim or is_def
    else:
        raise ValueError(f"Invalid combo: {combo}")

    return not cond if inverse else cond


# ------------------------------
# 3️⃣ Global 系统
# ------------------------------

def global_rule(A_labels: Dict[str, str], P_labels: Dict[str, str],
                feature: str, direction: str = "up") -> Dict[str, bool]:
    """
    全局系统：A/P 比较统一。
    direction:
        "up"   = A < P 时加
        "down" = A > P 时加
    """
    hier = HIERARCHY[feature]
    cmp_res = compare_labels(A_labels.get(feature), P_labels.get(feature), hier)

    res = {"A_mark": False, "P_mark": False}
    if direction == "up":
        should_mark = cmp_res == -1
    elif direction == "down":
        should_mark = cmp_res == 1
    else:
        raise ValueError(f"Invalid direction: {direction}")

    res["A_mark"] = should_mark
    res["P_mark"] = should_mark
    return res


# ------------------------------
# 工具函数：确定层级类型
# ------------------------------

def _hierarchy_of(mode: str) -> str:
    if mode in {"animate", "inanimate"}:
        return "animacy"
    elif mode in {"pronoun", "common"}:
        return "nptype"
    elif mode in {"definite", "indef"}:
        return "definiteness"
    elif mode == "none":
        return None
    else:
        raise ValueError(f"Invalid mode: {mode}")


# ------------------------------
# 快捷调用接口（供 run_perturb_v2 使用）
# ------------------------------

def apply_rule(system: str, A_labels: Dict[str, str], P_labels: Dict[str, str],
               A_mode: str = "none", P_mode: str = "none",
               inverse: bool = False, feature: str = None,
               combo: str = None, direction: str = None) -> Dict[str, bool]:
    """
    统一接口：
        system = independent / dualP / global
    返回 {"A_mark": bool, "P_mark": bool}
    """
    if system == "independent":
        return independent_rule(A_labels, P_labels, A_mode, P_mode, inverse)
    elif system == "dualP":
        mark_P = dualP_rule(P_labels, combo or "and", inverse)
        return {"A_mark": False, "P_mark": mark_P}
    elif system == "global":
        return global_rule(A_labels, P_labels, feature, direction or "up")
    else:
        raise ValueError(f"Unknown system: {system}")
