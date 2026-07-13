import random
import re
from collections import defaultdict

import numpy as np

from action_router.constants import LABEL2ID
from action_router.features import render_granite_sample


EXPLORATION_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
_HANGUL_RE = re.compile(r"[가-힣]")


PROMPT_TEMPLATES = {
    "read_file": [
        "open {path} so I can check the implementation",
        "{path} 파일 내용 먼저 확인해줘",
        "read {path} and tell me what it does",
        "{path} 열어서 현재 로직이 어떻게 되어 있는지 봐줘",
        "pull up {path}; I need to inspect the exact code",
        "{path} 내용을 보고 다음 수정 범위를 판단하자",
        "can you show me the contents of {path}?",
        "{path} 한번 읽어보고 구조를 알려줘",
    ],
    "grep_search": [
        "search the repo for usages of {symbol}",
        "{symbol} 어디에서 쓰이는지 전체 검색해줘",
        "find every reference to {symbol} before we edit it",
        "{symbol} 호출하는 곳들을 찾아봐",
        "grep for {symbol} across the project",
        "{symbol} 문자열이 들어간 코드 위치를 찾아줘",
        "where is {symbol} defined or imported?",
        "{symbol} 관련된 부분이 흩어져 있는지 검색해줘",
    ],
    "list_directory": [
        "list what is inside {dir}",
        "{dir} 디렉터리에 뭐가 있는지 보여줘",
        "show me the files under {dir}",
        "{dir} 폴더 구조부터 확인하자",
        "what entries are in {dir}?",
        "{dir} 안에 어떤 파일들이 있는지 봐줘",
        "scan the directory listing for {dir}",
        "{dir} 목록만 먼저 확인해줘",
    ],
    "glob_pattern": [
        "find files matching {pattern}",
        "{pattern} 패턴에 맞는 파일들을 찾아줘",
        "glob for {pattern} so we know the candidates",
        "{pattern} 파일이 어디 있는지 찾아봐",
        "show every path that matches {pattern}",
        "{pattern} 매칭되는 경로들을 모아줘",
        "use a file pattern search for {pattern}",
        "{pattern} 기준으로 관련 파일을 찾아줘",
    ],
    "run_bash": [
        "run {command} and show me the output",
        "{command} 명령 한번 실행해줘",
        "execute {command} so we can see what happens",
        "{command} 돌려서 현재 상태 확인해줘",
    ],
    "run_tests": [
        "run the test suite",
        "테스트 한번 돌려봐",
        "execute the focused tests for this change",
        "pytest로 회귀 없는지 확인해줘",
        "run npm test and report failures",
    ],
    "lint_or_typecheck": [
        "run the type checker",
        "타입 검사만 한번 돌려줘",
        "run the linter for this package",
        "mypy/ruff 쪽 정적 검사 확인해줘",
        "tsc로 타입 깨지는지 봐줘",
    ],
    "ask_user": [
        "before changing code, ask me which option I prefer",
        "수정 전에 어떤 방향이 맞는지 나한테 먼저 물어봐줘",
        "confirm the intended behavior with me before proceeding",
    ],
    "plan_task": [
        "break this into a step by step plan before touching files",
        "코드 수정 전에 작업 계획부터 세워줘",
        "help me decide the implementation order for this change",
    ],
    "web_search": [
        "look up the official docs for this behavior",
        "이 기능의 최신 문서나 사례를 웹에서 찾아봐줘",
        "search the web for the recommended approach",
    ],
}


TRANSLATION_PROMPT_TEMPLATES = {
    "read_file": [
        "open {path} and show me the exact contents first",
        "read {path}; I need to understand the current implementation before changing anything",
        "show me {path} so I can inspect the code directly",
        "pull up {path}; let's look at the file itself first",
        "display {path} and check how this is currently written",
        "open the file {path}; I want to verify the details in place",
    ],
    "grep_search": [
        "search the whole repo for references to {symbol}",
        "find every place that uses {symbol}",
        "grep for {symbol} across the project before we edit it",
        "look for all callers and definitions related to {symbol}",
        "search where {symbol} is mentioned or imported",
        "find usages of {symbol} in the codebase",
    ],
    "list_directory": [
        "list the entries under {dir} first",
        "show me the directory structure for {dir}",
        "what files and folders are inside {dir}?",
        "scan the contents of the {dir} directory",
        "list what is at {dir} so I can orient myself",
        "show the folder listing for {dir}",
    ],
    "glob_pattern": [
        "find files matching {pattern}",
        "show every path that matches the pattern {pattern}",
        "glob for {pattern} and list the matching files",
        "locate candidate files using {pattern}",
        "find all filenames that match {pattern}",
        "use a file pattern search for {pattern}",
    ],
}


SLOTS = {
    "path": [
        "src/auth.ts",
        "app/router.py",
        "README.md",
        "package.json",
        "cmd/root.go",
        "config.yaml",
        "tests/test_auth.py",
        "src/components/Header.tsx",
    ],
    "symbol": [
        "useAuth",
        "ClientSession",
        "fetchUser",
        "Pipeline",
        "validateToken",
        "createRouter",
        "USER_ID",
        "timeout",
    ],
    "dir": [
        "src",
        "tests",
        "app",
        "scripts",
        "cmd",
        "config",
        "components",
        "migrations",
    ],
    "pattern": [
        "*.py",
        "**/*_test.go",
        "src/**/*.tsx",
        "**/Dockerfile",
        "tests/test_*.py",
        "**/*.yaml",
        "**/auth*.ts",
        "scripts/*.sh",
    ],
    "command": [
        "npm run build",
        "python -m pytest tests/test_auth.py",
        "cargo check",
        "go test ./...",
        "python scripts/train.py --dry-run",
    ],
}


def _format_prompt(template, rng):
    values = {name: rng.choice(options) for name, options in SLOTS.items()}
    return template.format(**values)


def _clone_with_prompt(sample, prompt):
    cloned = dict(sample)
    cloned["current_prompt"] = prompt
    return cloned


def _looks_korean_prompt(sample):
    return bool(_HANGUL_RE.search(sample.get("current_prompt", "")))


def build_augmented_granite_examples(
    samples,
    label_names,
    source_indices,
    max_history_events,
    open_files_mode="count",
    classes=None,
    max_aug_per_class=1000,
    seed=42,
    mode="contrastive",
):
    """Create label-preserving synthetic train-only examples.

    The generated examples copy metadata/history from real train-fold samples
    of the same class and replace only the current prompt with class-specific
    high-precision templates. This keeps validation leakage out while making
    the most confused action boundaries more explicit.
    """
    rng = random.Random(seed)
    classes = list(classes or EXPLORATION_CLASSES)
    by_label = defaultdict(list)
    for idx in source_indices:
        label = label_names[int(idx)]
        if label in classes and (mode != "translation" or _looks_korean_prompt(samples[int(idx)])):
            by_label[label].append(int(idx))

    texts = []
    y = []
    stats = {}
    for label in classes:
        source_pool = by_label.get(label) or []
        templates = TRANSLATION_PROMPT_TEMPLATES[label] if mode == "translation" else PROMPT_TEMPLATES[label]
        if not source_pool:
            stats[label] = 0
            continue

        count = min(int(max_aug_per_class), len(source_pool) * len(templates))
        stats[label] = count
        for _ in range(count):
            sample = samples[rng.choice(source_pool)]
            prompt = _format_prompt(rng.choice(templates), rng)
            augmented = _clone_with_prompt(sample, prompt)
            texts.append(
                render_granite_sample(
                    augmented,
                    max_history_events=max_history_events,
                    open_files_mode=open_files_mode,
                )
            )
            y.append(LABEL2ID[label])

    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    y = np.array([y[i] for i in order], dtype=np.int64)
    return texts, y, stats
