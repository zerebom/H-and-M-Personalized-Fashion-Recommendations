import sys
from functools import wraps
from logging import (DEBUG, INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger)
from typing import Optional


def set_logger(
    name: str, slevel: int = DEBUG, flevel: int = INFO, fname: Optional[str] = None
) -> Logger:
    """ロガーを生成する

    Args:
        name (str): getLogger(name)として利用
        slevel (int, optional): streamhandler level. Defaults to INFO.
        flevel (int, optional): ログに出力するstreamhandler level. Defaults to DEBUG.
        fname (Optional[str], optional): 指定すれば， 指定先にlogファイルを出力する. Defaults to None.

    Returns:
        Logger: logger
    """
    logger = getLogger(name)

    if len(logger.handlers) == 0:
        # 出力のフォーマッタを作成
        formatter = Formatter(
            fmt=" %(name)s - %(levelname)s - %(message)s"
        )
        # 標準出力(コンソール)にログを出力するハンドラを生成
        sh = StreamHandler(sys.stdout)
        sh.setLevel(slevel)

        # フォーマッタをハンドラに紐づけ
        sh.setFormatter(formatter)

        # ハンドラをロガーに紐づけ
        logger.setLevel(DEBUG)
        logger.addHandler(sh)

        if fname is None:
            return logger

        # ----以降は、ログをファイルとして出力したい場合----
        # ログをファイルとして出力するハンドラを生成
        fh = FileHandler(fname)
        fh.setLevel(flevel)
        fh.setFormatter(formatter)

        # ハンドラをロガーに紐づけ
        logger.addHandler(fh)

    return logger


def logging_function(logger: Logger):
    def _logging_function(func):
        """関数が利用されたことをloggingするデコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    return _logging_function