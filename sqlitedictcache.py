from collections import OrderedDict
from collections.abc import Callable
from typing import Final, Generic, Hashable, NamedTuple, Optional, ParamSpec, TypeVar, Union, Any
from logging import getLogger
import datetime
import functools
import asyncio

T = TypeVar("T")
P = ParamSpec("P")


# sqlitedict cache
import sqlitedict
import json


class CacheEntry(NamedTuple):
    value: T
    expiry: datetime.datetime


class _SQLiteDictCacheFunctionWrapper(Generic[P, T]):
    def __init__(self, func: Callable[P, T], basename: str, expiry_days: Optional[float] = 7.0):
        self.__wrapped__ = func
        self.__basename = basename
        self.__expiry_days = expiry_days
        self.__logger = getLogger(__name__)
        self.__logger.debug(f"Initialized cache wrapper for {basename} with expiry_days={expiry_days}")

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        logger = self.__logger
        call_args = json.dumps(args + tuple(kwargs.items()))
        logger.debug(f"Function {self.__basename} called with args: {call_args}")
        
        cache_file = f"{self.__basename}.sqlite"
        logger.debug(f"Opening cache file: {cache_file}")
        try:
            with sqlitedict.open(cache_file) as shelf:
                logger.debug(f"Successfully opened cache file: {cache_file}")
                now = datetime.datetime.now()
                logger.debug(f"Current time: {now}")
                
                # キャッシュに存在するか確認
                if call_args in shelf:
                    cache_data = shelf[call_args]
                    logger.debug(f"Cache entry found for {self.__basename}")
                    
                    # 古いキャッシュ（単純な値）か新しいキャッシュ（CacheEntry）かを判定
                    if isinstance(cache_data, CacheEntry):
                        # 新しいキャッシュ形式
                        logger.debug(f"New cache format detected, expiry: {cache_data.expiry}")
                        if cache_data.expiry > now:
                            logger.debug(f"Cache hit for {self.__basename}, returning cached value")
                            return cache_data.value
                    else:
                        # 古いキャッシュ形式（単純な値）
                        logger.debug(f"Legacy cache format detected for {self.__basename}")
                        # 古いキャッシュを新しい形式に変換
                        expiry = now + datetime.timedelta(days=self.__expiry_days)
                        new_cache_entry = CacheEntry(value=cache_data, expiry=expiry)
                        shelf[call_args] = new_cache_entry
                        shelf.commit()
                        logger.debug(f"Legacy cache converted to new format with expiry {expiry}")
                        return cache_data
                else:
                    logger.debug(f"Cache miss for {self.__basename}")
                
                # キャッシュにないか期限切れの場合、関数を実行
                logger.debug(f"Executing function {self.__basename}")
                ret = self.__wrapped__(*args, **kwargs)
                if ret is None:
                    logger.info(f"Cache for {self.__basename} prevents storing None.")
                else:
                    # 有効期限を設定
                    expiry = now + datetime.timedelta(days=self.__expiry_days)
                    cache_entry = CacheEntry(value=ret, expiry=expiry)
                    shelf[call_args] = cache_entry
                    shelf.commit()
                    logger.debug(f"Cache set for {self.__basename} with expiry {expiry}")
                
                return ret
        except Exception as e:
            logger.error(f"Error opening cache file: {cache_file}, {e}")
            return self.__wrapped__(*args, **kwargs)


def sqlitedict_cache(
    basename: str,
    expiry_days: Optional[float] = 7.0,
) -> Callable[[Callable[P, T]], _SQLiteDictCacheFunctionWrapper[P, T]]:
    def decorator(func: Callable[P, T]) -> _SQLiteDictCacheFunctionWrapper[P, T]:
        wrapped = _SQLiteDictCacheFunctionWrapper(func, basename, expiry_days)
        wrapped.__doc__ = func.__doc__
        return wrapped
    return decorator


def cache_if_not_none(func):
    logger = getLogger(__name__)
    logger.debug(f"Creating cache_if_not_none wrapper for {func.__name__}")
    
    @functools.wraps(func)
    @sqlitedict_cache
    def wrapper(*args, **kwargs):
        logger.debug(f"cache_if_not_none wrapper called for {func.__name__}")
        result = func(*args, **kwargs)
        if result is None:
            logger.debug(f"Result is None, removing from cache for {func.__name__}")
            wrapper.__wrapped__.__cache__.pop(
                (args, tuple(kwargs.items())), None
            )  # キャッシュから削除
        return result

    return wrapper


# デコレータを適用する前に、関数を定義
def fib(n):
    return 1 if n in (0, 1) else fib(n - 1) + fib(n - 2)



if __name__ == "__main__":
    # ログレベルをDEBUGに設定
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print(fib(40))
