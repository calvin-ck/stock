"""
core.py — 네이버 금융에서 특정 종목의 일별 시세를 가져오는 핵심 모듈.

쉘에서 직접 실행하여 테스트할 수 있습니다.

사용 예:
    python core.py 005930 --days 30
    python core.py 005930 -d 90 --csv out.csv
"""

import sys
import re
import time
import argparse
from io import StringIO
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

SISE_DAY_URL = "https://finance.naver.com/item/sise_day.naver"
MAIN_URL = "https://finance.naver.com/item/main.naver"


def _decode_response(resp: requests.Response) -> str:
    """
    페이지의 실제 charset을 감지해서 디코딩한다.
    네이버 페이지는 종목/ETF에 따라 euc-kr, utf-8이 섞여 있어서
    인코딩을 하드코딩하면 한글이 깨질 수 있다 (mojibake).
    """
    raw = resp.content
    m = re.search(rb'charset=["\']?([\w-]+)', raw[:2000], re.IGNORECASE)
    encoding = m.group(1).decode("ascii").lower() if m else "euc-kr"
    try:
        return raw.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        # 감지된 인코딩으로 실패하면 다른 인코딩으로 재시도
        for fallback in ("utf-8", "euc-kr", "cp949"):
            try:
                return raw.decode(fallback)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")


def fetch_page(code: str, page: int) -> pd.DataFrame:
    """
    네이버 금융 일별시세 페이지 1개를 가져와 DataFrame으로 반환.

    전일비 컬럼은 상승/하락을 이미지 아이콘(alt='상승'/'하락')으로 표시하는데,
    pd.read_html은 이미지의 alt 텍스트를 읽지 못해 부호가 소실되거나 NaN이 될 수 있다.
    이를 피하기 위해 BeautifulSoup으로 각 셀을 직접 파싱한다.
    """
    resp = requests.get(
        SISE_DAY_URL,
        params={"code": code, "page": page},
        headers=HEADERS,
        timeout=5,
    )
    resp.raise_for_status()
    text = _decode_response(resp)

    soup = BeautifulSoup(text, "lxml")
    table = soup.select_one("table.type2") or soup.find("table")
    if table is None:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 7:
            continue  # 헤더/구분용 빈 행 건너뜀

        date_text = tds[0].get_text(strip=True)
        if not date_text:
            continue  # 날짜 없는 빈 행(스페이서) 건너뜀

        change_cell = tds[2]
        change_text = change_cell.get_text(strip=True)
        img = change_cell.find("img")
        alt = (img.get("alt") if img else "") or ""

        rows.append({
            "날짜": date_text,
            "종가": tds[1].get_text(strip=True),
            "전일비": change_text,
            "전일비_방향": alt,  # '상승' / '하락' / '보합'
            "시가": tds[3].get_text(strip=True),
            "고가": tds[4].get_text(strip=True),
            "저가": tds[5].get_text(strip=True),
            "거래량": tds[6].get_text(strip=True),
        })

    return pd.DataFrame(rows)


def get_stock_name(code: str) -> str:
    """종목명 조회 (메인 페이지 title 태그 파싱)."""
    resp = requests.get(MAIN_URL, params={"code": code}, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    text = _decode_response(resp)
    m = re.search(r"<title>(.*?)</title>", text)
    if m:
        return m.group(1).replace(":네이버 증권", "").strip()
    return code


def _parse_change_column(text_series: pd.Series, direction_series: pd.Series) -> pd.Series:
    """
    '전일비' 숫자 텍스트와 '전일비_방향'(alt: 상승/하락/보합)을 조합해
    부호 있는 숫자로 변환한다.
    """

    def parse_one(text_val, direction_val):
        if pd.isna(text_val):
            return None
        s = str(text_val)
        m = re.search(r"[\d,]+(?:\.\d+)?", s)
        if not m:
            return None
        num = float(m.group(0).replace(",", ""))

        direction = str(direction_val) if not pd.isna(direction_val) else ""
        # alt 속성으로 우선 판단, 없으면 원본 텍스트 안에 방향 표시가 섞여 있는지로 판단
        if ("하락" in direction) or ("하락" in s) or ("↓" in s) or ("▼" in s):
            num = -num
        return num

    return pd.Series(
        [parse_one(t, d) for t, d in zip(text_series, direction_series)],
        index=text_series.index,
    )


def get_stock_data(code: str, days: int, sleep: float = 0.2) -> pd.DataFrame:
    """
    오늘부터 `days`일 전까지의 일별 시세를 가져온다.

    Parameters
    ----------
    code : str
        종목 코드 (예: '005930' 삼성전자)
    days : int
        오늘부터 몇 일 전까지 데이터를 가져올지
    sleep : float
        페이지 요청 사이 대기 시간 (과도한 요청 방지용)

    Returns
    -------
    pd.DataFrame  columns=[날짜, 시가, 종가, 전일비, 고가, 저가, 거래량]
        최신 날짜가 맨 위로 정렬됨.
    """
    cutoff = datetime.now() - timedelta(days=days)

    all_rows = []
    page = 1
    max_pages = 500  # 무한루프 방지용 안전장치

    while page <= max_pages:
        df = fetch_page(code, page)
        if df.empty:
            break

        df["날짜"] = pd.to_datetime(df["날짜"], format="%Y.%m.%d")
        all_rows.append(df)

        oldest_in_page = df["날짜"].min()
        if oldest_in_page <= cutoff:
            break

        page += 1
        time.sleep(sleep)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result[result["날짜"] >= cutoff]
    result = result.drop_duplicates(subset="날짜")
    result = result.sort_values("날짜", ascending=False).reset_index(drop=True)

    for col in ["종가", "시가", "고가", "저가", "거래량"]:
        result[col] = pd.to_numeric(
            result[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    result["전일비"] = _parse_change_column(result["전일비"], result["전일비_방향"])
    result = result.drop(columns=["전일비_방향"])
    result = result[["날짜", "시가", "종가", "전일비", "고가", "저가", "거래량"]]

    return result


def _simulate_grid(
    prices: list,
    dates,
    sell_gap_percent: float,
    buy_gap_percent: float,
    trade_qty: int,
    initial_shares: int,
    sell_only: bool = False,
    allow_negative_cash: bool = False,
    record_log: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
) -> dict:
    """
    그리드 매매 전략의 실제 시뮬레이션 루프 (트레일링 고점/저점 방식).
    grid_trade_strategy()와 _run_grid_fast() 둘 다 이 함수를 사용해서 로직이 어긋나지 않게 한다.

    옵션
    ----
    sell_only : True면 매도만 하고 매수는 절대 하지 않는다.
    allow_negative_cash : True면 매수 시 현금 잔고를 확인하지 않고 trade_qty를 그대로 매수한다
        (초창기 모델 방식 — 현금이 마이너스가 될 수 있음).
        False(기본)면 "쌓인 현금으로 살 수 있는 만큼"과 trade_qty 중 작은 값만큼만 매수한다.
    record_log : True면 매매일지를 기록해서 반환한다 (느림). False면 최종 결과만 계산한다 (빠름,
        히트맵처럼 수천~수만 번 반복 계산할 때 사용).
    profit_gap_percent, profit_recover_percent : 둘 다 지정하면 "이익 회수"(백테스트2) 기능이
        추가로 동작한다. grid_trade_strategy()의 해당 파라미터 설명 참고. 하나만 None이 아니면
        호출자(grid_trade_strategy)가 이미 검증했다고 가정한다.
    """
    sell_gap_ratio = sell_gap_percent / 100.0
    buy_gap_ratio = buy_gap_percent / 100.0
    max_price = min_price = prices[0]
    prev_price = prices[0]
    shares = initial_shares
    cash = 0.0
    sell_count = 0
    buy_count = 0
    trade_log = [] if record_log else None

    profit_sweep = profit_gap_percent is not None
    if profit_sweep:
        profit_gap_ratio = profit_gap_percent / 100.0
        profit_recover_ratio = profit_recover_percent / 100.0
        profit_base_price = prices[0]  # 이익 회수 기준가도 max/min처럼 첫날 종가로 시작
        reserve = 0.0
        recover_count = 0
        recover_log = [] if record_log else None

    for i in range(1, len(prices)):
        price = prices[i]
        date = dates[i] if dates is not None else None

        # 최근 고점/저점 상시 갱신 (매매 발생 여부와 무관)
        if price > max_price:
            max_price = price
        if price < min_price:
            min_price = price

        # 이번 거래 판단에 쓰인 max/min 스냅샷 (트레이드로 리셋되기 전 값)
        max_snapshot = max_price
        min_snapshot = min_price

        # 이익 회수 이벤트 (백테스트2 전용): 매수/매도 판단(이하 sell/buy 로직)과는 완전히
        # 독립적으로, 매일 먼저 체크한다. 기준가(profit_base_price) 대비 profit_gap_percent%
        # 이상 오르면 발동 — 이때 "현금에서 떼어내는" 게 아니라, 필요한 만큼 주식을 직접
        # 매도해서 그 대금을 적립금으로 옮긴다 (일반 매도 로직의 max/횟수와는 무관).
        if profit_sweep and price >= profit_base_price * (1 + profit_gap_ratio):
            base_snapshot = profit_base_price
            profit = shares * (price - base_snapshot)
            if profit > 0 and shares > 0:
                # 회수해야 할 금액(평가차익 x 회수율)만큼을 현재가로 환산해 몇 주를 팔지 정한다.
                # 주식 수는 정수 단위라 목표 금액을 소수 없이 채우지 못할 수 있어 내림 처리한다.
                target_value = profit * profit_recover_ratio
                recover_shares = min(shares, int(target_value // price))
                if recover_shares > 0:
                    recovered = recover_shares * price
                    shares -= recover_shares
                    reserve += recovered
                    recover_count += 1
                    if record_log:
                        recover_log.append({
                            "날짜": date, "구분": "이익회수", "기준가": base_snapshot,
                            "가격": price, "평가차익": profit, "매도주식수": recover_shares,
                            "회수액": recovered, "적립금잔고": reserve, "현금잔고": cash,
                            "보유주식수": shares, "주식평가금액": price * shares,
                        })
            # 회수 여부와 무관하게, 조건이 충족되면 기준가를 그 시점 주가로 갱신한다.
            # (그렇지 않으면 오래된 기준가가 계속 남아 이후 작은 상승에도 이벤트가 반복 발동한다.)
            profit_base_price = price

        # 매도: max 대비 sell_gap% 하락, 단 전날보다 내려간 날에만
        if (
            price <= max_snapshot * (1 - sell_gap_ratio)
            and shares >= trade_qty
            and price < prev_price
        ):
            shares -= trade_qty
            cash += trade_qty * price
            max_price = price
            sell_count += 1
            if record_log:
                trade_log.append({
                    "날짜": date, "구분": "매도", "가격": price, "수량": trade_qty,
                    "현금잔고": cash, "보유주식수": shares,
                    "max": max_snapshot, "min": min_snapshot,
                    "등락률": (price - max_snapshot) / max_snapshot * 100,
                    "주식평가금액": price * shares,
                })

        # 매수: min 대비 buy_gap% 상승, 단 전날보다 올라간 날에만 (sell_only면 아예 건너뜀)
        if (not sell_only) and price >= min_snapshot * (1 + buy_gap_ratio) and price > prev_price:
            if allow_negative_cash:
                # 초창기 모델: 현금 부족 여부와 무관하게 trade_qty를 그대로 매수 (현금 마이너스 허용)
                buy_qty = trade_qty
            else:
                affordable_qty = int(cash // price) if price > 0 else 0
                buy_qty = min(trade_qty, affordable_qty)

            # 매수 신호가 뜨면 실제로 살 수 있었는지와 무관하게 min을 갱신한다.
            # 그래야 현금 부족으로 못 산 경우, 오래된 저점 기준이 계속 남아
            # 이후 아주 작은 반등에도 매수 조건이 계속 참이 되는 것을 막을 수 있다.
            min_price = price
            if buy_qty > 0:
                shares += buy_qty
                cash -= buy_qty * price
                buy_count += 1
                if record_log:
                    trade_log.append({
                        "날짜": date, "구분": "매수", "가격": price, "수량": buy_qty,
                        "현금잔고": cash, "보유주식수": shares,
                        "max": max_snapshot, "min": min_snapshot,
                        "등락률": (price - min_snapshot) / min_snapshot * 100,
                        "주식평가금액": price * shares,
                    })

        prev_price = price

    final_price = prices[-1]
    stock_value = shares * final_price
    reserve_final = reserve if profit_sweep else 0.0

    result = {
        "주가": final_price,
        "보유주식수": shares,
        "주식_평가금액": stock_value,
        "현금": cash,
        "적립금": reserve_final,
        "total": stock_value + cash + reserve_final,
        "매도횟수": sell_count,
        "매수횟수": buy_count,
        "매매일지": trade_log if record_log else [],
    }
    if profit_sweep:
        result["이익회수횟수"] = recover_count
        result["이익회수일지"] = recover_log if record_log else []
    return result


def grid_trade_strategy(
    df: pd.DataFrame,
    trade_qty: int,
    sell_gap_percent: float,
    buy_gap_percent: float = None,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    sell_only: bool = False,
    allow_negative_cash: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
) -> dict:
    """
    등락폭(gap) 기반 그리드 매매 전략 백테스트 (트레일링 고점/저점 방식).

    규칙
    ----
    - 시작: 주식 `initial_shares`주 보유, 현금 0원
    - max, min 기준값은 첫날 종가로 시작
    - 주가가 현재 max보다 오르면 max를 그 가격으로 갱신 (최근 고점을 계속 추적)
    - 주가가 현재 min보다 내리면 min을 그 가격으로 갱신 (최근 저점을 계속 추적)
    - 현재가가 max에서 sell_gap% 만큼 떨어지면 -> trade_qty 만큼 매도, max를 매도가로 갱신
      (이후 그 매도가부터 새로운 고점을 다시 추적). 단, **전날보다 가격이 내려간 날에만** 매도한다.
    - (sell_only=True가 아니면) 현재가가 min에서 buy_gap% 만큼 오르면 -> 매수를 시도한다.
      단, **전날보다 가격이 올라간 날에만** 매수한다.
    - 전날 대비 상승/하락 조건 덕분에 하루에 매도와 매수가 동시에 발생하는 일은 없다.

    이익 회수 (백테스트2, profit_gap_percent/profit_recover_percent를 둘 다 지정할 때만 동작)
    ----
    - 매수/매도 로직과는 완전히 독립적으로 동작한다 (max/min, 매도·매수 횟수에 영향 없음).
    - 기준가(profit_base_price)는 max/min과 마찬가지로 첫날 종가로 시작한다.
    - 매일, 매수/매도 판단보다 먼저 확인한다: 주가가 기준가보다 profit_gap_percent% 이상
      오르면 이벤트 발동.
    - 발동 시 평가차익 = 보유주식수 × (현재가 − 기준가). 이 중 profit_recover_percent%에
      해당하는 금액만큼 **주식을 직접 매도**해서(현재가 기준, 정수 주식수로 내림 처리) 그
      대금을 "적립금"으로 옮긴다 — 기존 현금과는 무관하게, 필요한 만큼 주식을 팔아서
      회수한다. 적립금은 이후 매수 자금으로 쓰이지 않고, 최종 total 계산에는 그대로
      합산된다.
    - 발동하면(실제로 회수됐는지와 무관하게) 기준가를 그 시점 주가로 갱신한다.

    Parameters
    ----------
    df : pd.DataFrame
        get_stock_data()로 얻은 일별 시세. 날짜 오름차순/내림차순 상관없이 내부에서 정렬함.
    trade_qty : int
        한 번에 매도할 주식 수 / 매수할 수 있는 최대 주식 수
    sell_gap_percent : float
        매도 판단 기준 등락폭 (%). 예: 5 -> 고점 대비 5% 하락 시 매도
    buy_gap_percent : float, optional
        매수 판단 기준 등락폭 (%). 지정 안 하면 sell_gap_percent와 동일한 값을 사용한다.
        (매수/매도 gap을 다르게 쓰고 싶을 때 지정)
    initial_shares : int
        시작 보유 주식 수 (기본 100)
    price_col : str
        기준으로 삼을 가격 컬럼명 (기본 '종가')
    date_col : str
        날짜 컬럼명 (기본 '날짜'), 매매일지에 사용
    sell_only : bool
        True면 매도만 하고 매수는 절대 하지 않는다 (기본 False)
    allow_negative_cash : bool
        True면 현금 잔고와 무관하게 trade_qty를 그대로 매수한다 (초창기 모델 방식,
        현금이 마이너스가 될 수 있음). False(기본)면 쌓인 현금 범위 내에서만 매수한다.
    profit_gap_percent : float, optional
        이익 회수 이벤트 트리거 등락폭 (%). 기준가 대비 이만큼 오르면 발동. 둘 다 지정해야
        이익 회수 기능이 켜진다 (지정 안 하면 기존과 완전히 동일하게 동작).
    profit_recover_percent : float, optional
        이벤트 발동 시 평가차익 중 주식을 매도해 회수할 비율 (1~100).

    Returns
    -------
    dict
        {
            "주가": 마지막 날 가격,
            "보유주식수": 최종 보유 주식 수,
            "주식_평가금액": 보유주식수 * 마지막 날 가격,
            "현금": 최종 현금,
            "적립금": 이익 회수로 모아둔 금액 (이익 회수 미사용 시 0),
            "total": 주식_평가금액 + 현금 + 적립금,
            "매도횟수": ...,
            "매수횟수": ...,
            "매매일지": [{"날짜":..., "구분":"매도/매수", "가격":..., "수량":...,
                       "현금잔고":..., "보유주식수":...,
                       "max": 매매 판단 시점의 고점, "min": 매매 판단 시점의 저점,
                       "등락률": 매도는 (가격-max)/max*100, 매수는 (가격-min)/min*100,
                       "주식평가금액": 가격 * 매매 직후 보유주식수}, ...],
            # 이익 회수 기능을 켰을 때만 아래 두 키가 추가된다.
            "이익회수횟수": ...,
            "이익회수일지": [{"날짜":..., "구분":"이익회수", "기준가":..., "가격":...,
                          "평가차익":..., "매도주식수":..., "회수액":..., "적립금잔고":...,
                          "현금잔고":..., "보유주식수":..., "주식평가금액":...}, ...],
        }
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")
    if buy_gap_percent is None:
        buy_gap_percent = sell_gap_percent

    if (profit_gap_percent is None) != (profit_recover_percent is None):
        raise ValueError("이익 회수 gap과 회수율은 함께 지정해야 합니다.")
    if profit_gap_percent is not None:
        if profit_gap_percent <= 0:
            raise ValueError("이익 회수 gap은 0보다 커야 합니다.")
        if not (1 <= profit_recover_percent <= 100):
            raise ValueError("이익 회수율은 1~100 사이여야 합니다.")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    dates = sorted_df[date_col].tolist()
    prices = sorted_df[price_col].tolist()

    return _simulate_grid(
        prices, dates, sell_gap_percent, buy_gap_percent, trade_qty, initial_shares,
        sell_only=sell_only, allow_negative_cash=allow_negative_cash, record_log=True,
        profit_gap_percent=profit_gap_percent, profit_recover_percent=profit_recover_percent,
    )


def _run_grid_fast(
    prices: list,
    sell_gap_percent: float,
    trade_qty: int,
    initial_shares: int = 100,
    buy_gap_percent: float = None,
    sell_only: bool = False,
    allow_negative_cash: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
) -> dict:
    """
    grid_trade_strategy()와 완전히 동일한 로직이지만, 매매일지를 기록하지 않아
    수천~수만 번 반복 계산(히트맵용)할 때 빠르게 동작한다.
    total/매도횟수/매수횟수(+이익 회수 사용 시 적립금/이익회수횟수)가 담긴 dict를 반환한다
    (매매일지는 항상 빈 리스트).
    """
    if buy_gap_percent is None:
        buy_gap_percent = sell_gap_percent
    return _simulate_grid(
        prices, None, sell_gap_percent, buy_gap_percent, trade_qty, initial_shares,
        sell_only=sell_only, allow_negative_cash=allow_negative_cash, record_log=False,
        profit_gap_percent=profit_gap_percent, profit_recover_percent=profit_recover_percent,
    )


def compute_profit_heatmap(
    df: pd.DataFrame,
    gap_values,
    qty_values,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    sell_only: bool = False,
    allow_negative_cash: bool = False,
) -> dict:
    """
    gap(%) x 매매수량 조합별 최종 수익률(%)을 계산해 히트맵용 데이터를 만든다.
    (여기서는 매수/매도 gap을 동일한 값으로 스윕한다.)

    Parameters
    ----------
    gap_values : iterable[float]
        gap(%) 값 목록 (예: range(1, 101) -> 1~100%)
    qty_values : iterable[int]
        매매 수량 값 목록 (예: range(1, 101) -> 1~100주)
    sell_only, allow_negative_cash : grid_trade_strategy() 참고

    Returns
    -------
    dict
        {
            "gaps": [...], "qtys": [...],
            "grid": [[qty별 수익률(%), ...], ...]  # grid[i][j] = gaps[i] x qtys[j] 조합
            "best": {"gap":..., "qty":..., "profit_pct":...},
            "worst": {"gap":..., "qty":..., "profit_pct":...},
            "top10": [{"gap":..., "qty":..., "profit_pct":..., "매수횟수":..., "매도횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [{"gap":..., "qty":..., "profit_pct":..., "매수횟수":..., "매도횟수":...}, ...],  # 하위 10 (오름차순)
            "ranked": [{"gap":..., "qty":..., "profit_pct":..., "매수횟수":..., "매도횟수":...}, ...],  # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [{"gap":..., "qty":..., "profit_pct":..., "매수횟수":..., "매도횟수":...}, ...],  # 전체 조합(중복 미제거), max -> min 순
            "initial_asset": 시작 자산,
        }

    top10/bottom10/ranked는 수익률이 같은 조합이 여러 개면 그중 하나만 남긴다(중복 제거).
    남기는 기준: gap이 가장 작은 조합 우선, gap도 같으면 수량이 가장 작은 조합.
    (grid 전체, best/worst, raw_ranked에는 중복 제거를 적용하지 않는다.)
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = initial_shares * first_price

    gaps = list(gap_values)
    qtys = list(qty_values)

    grid = []
    all_combos = []  # top10/bottom10 계산용 (gap, qty, profit_pct) 전체 모음
    best = {"gap": None, "qty": None, "profit_pct": float("-inf")}
    worst = {"gap": None, "qty": None, "profit_pct": float("inf")}

    for g in gaps:
        row = []
        for q in qtys:
            run_result = _run_grid_fast(
                prices, sell_gap_percent=g, trade_qty=q, initial_shares=initial_shares,
                sell_only=sell_only, allow_negative_cash=allow_negative_cash,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "gap": g, "qty": q, "profit_pct": profit_pct,
                "매수횟수": run_result["매수횟수"], "매도횟수": run_result["매도횟수"],
            }
            all_combos.append(combo)
            if profit_pct > best["profit_pct"]:
                best = combo
            if profit_pct < worst["profit_pct"]:
                worst = combo
        grid.append(row)

    # top10/bottom10/ranked는 수익률이 같은 조합을 중복 제거한 뒤 뽑는다: 같은 수익률이면
    # gap이 가장 작은 조합을, gap도 같으면 수량이 가장 작은 조합을 남긴다.
    # all_combos는 gap 오름차순(바깥 루프) -> qty 오름차순(안쪽 루프) 순서로 쌓이므로,
    # 특정 수익률이 처음 등장하는 조합이 곧 그 수익률 중 gap 최소/수량 최소 조합이다.
    seen_profit_pct = set()
    dedup_combos = []
    for combo in all_combos:
        key = round(combo["profit_pct"], 6)
        if key in seen_profit_pct:
            continue
        seen_profit_pct.add(key)
        dedup_combos.append(combo)

    dedup_combos.sort(key=lambda c: c["profit_pct"], reverse=True)
    top10 = dedup_combos[:10]
    bottom10 = list(reversed(dedup_combos[-10:]))  # 가장 낮은 수익률이 맨 위로 오도록
    ranked = dedup_combos  # 전체 조합을 max -> min 순으로 나열 (중복 제거 적용)
    raw_ranked = sorted(all_combos, key=lambda c: c["profit_pct"], reverse=True)  # 중복 미제거 버전

    return {
        "gaps": gaps,
        "qtys": qtys,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "initial_asset": initial_asset,
    }


def compute_profit_heatmap2(
    df: pd.DataFrame,
    gap_values,
    profit_gap_values,
    trade_qty: int,
    profit_recover_percent: float,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    sell_only: bool = False,
    allow_negative_cash: bool = False,
) -> dict:
    """
    백테스트2(이익 회수) 전용 히트맵: 매매 gap(%) x 이익회수 gap(%) 조합별 최종 수익률(%)을
    계산한다. 거래 수량(trade_qty)과 회수율(profit_recover_percent)은 고정 입력값이라 스윕
    대상이 아니다 (매매 gap은 매수/매도 동일한 값으로 스윕한다).

    Parameters
    ----------
    gap_values : iterable[float]
        매매 gap(%) 값 목록 (예: range(1, 51) -> 1~50%)
    profit_gap_values : iterable[float]
        이익 회수 gap(%) 값 목록 (예: range(1, 51) -> 1~50%)
    trade_qty : int
        고정 매수/매도 수량
    profit_recover_percent : float
        고정 이익 회수율 (1~100)
    sell_only, allow_negative_cash : grid_trade_strategy() 참고

    Returns
    -------
    dict
        {
            "gaps": [...], "profit_gaps": [...],
            "grid": [[이익gap별 수익률(%), ...], ...]  # grid[i][j] = gaps[i] x profit_gaps[j] 조합
            "best": {"gap":..., "profit_gap":..., "profit_pct":...},
            "worst": {"gap":..., "profit_gap":..., "profit_pct":...},
            "top10": [{"gap":..., "profit_gap":..., "profit_pct":...,
                       "매수횟수":..., "매도횟수":..., "이익회수횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [...],  # 하위 10 (오름차순)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "trade_qty": 고정 거래 수량,
            "profit_recover_percent": 고정 회수율,
            "initial_asset": 시작 자산,
        }

    top10/bottom10/ranked는 수익률이 같은 조합이 여러 개면 그중 하나만 남긴다(중복 제거).
    남기는 기준: gap이 가장 작은 조합 우선, gap도 같으면 profit_gap이 가장 작은 조합.
    (grid 전체, best/worst, raw_ranked에는 중복 제거를 적용하지 않는다.)
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = initial_shares * first_price

    gaps = list(gap_values)
    profit_gaps = list(profit_gap_values)

    grid = []
    all_combos = []  # top10/bottom10/ranked 계산용 전체 모음
    best = {"gap": None, "profit_gap": None, "profit_pct": float("-inf")}
    worst = {"gap": None, "profit_gap": None, "profit_pct": float("inf")}

    for g in gaps:
        row = []
        for pg in profit_gaps:
            run_result = _run_grid_fast(
                prices, sell_gap_percent=g, trade_qty=trade_qty, initial_shares=initial_shares,
                sell_only=sell_only, allow_negative_cash=allow_negative_cash,
                profit_gap_percent=pg, profit_recover_percent=profit_recover_percent,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "gap": g, "profit_gap": pg, "profit_pct": profit_pct,
                "매수횟수": run_result["매수횟수"], "매도횟수": run_result["매도횟수"],
                "이익회수횟수": run_result["이익회수횟수"],
            }
            all_combos.append(combo)
            if profit_pct > best["profit_pct"]:
                best = combo
            if profit_pct < worst["profit_pct"]:
                worst = combo
        grid.append(row)

    # top10/bottom10/ranked는 수익률이 같은 조합을 중복 제거한 뒤 뽑는다: 같은 수익률이면
    # gap이 가장 작은 조합을, gap도 같으면 profit_gap이 가장 작은 조합을 남긴다.
    # all_combos는 gap 오름차순(바깥 루프) -> profit_gap 오름차순(안쪽 루프) 순서로 쌓이므로,
    # 특정 수익률이 처음 등장하는 조합이 곧 그 수익률 중 gap 최소/profit_gap 최소 조합이다.
    seen_profit_pct = set()
    dedup_combos = []
    for combo in all_combos:
        key = round(combo["profit_pct"], 6)
        if key in seen_profit_pct:
            continue
        seen_profit_pct.add(key)
        dedup_combos.append(combo)

    dedup_combos.sort(key=lambda c: c["profit_pct"], reverse=True)
    top10 = dedup_combos[:10]
    bottom10 = list(reversed(dedup_combos[-10:]))
    ranked = dedup_combos
    raw_ranked = sorted(all_combos, key=lambda c: c["profit_pct"], reverse=True)

    return {
        "gaps": gaps,
        "profit_gaps": profit_gaps,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "trade_qty": trade_qty,
        "profit_recover_percent": profit_recover_percent,
        "initial_asset": initial_asset,
    }


def _main():
    parser = argparse.ArgumentParser(description="네이버 금융 일별 시세 조회 (쉘 테스트용)")
    parser.add_argument("code", help="종목 코드 (예: 005930)")
    parser.add_argument("-d", "--days", type=int, default=30, help="오늘부터 며칠 전까지 (기본 30)")
    parser.add_argument("--csv", help="CSV로 저장할 파일 경로 (선택)")
    parser.add_argument("--gap", type=float, help="매도 gap(%%). 지정하면 백테스트 실행 (매수 gap 별도 지정 없으면 동일하게 적용)")
    parser.add_argument("--buy-gap", type=float, help="매수 gap(%%) (지정 안 하면 --gap과 동일)")
    parser.add_argument("--qty", type=int, default=1, help="매수/매도 주식 개수 (기본 1, --gap과 함께 사용)")
    parser.add_argument("--init-shares", type=int, default=100, help="시작 보유 주식 수 (기본 100)")
    parser.add_argument("--sell-only", action="store_true", help="매도만 하고 매수는 하지 않음")
    parser.add_argument(
        "--allow-negative-cash", action="store_true",
        help="현금이 부족해도 거래수량 그대로 매수 (초창기 모델, 현금이 마이너스가 될 수 있음)",
    )
    args = parser.parse_args()

    print(f"[조회] 종목코드={args.code}  기간={args.days}일")
    try:
        name = get_stock_name(args.code)
        print(f"[종목명] {name}")
    except Exception as e:
        print(f"[경고] 종목명 조회 실패: {e}")

    df = get_stock_data(args.code, args.days)

    if df.empty:
        print("데이터를 가져오지 못했습니다. 종목코드를 확인하세요.")
        sys.exit(1)

    print(df.to_string(index=False))
    print(f"\n총 {len(df)}건")

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"[저장] {args.csv}")

    if args.gap is not None:
        result = grid_trade_strategy(
            df,
            trade_qty=args.qty,
            sell_gap_percent=args.gap,
            buy_gap_percent=args.buy_gap,
            initial_shares=args.init_shares,
            sell_only=args.sell_only,
            allow_negative_cash=args.allow_negative_cash,
        )
        trade_log = result.pop("매매일지")
        buy_gap_display = args.buy_gap if args.buy_gap is not None else args.gap
        print(
            f"\n[그리드 매매 백테스트] 매도gap={args.gap}%  매수gap={buy_gap_display}%  "
            f"거래수량={args.qty}  시작주식수={args.init_shares}  "
            f"매도전용={args.sell_only}  현금마이너스허용={args.allow_negative_cash}"
        )
        for k, v in result.items():
            print(f"  {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        if trade_log:
            print("\n[매매일지]")
            print(pd.DataFrame(trade_log).to_string(index=False))
        else:
            print("\n[매매일지] 거래 없음")


if __name__ == "__main__":
    _main()
