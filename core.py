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
import statistics
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
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    record_log: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
    capital: float = None,
) -> dict:
    """
    그리드 매매 전략의 실제 시뮬레이션 루프 (트레일링 고점/저점 방식).
    grid_trade_strategy()와 _run_grid_fast() 둘 다 이 함수를 사용해서 로직이 어긋나지 않게 한다.

    옵션
    ----
    no_sell : True면 매도를 하지 않는다 (매수는 정상 동작).
    no_buy : True면 매수를 하지 않는다 (매도는 정상 동작).
    allow_negative_cash : True면 매수 시 현금 잔고를 확인하지 않고 trade_qty를 그대로 매수한다
        (초창기 모델 방식 — 현금이 마이너스가 될 수 있음).
        False(기본)면 "쌓인 현금으로 살 수 있는 만큼"과 trade_qty 중 작은 값만큼만 매수한다.
    record_log : True면 매매일지를 기록해서 반환한다 (느림). False면 최종 결과만 계산한다 (빠름,
        히트맵처럼 수천~수만 번 반복 계산할 때 사용).
    profit_gap_percent, profit_recover_percent, capital : 앞의 둘을 함께 지정하면 "이익 회수"
        기능이 추가로 동작한다. grid_trade_strategy()의 해당 파라미터 설명 참고.
        profit_gap_percent만 None이 아니면 호출자(grid_trade_strategy)가 이미 검증했다고
        가정한다.
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
    asset_log = [] if record_log else None

    profit_sweep = profit_gap_percent is not None
    if profit_sweep:
        profit_gap_ratio = profit_gap_percent / 100.0
        profit_recover_ratio = profit_recover_percent / 100.0
        # 자본금 미지정 시 시작 자산(초기 보유주식수 x 첫날 종가)을 자본금으로 삼는다.
        capital = capital if capital is not None else initial_shares * prices[0]
        reserve = 0.0
        recover_count = 0
        recover_log = [] if record_log else None

    if record_log:
        # 첫날(거래 발생 전 시작 상태) 스냅샷도 자산 추이 그래프의 시작점으로 남겨둔다.
        asset_log.append({
            "날짜": dates[0] if dates is not None else None,
            "주가": prices[0], "현금": cash, "보유주식수": shares,
            "주식평가금액": prices[0] * shares, "적립금": 0.0,
            "total": prices[0] * shares + cash,
        })

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

        # 이익 회수 이벤트: 매수/매도 판단(이하 sell/buy 로직)과는 완전히
        # 독립적으로, 매일 먼저 체크한다. 평가금액(주식평가금액+현금)이 자본금 대비
        # profit_gap_percent% 이상 벌었으면 발동한다 (자본금은 고정값이라 별도 리셋이
        # 필요 없다 — 회수로 평가금액 자체가 줄어들며 자연히 다시 벌어야 재발동한다).
        if profit_sweep:
            current_value = shares * price + cash
            profit = current_value - capital
            if profit > 0 and profit >= capital * profit_gap_ratio:
                # 회수해야 할 금액(평가차익 x 회수율)을 현금에서 먼저 충당하고, 모자란
                # 만큼은 주식을 추가로 매도해 마련한다. 주식 수는 정수 단위라 목표 금액을
                # 소수 없이 채우지 못할 수 있어 내림 처리한다.
                target_value = profit * profit_recover_ratio
                cash_used = min(cash, target_value)
                remaining = target_value - cash_used
                shares_sold = min(shares, int(remaining // price)) if remaining > 0 else 0
                recovered = cash_used + shares_sold * price
                if recovered > 0:
                    cash -= cash_used
                    shares -= shares_sold
                    reserve += recovered
                    recover_count += 1
                    if record_log:
                        recover_log.append({
                            "날짜": date, "구분": "이익회수", "자본금": capital,
                            "가격": price, "평가금액": current_value, "평가차익": profit,
                            "현금사용액": cash_used, "매도주식수": shares_sold,
                            "회수액": recovered, "적립금잔고": reserve, "현금잔고": cash,
                            "보유주식수": shares, "주식평가금액": price * shares,
                        })

        # 매도: max 대비 sell_gap% 하락, 단 전날보다 내려간 날에만 (no_sell이면 아예 건너뜀)
        if (
            not no_sell
            and price <= max_snapshot * (1 - sell_gap_ratio)
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

        # 매수: min 대비 buy_gap% 상승, 단 전날보다 올라간 날에만 (no_buy면 아예 건너뜀)
        if (not no_buy) and price >= min_snapshot * (1 + buy_gap_ratio) and price > prev_price:
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

        if record_log:
            reserve_now = reserve if profit_sweep else 0.0
            asset_log.append({
                "날짜": date, "주가": price, "현금": cash, "보유주식수": shares,
                "주식평가금액": price * shares, "적립금": reserve_now,
                "total": price * shares + cash + reserve_now,
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
        "자산추이": asset_log if record_log else [],
    }
    if profit_sweep:
        result["자본금"] = capital
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
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
    capital: float = None,
) -> dict:
    """
    등락폭(gap) 기반 그리드 매매 전략 백테스트 (트레일링 고점/저점 방식).

    규칙
    ----
    - 시작: 주식 `initial_shares`주 보유, 현금 0원
    - max, min 기준값은 첫날 종가로 시작
    - 주가가 현재 max보다 오르면 max를 그 가격으로 갱신 (최근 고점을 계속 추적)
    - 주가가 현재 min보다 내리면 min을 그 가격으로 갱신 (최근 저점을 계속 추적)
    - (no_sell=True가 아니면) 현재가가 max에서 sell_gap% 만큼 떨어지면 -> trade_qty 만큼 매도,
      max를 매도가로 갱신(이후 그 매도가부터 새로운 고점을 다시 추적). 단, **전날보다 가격이
      내려간 날에만** 매도한다.
    - (no_buy=True가 아니면) 현재가가 min에서 buy_gap% 만큼 오르면 -> 매수를 시도한다.
      단, **전날보다 가격이 올라간 날에만** 매수한다.
    - 전날 대비 상승/하락 조건 덕분에 하루에 매도와 매수가 동시에 발생하는 일은 없다.

    이익 회수 (profit_gap_percent/profit_recover_percent를 둘 다 지정할 때만 동작)
    ----
    - 매수/매도 로직과는 완전히 독립적으로 동작한다 (max/min, 매도·매수 횟수에 영향 없음).
    - 자본금(capital)을 기준으로 삼는다. 지정하지 않으면 시작 자산(initial_shares × 첫날
      종가)을 자본금으로 사용한다.
    - 매일, 매수/매도 판단보다 먼저 확인한다: 평가금액(주식평가금액 + 현금)이 자본금 대비
      profit_gap_percent% 이상 벌었으면 이벤트 발동.
    - 발동 시 평가차익 = 평가금액 − 자본금. 이 중 profit_recover_percent%에 해당하는
      금액을 **현금에서 먼저 충당**하고, 모자란 만큼 **주식을 추가로 매도**해서(현재가 기준,
      정수 주식수로 내림 처리) 마련한 뒤 "적립금"으로 옮긴다. 적립금은 이후 매수 자금으로
      쓰이지 않고, 최종 total 계산에는 그대로 합산된다.
    - 자본금은 고정값이라 별도로 갱신하지 않는다. 회수로 평가금액 자체가 줄어들기 때문에
      다시 자본금 대비 profit_gap_percent%만큼 벌어야 재발동한다.

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
    no_sell : bool
        True면 매도를 하지 않는다 (매수는 정상 동작, 기본 False)
    no_buy : bool
        True면 매수를 하지 않는다 (매도는 정상 동작, 기본 False)
    allow_negative_cash : bool
        True면 현금 잔고와 무관하게 trade_qty를 그대로 매수한다 (초창기 모델 방식,
        현금이 마이너스가 될 수 있음). False(기본)면 쌓인 현금 범위 내에서만 매수한다.
    profit_gap_percent : float, optional
        이익 회수 이벤트 트리거 등락폭 (%). 자본금 대비 이만큼 벌면 발동. 둘 다 지정해야
        이익 회수 기능이 켜진다 (지정 안 하면 기존과 완전히 동일하게 동작).
    profit_recover_percent : float, optional
        이벤트 발동 시 평가차익 중 회수할 비율 (1~100).
    capital : float, optional
        이익 회수 기준이 되는 자본금. 지정하지 않으면 시작 자산(initial_shares × 첫날 종가)을
        자본금으로 사용한다.

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
            "자산추이": [{"날짜":..., "주가":..., "현금":..., "보유주식수":...,
                       "주식평가금액":..., "적립금":..., "total":...}, ...],
                # 첫날부터 마지막 날까지 매일의 스냅샷 (거래 발생 여부와 무관, 그래프용).
                # 이익 회수 미사용 시 "적립금"은 항상 0.
            # 이익 회수 기능을 켰을 때만 아래 세 키가 추가된다.
            "자본금": 이익 회수에 사용된 자본금 (지정 안 했으면 자동 계산된 시작 자산),
            "이익회수횟수": ...,
            "이익회수일지": [{"날짜":..., "구분":"이익회수", "자본금":..., "가격":...,
                          "평가금액":..., "평가차익":..., "현금사용액":..., "매도주식수":...,
                          "회수액":..., "적립금잔고":..., "현금잔고":..., "보유주식수":...,
                          "주식평가금액":...}, ...],
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
        no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash, record_log=True,
        profit_gap_percent=profit_gap_percent, profit_recover_percent=profit_recover_percent,
        capital=capital,
    )


def _run_grid_fast(
    prices: list,
    sell_gap_percent: float,
    trade_qty: int,
    initial_shares: int = 100,
    buy_gap_percent: float = None,
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    profit_gap_percent: float = None,
    profit_recover_percent: float = None,
    capital: float = None,
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
        no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash, record_log=False,
        profit_gap_percent=profit_gap_percent, profit_recover_percent=profit_recover_percent,
        capital=capital,
    )


def resolve_trade_qty(initial_shares: int, qty_percent: float) -> int:
    """
    시작 보유 주식수 대비 비율(%)로 매매 수량(주)을 계산한다 (최소 1주).
    시작 보유 주식수를 기준으로 **한 번만** 계산하며, 이후 보유 주식수가 매매로 바뀌어도
    다시 계산하지 않는다 — 지금까지의 "고정 수량" 매매 로직과 동일하게 동작한다.
    """
    return max(1, round(initial_shares * qty_percent / 100))


def _simulate_daily(
    prices: list,
    dates,
    sell_qty: int,
    buy_qty: int,
    initial_shares: int,
    allow_negative_cash: bool = False,
    sell_above_start_asset_only: bool = False,
    record_log: bool = False,
) -> dict:
    """
    "일별 방향 매매" 전략의 실제 시뮬레이션 루프. daily_reversal_strategy()와
    _run_daily_fast() 둘 다 이 함수를 사용해서 로직이 어긋나지 않게 한다.

    record_log : True면 매매일지/자산추이를 기록해서 반환한다 (느림). False면 최종 결과만
        계산한다 (빠름, 히트맵처럼 수천 번 반복 계산할 때 사용).
    """
    shares = initial_shares
    cash = 0.0
    sell_count = 0
    buy_count = 0
    trade_log = [] if record_log else None
    asset_log = [] if record_log else None
    initial_asset = initial_shares * prices[0]  # 시작 자산 (매도 제한 옵션의 비교 기준)

    if record_log:
        asset_log.append({
            "날짜": dates[0], "주가": prices[0], "현금": cash, "보유주식수": shares,
            "주식평가금액": prices[0] * shares, "total": prices[0] * shares + cash,
        })

    prev_price = prices[0]
    for i in range(1, len(prices)):
        price = prices[i]
        date = dates[i] if dates is not None else None
        current_asset = shares * price + cash  # 오늘 거래를 반영하기 전, 그 시점 평가자산

        if (
            price > prev_price
            and shares >= sell_qty
            and (not sell_above_start_asset_only or current_asset > initial_asset)
        ):
            shares -= sell_qty
            cash += sell_qty * price
            sell_count += 1
            if record_log:
                hold_only_asset = initial_shares * price
                trade_log.append({
                    "날짜": date, "구분": "매도", "가격": price, "수량": sell_qty,
                    "현금잔고": cash, "보유주식수": shares, "주식평가금액": price * shares,
                    "매매안했을때자산": hold_only_asset,
                    "차이": (price * shares + cash) - hold_only_asset,
                })
        elif price < prev_price:
            if allow_negative_cash:
                # 현금 잔고와 무관하게 buy_qty를 그대로 매수 (현금 마이너스 허용)
                actual_buy_qty = buy_qty
            else:
                affordable_qty = int(cash // price) if price > 0 else 0
                actual_buy_qty = min(buy_qty, affordable_qty)

            if actual_buy_qty > 0:
                shares += actual_buy_qty
                cash -= actual_buy_qty * price
                buy_count += 1
                if record_log:
                    hold_only_asset = initial_shares * price
                    trade_log.append({
                        "날짜": date, "구분": "매수", "가격": price, "수량": actual_buy_qty,
                        "현금잔고": cash, "보유주식수": shares, "주식평가금액": price * shares,
                        "매매안했을때자산": hold_only_asset,
                        "차이": (price * shares + cash) - hold_only_asset,
                    })

        if record_log:
            asset_log.append({
                "날짜": date, "주가": price, "현금": cash, "보유주식수": shares,
                "주식평가금액": price * shares, "total": price * shares + cash,
            })
        prev_price = price

    final_price = prices[-1]
    stock_value = shares * final_price

    return {
        "주가": final_price,
        "보유주식수": shares,
        "주식_평가금액": stock_value,
        "현금": cash,
        "total": stock_value + cash,
        "매도횟수": sell_count,
        "매수횟수": buy_count,
        "매매일지": trade_log if record_log else [],
        "자산추이": asset_log if record_log else [],
    }


def daily_reversal_strategy(
    df: pd.DataFrame,
    sell_qty: int,
    buy_qty: int,
    initial_shares: int = 100,
    allow_negative_cash: bool = False,
    sell_above_start_asset_only: bool = False,
    price_col: str = "종가",
    date_col: str = "날짜",
) -> dict:
    """
    "일별 방향 매매" 전략 — gap이나 고점/저점 추적 없이, 오직 전날 종가 대비 오늘 종가만 본다.

    규칙
    ----
    - 전날보다 오른 날: sell_qty만큼 매도한다. 단, 보유 주식수가 sell_qty보다 적으면
      매도하지 않는다 (공매도 없음). sell_above_start_asset_only=True면 추가로 **그 시점
      평가자산(그날 가격 기준 주식평가금액+현금)이 시작 자산(initial_shares × 첫날
      종가)보다 높을 때만** 매도한다 (아직 시작 자산을 회복하지 못한 상태의 반등에는 팔지
      않고 계속 들고 있음).
    - 전날보다 내린 날: buy_qty만큼 매수를 시도한다.
      - allow_negative_cash=True면 현금 잔고와 무관하게 항상 buy_qty만큼 그대로 매수한다
        (현금이 마이너스가 될 수 있음).
      - False(기본)면 "쌓인 현금으로 살 수 있는 만큼"과 buy_qty 중 작은 값만큼만 매수한다
        (grid_trade_strategy()의 기본 매수 로직과 동일). 그마저도 0이면 매수하지 않는다.
    - 전날과 같은 날: 아무 것도 하지 않는다.
    - grid_trade_strategy()와 달리 max/min 트레일링 기준, gap 임계값, 이익 회수 기능이
      전혀 없는 가장 단순한 형태다. 매도/매수 수량을 독립적으로 지정할 수 있다.

    Parameters
    ----------
    df : pd.DataFrame
        get_stock_data()로 얻은 일별 시세. 날짜 오름차순/내림차순 상관없이 내부에서 정렬함.
    sell_qty : int
        매도 시 거래할 주식 수 (보통 resolve_trade_qty()로 시작 보유 주식수 대비 %에서
        계산해서 넘긴다).
    buy_qty : int
        매수 시도 시 거래할 주식 수 (마찬가지로 resolve_trade_qty() 사용 권장).
    initial_shares : int
        시작 보유 주식 수 (기본 100)
    allow_negative_cash : bool
        True면 현금 잔고와 무관하게 buy_qty를 그대로 매수한다 (현금이 마이너스가 될 수
        있음). False(기본)면 쌓인 현금 범위 내에서만 매수한다.
    sell_above_start_asset_only : bool
        True면 그 시점 평가자산(주식평가금액+현금)이 시작 자산보다 높을 때만 매도한다
        (기본 False — 전날 대비 상승이기만 하면 자산 수준과 무관하게 매도).
    price_col : str
        기준으로 삼을 가격 컬럼명 (기본 '종가')
    date_col : str
        날짜 컬럼명 (기본 '날짜'), 매매일지에 사용

    Returns
    -------
    dict
        {
            "주가": 마지막 날 가격,
            "보유주식수": 최종 보유 주식 수,
            "주식_평가금액": 보유주식수 * 마지막 날 가격,
            "현금": 최종 현금 (allow_negative_cash=True면 마이너스일 수 있음),
            "total": 주식_평가금액 + 현금,
            "매도횟수": ...,
            "매수횟수": ...,
            "매매일지": [{"날짜":..., "구분":"매도/매수", "가격":..., "수량":...,
                       "현금잔고":..., "보유주식수":..., "주식평가금액":...,
                       "매매안했을때자산": initial_shares * 그날 가격,
                       "차이": (주식평가금액+현금잔고) - 매매안했을때자산}, ...],
            "자산추이": [{"날짜":..., "주가":..., "현금":..., "보유주식수":...,
                       "주식평가금액":..., "total":...}, ...],
                # 첫날부터 마지막 날까지 매일의 스냅샷 (거래 발생 여부와 무관, 그래프용).
        }
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    dates = sorted_df[date_col].tolist()
    prices = sorted_df[price_col].tolist()

    return _simulate_daily(
        prices, dates, sell_qty, buy_qty, initial_shares,
        allow_negative_cash=allow_negative_cash,
        sell_above_start_asset_only=sell_above_start_asset_only,
        record_log=True,
    )


def _run_daily_fast(
    prices: list,
    sell_qty: int,
    buy_qty: int,
    initial_shares: int = 100,
    allow_negative_cash: bool = False,
    sell_above_start_asset_only: bool = False,
) -> dict:
    """
    daily_reversal_strategy()와 완전히 동일한 로직이지만, 매매일지를 기록하지 않아
    수천 번 반복 계산(히트맵용)할 때 빠르게 동작한다.
    """
    return _simulate_daily(
        prices, None, sell_qty, buy_qty, initial_shares,
        allow_negative_cash=allow_negative_cash,
        sell_above_start_asset_only=sell_above_start_asset_only,
        record_log=False,
    )


def _simulate_daily_gap(
    prices: list,
    dates,
    trade_qty: int,
    gap_percent: float,
    initial_shares: int,
    no_sell: bool = False,
    no_buy: bool = False,
    record_log: bool = False,
) -> dict:
    """
    "일별 매매 2" 전략의 실제 시뮬레이션 루프 (트레일링 고점/저점 방식).
    daily_gap_strategy()와 _run_daily_gap_fast() 둘 다 이 함수를 사용해서 로직이 어긋나지
    않게 한다.

    규칙 (grid_trade_strategy()의 매도/매수를 서로 바꾼 형태 — 자세한 설명은
    daily_gap_strategy() 참고)
    ----
    - max/min은 첫날 종가로 시작하고, 주가가 새 고점/저점을 찍을 때마다 계속 갱신된다
      (매매 발생 여부와 무관).
    - **매수**: 현재가가 max에서 gap% 만큼 떨어지면 매수하고 max를 그 매수가로 리셋한다.
      단, 전날보다 가격이 내려간 날에만 매수한다. no_buy=True면 매수 자체를 하지 않는다
      (이 경우 max는 상시 갱신만 되고 매수 시 리셋은 일어나지 않는다).
    - **매도**: 현재가가 min에서 gap% 만큼 오르면 매도하고 min을 그 매도가로 리셋한다.
      단, 전날보다 가격이 올라간 날에만 매도한다. no_sell=True면 매도 자체를 하지 않는다
      (min도 마찬가지로 상시 갱신만 된다).

    record_log : True면 매매일지/자산추이를 기록해서 반환한다 (느림). False면 최종 결과만
        계산한다 (빠름, 히트맵처럼 수천 번 반복 계산할 때 사용).
    """
    gap_ratio = gap_percent / 100.0
    max_price = min_price = prices[0]
    shares = initial_shares
    cash = 0.0
    sell_count = 0
    buy_count = 0
    trade_log = [] if record_log else None
    asset_log = [] if record_log else None

    if record_log:
        asset_log.append({
            "날짜": dates[0], "주가": prices[0], "현금": cash, "보유주식수": shares,
            "주식평가금액": prices[0] * shares, "total": prices[0] * shares + cash,
        })

    prev_price = prices[0]
    for i in range(1, len(prices)):
        price = prices[i]
        date = dates[i] if dates is not None else None

        # 최근 고점/저점 상시 갱신 (매매 발생 여부와 무관)
        if price > max_price:
            max_price = price
        if price < min_price:
            min_price = price

        # 이번 판단에 쓰인 max/min 스냅샷 (리셋되기 전 값)
        max_snapshot = max_price
        min_snapshot = min_price

        # 매수: max 대비 gap% 하락, 단 전날보다 내려간 날에만 (no_buy면 아예 건너뜀)
        if (not no_buy) and price <= max_snapshot * (1 - gap_ratio) and price < prev_price:
            affordable_qty = int(cash // price) if price > 0 else 0
            buy_qty = min(trade_qty, affordable_qty)

            # 매수 신호가 뜨면 실제로 살 수 있었는지와 무관하게 max를 갱신한다.
            # 그래야 현금 부족으로 못 산 경우, 오래된 고점 기준이 계속 남아
            # 이후 아주 작은 하락에도 매수 조건이 계속 참이 되는 것을 막을 수 있다.
            max_price = price
            if buy_qty > 0:
                shares += buy_qty
                cash -= buy_qty * price
                buy_count += 1
                if record_log:
                    hold_only_asset = initial_shares * price
                    trade_log.append({
                        "날짜": date, "구분": "매수", "가격": price, "수량": buy_qty,
                        "현금잔고": cash, "보유주식수": shares, "주식평가금액": price * shares,
                        "max": max_snapshot, "min": min_snapshot,
                        "등락률": (price - max_snapshot) / max_snapshot * 100,
                        "매매안했을때자산": hold_only_asset,
                        "차이": (price * shares + cash) - hold_only_asset,
                    })

        # 매도: min 대비 gap% 상승, 단 전날보다 올라간 날에만 (no_sell이면 아예 건너뜀)
        if (
            not no_sell
            and price >= min_snapshot * (1 + gap_ratio)
            and price > prev_price
            and shares >= trade_qty
        ):
            shares -= trade_qty
            cash += trade_qty * price
            min_price = price
            sell_count += 1
            if record_log:
                hold_only_asset = initial_shares * price
                trade_log.append({
                    "날짜": date, "구분": "매도", "가격": price, "수량": trade_qty,
                    "현금잔고": cash, "보유주식수": shares, "주식평가금액": price * shares,
                    "max": max_snapshot, "min": min_snapshot,
                    "등락률": (price - min_snapshot) / min_snapshot * 100,
                    "매매안했을때자산": hold_only_asset,
                    "차이": (price * shares + cash) - hold_only_asset,
                })

        if record_log:
            asset_log.append({
                "날짜": date, "주가": price, "현금": cash, "보유주식수": shares,
                "주식평가금액": price * shares, "total": price * shares + cash,
            })
        prev_price = price

    final_price = prices[-1]
    stock_value = shares * final_price

    return {
        "주가": final_price,
        "보유주식수": shares,
        "주식_평가금액": stock_value,
        "현금": cash,
        "total": stock_value + cash,
        "매도횟수": sell_count,
        "매수횟수": buy_count,
        "매매일지": trade_log if record_log else [],
        "자산추이": asset_log if record_log else [],
    }


def daily_gap_strategy(
    df: pd.DataFrame,
    trade_qty: int,
    gap_percent: float,
    initial_shares: int = 100,
    no_sell: bool = False,
    no_buy: bool = False,
    price_col: str = "종가",
    date_col: str = "날짜",
) -> dict:
    """
    "일별 매매 2" 전략 — 트레일링 고점(max)/저점(min) 기준. max, min은 첫날 종가로 시작해서
    새 고점/저점을 찍을 때마다 계속 갱신된다 (매매 발생 여부와 무관). grid_trade_strategy()와
    똑같은 트레일링 구조지만 **매도/매수가 서로 뒤바뀌어 있다**: max에서 떨어지면 매도가
    아니라 매수, min에서 오르면 매수가 아니라 매도한다.

    규칙
    ----
    - **매수**: 현재가가 max에서 gap_percent% 만큼 떨어지면 매수를 시도한다. "쌓인 현금으로
      살 수 있는 만큼"과 trade_qty 중 작은 값만큼만 매수한다 (현금 마이너스 허용 안 함).
      매수 후(또는 매수 신호만 뜨고 현금 부족으로 못 샀어도) max를 그 시점 가격으로 리셋한다.
      단, **전날보다 가격이 내려간 날에만** 매수한다.
    - **매도**: 현재가가 min에서 gap_percent% 만큼 오르면 trade_qty만큼 매도한다. 단, 보유
      주식수가 trade_qty보다 적으면 매도하지 않는다 (공매도 없음). 매도 후 min을 그 매도가로
      리셋한다. 단, **전날보다 가격이 올라간 날에만** 매도한다.
    - 전날 대비 상승/하락 조건 덕분에 하루에 매도와 매수가 동시에 발생하는 일은 없다
      (grid_trade_strategy()와 같은 이유).
    - 매도/매수 수량이 trade_qty 하나로 공유된다 (daily_reversal_strategy()는 매도/매수
      수량을 독립적으로 지정할 수 있었던 것과 다르다).

    Parameters
    ----------
    df : pd.DataFrame
        get_stock_data()로 얻은 일별 시세. 날짜 오름차순/내림차순 상관없이 내부에서 정렬함.
    trade_qty : int
        매도/매수 공통 거래 수량 (보통 resolve_trade_qty()로 시작 보유 주식수 대비 %에서
        계산해서 넘긴다).
    gap_percent : float
        매매를 촉발하는 트레일링 고점/저점 대비 등락폭(%) 임계값. 예: 3 -> 고점 대비 3%
        하락하면 매수, 저점 대비 3% 상승하면 매도.
    initial_shares : int
        시작 보유 주식 수 (기본 100)
    no_sell : bool
        True면 매도를 하지 않는다 (매수는 정상 동작, 기본 False)
    no_buy : bool
        True면 매수를 하지 않는다 (매도는 정상 동작, 기본 False)
    price_col : str
        기준으로 삼을 가격 컬럼명 (기본 '종가')
    date_col : str
        날짜 컬럼명 (기본 '날짜'), 매매일지에 사용

    Returns
    -------
    dict
        {
            "주가": 마지막 날 가격,
            "보유주식수": 최종 보유 주식 수,
            "주식_평가금액": 보유주식수 * 마지막 날 가격,
            "현금": 최종 현금,
            "total": 주식_평가금액 + 현금,
            "매도횟수": ...,
            "매수횟수": ...,
            "매매일지": [{"날짜":..., "구분":"매도/매수", "가격":..., "수량":...,
                       "현금잔고":..., "보유주식수":..., "주식평가금액":...,
                       "max": 그 거래를 판단할 때 쓰인 고점 스냅샷,
                       "min": 그 거래를 판단할 때 쓰인 저점 스냅샷,
                       "등락률": 매수는 (가격-max)/max*100, 매도는 (가격-min)/min*100,
                       "매매안했을때자산": initial_shares * 그날 가격,
                       "차이": (주식평가금액+현금잔고) - 매매안했을때자산}, ...],
            "자산추이": [{"날짜":..., "주가":..., "현금":..., "보유주식수":...,
                       "주식평가금액":..., "total":...}, ...],
                # 첫날부터 마지막 날까지 매일의 스냅샷 (거래 발생 여부와 무관, 그래프용).
        }
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")
    if gap_percent <= 0:
        raise ValueError("등락폭 gap은 0보다 커야 합니다.")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    dates = sorted_df[date_col].tolist()
    prices = sorted_df[price_col].tolist()

    return _simulate_daily_gap(
        prices, dates, trade_qty, gap_percent, initial_shares,
        no_sell=no_sell, no_buy=no_buy, record_log=True,
    )


def _run_daily_gap_fast(
    prices: list,
    trade_qty: int,
    gap_percent: float,
    initial_shares: int = 100,
    no_sell: bool = False,
    no_buy: bool = False,
) -> dict:
    """
    daily_gap_strategy()와 완전히 동일한 로직이지만, 매매일지를 기록하지 않아 수천 번
    반복 계산(히트맵용)할 때 빠르게 동작한다.
    """
    return _simulate_daily_gap(
        prices, None, trade_qty, gap_percent, initial_shares,
        no_sell=no_sell, no_buy=no_buy, record_log=False,
    )


def compute_daily_gap_heatmap(
    df: pd.DataFrame,
    gap_values,
    qty_percent_values,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    no_sell: bool = False,
    no_buy: bool = False,
) -> dict:
    """
    daily_gap_strategy() 전용 히트맵: 등락폭 gap(%) x 매매수량(%) 조합별 최종 수익률(%)을
    계산한다. 매매수량은 시작 보유 주식수 대비 비율로, resolve_trade_qty()로 절대 수량으로
    변환한 뒤 스윕한다 (매도/매수 공통 수량).

    Parameters
    ----------
    gap_values : iterable[float]
        등락폭 gap(%) 값 목록 (예: range(1, 51) -> 1~50%)
    qty_percent_values : iterable[float]
        매매 수량 비율(%) 값 목록 (예: range(1, 51) -> 시작 보유 주식수의 1~50%)
    no_sell, no_buy : daily_gap_strategy() 참고

    Returns
    -------
    dict
        {
            "gaps": [...], "qty_pcts": [...],
            "grid": [[qty%별 수익률(%), ...], ...]  # grid[i][j] = gaps[i] x qty_pcts[j] 조합
            "best": {"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...},
            "worst": {"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...},
            "top10": [{"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...,
                       "매수횟수":..., "매도횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [...],  # 하위 10 (오름차순)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "initial_asset": 시작 자산,
            "hold_only_asset": 매매 안 했을 때 최종 자산,
        }

    top10/bottom10/ranked는 수익률이 같은 조합이 여러 개면 그중 하나만 남긴다(중복 제거).
    남기는 기준: gap이 가장 작은 조합 우선, gap도 같으면 수량비율이 가장 작은 조합.
    (grid 전체, best/worst, raw_ranked에는 중복 제거를 적용하지 않는다.)
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = initial_shares * first_price
    hold_only_asset = initial_shares * prices[-1]

    gaps = list(gap_values)
    qty_pcts = list(qty_percent_values)

    grid = []
    all_combos = []
    best = {"gap": None, "qty_pct": None, "qty": None, "profit_pct": float("-inf")}
    worst = {"gap": None, "qty_pct": None, "qty": None, "profit_pct": float("inf")}

    for g in gaps:
        row = []
        for qp in qty_pcts:
            resolved_qty = resolve_trade_qty(initial_shares, qp)
            run_result = _run_daily_gap_fast(
                prices, trade_qty=resolved_qty, gap_percent=g, initial_shares=initial_shares,
                no_sell=no_sell, no_buy=no_buy,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "gap": g, "qty_pct": qp, "qty": resolved_qty, "profit_pct": profit_pct, "total": total,
                "매매안했을때자산": hold_only_asset, "차이": total - hold_only_asset,
                "매수횟수": run_result["매수횟수"], "매도횟수": run_result["매도횟수"],
            }
            all_combos.append(combo)
            if profit_pct > best["profit_pct"]:
                best = combo
            if profit_pct < worst["profit_pct"]:
                worst = combo
        grid.append(row)

    # top10/bottom10/ranked는 수익률이 같은 조합을 중복 제거한 뒤 뽑는다: 같은 수익률이면
    # gap이 가장 작은 조합을, gap도 같으면 수량비율이 가장 작은 조합을 남긴다.
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
        "qty_pcts": qty_pcts,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "initial_asset": initial_asset,
        "hold_only_asset": hold_only_asset,
    }


def compute_daily_heatmap(
    df: pd.DataFrame,
    sell_qty_pct_values,
    buy_qty_pct_values,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    allow_negative_cash: bool = False,
    sell_above_start_asset_only: bool = False,
) -> dict:
    """
    daily_reversal_strategy() 전용 히트맵: 매도수량%(sell) x 매수수량%(buy) 조합별 최종
    수익률(%)을 계산한다. 둘 다 시작 보유 주식수 대비 비율로, resolve_trade_qty()로 각각
    독립적인 절대 수량으로 변환한 뒤 스윕한다.

    Parameters
    ----------
    sell_qty_pct_values : iterable[float]
        매도수량(%) 값 목록 (예: range(1, 51) -> 1~50%)
    buy_qty_pct_values : iterable[float]
        매수수량(%) 값 목록 (예: range(1, 51) -> 1~50%)
    allow_negative_cash, sell_above_start_asset_only : daily_reversal_strategy() 참고

    Returns
    -------
    dict
        {
            "sell_pcts": [...], "buy_pcts": [...],
            "grid": [[buy%별 수익률(%), ...], ...]  # grid[i][j] = sell_pcts[i] x buy_pcts[j] 조합
            "best": {"sell_pct":..., "buy_pct":..., "sell_qty":..., "buy_qty":..., "profit_pct":..., "total":...},
            "worst": {"sell_pct":..., "buy_pct":..., "sell_qty":..., "buy_qty":..., "profit_pct":..., "total":...},
            "top10": [{"sell_pct":..., "buy_pct":..., "sell_qty":..., "buy_qty":...,
                       "profit_pct":..., "total":..., "매수횟수":..., "매도횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [...],  # 하위 10 (오름차순)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "initial_asset": 시작 자산,
        }

    top10/bottom10/ranked는 수익률이 같은 조합이 여러 개면 그중 하나만 남긴다(중복 제거).
    남기는 기준: 매도수량%이 가장 작은 조합 우선, 같으면 매수수량%이 가장 작은 조합.
    (grid 전체, best/worst, raw_ranked에는 중복 제거를 적용하지 않는다.)
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = initial_shares * first_price
    hold_only_asset = initial_shares * prices[-1]  # 매매 안 했을 때(그냥 보유) 최종 자산

    sell_pcts = list(sell_qty_pct_values)
    buy_pcts = list(buy_qty_pct_values)

    grid = []
    all_combos = []
    best = {"sell_pct": None, "buy_pct": None, "profit_pct": float("-inf")}
    worst = {"sell_pct": None, "buy_pct": None, "profit_pct": float("inf")}

    for sp in sell_pcts:
        sell_qty = resolve_trade_qty(initial_shares, sp)
        row = []
        for bp in buy_pcts:
            buy_qty = resolve_trade_qty(initial_shares, bp)
            run_result = _run_daily_fast(
                prices, sell_qty=sell_qty, buy_qty=buy_qty, initial_shares=initial_shares,
                allow_negative_cash=allow_negative_cash,
                sell_above_start_asset_only=sell_above_start_asset_only,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "sell_pct": sp, "buy_pct": bp, "sell_qty": sell_qty, "buy_qty": buy_qty,
                "profit_pct": profit_pct, "total": total,
                "매매안했을때자산": hold_only_asset, "차이": total - hold_only_asset,
                "매도횟수": run_result["매도횟수"], "매수횟수": run_result["매수횟수"],
            }
            all_combos.append(combo)
            if profit_pct > best["profit_pct"]:
                best = combo
            if profit_pct < worst["profit_pct"]:
                worst = combo
        grid.append(row)

    # top10/bottom10/ranked는 수익률이 같은 조합을 중복 제거한 뒤 뽑는다: 같은 수익률이면
    # 매도수량%이 가장 작은 조합을, 같으면 매수수량%이 가장 작은 조합을 남긴다.
    # all_combos는 sell_pct 오름차순(바깥 루프) -> buy_pct 오름차순(안쪽 루프) 순서로
    # 쌓이므로, 특정 수익률이 처음 등장하는 조합이 곧 그 기준을 만족한다.
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
        "sell_pcts": sell_pcts,
        "buy_pcts": buy_pcts,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "initial_asset": initial_asset,
        "hold_only_asset": hold_only_asset,
    }


def compute_profit_heatmap(
    df: pd.DataFrame,
    gap_values,
    qty_percent_values,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    capital: float = None,
) -> dict:
    """
    gap(%) x 매매수량(%) 조합별 최종 수익률(%)을 계산해 히트맵용 데이터를 만든다.
    (여기서는 매수/매도 gap을 동일한 값으로 스윕한다.)

    매매수량은 절대 주식수가 아니라 **시작 보유 주식수 대비 비율(%)**로 지정한다
    (`resolve_trade_qty()`로 시작 시점에 한 번만 절대 수량으로 변환) — 보유 주식수가
    커져도 절대 수량 범위를 그에 맞춰 새로 정할 필요 없이 항상 같은 %범위로 의미 있는
    스윕이 가능하다.

    Parameters
    ----------
    gap_values : iterable[float]
        gap(%) 값 목록 (예: range(1, 101) -> 1~100%)
    qty_percent_values : iterable[float]
        매매 수량 비율(%) 값 목록 (예: range(1, 51) -> 시작 보유 주식수의 1~50%)
    capital : float, optional
        수익률(%) 계산 기준이 되는 시작 자산. 지정하지 않으면 시작 자산
        (initial_shares × 첫날 종가)을 그대로 사용한다.
    no_sell, no_buy, allow_negative_cash : grid_trade_strategy() 참고

    Returns
    -------
    dict
        {
            "gaps": [...], "qty_pcts": [...],
            "grid": [[qty%별 수익률(%), ...], ...]  # grid[i][j] = gaps[i] x qty_pcts[j] 조합
            "best": {"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...},
            "worst": {"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...},
            "top10": [{"gap":..., "qty_pct":..., "qty":..., "profit_pct":..., "total":...,
                       "매수횟수":..., "매도횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [...],  # 하위 10 (오름차순)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "initial_asset": 수익률 계산에 쓰인 시작 자산 (capital 지정 시 그 값, 아니면
                initial_shares × 첫날 종가),
            "qty_stats": [{"qty_pct":..., "median_profit_pct":..., "traded_gap_count":...}, ...],
                # 수량%별로, 매매가 1회 이상 발생한 gap들만 모아 수익률의 중앙값을 낸 값.
                # 매매가 전혀 발생하지 않은 수량%는 median_profit_pct가 None.
            "recommended_qty": qty_stats 중 median_profit_pct가 가장 높은 항목 (전부 None이면 None),
                # "gap을 모르는 상태에서 이 수량%가 대체로 가장 좋은 성과를 낸다"는 의미의 대표 수량.
        }
        (각 조합의 "qty"는 "qty_pct"를 resolve_trade_qty()로 변환한 실제 매매 주식수,
        "total"은 그 조합으로 백테스트했을 때의 최종 자산)

    top10/bottom10/ranked는 수익률이 같은 조합이 여러 개면 그중 하나만 남긴다(중복 제거).
    남기는 기준: gap이 가장 작은 조합 우선, gap도 같으면 수량비율이 가장 작은 조합.
    (grid 전체, best/worst, raw_ranked에는 중복 제거를 적용하지 않는다.)
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = capital if capital is not None else initial_shares * first_price
    hold_only_asset = initial_shares * prices[-1]  # 매매 안 했을 때(그냥 보유) 최종 자산

    gaps = list(gap_values)
    qty_pcts = list(qty_percent_values)

    grid = []
    all_combos = []  # top10/bottom10 계산용 (gap, qty_pct, profit_pct) 전체 모음
    best = {"gap": None, "qty_pct": None, "qty": None, "profit_pct": float("-inf")}
    worst = {"gap": None, "qty_pct": None, "qty": None, "profit_pct": float("inf")}

    for g in gaps:
        row = []
        for qp in qty_pcts:
            resolved_qty = resolve_trade_qty(initial_shares, qp)
            run_result = _run_grid_fast(
                prices, sell_gap_percent=g, trade_qty=resolved_qty, initial_shares=initial_shares,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "gap": g, "qty_pct": qp, "qty": resolved_qty, "profit_pct": profit_pct, "total": total,
                "매매안했을때자산": hold_only_asset, "차이": total - hold_only_asset,
                "매수횟수": run_result["매수횟수"], "매도횟수": run_result["매도횟수"],
            }
            all_combos.append(combo)
            if profit_pct > best["profit_pct"]:
                best = combo
            if profit_pct < worst["profit_pct"]:
                worst = combo
        grid.append(row)

    # top10/bottom10/ranked는 수익률이 같은 조합을 중복 제거한 뒤 뽑는다: 같은 수익률이면
    # gap이 가장 작은 조합을, gap도 같으면 수량비율이 가장 작은 조합을 남긴다.
    # all_combos는 gap 오름차순(바깥 루프) -> qty_pct 오름차순(안쪽 루프) 순서로 쌓이므로,
    # 특정 수익률이 처음 등장하는 조합이 곧 그 수익률 중 gap 최소/수량비율 최소 조합이다.
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

    # 수량%별 대표값: gap 축을 모아 "이 수량이면 gap을 모르는 상태에서 대체로 어떤 성과를
    # 내는지"를 중앙값으로 요약한다. gap이 커지면 매매가 거의 발생하지 않아 수량과 무관하게
    # 전부 단순보유 수익률로 수렴해버리므로, 그런 조합은 제외하고 실제로 매매가 1회 이상
    # 발생한 gap만 사용해야 수량별 차이가 흐려지지 않는다.
    qty_stats = []
    for qp in qty_pcts:
        traded = [
            c for c in all_combos
            if c["qty_pct"] == qp and (c["매수횟수"] + c["매도횟수"]) > 0
        ]
        median_profit_pct = statistics.median(c["profit_pct"] for c in traded) if traded else None
        qty_stats.append({
            "qty_pct": qp,
            "median_profit_pct": median_profit_pct,
            "traded_gap_count": len(traded),
        })
    recommended_qty = max(
        (s for s in qty_stats if s["median_profit_pct"] is not None),
        key=lambda s: s["median_profit_pct"],
        default=None,
    )

    return {
        "gaps": gaps,
        "qty_pcts": qty_pcts,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "initial_asset": initial_asset,
        "hold_only_asset": hold_only_asset,
        "qty_stats": qty_stats,
        "recommended_qty": recommended_qty,
    }


def compute_profit_heatmap2(
    df: pd.DataFrame,
    gap_values,
    profit_gap_values,
    trade_qty_percent: float,
    profit_recover_percent: float,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    capital: float = None,
) -> dict:
    """
    이익 회수 전용 히트맵: 매매 gap(%) x 이익회수 gap(%) 조합별 최종 수익률(%)을
    계산한다. 거래 수량비율(trade_qty_percent), 회수율(profit_recover_percent), 자본금
    (capital)은 고정 입력값이라 스윕 대상이 아니다 (매매 gap은 매수/매도 동일한 값으로
    스윕한다). 수익률(%)은 이익 회수의 자본금(지정 안 했으면 시작 자산)을 기준으로 계산한다.

    Parameters
    ----------
    gap_values : iterable[float]
        매매 gap(%) 값 목록 (예: range(1, 51) -> 1~50%)
    profit_gap_values : iterable[float]
        이익 회수 gap(%) 값 목록 (예: range(1, 51) -> 1~50%). 자본금 대비 벌어야 할 비율.
    trade_qty_percent : float
        고정 매수/매도 수량 비율(%) — 시작 보유 주식수 대비. resolve_trade_qty()로 시작
        시점에 한 번만 절대 수량으로 변환해 사용한다.
    profit_recover_percent : float
        고정 이익 회수율 (1~100)
    capital : float, optional
        고정 자본금. 지정하지 않으면 시작 자산(initial_shares × 첫날 종가)을 사용한다.
    no_sell, no_buy, allow_negative_cash : grid_trade_strategy() 참고

    Returns
    -------
    dict
        {
            "gaps": [...], "profit_gaps": [...],
            "grid": [[이익gap별 수익률(%), ...], ...]  # grid[i][j] = gaps[i] x profit_gaps[j] 조합
            "best": {"gap":..., "profit_gap":..., "profit_pct":..., "total":...},
            "worst": {"gap":..., "profit_gap":..., "profit_pct":..., "total":...},
            "top10": [{"gap":..., "profit_gap":..., "profit_pct":..., "total":...,
                       "매수횟수":..., "매도횟수":..., "이익회수횟수":...}, ...],  # 상위 10 (내림차순)
            "bottom10": [...],  # 하위 10 (오름차순)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "trade_qty_percent": 고정 거래 수량 비율(%),
            "trade_qty": 위 비율을 시작 보유 주식수 기준으로 변환한 실제 거래 수량,
            "profit_recover_percent": 고정 회수율,
            "capital": 이익 회수에 사용된 자본금 (지정 안 했으면 자동 계산된 시작 자산),
            "initial_asset": 시작 자산 (= capital과 동일한 값. 수익률 계산 기준),
        }
        (각 조합의 "total"은 그 조합으로 백테스트했을 때의 최종 자산)

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
    capital_resolved = capital if capital is not None else initial_asset
    trade_qty_resolved = resolve_trade_qty(initial_shares, trade_qty_percent)
    hold_only_asset = initial_shares * prices[-1]  # 매매 안 했을 때(그냥 보유) 최종 자산

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
                prices, sell_gap_percent=g, trade_qty=trade_qty_resolved, initial_shares=initial_shares,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
                profit_gap_percent=pg, profit_recover_percent=profit_recover_percent,
                capital=capital_resolved,
            )
            total = run_result["total"]
            # 시작 자산은 이익 회수의 자본금(지정 안 했으면 주식수 x 첫날 종가)을 그대로 사용한다.
            profit_pct = (total - capital_resolved) / capital_resolved * 100 if capital_resolved else 0.0
            row.append(profit_pct)
            combo = {
                "gap": g, "profit_gap": pg, "profit_pct": profit_pct, "total": total,
                "매매안했을때자산": hold_only_asset, "차이": total - hold_only_asset,
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
        "trade_qty_percent": trade_qty_percent,
        "trade_qty": trade_qty_resolved,
        "profit_recover_percent": profit_recover_percent,
        "capital": capital_resolved,
        "initial_asset": capital_resolved,
        "hold_only_asset": hold_only_asset,
    }


# 히트맵에서 축/고정값으로 고를 수 있는 4개 피쳐와 기본값.
# "sweep_default": 그 피쳐를 축으로 골랐을 때 스윕 범위(하한, 상한) 기본값 (1% 단위).
# "fixed_default": 그 피쳐를 축으로 안 골랐을 때 적용할 고정값 기본값.
# (/backtest 폼의 기본값과 동일하게 맞춰 두 화면을 오갈 때 값이 낯설지 않게 했다.)
HEATMAP_FEATURES = {
    "gap": {"label": "주가 gap (%)", "sweep_default": (1, 50), "fixed_default": 10},
    "qty_pct": {"label": "매매 수량 (%)", "sweep_default": (1, 50), "fixed_default": 10},
    "profit_gap": {"label": "이익 gap (%)", "sweep_default": (1, 100), "fixed_default": 100},
    "profit_recover": {"label": "이익 회수율 (%)", "sweep_default": (1, 100), "fixed_default": 100},
}


def compute_profit_heatmap_2d(
    df: pd.DataFrame,
    x_feature: str,
    x_values,
    y_feature: str,
    y_values,
    fixed: dict,
    initial_shares: int = 100,
    price_col: str = "종가",
    date_col: str = "날짜",
    no_sell: bool = False,
    no_buy: bool = False,
    allow_negative_cash: bool = False,
    capital: float = None,
) -> dict:
    """
    HEATMAP_FEATURES의 4개 피쳐(주가gap/매매수량/이익gap/이익회수율) 중 2개를 x/y 축으로 골라
    그 조합별 수익률·최종자산을 계산하는 통합 히트맵. `compute_profit_heatmap()`(gap x 수량)과
    `compute_profit_heatmap2()`(gap x 이익gap)를 일반화한 버전이다.

    이익 회수 로직은 축으로 선택되지 않았더라도 **항상 켜진 채로** 계산된다 — 축이 아닌
    이익gap/이익회수율은 `fixed`에 담긴 고정값을 그대로 쓴다. (매매gap/매매수량이 축이 아닐
    때도 마찬가지로 `fixed`의 고정값을 쓴다.)

    Parameters
    ----------
    x_feature, y_feature : str
        HEATMAP_FEATURES의 키 중 하나씩, 서로 달라야 한다 ("gap", "qty_pct", "profit_gap",
        "profit_recover").
    x_values, y_values : iterable[float]
        각 축으로 스윕할 값 목록.
    fixed : dict
        x_feature/y_feature가 아닌 나머지 두 피쳐의 고정값. 예: x_feature="gap",
        y_feature="qty_pct"라면 {"profit_gap": 100, "profit_recover": 100} 형태로 넘긴다.
        (x_feature/y_feature에 해당하는 키가 fixed에 있어도 무시되고 x_values/y_values로
        덮어써진다.)
    capital : float, optional
        수익률(%) 계산 기준이 되는 시작 자산. 지정하지 않으면 시작 자산
        (initial_shares × 첫날 종가)을 그대로 사용한다.
    no_sell, no_buy, allow_negative_cash : grid_trade_strategy() 참고

    Returns
    -------
    dict
        {
            "x_feature": ..., "y_feature": ..., "xs": [...], "ys": [...],
            "grid": [[y별 수익률(%), ...], ...],  # grid[i][j] = xs[i] x ys[j] 조합
            "best": {"x":..., "y":..., "gap":..., "qty_pct":..., "qty":..., "profit_gap":...,
                     "profit_recover":..., "profit_pct":..., "total":...,
                     "매수횟수":..., "매도횟수":..., "이익회수횟수":...},
            "worst": {...동일 구조...},
            "top10": [...],  # 상위 10 (내림차순, 수익률 중복 제거)
            "bottom10": [...],  # 하위 10 (오름차순, 수익률 중복 제거)
            "ranked": [...],   # 전체 조합(중복 제거), max -> min 순
            "raw_ranked": [...],  # 전체 조합(중복 미제거), max -> min 순
            "initial_asset": 시작 자산,
            "fixed": 실제로 적용된 고정값 dict (x_feature/y_feature 제외 2개),
        }
    """
    if df.empty:
        raise ValueError("데이터가 없습니다.")
    if x_feature == y_feature:
        raise ValueError("x축과 y축은 서로 다른 항목이어야 합니다.")
    if x_feature not in HEATMAP_FEATURES or y_feature not in HEATMAP_FEATURES:
        raise ValueError("알 수 없는 히트맵 축입니다.")

    sorted_df = df.sort_values(date_col)
    prices = sorted_df[price_col].tolist()

    first_price = prices[0]
    initial_asset = capital if capital is not None else initial_shares * first_price
    hold_only_asset = initial_shares * prices[-1]  # 매매 안 했을 때(그냥 보유) 최종 자산

    xs = list(x_values)
    ys = list(y_values)

    grid = []
    all_combos = []
    best = {"x": None, "y": None, "profit_pct": float("-inf")}
    worst = {"x": None, "y": None, "profit_pct": float("inf")}

    for xv in xs:
        row = []
        for yv in ys:
            # 4개 피쳐값을 확정: 이번 조합의 x/y 값 + 나머지 두 피쳐는 고정값.
            params = dict(fixed)
            params[x_feature] = xv
            params[y_feature] = yv

            resolved_qty = resolve_trade_qty(initial_shares, params["qty_pct"])
            run_result = _run_grid_fast(
                prices, sell_gap_percent=params["gap"], trade_qty=resolved_qty,
                initial_shares=initial_shares,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
                profit_gap_percent=params["profit_gap"],
                profit_recover_percent=params["profit_recover"],
                capital=initial_asset,
            )
            total = run_result["total"]
            profit_pct = (total - initial_asset) / initial_asset * 100 if initial_asset else 0.0
            row.append(profit_pct)
            combo = {
                "x": xv, "y": yv, "profit_pct": profit_pct, "total": total,
                "매매안했을때자산": hold_only_asset, "차이": total - hold_only_asset,
                "gap": params["gap"], "qty_pct": params["qty_pct"], "qty": resolved_qty,
                "profit_gap": params["profit_gap"], "profit_recover": params["profit_recover"],
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
    # x가 가장 작은 조합을, x도 같으면 y가 가장 작은 조합을 남긴다. all_combos는
    # x 오름차순(바깥 루프) -> y 오름차순(안쪽 루프) 순서로 쌓이므로, 특정 수익률이 처음
    # 등장하는 조합이 곧 그 수익률 중 x 최소/y 최소 조합이다.
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
        "x_feature": x_feature,
        "y_feature": y_feature,
        "xs": xs,
        "ys": ys,
        "grid": grid,
        "best": best,
        "worst": worst,
        "top10": top10,
        "bottom10": bottom10,
        "ranked": ranked,
        "raw_ranked": raw_ranked,
        "initial_asset": initial_asset,
        "hold_only_asset": hold_only_asset,
        "fixed": fixed,
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
    parser.add_argument("--no-sell", action="store_true", help="매도를 하지 않음")
    parser.add_argument("--no-buy", action="store_true", help="매수를 하지 않음")
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
            no_sell=args.no_sell,
            no_buy=args.no_buy,
            allow_negative_cash=args.allow_negative_cash,
        )
        trade_log = result.pop("매매일지")
        buy_gap_display = args.buy_gap if args.buy_gap is not None else args.gap
        print(
            f"\n[그리드 매매 백테스트] 매도gap={args.gap}%  매수gap={buy_gap_display}%  "
            f"거래수량={args.qty}  시작주식수={args.init_shares}  "
            f"매도안함={args.no_sell}  매수안함={args.no_buy}  현금마이너스허용={args.allow_negative_cash}"
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
