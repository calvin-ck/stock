"""
app.py — Flask 웹 앱: 종목 코드와 기간을 입력받아 네이버 금융 데이터를 보여줌.

실행:
    python app.py
접속:
    http://localhost:8000
"""

import os
import re
from datetime import datetime, timedelta
from io import StringIO
from urllib.parse import urlencode

import pandas as pd
from flask import Flask, render_template, request, Response
from core import (
    get_stock_data, get_stock_data_range, get_stock_name, grid_trade_strategy, resolve_trade_qty,
    trend_trade_strategy, compute_trend_heatmap,
    daily_reversal_strategy, compute_daily_heatmap,
    daily_gap_strategy, compute_daily_gap_heatmap,
    daily_reference_strategy, compute_daily_reference_heatmap_2d,
    capital_recovery_strategy, compute_capital_recovery_heatmap,
    compute_profit_heatmap, compute_profit_heatmap2, compute_profit_heatmap_2d,
    compute_price_stats,
    HEATMAP_FEATURES, DAILY3_HEATMAP_FEATURES, SISE_DAY_URL,
)

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
_CODE_CSV_RE = re.compile(r"^(\d+)\.csv$")  # 새 파일명 규칙: 종목당 파일 하나
_LEGACY_DAILY_CSV_RE = re.compile(r"^(\d+)_(\d{8})\.csv$")  # 예전 규칙: 종목_생성일자.csv (마이그레이션 전용)
_LEGACY_CSV_NAME_RE = re.compile(r"^(\d+)_(\d+)days_(\d{8})\.csv$")  # 그 이전 규칙 (마이그레이션 전용)
MAX_RETENTION_DAYS = 730  # 로컬 캐시 최대 보관 기간 (2년) — 이보다 오래된 데이터는 저장 시 잘라냄
DEFAULT_PERIOD_DAYS = 30  # 종료일만 있고 시작일/기간이 둘 다 없을 때 기본 조회 기간


def _first_page_url(code: str) -> str:
    """데이터를 가져오는 첫 번째 요청 URL (page=1)을 만들어 반환."""
    return f"{SISE_DAY_URL}?{urlencode({'code': code, 'page': 1})}"


def _local_csv_path(code: str) -> str:
    return os.path.join(DATA_DIR, f"{code}.csv")


def _read_local_csv(code: str):
    """종목당 파일 하나(data/{code}.csv)를 읽어 반환한다. 없거나 읽을 수 없으면 None."""
    path = _local_csv_path(code)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
    except Exception:
        return None


def _save_local_csv(df: pd.DataFrame, code: str) -> str:
    """종목당 파일 하나(data/{code}.csv)에 저장한다. MAX_RETENTION_DAYS(2년)보다 오래된
    행은 잘라내고, 날짜 기준 중복을 제거한 뒤 내림차순으로 정렬해서 쓴다."""
    cutoff = datetime.now() - timedelta(days=MAX_RETENTION_DAYS)
    trimmed = df[df["날짜"] >= cutoff].drop_duplicates(subset="날짜")
    trimmed = trimmed.sort_values("날짜", ascending=False).reset_index(drop=True)
    path = _local_csv_path(code)
    trimmed.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _migrate_legacy_csvs() -> None:
    """예전 파일명 규칙(`{code}_{days}days_{created}.csv`)으로 저장된 파일들을, 한 단계
    최신인 규칙(`{code}_{생성일자}.csv`)으로 1회 통합한다. 같은 (code, created) 중 days가
    가장 큰 파일만 남기고(그날 조회한 최대범위이므로 나머지의 상위집합) 새 이름으로 바꾸고
    나머지는 지운다. `_migrate_per_day_csvs()`가 이어서 그 결과를 종목당 파일 하나로 다시
    통합한다. 모듈 로드 시 한 번 실행되며, 이미 정리된 상태에서 다시 실행해도 아무 일도
    일어나지 않는다(Flask 개발 서버의 리로더 자식 프로세스가 모듈을 다시 임포트해도 안전)."""
    groups = {}
    for fname in os.listdir(DATA_DIR):
        m = _LEGACY_CSV_NAME_RE.match(fname)
        if not m:
            continue
        code, days, created = m.group(1), int(m.group(2)), m.group(3)
        groups.setdefault((code, created), []).append((days, fname))

    for (code, created), items in groups.items():
        new_path = os.path.join(DATA_DIR, f"{code}_{created}.csv")
        items.sort(key=lambda t: t[0], reverse=True)  # days 내림차순 -> [0]이 최대범위
        _, largest_fname = items[0]
        largest_path = os.path.join(DATA_DIR, largest_fname)

        try:
            if os.path.isfile(largest_path):
                if not os.path.isfile(new_path):
                    os.replace(largest_path, new_path)
                elif os.path.abspath(largest_path) != os.path.abspath(new_path):
                    os.remove(largest_path)
        except FileNotFoundError:
            pass

        for _, fname in items[1:]:
            try:
                os.remove(os.path.join(DATA_DIR, fname))
            except FileNotFoundError:
                pass


def _migrate_per_day_csvs() -> None:
    """예전 규칙(`{code}_{생성일자}.csv`, 하루 단위 스냅샷)으로 저장된 파일들을 종목당 파일
    하나(`{code}.csv`)로 1회 통합한다. 같은 종목의 스냅샷들을 모두 합쳐 날짜 기준 중복을
    제거하고(겹치는 날짜는 어느 스냅샷 것이든 값이 같다 — 실제 시세는 조회 시점과 무관)
    _save_local_csv()로 저장(2년 트림 포함)한 뒤 원본 스냅샷 파일은 지운다. 모듈 로드 시
    `_migrate_legacy_csvs()` 다음에 한 번 실행된다."""
    groups = {}
    for fname in os.listdir(DATA_DIR):
        m = _LEGACY_DAILY_CSV_RE.match(fname)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append(fname)

    for code, fnames in groups.items():
        frames = []
        for fname in fnames:
            try:
                frames.append(pd.read_csv(os.path.join(DATA_DIR, fname), encoding="utf-8-sig", parse_dates=["날짜"]))
            except Exception:
                continue
        existing = _read_local_csv(code)
        if existing is not None and not existing.empty:
            frames.append(existing)
        if frames:
            _save_local_csv(pd.concat(frames, ignore_index=True), code)

        for fname in fnames:
            try:
                os.remove(os.path.join(DATA_DIR, fname))
            except FileNotFoundError:
                pass


_migrate_legacy_csvs()
_migrate_per_day_csvs()


def _resolve_query_range(end_date_str: str, start_date_str: str, period_str: str) -> tuple:
    """
    종료일(없으면 오늘) + (시작일 또는 기간 중 하나, 둘 다 없으면 DEFAULT_PERIOD_DAYS)로
    [start_date, end_date] 범위와 적용된 기간(일)을 계산한다. **시작일이 주어지면 그것을
    우선**해 기간을 역산하고, 아니면 기간(또는 기본값)으로 시작일을 역산한다.
    """
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else datetime.now()
    end_date = min(end_date, datetime.now())  # 미래 날짜는 오늘로 clamp

    if start_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        if start_date > end_date:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
        period = (end_date - start_date).days
    else:
        period = int(period_str) if period_str else DEFAULT_PERIOD_DAYS
        if period <= 0:
            raise ValueError("기간은 1 이상이어야 합니다.")
        start_date = end_date - timedelta(days=period)

    return start_date, end_date, period


def _default_display_range(end_date_str: str, start_date_str: str, period_str: str) -> tuple:
    """
    폼에 채워 보여줄 종료일/시작일/기간 **표시용** 기본값을 계산한다: 종료일이 비어 있으면
    오늘로, 시작일이 비어 있으면(기간 또는 기본값 DEFAULT_PERIOD_DAYS로 역산해) 채운다.
    시작일이 이미 있으면 기간은 건드리지 않는다(시작일이 우선이라 기간은 무시되므로,
    괜히 30을 채워 헷갈리게 하지 않음). 실제 유효성 검사·범위 계산은 _resolve_query_range()
    가 담당하므로, 여기서는 값이 이상해도 예외를 던지지 않고 원래 문자열을 그대로 둔다.
    """
    end_date_str = end_date_str or datetime.now().strftime("%Y-%m-%d")
    if not start_date_str:
        period_str = period_str or str(DEFAULT_PERIOD_DAYS)
        try:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            start_date_str = (end_dt - timedelta(days=int(period_str))).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return end_date_str, start_date_str, period_str


def _ensure_range_df(code: str, start_date: datetime, end_date: datetime) -> tuple:
    """
    종목당 파일 하나(data/{code}.csv)를 기준으로 [start_date, end_date] 범위 데이터를
    반환한다. 캐시가 없으면 그 범위를 통째로 받아 저장하고, 있으면 부족한 구간(최신 쪽/과거
    쪽, 필요하면 둘 다)만 추가로 받아 병합한다 — 네이버 시세는 조회 시점과 무관하게 같은
    과거 날짜는 항상 같은 값이라, 캐시에 이미 있는 구간은 다시 받을 필요가 없다. start_date
    가 MAX_RETENTION_DAYS(2년)보다 과거면 그만큼만 받고 안내 메시지를 반환한다.

    Returns
    -------
    (요청 범위로 슬라이스된 df, 안내 메시지 or None, 네이버를 한 번도 부르지 않았는지,
     실제 적용된 start_date — 2년 clamp가 적용됐으면 그 조정된 값)
    """
    note = None
    oldest_allowed = datetime.now() - timedelta(days=MAX_RETENTION_DAYS)
    if start_date < oldest_allowed:
        note = (
            f"최대 보관 기간(2년)을 넘는 과거 데이터는 제공하지 않아 "
            f"{oldest_allowed.strftime('%Y-%m-%d')}부터로 조정했습니다."
        )
        start_date = oldest_allowed

    cached = _read_local_csv(code)
    used_cache_only = cached is not None and not cached.empty
    fetched = []

    if cached is None or cached.empty:
        fetched.append(get_stock_data_range(code, start_date, end_date))
        used_cache_only = False
    else:
        cached_max = cached["날짜"].max()
        cached_min = cached["날짜"].min()
        if end_date > cached_max:
            fetched.append(get_stock_data_range(code, cached_max + timedelta(days=1), end_date))
            used_cache_only = False
        if start_date < cached_min:
            fetched.append(get_stock_data_range(code, start_date, cached_min - timedelta(days=1)))
            used_cache_only = False

    frames = [f for f in ([cached] if cached is not None else []) + fetched if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), note, used_cache_only, start_date

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="날짜") if len(frames) > 1 else frames[0]
    if fetched:
        _save_local_csv(merged, code)

    sliced = merged[(merged["날짜"] >= start_date) & (merged["날짜"] <= end_date)]
    sliced = sliced.sort_values("날짜", ascending=False).reset_index(drop=True)
    return sliced, note, used_cache_only, start_date


def _list_local_codes():
    """data/ 폴더에 종목당 파일 하나로 저장된 로컬 데이터 목록을 반환한다 (선택 드롭다운용)."""
    codes = []
    for fname in sorted(os.listdir(DATA_DIR)):
        m = _CODE_CSV_RE.match(fname)
        if not m:
            continue
        code = m.group(1)
        path = os.path.join(DATA_DIR, fname)
        try:
            dates = pd.read_csv(path, encoding="utf-8-sig", usecols=["날짜"], parse_dates=["날짜"])["날짜"]
        except Exception:
            continue
        if dates.empty:
            continue
        codes.append({
            "code": code,
            "min_date": dates.min().strftime("%Y-%m-%d"),
            "max_date": dates.max().strftime("%Y-%m-%d"),
            "max_days": (dates.max() - dates.min()).days,
        })
    codes.sort(key=lambda c: c["code"])
    return codes


def _profit_color(value: float, vmax: float) -> str:
    """
    수익률(value, %)을 vmax 기준으로 정규화해 발산형 색상으로 변환.
    상승(수익)은 붉은색, 하락(손실)은 파란색 (이 앱의 상승/하락 색 규칙과 동일).
    """
    if vmax <= 0:
        return "rgb(245,246,248)"
    t = max(-1.0, min(1.0, value / vmax))
    white = (245, 246, 248)
    if t >= 0:
        target = (210, 67, 67)  # --up
    else:
        target = (42, 111, 214)  # --down
        t = -t
    r = round(white[0] + (target[0] - white[0]) * t)
    g = round(white[1] + (target[1] - white[1]) * t)
    b = round(white[2] + (target[2] - white[2]) * t)
    return f"rgb({r},{g},{b})"


def _build_backtest_link(
    code, gap, qty_pct, init_shares, no_sell, no_buy, allow_negative_cash,
    profit_gap=None, profit_recover=None, capital=None, end_date=None, period=None,
):
    """
    히트맵/히트맵2 셀·요약 클릭 시 그 조건 그대로 백테스트 페이지로 이동하는 링크를 만든다.
    profit_gap/profit_recover를 함께 넘기면(히트맵2용) "이익 회수 사용"을 켠 상태로 연결한다.
    """
    params = {
        "code": code,
        "sell_gap": gap,
        "buy_gap": gap,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if no_sell:
        params["no_sell"] = "on"
    if no_buy:
        params["no_buy"] = "on"
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    if profit_gap is not None and profit_recover is not None:
        params["recover_enabled"] = "on"
        params["profit_gap"] = profit_gap
        params["profit_recover"] = profit_recover
        if capital:
            params["capital"] = capital
    return f"/backtest?{urlencode(params)}"


def _build_daily_link(
    code, sell_qty_pct, buy_qty_pct, init_shares,
    allow_negative_cash, sell_above_start_asset_only, end_date=None, period=None,
):
    """히트맵4 셀·요약 클릭 시 그 조건 그대로 /daily 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "sell_qty_pct": sell_qty_pct,
        "buy_qty_pct": buy_qty_pct,
        "init_shares": init_shares,
        # "submitted"을 명시해서, /daily의 기본값(체크박스 True) 추정 로직 대신
        # 여기서 넘긴 sell_above_start_asset_only 값을 그대로 쓰게 한다.
        "submitted": "1",
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    if sell_above_start_asset_only:
        params["sell_above_start_asset_only"] = "on"
    return f"/daily?{urlencode(params)}"


def _build_trend_link(
    code, sell_qty_pct, buy_qty_pct, init_shares,
    no_sell=False, no_buy=False, allow_negative_cash=False, end_date=None, period=None,
):
    """히트맵8 셀·요약 클릭 시 그 조건 그대로 /trend 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "sell_qty_pct": sell_qty_pct,
        "buy_qty_pct": buy_qty_pct,
        "init_shares": init_shares,
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if no_sell:
        params["no_sell"] = "on"
    if no_buy:
        params["no_buy"] = "on"
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    return f"/trend?{urlencode(params)}"


def _build_daily2_link(code, gap_pct, qty_pct, init_shares, no_sell=False, no_buy=False, end_date=None, period=None):
    """히트맵5 셀·요약 클릭 시 그 조건 그대로 /daily2 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if no_sell:
        params["no_sell"] = "on"
    if no_buy:
        params["no_buy"] = "on"
    return f"/daily2?{urlencode(params)}"


def _build_daily3_link(
    code, up_gap_pct, down_gap_pct, qty_pct, init_shares,
    allow_negative_cash=False, no_sell=False, no_buy=False, end_date=None, period=None,
):
    """히트맵6 셀·요약 클릭 시 그 조건 그대로 /daily3 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "up_gap_pct": up_gap_pct,
        "down_gap_pct": down_gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    if no_sell:
        params["no_sell"] = "on"
    if no_buy:
        params["no_buy"] = "on"
    return f"/daily3?{urlencode(params)}"


def _build_recovery_link(
    code, buy_trigger_pct, buy_recover_pct, init_shares,
    base_price=None, allow_negative_cash=False, end_date=None, period=None,
):
    """히트맵7 셀·요약 클릭 시 그 조건 그대로 /recovery 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "buy_trigger_pct": buy_trigger_pct,
        "buy_recover_pct": buy_recover_pct,
        "init_shares": init_shares,
    }
    if base_price is not None:
        params["base_price"] = base_price
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    return f"/recovery?{urlencode(params)}"


@app.route("/", methods=["GET"])
def index():
    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)

    context = {
        "active": "index",
        "code": code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "applied_period": None,
        "error": None,
        "table": None,
        "name": None,
        "chart_labels": None,
        "chart_prices": None,
        "source_url": None,
        "from_cache": False,
    }

    if code:
        try:
            start_date, end_date, _ = _resolve_query_range(end_date_str, start_date_str, period_str)

            context["source_url"] = _first_page_url(code)

            try:
                context["name"] = get_stock_name(code)
            except Exception:
                context["name"] = None

            df, fetch_note, used_cache_only, start_date = _ensure_range_df(code, start_date, end_date)
            context["from_cache"] = used_cache_only
            context["fetch_note"] = fetch_note
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period

            if df.empty:
                context["error"] = fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요."
            else:
                display_df = df.copy()
                display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
                context["table"] = display_df.to_dict(orient="records")

                chart_df = df.sort_values("날짜")
                context["chart_labels"] = chart_df["날짜"].dt.strftime("%m/%d").tolist()
                context["chart_prices"] = chart_df["종가"].tolist()

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"데이터를 가져오는 중 오류가 발생했습니다: {e}"

    return render_template("index.html", **context)


@app.route("/download.csv")
def download_csv():
    """화면에 표시된 것과 동일한 조건(code, end_date, start_date/period)으로 다시 조회해
    CSV로 내려준다."""
    code = request.args.get("code", "").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)

    if not code:
        return "code 파라미터가 필요합니다.", 400

    try:
        start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)
    except ValueError as e:
        return f"입력 오류: {e}", 400

    df, _, _, start_date = _ensure_range_df(code, start_date, end_date)
    if df.empty:
        return "데이터가 없습니다. 종목 코드를 확인해주세요.", 400

    buf = StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)

    filename = f"{code}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/trend", methods=["GET"])
def trend():
    """
    "추세 매매" 백테스트 페이지. grid_trade_strategy()에서 트레일링 고점/저점(max/min)과
    매도gap/매수gap 조건만 뺀 가장 단순한 형태 — 오직 전날 종가 대비 오늘 종가만 보고,
    전날보다 내리면 매도, 오르면 매수한다(하락에 함께 팔고 상승에 함께 사는 추세추종
    방식). 매도/매수 안 함, 현금 부족해도 매수 옵션은 그리드 매매와 동일하게 유지한다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    sell_qty_pct = request.args.get("sell_qty_pct", "10").strip()
    buy_qty_pct = request.args.get("buy_qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"

    context = {
        "active": "trend",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "sell_qty_pct": sell_qty_pct,
        "buy_qty_pct": buy_qty_pct,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "error": None,
        "summary": None,
        "trade_log": None,
        "sell_qty": None,
        "buy_qty": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_hold_only": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            sell_qty_pct_f = float(sell_qty_pct)
            buy_qty_pct_f = float(buy_qty_pct)
            init_i = int(init_shares)
            if sell_qty_pct_f <= 0:
                raise ValueError("매도 수량(%)은 0보다 커야 합니다.")
            if buy_qty_pct_f <= 0:
                raise ValueError("매수 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            sell_qty_i = resolve_trade_qty(init_i, sell_qty_pct_f)
            buy_qty_i = resolve_trade_qty(init_i, buy_qty_pct_f)
            context["sell_qty"] = sell_qty_i
            context["buy_qty"] = buy_qty_i

            result = trend_trade_strategy(
                df, sell_qty=sell_qty_i, buy_qty=buy_qty_i, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
            )

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = init_i * first_price

            hold_only_asset = init_i * result["주가"]

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            context["summary"] = result
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_hold_only"] = chart_hold_only

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("trend.html", **context)


@app.route("/backtest", methods=["GET"])
def backtest():
    """
    저장된 로컬 CSV(data/ 폴더)만 읽어서 그리드 매매 백테스트를 계산하는 페이지.
    네이버에 다시 접속하지 않는다 (부하/차단 방지). "이익 회수 사용"을 체크하면 추가로:
    자본금(비우면 시작 자산) 대비 평가금액(주식평가금액+현금)이 profit_gap% 이상 벌면 그 시점
    평가차익 중 profit_recover%를 현금에서 먼저 충당하고 모자라면 주식을 추가로 매도해서
    마련한 뒤 별도 적립금으로 옮긴다 (이후 매매에 쓰이지 않음). 자본금은 고정값이라 별도로
    갱신되지 않는다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    sell_gap = request.args.get("sell_gap", "10").strip()
    buy_gap = request.args.get("buy_gap", "").strip()  # 비워두면 sell_gap과 동일하게 처리
    qty_pct = request.args.get("qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    recover_enabled = request.args.get("recover_enabled") == "on"
    profit_gap = request.args.get("profit_gap", "100").strip()
    profit_recover = request.args.get("profit_recover", "100").strip()
    capital = request.args.get("capital", "").strip()

    context = {
        "active": "backtest",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "sell_gap": sell_gap,
        "buy_gap": buy_gap,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "recover_enabled": recover_enabled,
        "profit_gap": profit_gap,
        "profit_recover": profit_recover,
        "capital": capital,
        "error": None,
        "summary": None,
        "trade_log": None,
        "recover_log": None,
        "qty": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "effective_buy_gap": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_recover_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_reserve": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            sell_gap_f = float(sell_gap)
            buy_gap_f = float(buy_gap) if buy_gap else None
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if sell_gap_f <= 0:
                raise ValueError("매도 gap은 0보다 커야 합니다.")
            if buy_gap_f is not None and buy_gap_f <= 0:
                raise ValueError("매수 gap은 0보다 커야 합니다.")
            if qty_pct_f <= 0:
                raise ValueError("매수/매도 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            profit_gap_f = None
            profit_recover_f = None
            capital_f = None
            if recover_enabled:
                profit_gap_f = float(profit_gap)
                profit_recover_f = float(profit_recover)
                capital_f = float(capital) if capital else None
                if profit_gap_f <= 0:
                    raise ValueError("이익 회수 gap은 0보다 커야 합니다.")
                if not (1 <= profit_recover_f <= 100):
                    raise ValueError("이익 회수율은 1~100 사이여야 합니다.")
                if capital_f is not None and capital_f <= 0:
                    raise ValueError("자본금은 0보다 커야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            # 매수/매도 수량은 시작 보유 주식수 대비 비율(%)로 입력받아, 시작 시점에 한 번만
            # 절대 수량으로 변환한다 (보유 주식수가 커져도 매번 큰 절대 수량을 입력할 필요 없음).
            qty_i = resolve_trade_qty(init_i, qty_pct_f)
            context["qty"] = qty_i

            result = grid_trade_strategy(
                df,
                trade_qty=qty_i,
                sell_gap_percent=sell_gap_f,
                buy_gap_percent=buy_gap_f,
                initial_shares=init_i,
                no_sell=no_sell,
                no_buy=no_buy,
                allow_negative_cash=allow_negative_cash,
                profit_gap_percent=profit_gap_f,
                profit_recover_percent=profit_recover_f,
                capital=capital_f,
            )
            context["effective_buy_gap"] = buy_gap_f if buy_gap_f is not None else sell_gap_f

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            recover_log = result.pop("이익회수일지", [])
            for row in recover_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")
            result.setdefault("이익회수횟수", 0)

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_recover_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in recover_log
            ]
            # 자산추이(일별 스냅샷)는 chart_labels와 같은 날짜 순서로 쌓이므로 그대로 병렬 배열로 뽑는다.
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_reserve = [row["적립금"] for row in asset_log]

            if recover_enabled:
                # 시작 자산은 이익 회수의 자본금(지정 안 했으면 주식수 x 첫날 종가)을 그대로 사용한다.
                initial_asset = result["자본금"]
            else:
                first_price = float(df.sort_values("날짜")["종가"].iloc[0])
                initial_asset = init_i * first_price

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset  # 그리드 매매 vs 단순 보유 차이

            context["summary"] = result
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["recover_log"] = recover_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_recover_points"] = chart_recover_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_reserve"] = chart_reserve

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("backtest.html", **context)


@app.route("/daily", methods=["GET"])
def daily():
    """
    "일별 방향 매매" 백테스트 페이지. gap이나 고점/저점 추적 없이, 전날 종가보다 오르면
    매도(보유 주식수가 모자라면 건너뜀), 내리면 매수를 시도한다.
    "현금 부족해도 매수" 체크 시 현금 잔고와 무관하게 항상 그대로 매수하고(현금 마이너스
    허용), 체크 안 하면 쌓인 현금 범위 내에서만 매수한다.
    "시작 자산보다 높을 때만 팔기" 체크 시 전날보다 올랐어도 그 시점 평가자산이 시작 자산을
    회복하지 못한 상태면 매도하지 않는다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    sell_qty_pct = request.args.get("sell_qty_pct", "10").strip()
    buy_qty_pct = request.args.get("buy_qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    # 체크박스가 기본 켜짐이라, GET 파라미터가 없는 상태(폼 제출 전 최초 진입)와
    # "사용자가 직접 체크 해제하고 제출"을 구분해야 한다. "submitted" 히든 필드로 폼 제출
    # 여부를 판별해서, 제출 전에는 기본값(True)을, 제출 후에는 실제 체크 여부를 사용한다.
    form_submitted = "submitted" in request.args
    if form_submitted:
        sell_above_start_asset_only = request.args.get("sell_above_start_asset_only") == "on"
    else:
        sell_above_start_asset_only = True

    context = {
        "active": "daily",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "sell_qty_pct": sell_qty_pct,
        "buy_qty_pct": buy_qty_pct,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "sell_above_start_asset_only": sell_above_start_asset_only,
        "error": None,
        "summary": None,
        "trade_log": None,
        "sell_qty": None,
        "buy_qty": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_hold_only": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            sell_qty_pct_f = float(sell_qty_pct)
            buy_qty_pct_f = float(buy_qty_pct)
            init_i = int(init_shares)
            if sell_qty_pct_f <= 0:
                raise ValueError("매도 수량(%)은 0보다 커야 합니다.")
            if buy_qty_pct_f <= 0:
                raise ValueError("매수 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            sell_qty_i = resolve_trade_qty(init_i, sell_qty_pct_f)
            buy_qty_i = resolve_trade_qty(init_i, buy_qty_pct_f)
            context["sell_qty"] = sell_qty_i
            context["buy_qty"] = buy_qty_i

            result = daily_reversal_strategy(
                df, sell_qty=sell_qty_i, buy_qty=buy_qty_i, initial_shares=init_i,
                allow_negative_cash=allow_negative_cash,
                sell_above_start_asset_only=sell_above_start_asset_only,
            )

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            # 매매 안 했을 때(시작 주식수만 계속 보유) 총자산의 날짜별 추이 — 그래프 비교용
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = init_i * first_price

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            context["summary"] = result
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_hold_only"] = chart_hold_only

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("daily.html", **context)


@app.route("/daily2", methods=["GET"])
def daily2():
    """
    "일별 매매 2" 백테스트 페이지. 트레일링 고점(max)/저점(min) 기준으로, max에서 gap%
    떨어지면 매수, min에서 gap% 오르면 매도한다 (grid_trade_strategy()와 매도/매수가
    서로 뒤바뀐 구조). 매도/매수 수량은 하나의 값(시작 보유 주식수 대비 %)을 공유한다.
    변수는 gap%와 수량% 딱 2개뿐이다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    gap_pct = request.args.get("gap_pct", "3").strip()
    qty_pct = request.args.get("qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"

    context = {
        "active": "daily2",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "error": None,
        "summary": None,
        "trade_log": None,
        "qty": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_hold_only": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            gap_pct_f = float(gap_pct)
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if gap_pct_f <= 0:
                raise ValueError("등락폭 gap(%)은 0보다 커야 합니다.")
            if qty_pct_f <= 0:
                raise ValueError("매수/매도 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            qty_i = resolve_trade_qty(init_i, qty_pct_f)
            context["qty"] = qty_i

            result = daily_gap_strategy(
                df, trade_qty=qty_i, gap_percent=gap_pct_f, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy,
            )

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = init_i * first_price

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            context["summary"] = result
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_hold_only"] = chart_hold_only

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("daily2.html", **context)


@app.route("/daily3", methods=["GET"])
def daily3():
    """
    "일별 방향3" 백테스트 페이지. 정적인 기준가 기준으로, 현재가가 기준가 대비 up_gap%
    이상 오르면 매도, down_gap% 이상 내리면 매수한다. 기준가는 첫날 종가로 시작해 매매가
    일어날 때만 그 거래가로 갱신된다(트레일링 고점/저점을 계속 따라가는 그리드 매매·
    일별 매매2와 달리, 매매 없이는 절대 움직이지 않는다). down_gap을 비워두면 up_gap과
    동일하게 취급한다. 매도/매수 수량은 하나의 값(시작 보유 주식수 대비 %)을 공유한다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    up_gap_pct = request.args.get("up_gap_pct", "5").strip()
    down_gap_pct = request.args.get("down_gap_pct", "").strip()  # 비워두면 up_gap_pct와 동일
    qty_pct = request.args.get("qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"

    context = {
        "active": "daily3",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "up_gap_pct": up_gap_pct,
        "down_gap_pct": down_gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "error": None,
        "summary": None,
        "trade_log": None,
        "qty": None,
        "effective_down_gap": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_hold_only": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            up_gap_f = float(up_gap_pct)
            down_gap_f = float(down_gap_pct) if down_gap_pct else None
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if up_gap_f <= 0:
                raise ValueError("상승 gap은 0보다 커야 합니다.")
            if down_gap_f is not None and down_gap_f <= 0:
                raise ValueError("하락 gap은 0보다 커야 합니다.")
            if qty_pct_f <= 0:
                raise ValueError("매수/매도 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            qty_i = resolve_trade_qty(init_i, qty_pct_f)
            context["qty"] = qty_i

            result = daily_reference_strategy(
                df, trade_qty=qty_i, up_gap_percent=up_gap_f, down_gap_percent=down_gap_f,
                initial_shares=init_i, allow_negative_cash=allow_negative_cash,
                no_sell=no_sell, no_buy=no_buy,
            )
            context["effective_down_gap"] = down_gap_f if down_gap_f is not None else up_gap_f

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = init_i * first_price

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            context["summary"] = result
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_hold_only"] = chart_hold_only

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("daily3.html", **context)


@app.route("/recovery", methods=["GET"])
def recovery():
    """
    "자본 회수" 백테스트 페이지. 기준가(비우면 첫날 종가, 입력하면 그 값)로 자본금
    (기준가 × 시작 보유 주식수)을 정하고, 주식 평가금액이 자본금을 넘어서면(주가가 기준가
    보다 오르면) 넘어선 만큼만 정수 주식 단위로 팔아 현금으로 회수한다. 반대로 주식
    평가금액이 자본금 대비 매수 트리거 gap(%) 이상 내려가면 부족분 중 매수 회복률(%)만큼만
    사서 채운다(자본금을 넘어서 사지는 않음). 기준가는 매매가 일어나도 갱신되지 않고 처음
    값 그대로 고정된다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    base_price = request.args.get("base_price", "").strip()  # 비우면 첫날 종가
    buy_trigger_pct = request.args.get("buy_trigger_pct", "10").strip()
    buy_recover_pct = request.args.get("buy_recover_pct", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"

    context = {
        "active": "recovery",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "base_price": base_price,
        "buy_trigger_pct": buy_trigger_pct,
        "buy_recover_pct": buy_recover_pct,
        "allow_negative_cash": allow_negative_cash,
        "error": None,
        "summary": None,
        "trade_log": None,
        "resolved_base_price": None,
        "capital": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "vs_hold": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_sell_points": None,
        "chart_buy_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_hold_only": None,
        "chart_capital": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            base_price_f = float(base_price) if base_price else None
            buy_trigger_f = float(buy_trigger_pct)
            buy_recover_f = float(buy_recover_pct)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            if base_price_f is not None and base_price_f <= 0:
                raise ValueError("기준가는 0보다 커야 합니다.")
            if buy_trigger_f <= 0:
                raise ValueError("매수 트리거 gap은 0보다 커야 합니다.")
            if buy_recover_f <= 0:
                raise ValueError("매수 회복률은 0보다 커야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            result = capital_recovery_strategy(
                df, initial_shares=init_i, base_price=base_price_f,
                buy_trigger_percent=buy_trigger_f, buy_recover_percent=buy_recover_f,
                allow_negative_cash=allow_negative_cash,
            )

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            resolved_base = result["기준가"]
            capital = result["자본금"]
            chart_capital = [capital] * len(asset_log)

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = init_i * first_price
            hold_only_asset = init_i * result["주가"]

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0
            vs_hold = result["total"] - hold_only_asset

            context["summary"] = result
            context["resolved_base_price"] = resolved_base
            context["capital"] = capital
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["vs_hold"] = vs_hold
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["trade_log"] = trade_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_hold_only"] = chart_hold_only
            context["chart_capital"] = chart_capital

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("recovery.html", **context)


@app.route("/heatmap", methods=["GET"])
def heatmap():
    """
    gap 1~50%(1% 단위) x 매매수량(시작 보유 주식수 대비 %) 1~50%(1% 단위) = 2,500가지 조합의
    수익률을 계산해 히트맵으로 보여준다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    capital = request.args.get("capital", "").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    gap_min = request.args.get("gap_min", "1").strip()
    gap_max = request.args.get("gap_max", "50").strip()
    qty_pct_min = request.args.get("qty_pct_min", "1").strip()
    qty_pct_max = request.args.get("qty_pct_max", "50").strip()

    context = {
        "active": "heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "capital": capital,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "qty_pct_min": qty_pct_min,
        "qty_pct_max": qty_pct_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "gaps": None,
        "qty_pcts": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "qty_stats": None,
        "recommended_qty": None,
        "recommended_qty_link": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            capital_f = float(capital) if capital else None
            if capital_f is not None and capital_f <= 0:
                raise ValueError("자본금은 0보다 커야 합니다.")

            gap_min_i = int(gap_min)
            gap_max_i = int(gap_max)
            qty_pct_min_i = int(qty_pct_min)
            qty_pct_max_i = int(qty_pct_max)
            if gap_min_i < 1 or qty_pct_min_i < 1:
                raise ValueError("gap/수량 하한은 1 이상이어야 합니다.")
            if gap_max_i < gap_min_i:
                raise ValueError("gap 상한은 하한보다 크거나 같아야 합니다.")
            if qty_pct_max_i < qty_pct_min_i:
                raise ValueError("수량 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            gap_values = range(gap_min_i, gap_max_i + 1)  # 1% 단위
            qty_percent_values = range(qty_pct_min_i, qty_pct_max_i + 1)  # 시작 보유 주식수 대비 1% 단위

            result = compute_profit_heatmap(
                df, gap_values, qty_percent_values, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
                capital=capital_f,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for gi, g in enumerate(result["gaps"]):
                for qi, qp in enumerate(result["qty_pcts"]):
                    pct = result["grid"][gi][qi]
                    is_best = (g == result["best"]["gap"] and qp == result["best"]["qty_pct"])
                    is_worst = (g == result["worst"]["gap"] and qp == result["worst"]["qty_pct"])
                    cells.append({
                        "gap": g, "qty_pct": qp, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _build_backtest_link(code, g, qp, init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["qty_pcts"] = result["qty_pcts"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _build_backtest_link(
                code, result["best"]["gap"], result["best"]["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period
            )
            context["worst_link"] = _build_backtest_link(
                code, result["worst"]["gap"], result["worst"]["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period
            )

            context["qty_stats"] = result["qty_stats"]
            context["recommended_qty"] = result["recommended_qty"]
            if result["recommended_qty"] is not None:
                recover_params = {
                    "code": code,
                    "end_date": end_date_iso,
                    "period": applied_period,
                    "init_shares": init_i,
                    "qty_pct": result["recommended_qty"]["qty_pct"],
                }
                if no_sell:
                    recover_params["no_sell"] = "on"
                if no_buy:
                    recover_params["no_buy"] = "on"
                if allow_negative_cash:
                    recover_params["allow_negative_cash"] = "on"
                if capital_f:
                    recover_params["capital"] = capital_f
                context["recommended_qty_link"] = f"/heatmap2?{urlencode(recover_params)}"

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_backtest_link(
                        code, combo["gap"], combo["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period
                    ),
                }

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap.html", **context)


@app.route("/heatmap2", methods=["GET"])
def heatmap2():
    """
    이익 회수 전용 히트맵: 매매 gap(%) x 이익회수 gap(%) 조합의 수익률을 계산한다.
    거래 수량과 회수율은 폼에서 고정 숫자로 입력받는다 (스윕 대상이 아님).
    저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    qty_pct = request.args.get("qty_pct", "10").strip()
    profit_recover = request.args.get("profit_recover", "100").strip()
    capital = request.args.get("capital", "").strip()
    gap_min = request.args.get("gap_min", "1").strip()
    gap_max = request.args.get("gap_max", "50").strip()
    profit_gap_min = request.args.get("profit_gap_min", "1").strip()
    profit_gap_max = request.args.get("profit_gap_max", "100").strip()

    context = {
        "active": "heatmap2",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "qty_pct": qty_pct,
        "profit_recover": profit_recover,
        "capital": capital,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "profit_gap_min": profit_gap_min,
        "profit_gap_max": profit_gap_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "gaps": None,
        "profit_gaps": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "qty": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "capital_used": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            qty_pct_f = float(qty_pct)
            if qty_pct_f <= 0:
                raise ValueError("거래 수량(%)은 0보다 커야 합니다.")

            profit_recover_f = float(profit_recover)
            if not (1 <= profit_recover_f <= 100):
                raise ValueError("이익 회수율은 1~100 사이여야 합니다.")

            capital_f = float(capital) if capital else None
            if capital_f is not None and capital_f <= 0:
                raise ValueError("자본금은 0보다 커야 합니다.")

            gap_min_i = int(gap_min)
            gap_max_i = int(gap_max)
            profit_gap_min_i = int(profit_gap_min)
            profit_gap_max_i = int(profit_gap_max)
            if gap_min_i < 1 or profit_gap_min_i < 1:
                raise ValueError("gap/이익 하한은 1 이상이어야 합니다.")
            if gap_max_i < gap_min_i:
                raise ValueError("gap 상한은 하한보다 크거나 같아야 합니다.")
            if profit_gap_max_i < profit_gap_min_i:
                raise ValueError("이익 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            gap_values = range(gap_min_i, gap_max_i + 1)  # 1% 단위
            profit_gap_values = range(profit_gap_min_i, profit_gap_max_i + 1)  # 1% 단위

            result = compute_profit_heatmap2(
                df, gap_values, profit_gap_values, trade_qty_percent=qty_pct_f,
                profit_recover_percent=profit_recover_f, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
                capital=capital_f,
            )
            context["qty"] = result["trade_qty"]

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for gi, g in enumerate(result["gaps"]):
                for pi, pg in enumerate(result["profit_gaps"]):
                    pct = result["grid"][gi][pi]
                    is_best = (g == result["best"]["gap"] and pg == result["best"]["profit_gap"])
                    is_worst = (g == result["worst"]["gap"] and pg == result["worst"]["profit_gap"])
                    cells.append({
                        "gap": g, "profit_gap": pg, "pct": pct,
                        "color": _profit_color(pct, vmax),
                        "link": _build_backtest_link(
                            code, g, qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                            profit_gap=pg, profit_recover=profit_recover_f, capital=capital_f,
                            end_date=end_date_iso, period=applied_period,
                        ),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["profit_gaps"] = result["profit_gaps"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["capital_used"] = result["capital"]
            context["best_link"] = _build_backtest_link(
                code, result["best"]["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                profit_gap=result["best"]["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
                end_date=end_date_iso, period=applied_period,
            )
            context["worst_link"] = _build_backtest_link(
                code, result["worst"]["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                profit_gap=result["worst"]["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
                end_date=end_date_iso, period=applied_period,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_backtest_link(
                        code, combo["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                        profit_gap=combo["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
                        end_date=end_date_iso, period=applied_period,
                    ),
                }

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap2.html", **context)


@app.route("/heatmap3", methods=["GET"])
def heatmap3():
    """
    주가gap · 매매수량 · 이익gap · 이익회수율 4개 피쳐 중 2개를 x/y 축으로 골라 그
    조합별 수익률·최종자산을 계산하는 통합 히트맵. /heatmap(gap x 수량), /heatmap2
    (gap x 이익gap)를 일반화한 페이지 — 축으로 고르지 않은 나머지 두 피쳐는 고정값으로
    적용되며, 이익 회수 로직은 축 선택과 무관하게 항상 켜진 채로 계산된다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    capital = request.args.get("capital", "").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"

    x_feature = request.args.get("x_feature", "gap").strip()
    y_feature = request.args.get("y_feature", "qty_pct").strip()

    # 4개 피쳐 전부에 대해 스윕범위(min/max)와 고정값 입력을 함께 받아둔다 — 실제로는
    # x_feature/y_feature에 해당하는 두 개만 스윕범위로, 나머지 두 개만 고정값으로 쓰인다.
    feature_inputs = {}
    for feat, meta in HEATMAP_FEATURES.items():
        sweep_min_default, sweep_max_default = meta["sweep_default"]
        feature_inputs[feat] = {
            "min": request.args.get(f"{feat}_min", str(sweep_min_default)).strip(),
            "max": request.args.get(f"{feat}_max", str(sweep_max_default)).strip(),
            "fixed": request.args.get(f"{feat}_fixed", str(meta["fixed_default"])).strip(),
        }

    context = {
        "active": "heatmap3",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "capital": capital,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "features": HEATMAP_FEATURES,
        "feature_inputs": feature_inputs,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "x_label": None,
        "y_label": None,
        "xs": None,
        "ys": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            if x_feature == y_feature:
                raise ValueError("x축과 y축은 서로 다른 항목이어야 합니다.")
            if x_feature not in HEATMAP_FEATURES or y_feature not in HEATMAP_FEATURES:
                raise ValueError("알 수 없는 축입니다.")

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            capital_f = float(capital) if capital else None
            if capital_f is not None and capital_f <= 0:
                raise ValueError("자본금은 0보다 커야 합니다.")

            # 축(x/y)은 min~max 스윕 범위로, 나머지 두 피쳐는 고정값 하나로 파싱한다.
            sweep_ranges = {}
            fixed_values = {}
            for feat, meta in HEATMAP_FEATURES.items():
                if feat in (x_feature, y_feature):
                    lo = int(feature_inputs[feat]["min"])
                    hi = int(feature_inputs[feat]["max"])
                    if lo < 1:
                        raise ValueError(f"{meta['label']} 하한은 1 이상이어야 합니다.")
                    if hi < lo:
                        raise ValueError(f"{meta['label']} 상한은 하한보다 크거나 같아야 합니다.")
                    sweep_ranges[feat] = range(lo, hi + 1)
                else:
                    val = float(feature_inputs[feat]["fixed"])
                    if val <= 0:
                        raise ValueError(f"{meta['label']} 고정값은 0보다 커야 합니다.")
                    if feat == "profit_recover" and not (1 <= val <= 100):
                        raise ValueError("이익 회수율 고정값은 1~100 사이여야 합니다.")
                    fixed_values[feat] = val

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            result = compute_profit_heatmap_2d(
                df, x_feature, sweep_ranges[x_feature], y_feature, sweep_ranges[y_feature],
                fixed=fixed_values, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
                capital=capital_f,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _params_for(xv, yv):
                params = dict(fixed_values)
                params[x_feature] = xv
                params[y_feature] = yv
                return params

            def _link_for(params):
                return _build_backtest_link(
                    code, params["gap"], params["qty_pct"], init_i, no_sell, no_buy,
                    allow_negative_cash, profit_gap=params["profit_gap"],
                    profit_recover=params["profit_recover"], capital=capital_f,
                    end_date=end_date_iso, period=applied_period,
                )

            def _link_for_combo(combo):
                return _link_for({
                    "gap": combo["gap"], "qty_pct": combo["qty_pct"],
                    "profit_gap": combo["profit_gap"], "profit_recover": combo["profit_recover"],
                })

            cells = []
            for xi, xv in enumerate(result["xs"]):
                for yi, yv in enumerate(result["ys"]):
                    pct = result["grid"][xi][yi]
                    is_best = (xv == result["best"]["x"] and yv == result["best"]["y"])
                    is_worst = (xv == result["worst"]["x"] and yv == result["worst"]["y"])
                    cells.append({
                        "x": xv, "y": yv, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _link_for(_params_for(xv, yv)),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["x_label"] = HEATMAP_FEATURES[x_feature]["label"]
            context["y_label"] = HEATMAP_FEATURES[y_feature]["label"]
            context["xs"] = result["xs"]
            context["ys"] = result["ys"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _link_for_combo(result["best"])
            context["worst_link"] = _link_for_combo(result["worst"])

            def _with_link(combo):
                return {**combo, "link": _link_for_combo(combo)}

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap3.html", **context)


@app.route("/heatmap4", methods=["GET"])
def heatmap4():
    """
    daily_reversal_strategy() 전용 히트맵: 매도수량% 1~50% x 매수수량% 1~50%(둘 다 1% 단위,
    시작 보유 주식수 대비) = 2,500가지 조합의 수익률을 계산해 히트맵으로 보여준다.
    저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    sell_above_start_asset_only = request.args.get("sell_above_start_asset_only") == "on"
    sell_qty_pct_min = request.args.get("sell_qty_pct_min", "1").strip()
    sell_qty_pct_max = request.args.get("sell_qty_pct_max", "50").strip()
    buy_qty_pct_min = request.args.get("buy_qty_pct_min", "1").strip()
    buy_qty_pct_max = request.args.get("buy_qty_pct_max", "50").strip()

    context = {
        "active": "heatmap4",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "sell_above_start_asset_only": sell_above_start_asset_only,
        "sell_qty_pct_min": sell_qty_pct_min,
        "sell_qty_pct_max": sell_qty_pct_max,
        "buy_qty_pct_min": buy_qty_pct_min,
        "buy_qty_pct_max": buy_qty_pct_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "sell_pcts": None,
        "buy_pcts": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            sell_min_i = int(sell_qty_pct_min)
            sell_max_i = int(sell_qty_pct_max)
            buy_min_i = int(buy_qty_pct_min)
            buy_max_i = int(buy_qty_pct_max)
            if sell_min_i < 1 or buy_min_i < 1:
                raise ValueError("매도/매수 수량 하한은 1 이상이어야 합니다.")
            if sell_max_i < sell_min_i:
                raise ValueError("매도 수량 상한은 하한보다 크거나 같아야 합니다.")
            if buy_max_i < buy_min_i:
                raise ValueError("매수 수량 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            sell_qty_pct_values = range(sell_min_i, sell_max_i + 1)  # 시작 보유 주식수 대비 1% 단위
            buy_qty_pct_values = range(buy_min_i, buy_max_i + 1)

            result = compute_daily_heatmap(
                df, sell_qty_pct_values, buy_qty_pct_values, initial_shares=init_i,
                allow_negative_cash=allow_negative_cash,
                sell_above_start_asset_only=sell_above_start_asset_only,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for si, sp in enumerate(result["sell_pcts"]):
                for bi, bp in enumerate(result["buy_pcts"]):
                    pct = result["grid"][si][bi]
                    is_best = (sp == result["best"]["sell_pct"] and bp == result["best"]["buy_pct"])
                    is_worst = (sp == result["worst"]["sell_pct"] and bp == result["worst"]["buy_pct"])
                    cells.append({
                        "sell_pct": sp, "buy_pct": bp, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _build_daily_link(
                            code, sp, bp, init_i, allow_negative_cash, sell_above_start_asset_only,
                            end_date=end_date_iso, period=applied_period,
                        ),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["sell_pcts"] = result["sell_pcts"]
            context["buy_pcts"] = result["buy_pcts"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _build_daily_link(
                code, result["best"]["sell_pct"], result["best"]["buy_pct"],
                init_i, allow_negative_cash, sell_above_start_asset_only, end_date=end_date_iso, period=applied_period,
            )
            context["worst_link"] = _build_daily_link(
                code, result["worst"]["sell_pct"], result["worst"]["buy_pct"],
                init_i, allow_negative_cash, sell_above_start_asset_only, end_date=end_date_iso, period=applied_period,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_daily_link(
                        code, combo["sell_pct"], combo["buy_pct"],
                        init_i, allow_negative_cash, sell_above_start_asset_only, end_date=end_date_iso, period=applied_period,
                    ),
                }

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap4.html", **context)


@app.route("/heatmap5", methods=["GET"])
def heatmap5():
    """
    daily_gap_strategy() 전용 히트맵: 등락폭 gap% 1~50% x 매매수량% 1~50%(둘 다 1% 단위,
    시작 보유 주식수 대비) = 2,500가지 조합의 수익률을 계산해 히트맵으로 보여준다.
    저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    gap_pct_min = request.args.get("gap_pct_min", "1").strip()
    gap_pct_max = request.args.get("gap_pct_max", "50").strip()
    qty_pct_min = request.args.get("qty_pct_min", "1").strip()
    qty_pct_max = request.args.get("qty_pct_max", "50").strip()

    context = {
        "active": "heatmap5",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "gap_pct_min": gap_pct_min,
        "gap_pct_max": gap_pct_max,
        "qty_pct_min": qty_pct_min,
        "qty_pct_max": qty_pct_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "gaps": None,
        "qty_pcts": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            gap_min_i = int(gap_pct_min)
            gap_max_i = int(gap_pct_max)
            qty_min_i = int(qty_pct_min)
            qty_max_i = int(qty_pct_max)
            if gap_min_i < 1 or qty_min_i < 1:
                raise ValueError("gap/수량 하한은 1 이상이어야 합니다.")
            if gap_max_i < gap_min_i:
                raise ValueError("gap 상한은 하한보다 크거나 같아야 합니다.")
            if qty_max_i < qty_min_i:
                raise ValueError("수량 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            gap_values = range(gap_min_i, gap_max_i + 1)  # 1% 단위
            qty_percent_values = range(qty_min_i, qty_max_i + 1)  # 시작 보유 주식수 대비 1% 단위

            result = compute_daily_gap_heatmap(
                df, gap_values, qty_percent_values, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for gi, g in enumerate(result["gaps"]):
                for qi, qp in enumerate(result["qty_pcts"]):
                    pct = result["grid"][gi][qi]
                    is_best = (g == result["best"]["gap"] and qp == result["best"]["qty_pct"])
                    is_worst = (g == result["worst"]["gap"] and qp == result["worst"]["qty_pct"])
                    cells.append({
                        "gap": g, "qty_pct": qp, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _build_daily2_link(code, g, qp, init_i, no_sell, no_buy, end_date=end_date_iso, period=applied_period),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["qty_pcts"] = result["qty_pcts"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _build_daily2_link(
                code, result["best"]["gap"], result["best"]["qty_pct"], init_i, no_sell, no_buy, end_date=end_date_iso, period=applied_period
            )
            context["worst_link"] = _build_daily2_link(
                code, result["worst"]["gap"], result["worst"]["qty_pct"], init_i, no_sell, no_buy, end_date=end_date_iso, period=applied_period
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_daily2_link(
                        code, combo["gap"], combo["qty_pct"], init_i, no_sell, no_buy, end_date=end_date_iso, period=applied_period
                    ),
                }

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap5.html", **context)


@app.route("/heatmap6", methods=["GET"])
def heatmap6():
    """
    상승gap · 하락gap · 매도/매수수량 3개 피쳐 중 2개를 x/y 축으로 골라 그 조합별
    수익률·최종자산을 계산하는 daily_reference_strategy() 전용 통합 히트맵
    (/heatmap3처럼 그리드 매매 통합 히트맵과 같은 방식). 기본은 x=상승gap, y=수량%이고,
    축으로 고르지 않은 하락gap은 값을 비워두면 daily_reference_strategy()의 기본 동작과
    동일하게 그 셀의 상승gap과 같은 값을 쓴다("하락gap을 상승gap과 동일하게").
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"

    x_feature = request.args.get("x_feature", "up_gap").strip()
    y_feature = request.args.get("y_feature", "qty_pct").strip()

    # 3개 피쳐 전부에 대해 스윕범위(min/max)와 고정값 입력을 함께 받아둔다 — 실제로는
    # x_feature/y_feature에 해당하는 두 개만 스윕범위로, 나머지 한 개만 고정값으로 쓰인다.
    feature_inputs = {}
    for feat, meta in DAILY3_HEATMAP_FEATURES.items():
        sweep_min_default, sweep_max_default = meta["sweep_default"]
        fixed_default_str = "" if meta["fixed_default"] is None else str(meta["fixed_default"])
        feature_inputs[feat] = {
            "min": request.args.get(f"{feat}_min", str(sweep_min_default)).strip(),
            "max": request.args.get(f"{feat}_max", str(sweep_max_default)).strip(),
            "fixed": request.args.get(f"{feat}_fixed", fixed_default_str).strip(),
        }

    context = {
        "active": "heatmap6",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "features": DAILY3_HEATMAP_FEATURES,
        "feature_inputs": feature_inputs,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "x_label": None,
        "y_label": None,
        "xs": None,
        "ys": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            if x_feature == y_feature:
                raise ValueError("x축과 y축은 서로 다른 항목이어야 합니다.")
            if x_feature not in DAILY3_HEATMAP_FEATURES or y_feature not in DAILY3_HEATMAP_FEATURES:
                raise ValueError("알 수 없는 축입니다.")

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            # 축(x/y)은 min~max 스윕 범위로, 나머지 한 피쳐는 고정값 하나로 파싱한다.
            # 하락gap이 고정값이고 비어 있으면 None으로 둬서(상승gap과 동일 처리) 넘긴다.
            sweep_ranges = {}
            fixed_values = {}
            for feat, meta in DAILY3_HEATMAP_FEATURES.items():
                if feat in (x_feature, y_feature):
                    lo = int(feature_inputs[feat]["min"])
                    hi = int(feature_inputs[feat]["max"])
                    if lo < 1:
                        raise ValueError(f"{meta['label']} 하한은 1 이상이어야 합니다.")
                    if hi < lo:
                        raise ValueError(f"{meta['label']} 상한은 하한보다 크거나 같아야 합니다.")
                    sweep_ranges[feat] = range(lo, hi + 1)
                else:
                    raw = feature_inputs[feat]["fixed"]
                    if not raw:
                        if feat != "down_gap":
                            raise ValueError(f"{meta['label']} 고정값을 입력해주세요.")
                        fixed_values[feat] = None
                    else:
                        val = float(raw)
                        if val <= 0:
                            raise ValueError(f"{meta['label']} 고정값은 0보다 커야 합니다.")
                        fixed_values[feat] = val

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            result = compute_daily_reference_heatmap_2d(
                df, x_feature, sweep_ranges[x_feature], y_feature, sweep_ranges[y_feature],
                fixed=fixed_values, initial_shares=init_i,
                allow_negative_cash=allow_negative_cash, no_sell=no_sell, no_buy=no_buy,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _params_for(xv, yv):
                # 3개 피쳐값을 확정: 이번 조합의 x/y 값 + 나머지 한 피쳐는 고정값.
                # down_gap이 고정값이면서 비어 있으면(None) 상승gap 값을 그대로 미러링한다.
                params = dict(fixed_values)
                params[x_feature] = xv
                params[y_feature] = yv
                if params.get("down_gap") is None:
                    params["down_gap"] = params["up_gap"]
                return params

            def _link_for(params):
                return _build_daily3_link(
                    code, params["up_gap"], params["down_gap"], params["qty_pct"], init_i,
                    allow_negative_cash, no_sell, no_buy, end_date=end_date_iso, period=applied_period,
                )

            def _link_for_combo(combo):
                return _link_for({
                    "up_gap": combo["up_gap"], "down_gap": combo["down_gap"], "qty_pct": combo["qty_pct"],
                })

            cells = []
            for xi, xv in enumerate(result["xs"]):
                for yi, yv in enumerate(result["ys"]):
                    pct = result["grid"][xi][yi]
                    is_best = (xv == result["best"]["x"] and yv == result["best"]["y"])
                    is_worst = (xv == result["worst"]["x"] and yv == result["worst"]["y"])
                    cells.append({
                        "x": xv, "y": yv, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _link_for(_params_for(xv, yv)),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["x_label"] = DAILY3_HEATMAP_FEATURES[x_feature]["label"]
            context["y_label"] = DAILY3_HEATMAP_FEATURES[y_feature]["label"]
            context["xs"] = result["xs"]
            context["ys"] = result["ys"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _link_for_combo(result["best"])
            context["worst_link"] = _link_for_combo(result["worst"])

            def _with_link(combo):
                return {**combo, "link": _link_for_combo(combo)}

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap6.html", **context)


@app.route("/heatmap7", methods=["GET"])
def heatmap7():
    """
    capital_recovery_strategy() 전용 히트맵: 매수 트리거 gap(%) x 매수 회복률(%) 조합별
    수익률을 계산해 히트맵으로 보여준다. 기준가/시작 보유 주식수/현금 부족해도 매수 여부는
    폼에서 고정값으로 입력한다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    base_price = request.args.get("base_price", "").strip()  # 비우면 첫날 종가
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    trigger_min = request.args.get("trigger_min", "1").strip()
    trigger_max = request.args.get("trigger_max", "50").strip()
    recover_min = request.args.get("recover_min", "1").strip()
    recover_max = request.args.get("recover_max", "100").strip()

    context = {
        "active": "heatmap7",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "base_price": base_price,
        "allow_negative_cash": allow_negative_cash,
        "trigger_min": trigger_min,
        "trigger_max": trigger_max,
        "recover_min": recover_min,
        "recover_max": recover_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "triggers": None,
        "recovers": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "resolved_base_price": None,
        "capital": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            base_price_f = float(base_price) if base_price else None
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            if base_price_f is not None and base_price_f <= 0:
                raise ValueError("기준가는 0보다 커야 합니다.")

            trigger_min_i = int(trigger_min)
            trigger_max_i = int(trigger_max)
            recover_min_i = int(recover_min)
            recover_max_i = int(recover_max)
            if trigger_min_i < 1 or recover_min_i < 1:
                raise ValueError("트리거/회복률 하한은 1 이상이어야 합니다.")
            if trigger_max_i < trigger_min_i:
                raise ValueError("트리거 상한은 하한보다 크거나 같아야 합니다.")
            if recover_max_i < recover_min_i:
                raise ValueError("회복률 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            trigger_values = range(trigger_min_i, trigger_max_i + 1)  # 1% 단위
            recover_values = range(recover_min_i, recover_max_i + 1)  # 1% 단위

            result = compute_capital_recovery_heatmap(
                df, trigger_values, recover_values, initial_shares=init_i,
                base_price=base_price_f, allow_negative_cash=allow_negative_cash,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _link_for_combo(combo):
                return _build_recovery_link(
                    code, combo["buy_trigger_pct"], combo["buy_recover_pct"], init_i,
                    base_price=base_price_f, allow_negative_cash=allow_negative_cash,
                    end_date=end_date_iso, period=applied_period,
                )

            cells = []
            for ti, t in enumerate(result["triggers"]):
                for ri, r in enumerate(result["recovers"]):
                    pct = result["grid"][ti][ri]
                    is_best = (t == result["best"]["buy_trigger_pct"] and r == result["best"]["buy_recover_pct"])
                    is_worst = (t == result["worst"]["buy_trigger_pct"] and r == result["worst"]["buy_recover_pct"])
                    cells.append({
                        "buy_trigger_pct": t, "buy_recover_pct": r, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _link_for_combo({"buy_trigger_pct": t, "buy_recover_pct": r}),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["triggers"] = result["triggers"]
            context["recovers"] = result["recovers"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["resolved_base_price"] = result["기준가"]
            context["capital"] = result["자본금"]
            context["best_link"] = _link_for_combo(result["best"])
            context["worst_link"] = _link_for_combo(result["worst"])

            def _with_link(combo):
                return {**combo, "link": _link_for_combo(combo)}

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap7.html", **context)


@app.route("/heatmap8", methods=["GET"])
def heatmap8():
    """
    trend_trade_strategy() 전용 히트맵: 매도수량% 1~50% x 매수수량% 1~50%(둘 다 1% 단위,
    시작 보유 주식수 대비) = 2,500가지 조합의 수익률을 계산해 히트맵으로 보여준다.
    저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    sell_qty_pct_min = request.args.get("sell_qty_pct_min", "1").strip()
    sell_qty_pct_max = request.args.get("sell_qty_pct_max", "50").strip()
    buy_qty_pct_min = request.args.get("buy_qty_pct_min", "1").strip()
    buy_qty_pct_max = request.args.get("buy_qty_pct_max", "50").strip()

    context = {
        "active": "heatmap8",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "sell_qty_pct_min": sell_qty_pct_min,
        "sell_qty_pct_max": sell_qty_pct_max,
        "buy_qty_pct_min": buy_qty_pct_min,
        "buy_qty_pct_max": buy_qty_pct_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "sell_pcts": None,
        "buy_pcts": None,
        "cells": None,
        "best": None,
        "worst": None,
        "best_link": None,
        "worst_link": None,
        "top10": None,
        "bottom10": None,
        "ranked": None,
        "ranked_raw": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            sell_min_i = int(sell_qty_pct_min)
            sell_max_i = int(sell_qty_pct_max)
            buy_min_i = int(buy_qty_pct_min)
            buy_max_i = int(buy_qty_pct_max)
            if sell_min_i < 1 or buy_min_i < 1:
                raise ValueError("매도/매수 수량 하한은 1 이상이어야 합니다.")
            if sell_max_i < sell_min_i:
                raise ValueError("매도 수량 상한은 하한보다 크거나 같아야 합니다.")
            if buy_max_i < buy_min_i:
                raise ValueError("매수 수량 상한은 하한보다 크거나 같아야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            sell_qty_pct_values = range(sell_min_i, sell_max_i + 1)  # 시작 보유 주식수 대비 1% 단위
            buy_qty_pct_values = range(buy_min_i, buy_max_i + 1)

            result = compute_trend_heatmap(
                df, sell_qty_pct_values, buy_qty_pct_values, initial_shares=init_i,
                no_sell=no_sell, no_buy=no_buy, allow_negative_cash=allow_negative_cash,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for si, sp in enumerate(result["sell_pcts"]):
                for bi, bp in enumerate(result["buy_pcts"]):
                    pct = result["grid"][si][bi]
                    is_best = (sp == result["best"]["sell_pct"] and bp == result["best"]["buy_pct"])
                    is_worst = (sp == result["worst"]["sell_pct"] and bp == result["worst"]["buy_pct"])
                    cells.append({
                        "sell_pct": sp, "buy_pct": bp, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _build_trend_link(
                            code, sp, bp, init_i, no_sell, no_buy, allow_negative_cash,
                            end_date=end_date_iso, period=applied_period,
                        ),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["sell_pcts"] = result["sell_pcts"]
            context["buy_pcts"] = result["buy_pcts"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _build_trend_link(
                code, result["best"]["sell_pct"], result["best"]["buy_pct"],
                init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period,
            )
            context["worst_link"] = _build_trend_link(
                code, result["worst"]["sell_pct"], result["worst"]["buy_pct"],
                init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_trend_link(
                        code, combo["sell_pct"], combo["buy_pct"],
                        init_i, no_sell, no_buy, allow_negative_cash, end_date=end_date_iso, period=applied_period,
                    ),
                }

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("heatmap8.html", **context)


@app.route("/best", methods=["GET"])
def best():
    """
    각 전략별 히트맵(그리드매매/이익회수/일별매매/일별매매2/일별매매3/자본회수)을 각
    페이지의 기본 스윕범위·고정값 그대로 한 번씩 계산해서, 그 안에서 나온 **최고 수익률
    조합**만 한 화면에 모아 비교하는 요약 페이지. 통합 히트맵(/heatmap3)은 그리드매매·
    이익회수의 일반화 버전이라 중복이므로 뺀다. 저장된 로컬 CSV만 사용.

    sweep_max_pct: gap·수량·매수 트리거처럼 기본 상한이 50%였던 축들에 공통으로 적용되는
    스윕 상한(%) — 하나의 값을 6개 히트맵의 해당 축에 그대로 전달한다. 이미 100%가 자연스러운
    축(이익회수 gap, 자본회수 매수 회복률)은 그대로 둔다(스윕 대상 아니거나 이미 100%).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()
    sweep_max_pct = request.args.get("sweep_max_pct", "100").strip()

    context = {
        "active": "best",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "sweep_max_pct": sweep_max_pct,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "rows": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            sweep_max_i = int(sweep_max_pct)
            if sweep_max_i < 1:
                raise ValueError("스윕 상한(%)은 1 이상이어야 합니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            last_price = float(df.sort_values("날짜")["종가"].iloc[-1])
            initial_asset = init_i * first_price
            hold_only_asset = init_i * last_price

            rows = []

            sweep_values = range(1, sweep_max_i + 1)

            # 1) 그리드 매매 — /heatmap과 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r1 = compute_profit_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b1 = r1["best"]
            rows.append({
                "key": "grid", "icon": "🔲", "label": "그리드 매매",
                "profit_pct": b1["profit_pct"], "total": b1["total"],
                "condition": f"gap {b1['gap']}% / 수량 {b1['qty_pct']}% ({b1['qty']}주)",
                "counts": f"매도 {b1['매도횟수']}회 / 매수 {b1['매수횟수']}회",
                "strategy_link": _build_backtest_link(
                    code, b1["gap"], b1["qty_pct"], init_i, False, False, False, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_max': sweep_max_i, 'qty_pct_max': sweep_max_i})}",
            })

            # 2) 이익 회수 — /heatmap2와 동일한 기본값 (수량 10% 고정, 회수율 100%, gap 1~sweep_max_pct% / 이익gap 1~100%)
            r2 = compute_profit_heatmap2(
                df, sweep_values, range(1, 101), trade_qty_percent=10, profit_recover_percent=100,
                initial_shares=init_i,
            )
            b2 = r2["best"]
            rows.append({
                "key": "recover_gap", "icon": "🏦", "label": "그리드 매매 + 이익 회수",
                "profit_pct": b2["profit_pct"], "total": b2["total"],
                "condition": f"gap {b2['gap']}% / 이익gap {b2['profit_gap']}% (수량 10% 고정, 회수율 100% 고정)",
                "counts": f"매도 {b2['매도횟수']}회 / 매수 {b2['매수횟수']}회 / 이익회수 {b2['이익회수횟수']}회",
                "strategy_link": _build_backtest_link(
                    code, b2["gap"], 10, init_i, False, False, False,
                    profit_gap=b2["profit_gap"], profit_recover=100, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/heatmap2?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_max': sweep_max_i})}",
            })

            # 3) 일별 매매 — /heatmap4와 동일한 기본값 (매도/매수 수량 각각 1~sweep_max_pct%)
            r3 = compute_daily_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b3 = r3["best"]
            rows.append({
                "key": "daily", "icon": "📅1️⃣", "label": "일별 매매",
                "profit_pct": b3["profit_pct"], "total": b3["total"],
                "condition": f"매도 {b3['sell_pct']}% ({b3['sell_qty']}주) / 매수 {b3['buy_pct']}% ({b3['buy_qty']}주)",
                "counts": f"매도 {b3['매도횟수']}회 / 매수 {b3['매수횟수']}회",
                "strategy_link": _build_daily_link(
                    code, b3["sell_pct"], b3["buy_pct"], init_i, False, False, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/heatmap4?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'sell_qty_pct_max': sweep_max_i, 'buy_qty_pct_max': sweep_max_i})}",
            })

            # 4) 일별 매매2 — /heatmap5와 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r4 = compute_daily_gap_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b4 = r4["best"]
            rows.append({
                "key": "daily2", "icon": "📅2️⃣", "label": "일별 매매2",
                "profit_pct": b4["profit_pct"], "total": b4["total"],
                "condition": f"gap {b4['gap']}% / 수량 {b4['qty_pct']}% ({b4['qty']}주)",
                "counts": f"매도 {b4['매도횟수']}회 / 매수 {b4['매수횟수']}회",
                "strategy_link": _build_daily2_link(code, b4["gap"], b4["qty_pct"], init_i, end_date=end_date_iso, period=applied_period),
                "heatmap_link": f"/heatmap5?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_pct_max': sweep_max_i, 'qty_pct_max': sweep_max_i})}",
            })

            # 5) 일별 매매3 — /heatmap6과 동일한 기본값 (x=상승gap 1~sweep_max_pct%, y=수량 1~sweep_max_pct%, 하락gap은 상승gap과 동일)
            r5 = compute_daily_reference_heatmap_2d(
                df, "up_gap", sweep_values, "qty_pct", sweep_values, fixed={"down_gap": None},
                initial_shares=init_i,
            )
            b5 = r5["best"]
            rows.append({
                "key": "daily3", "icon": "📅3️⃣", "label": "일별 매매3",
                "profit_pct": b5["profit_pct"], "total": b5["total"],
                "condition": f"상승gap {b5['up_gap']}% / 하락gap {b5['down_gap']}% / 수량 {b5['qty_pct']}% ({b5['qty']}주)",
                "counts": f"매도 {b5['매도횟수']}회 / 매수 {b5['매수횟수']}회",
                "strategy_link": _build_daily3_link(
                    code, b5["up_gap"], b5["down_gap"], b5["qty_pct"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/heatmap6?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'up_gap_max': sweep_max_i, 'qty_pct_max': sweep_max_i})}",
            })

            # 6) 자본 회수 — /heatmap7과 동일한 기본값 (트리거 1~sweep_max_pct%, 회복률 1~100%, 기준가=첫날 종가)
            r6 = compute_capital_recovery_heatmap(df, sweep_values, range(1, 101), initial_shares=init_i)
            b6 = r6["best"]
            rows.append({
                "key": "recovery", "icon": "💰", "label": "자본 회수",
                "profit_pct": b6["profit_pct"], "total": b6["total"],
                "condition": f"매수 트리거 {b6['buy_trigger_pct']}% / 매수 회복률 {b6['buy_recover_pct']}%",
                "counts": f"매도 {b6['매도횟수']}회 / 매수 {b6['매수횟수']}회",
                "strategy_link": _build_recovery_link(
                    code, b6["buy_trigger_pct"], b6["buy_recover_pct"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/heatmap7?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'trigger_max': sweep_max_i})}",
            })

            rows.sort(key=lambda r: r["profit_pct"], reverse=True)
            for i, row in enumerate(rows):
                row["rank"] = i + 1
                row["vs_hold"] = row["total"] - hold_only_asset

            context["rows"] = rows
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"계산 중 오류가 발생했습니다: {e}"

    return render_template("best.html", **context)


@app.route("/stats", methods=["GET"])
def stats():
    """
    상승/하락 일수, 연속 상승/하락(streak) 일수 분포, 상승일·하락일 각각의 등락률(%) 분포를
    보여주는 통계 페이지. gap이나 매매 로직과 무관한 순수 가격 통계이며, 저장된 로컬 CSV를
    사용한다(없거나 기간이 부족하면 다른 페이지들과 동일하게 네이버에서 자동으로 채운다).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)

    context = {
        "active": "stats",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "result": None,
        "chart_labels": None,
        "chart_prices": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            context["end_date"] = end_date.strftime("%Y-%m-%d")
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            chart_df = df.sort_values("날짜")
            context["chart_labels"] = chart_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            context["chart_prices"] = chart_df["종가"].tolist()

            # 당일 갭 = 그날 종가 - 그날 시가, 전일 갭 = 그날 시가 - 전날 종가.
            # 둘을 더하면 전날 종가 대비 그날 종가의 전체 변화폭과 같다.
            intraday_gap = chart_df["종가"] - chart_df["시가"]
            overnight_gap = chart_df["시가"] - chart_df["종가"].shift(1)
            context["chart_intraday_gap"] = [float(v) for v in intraday_gap.tolist()]
            context["chart_overnight_gap"] = [
                None if pd.isna(v) else float(v) for v in overnight_gap.tolist()
            ]

            context["result"] = compute_price_stats(df)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"통계 계산 중 오류가 발생했습니다: {e}"

    return render_template("stats.html", **context)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
