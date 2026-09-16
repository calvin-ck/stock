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
    daily_reversal_strategy, compute_daily_heatmap,
    daily_gap_strategy, compute_daily_gap_heatmap,
    daily_reference_strategy, compute_daily_reference_heatmap,
    capital_recovery_strategy, compute_capital_recovery_heatmap,
    compute_profit_heatmap, compute_profit_recovery_heatmap,
    compute_price_stats,
    SISE_DAY_URL,
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
    # 시각(시:분:초)까지 들어가면 기간 계산의 경계일이 요청 시각에 따라 하루씩 밀릴 수
    # 있으므로(예: 히트맵이 내부에서 쓴 "오늘"과, 그 결과 링크를 다시 열었을 때 파싱되는
    # "오늘"이 달라짐), 자정으로 정규화해 날짜 단위로만 계산한다.
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else today
    end_date = min(end_date, today)  # 미래 날짜는 오늘로 clamp

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


def _build_grid_link(
    code, gap, qty_pct, init_shares, no_sell=False, no_buy=False, allow_negative_cash=False,
    capital=None, end_date=None, period=None,
):
    """그리드 트레이드 히트맵 셀·요약 클릭 시 그 조건 그대로 /grid_trade 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "gap": gap,
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
    if capital:
        params["capital"] = capital
    return f"/grid_trade?{urlencode(params)}"


def _build_grid_recover_link(code, profit_gap, profit_recover, init_shares, capital=None, end_date=None, period=None):
    """이익회수 히트맵 셀·요약 클릭 시 그 조건 그대로 /profit_recovery 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code, "profit_gap": profit_gap, "profit_recover": profit_recover, "init_shares": init_shares,
    }
    if end_date is not None:
        params["end_date"] = end_date
    if period is not None:
        params["period"] = period
    if capital:
        params["capital"] = capital
    return f"/profit_recovery?{urlencode(params)}"


def _build_daily_link(
    code, gap_pct, qty_pct, init_shares,
    allow_negative_cash, sell_above_start_asset_only, end_date=None, period=None,
):
    """상승 매도 하락 매수 - 전날 기준 히트맵 셀·요약 클릭 시 그 조건 그대로 /daily_reversal 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
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
    return f"/daily_reversal?{urlencode(params)}"


def _build_daily2_link(code, gap_pct, qty_pct, init_shares, no_sell=False, no_buy=False, end_date=None, period=None):
    """상승 매도 하락 매수 - min,max 기준 히트맵 셀·요약 클릭 시 그 조건 그대로 /daily_gap 페이지로 이동하는 링크를 만든다."""
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
    return f"/daily_gap?{urlencode(params)}"


def _build_daily3_link(
    code, gap_pct, qty_pct, init_shares,
    allow_negative_cash=False, no_sell=False, no_buy=False, keep_base_if_no_trade=True,
    end_date=None, period=None,
):
    """상승 매도 하락 매수 - 매매시 기준 히트맵 셀·요약 클릭 시 그 조건 그대로 /daily_reference 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        # "submitted"을 명시해서, /daily_reference의 기본값(체크박스 True) 추정 로직 대신
        # 여기서 넘긴 keep_base_if_no_trade 값을 그대로 쓰게 한다.
        "submitted": "1",
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
    if keep_base_if_no_trade:
        params["keep_base_if_no_trade"] = "on"
    return f"/daily_reference?{urlencode(params)}"


def _build_recovery_link(
    code, gap_pct, ratio_pct, init_shares,
    base_price=None, allow_negative_cash=False, end_date=None, period=None,
):
    """주식평가금액 유지 히트맵 셀·요약 클릭 시 그 조건 그대로 /capital_recovery 페이지로 이동하는 링크를 만든다."""
    params = {
        "code": code,
        "gap_pct": gap_pct,
        "ratio_pct": ratio_pct,
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
    return f"/capital_recovery?{urlencode(params)}"


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
        "chart_intraday_gap": None,
        "chart_overnight_gap": None,
        "result": None,
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


@app.route("/grid_trade", methods=["GET"])
def grid_trade():
    """
    그리드 매매 백테스트 페이지(이익 회수 없음). 트레일링 고점(max)/저점(min) 기준으로
    매매한다: 시작 시 max/min은 첫날 종가로 시작하고, 주가가 오르면 max를, 내리면 min을
    매매 여부와 무관하게 계속 갱신한다. max에서 매매gap%만큼 떨어지면(전날보다 하락한
    날에 한해) 매매수량만큼 매도하고 max를 그 매도가로 리셋한다. min에서 매매gap%만큼
    오르면(전날보다 상승한 날에 한해) 매수를 시도하고(현금 부족해도 매수=False면 살 수
    있는 만큼만) min을 그 매수가로 갱신한다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    gap = request.args.get("gap", "10").strip()
    qty_pct = request.args.get("qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    capital = request.args.get("capital", "").strip()

    context = {
        "active": "grid_trade",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "gap": gap,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "allow_negative_cash": allow_negative_cash,
        "capital": capital,
        "error": None,
        "summary": None,
        "trade_log": None,
        "qty": None,
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
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

            gap_f = float(gap)
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if gap_f <= 0:
                raise ValueError("매매 gap은 0보다 커야 합니다.")
            if qty_pct_f <= 0:
                raise ValueError("매수/매도 수량(%)은 0보다 커야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            capital_f = float(capital) if capital else None
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
                sell_gap_percent=gap_f,
                buy_gap_percent=gap_f,
                initial_shares=init_i,
                no_sell=no_sell,
                no_buy=no_buy,
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

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])
            initial_asset = capital_f if capital_f else init_i * first_price
            chart_capital = [initial_asset] * len(asset_log)

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset  # 그리드 매매 vs 단순 보유 차이

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
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

    return render_template("grid_trade.html", **context)


@app.route("/profit_recovery", methods=["GET"])
def profit_recovery():
    """
    "이익회수" 백테스트 페이지. 그리드 매수/매도는 전혀 하지 않고
    (grid_trade_strategy()를 no_sell/no_buy 고정으로 호출), 오직 "이익 회수" 이벤트만
    동작한다: 자본금(비우면 시작 자산) 대비 평가금액(주식평가금액+현금)이 이익gap%만큼
    벌면 그 시점 평가차익 중 회수율%만큼을 현금에서 먼저 충당하고, 모자라면 주식을 추가로
    매도해서 마련한 뒤 별도 적립금으로 옮긴다. 자본금은 고정값이라 회수 후 다시 벌어야
    재발동한다.
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
    profit_gap = request.args.get("profit_gap", "20").strip()
    profit_recover = request.args.get("profit_recover", "50").strip()

    context = {
        "active": "profit_recovery",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "capital": capital,
        "profit_gap": profit_gap,
        "profit_recover": profit_recover,
        "error": None,
        "summary": None,
        "recover_log": None,
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
        "profit": None,
        "profit_pct": None,
        "applied_period": None,
        "fetch_note": None,
        "chart_labels": None,
        "chart_prices": None,
        "chart_recover_points": None,
        "chart_total": None,
        "chart_stock_value": None,
        "chart_cash": None,
        "chart_reserve": None,
        "chart_hold_only": None,
        "chart_capital": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            profit_gap_f = float(profit_gap)
            profit_recover_f = float(profit_recover)
            if profit_gap_f <= 0:
                raise ValueError("이익 gap은 0보다 커야 합니다.")
            if not (1 <= profit_recover_f <= 100):
                raise ValueError("회수율은 1~100 사이여야 합니다.")

            capital_f = float(capital) if capital else None
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

            result = grid_trade_strategy(
                df,
                trade_qty=0,
                sell_gap_percent=1.0,
                initial_shares=init_i,
                no_sell=True,
                no_buy=True,
                profit_gap_percent=profit_gap_f,
                profit_recover_percent=profit_recover_f,
                capital=capital_f,
            )

            result.pop("매매일지")  # 그리드 매수/매도가 없어 항상 빈 리스트
            recover_log = result.pop("이익회수일지", [])
            for row in recover_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")
            result.setdefault("이익회수횟수", 0)

            asset_log = result.pop("자산추이")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_recover_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in recover_log
            ]
            chart_total = [row["total"] for row in asset_log]
            chart_stock_value = [row["주식평가금액"] for row in asset_log]
            chart_cash = [row["현금"] for row in asset_log]
            chart_reserve = [row["적립금"] for row in asset_log]
            chart_hold_only = [row["주가"] * init_i for row in asset_log]

            first_price = float(df.sort_values("날짜")["종가"].iloc[0])

            # 시작 자산은 이익 회수의 자본금(지정 안 했으면 주식수 x 첫날 종가)을 그대로 사용한다.
            initial_asset = result["자본금"]
            chart_capital = [initial_asset] * len(asset_log)
            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0
            vs_hold = result["total"] - hold_only_asset  # 이익회수 vs 단순 보유 차이

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
            context["profit"] = profit
            context["profit_pct"] = profit_pct
            context["recover_log"] = recover_log
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_recover_points"] = chart_recover_points
            context["chart_total"] = chart_total
            context["chart_stock_value"] = chart_stock_value
            context["chart_cash"] = chart_cash
            context["chart_reserve"] = chart_reserve
            context["chart_hold_only"] = chart_hold_only
            context["chart_capital"] = chart_capital

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("profit_recovery.html", **context)


@app.route("/daily_reversal", methods=["GET"])
def daily_reversal():
    """
    "상승 매도 하락 매수 - 전날 기준" 백테스트 페이지. gap이나 고점/저점 추적 없이, 전날 종가 대비
    gap%만큼 오르면 매도, 내리면 매수한다. 매도/매수 수량은 하나의 값(시작 보유 주식수
    대비 %)을 공유한다.
    "현금 부족해도 매수" 체크 시 현금 잔고와 무관하게 매수 수량을 그대로 매수하고(현금
    마이너스 허용), 체크 안 하면 쌓인 현금 범위 내에서만 매수한다.
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
    gap_pct = request.args.get("gap_pct", "3").strip()
    qty_pct = request.args.get("qty_pct", "10").strip()
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
        "active": "daily_reversal",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "sell_above_start_asset_only": sell_above_start_asset_only,
        "error": None,
        "summary": None,
        "trade_log": None,
        "qty": None,
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
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

            gap_pct_f = float(gap_pct)
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if gap_pct_f <= 0:
                raise ValueError("gap은 0보다 커야 합니다.")
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

            result = daily_reversal_strategy(
                df, gap_percent=gap_pct_f, trade_qty=qty_i, initial_shares=init_i,
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
            chart_capital = [initial_asset] * len(asset_log)

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
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

    return render_template("daily_reversal.html", **context)


@app.route("/daily_gap", methods=["GET"])
def daily_gap():
    """
    "상승 매도 하락 매수 - min,max 기준" 백테스트 페이지. 트레일링 고점(max)/저점(min)
    기준으로, max에서 gap% 떨어지면 매수, min에서 gap% 오르면 매도한다. 매도/매수 수량은
    하나의 값(시작 보유 주식수 대비 %)을 공유한다. 변수는 gap%와 수량% 딱 2개뿐이다.
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
        "active": "daily_gap",
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
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
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
            chart_capital = [initial_asset] * len(asset_log)

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
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

    return render_template("daily_gap.html", **context)


@app.route("/daily_reference", methods=["GET"])
def daily_reference():
    """
    "상승 매도 하락 매수 - 매매시 기준" 백테스트 페이지. 정적인 기준가 기준으로, 현재가가
    기준가 대비 gap% 이상 오르면 매도, gap% 이상 내리면 매수한다. 기준가는 첫날 종가로
    시작해 매매가 체결될 때만 그 거래가로 갱신된다(트레일링 고점/저점을 계속 따라가는
    다른 전략들과 달리, 매매 없이는 절대 움직이지 않는다). 매도/매수 수량은 하나의
    값(시작 보유 주식수 대비 %)을 공유한다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    gap_pct = request.args.get("gap_pct", "5").strip()
    qty_pct = request.args.get("qty_pct", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    no_sell = request.args.get("no_sell") == "on"
    no_buy = request.args.get("no_buy") == "on"
    # 체크박스가 기본 켜짐이라, GET 파라미터가 없는 상태(폼 제출 전 최초 진입)와
    # "사용자가 직접 체크 해제하고 제출"을 구분해야 한다. "submitted" 히든 필드로 폼 제출
    # 여부를 판별해서, 제출 전에는 기본값(True)을, 제출 후에는 실제 체크 여부를 사용한다.
    form_submitted = "submitted" in request.args
    if form_submitted:
        keep_base_if_no_trade = request.args.get("keep_base_if_no_trade") == "on"
    else:
        keep_base_if_no_trade = True

    context = {
        "active": "daily_reference",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "gap_pct": gap_pct,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "no_sell": no_sell,
        "no_buy": no_buy,
        "keep_base_if_no_trade": keep_base_if_no_trade,
        "error": None,
        "summary": None,
        "trade_log": None,
        "qty": None,
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
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

            gap_pct_f = float(gap_pct)
            qty_pct_f = float(qty_pct)
            init_i = int(init_shares)
            if gap_pct_f <= 0:
                raise ValueError("gap은 0보다 커야 합니다.")
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
                df, trade_qty=qty_i, gap_percent=gap_pct_f,
                initial_shares=init_i, allow_negative_cash=allow_negative_cash,
                no_sell=no_sell, no_buy=no_buy,
                keep_base_if_no_trade=keep_base_if_no_trade,
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
            chart_capital = [initial_asset] * len(asset_log)

            hold_only_asset = init_i * result["주가"]  # 매매 없이 그냥 들고만 있었을 때 최종 자산

            profit = result["total"] - initial_asset
            profit_pct = (profit / initial_asset * 100) if initial_asset else 0.0

            vs_hold = result["total"] - hold_only_asset

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
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

    return render_template("daily_reference.html", **context)


@app.route("/capital_recovery", methods=["GET"])
def capital_recovery():
    """
    "주식평가금액 유지" 백테스트 페이지. 기준가(비우면 첫날 종가, 입력하면 그 값)로 자본금
    (기준가 × 시작 보유 주식수)을 정하고, 주식 평가금액이 자본금과 매매gap(%) 이상 차이
    날 때만 매매한다. 자본금보다 높으면 초과분의 매매율(%)만큼 매도하고, 낮으면 부족분의
    매매율(%)만큼 매수한다. 기준가는 매매가 일어나도 갱신되지 않고 처음 값 그대로
    고정된다.
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
    gap_pct = request.args.get("gap_pct", "10").strip()
    ratio_pct = request.args.get("ratio_pct", "100").strip()
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"

    context = {
        "active": "capital_recovery",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "base_price": base_price,
        "gap_pct": gap_pct,
        "ratio_pct": ratio_pct,
        "allow_negative_cash": allow_negative_cash,
        "error": None,
        "summary": None,
        "trade_log": None,
        "resolved_base_price": None,
        "capital": None,
        "first_price": None,
        "initial_asset": None,
        "hold_only_asset": None,
        "hold_only_pct": None,
        "vs_hold": None,
        "vs_hold_pct": None,
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
            gap_pct_f = float(gap_pct)
            ratio_pct_f = float(ratio_pct)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            if base_price_f is not None and base_price_f <= 0:
                raise ValueError("기준가는 0보다 커야 합니다.")
            if gap_pct_f <= 0:
                raise ValueError("매매 gap은 0보다 커야 합니다.")
            if not (0 < ratio_pct_f <= 100):
                raise ValueError("매매율(%)은 0보다 크고 100 이하여야 합니다.")

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
                gap_percent=gap_pct_f, trade_ratio_percent=ratio_pct_f,
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

            hold_only_pct = ((hold_only_asset - initial_asset) / initial_asset * 100) if initial_asset else 0.0
            vs_hold_pct = (vs_hold / hold_only_asset * 100) if hold_only_asset else 0.0

            context["summary"] = result
            context["first_price"] = first_price
            context["resolved_base_price"] = resolved_base
            context["capital"] = capital
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset
            context["hold_only_pct"] = hold_only_pct
            context["vs_hold"] = vs_hold
            context["vs_hold_pct"] = vs_hold_pct
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

    return render_template("capital_recovery.html", **context)


@app.route("/daily_reversal_heatmap", methods=["GET"])
def daily_reversal_heatmap():
    """
    daily_reversal_strategy() 전용 히트맵: 등락폭 gap%(기본 1~50%) x 매매수량%(기본
    1~100%, 둘 다 1% 단위, 시작 보유 주식수 대비) 조합의 수익률을 계산해 히트맵으로 보여준다.
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
    gap_pct_min = request.args.get("gap_pct_min", "1").strip()
    gap_pct_max = request.args.get("gap_pct_max", "50").strip()
    qty_pct_min = request.args.get("qty_pct_min", "1").strip()
    qty_pct_max = request.args.get("qty_pct_max", "100").strip()

    context = {
        "active": "daily_reversal_heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "allow_negative_cash": allow_negative_cash,
        "sell_above_start_asset_only": sell_above_start_asset_only,
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

            result = compute_daily_heatmap(
                df, gap_values, qty_percent_values, initial_shares=init_i,
                allow_negative_cash=allow_negative_cash,
                sell_above_start_asset_only=sell_above_start_asset_only,
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
                        "link": _build_daily_link(
                            code, g, qp, init_i, allow_negative_cash, sell_above_start_asset_only,
                            end_date=end_date_iso, period=applied_period,
                        ),
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
            context["best_link"] = _build_daily_link(
                code, result["best"]["gap"], result["best"]["qty_pct"],
                init_i, allow_negative_cash, sell_above_start_asset_only, end_date=end_date_iso, period=applied_period,
            )
            context["worst_link"] = _build_daily_link(
                code, result["worst"]["gap"], result["worst"]["qty_pct"],
                init_i, allow_negative_cash, sell_above_start_asset_only, end_date=end_date_iso, period=applied_period,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_daily_link(
                        code, combo["gap"], combo["qty_pct"],
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

    return render_template("daily_reversal_heatmap.html", **context)


@app.route("/daily_gap_heatmap", methods=["GET"])
def daily_gap_heatmap():
    """
    "상승 매도 하락 매수 - min,max 기준"(daily_gap_strategy()) 전용 히트맵: 등락폭 gap%
    (기본 1~50%) x 매매수량%(기본 1~100%, 둘 다 1% 단위, 시작 보유 주식수 대비) 조합의
    수익률을 계산해 히트맵으로 보여준다.
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
    qty_pct_max = request.args.get("qty_pct_max", "100").strip()

    context = {
        "active": "daily_gap_heatmap",
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

    return render_template("daily_gap_heatmap.html", **context)


@app.route("/daily_reference_heatmap", methods=["GET"])
def daily_reference_heatmap():
    """
    "상승 매도 하락 매수 - 매매시 기준"(daily_reference_strategy()) 전용 히트맵: 등락폭
    gap%(기본 1~50%) x 매매수량%(기본 1~100%, 둘 다 1% 단위, 시작 보유 주식수 대비)
    조합의 수익률을 계산해 히트맵으로 보여준다.
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
    keep_base_if_no_trade = request.args.get("keep_base_if_no_trade") == "on"
    gap_pct_min = request.args.get("gap_pct_min", "1").strip()
    gap_pct_max = request.args.get("gap_pct_max", "50").strip()
    qty_pct_min = request.args.get("qty_pct_min", "1").strip()
    qty_pct_max = request.args.get("qty_pct_max", "100").strip()

    context = {
        "active": "daily_reference_heatmap",
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
        "keep_base_if_no_trade": keep_base_if_no_trade,
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

            result = compute_daily_reference_heatmap(
                df, gap_values, qty_percent_values, initial_shares=init_i,
                allow_negative_cash=allow_negative_cash, no_sell=no_sell, no_buy=no_buy,
                keep_base_if_no_trade=keep_base_if_no_trade,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _link_for_combo(combo):
                return _build_daily3_link(
                    code, combo["gap"], combo["qty_pct"], init_i,
                    allow_negative_cash, no_sell, no_buy, keep_base_if_no_trade,
                    end_date=end_date_iso, period=applied_period,
                )

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
                        "link": _link_for_combo({"gap": g, "qty_pct": qp}),
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

    return render_template("daily_reference_heatmap.html", **context)


@app.route("/capital_recovery_heatmap", methods=["GET"])
def capital_recovery_heatmap():
    """
    capital_recovery_strategy() 전용 히트맵: 매매gap(%) x 매매율(%) 조합별 수익률을
    계산해 히트맵으로 보여준다. 기준가/시작 보유 주식수/현금 부족해도 매수 여부는 폼에서
    고정값으로 입력한다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
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
    gap_pct_min = request.args.get("gap_pct_min", "1").strip()
    gap_pct_max = request.args.get("gap_pct_max", "50").strip()
    ratio_pct_min = request.args.get("ratio_pct_min", "1").strip()
    ratio_pct_max = request.args.get("ratio_pct_max", "100").strip()

    context = {
        "active": "capital_recovery_heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "base_price": base_price,
        "allow_negative_cash": allow_negative_cash,
        "gap_pct_min": gap_pct_min,
        "gap_pct_max": gap_pct_max,
        "ratio_pct_min": ratio_pct_min,
        "ratio_pct_max": ratio_pct_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "gaps": None,
        "ratio_pcts": None,
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

            gap_min_i = int(gap_pct_min)
            gap_max_i = int(gap_pct_max)
            ratio_min_i = int(ratio_pct_min)
            ratio_max_i = int(ratio_pct_max)
            if gap_min_i < 1 or ratio_min_i < 1:
                raise ValueError("gap/매매율 하한은 1 이상이어야 합니다.")
            if gap_max_i < gap_min_i:
                raise ValueError("gap 상한은 하한보다 크거나 같아야 합니다.")
            if ratio_max_i < ratio_min_i:
                raise ValueError("매매율 상한은 하한보다 크거나 같아야 합니다.")
            if ratio_max_i > 100:
                raise ValueError("매매율은 100을 넘을 수 없습니다.")

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
            ratio_percent_values = range(ratio_min_i, ratio_max_i + 1)  # 1% 단위

            result = compute_capital_recovery_heatmap(
                df, gap_values, ratio_percent_values, initial_shares=init_i,
                base_price=base_price_f, allow_negative_cash=allow_negative_cash,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _link_for_combo(combo):
                return _build_recovery_link(
                    code, combo["gap"], combo["ratio_pct"], init_i,
                    base_price=base_price_f, allow_negative_cash=allow_negative_cash,
                    end_date=end_date_iso, period=applied_period,
                )

            cells = []
            for gi, g in enumerate(result["gaps"]):
                for ri, rp in enumerate(result["ratio_pcts"]):
                    pct = result["grid"][gi][ri]
                    is_best = (g == result["best"]["gap"] and rp == result["best"]["ratio_pct"])
                    is_worst = (g == result["worst"]["gap"] and rp == result["worst"]["ratio_pct"])
                    cells.append({
                        "gap": g, "ratio_pct": rp, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _link_for_combo({"gap": g, "ratio_pct": rp}),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["ratio_pcts"] = result["ratio_pcts"]
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

    return render_template("capital_recovery_heatmap.html", **context)


@app.route("/grid_trade_heatmap", methods=["GET"])
def grid_trade_heatmap():
    """
    /grid_trade(하락 매도 상승 매수(익절), 이익 회수 없음) 전용 히트맵. gap(기본 1~50%,
    1% 단위) x 매매수량(시작 보유 주식수 대비 %, 기본 1~100%, 1% 단위) 조합의 수익률을 계산해
    히트맵으로 보여준다. compute_profit_heatmap()은 애초에 이익 회수와 무관하게 동작하므로
    /heatmap과 완전히 같은 계산을 쓰고, 셀/순위 링크만 /grid로 연결한다. 저장된 로컬 CSV만
    사용 (네이버 재접속 없음).
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
    qty_pct_max = request.args.get("qty_pct_max", "100").strip()

    context = {
        "active": "grid_trade_heatmap",
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

            def _link_for(g, qp):
                return _build_grid_link(
                    code, g, qp, init_i, no_sell, no_buy, allow_negative_cash,
                    capital=capital_f, end_date=end_date_iso, period=applied_period,
                )

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
                        "link": _link_for(g, qp),
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
            context["best_link"] = _link_for(result["best"]["gap"], result["best"]["qty_pct"])
            context["worst_link"] = _link_for(result["worst"]["gap"], result["worst"]["qty_pct"])

            def _with_link(combo):
                return {**combo, "link": _link_for(combo["gap"], combo["qty_pct"])}

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]  # 순위별 수익률 그래프의 "중복 제거 off" 데이터 (링크 불필요)

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("grid_trade_heatmap.html", **context)


@app.route("/profit_recovery_heatmap", methods=["GET"])
def profit_recovery_heatmap():
    """
    /profit_recovery(이익회수, 그리드 매수/매도 없음) 전용 히트맵. 이익 gap(%) x
    회수율(%) 조합별 수익률을 계산해 히트맵으로 보여준다. 저장된 로컬 CSV만 사용한다.
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
    profit_gap_min = request.args.get("profit_gap_min", "1").strip()
    profit_gap_max = request.args.get("profit_gap_max", "50").strip()
    profit_recover_min = request.args.get("profit_recover_min", "1").strip()
    profit_recover_max = request.args.get("profit_recover_max", "100").strip()

    context = {
        "active": "profit_recovery_heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "capital": capital,
        "profit_gap_min": profit_gap_min,
        "profit_gap_max": profit_gap_max,
        "profit_recover_min": profit_recover_min,
        "profit_recover_max": profit_recover_max,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "profit_gaps": None,
        "profit_recovers": None,
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

            capital_f = float(capital) if capital else None
            if capital_f is not None and capital_f <= 0:
                raise ValueError("자본금은 0보다 커야 합니다.")

            profit_gap_min_i = int(profit_gap_min)
            profit_gap_max_i = int(profit_gap_max)
            profit_recover_min_i = int(profit_recover_min)
            profit_recover_max_i = int(profit_recover_max)
            if profit_gap_min_i < 1 or profit_recover_min_i < 1:
                raise ValueError("이익gap/회수율 하한은 1 이상이어야 합니다.")
            if profit_gap_max_i < profit_gap_min_i:
                raise ValueError("이익gap 상한은 하한보다 크거나 같아야 합니다.")
            if profit_recover_max_i < profit_recover_min_i:
                raise ValueError("회수율 상한은 하한보다 크거나 같아야 합니다.")
            if profit_recover_max_i > 100:
                raise ValueError("회수율은 100을 넘을 수 없습니다.")

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["start_date"] = start_date.strftime("%Y-%m-%d")
            applied_period = (end_date - start_date).days
            context["applied_period"] = applied_period
            context["fetch_note"] = fetch_note

            profit_gap_values = range(profit_gap_min_i, profit_gap_max_i + 1)
            profit_recover_values = range(profit_recover_min_i, profit_recover_max_i + 1)

            result = compute_profit_recovery_heatmap(
                df, profit_gap_values, profit_recover_values, initial_shares=init_i,
                capital=capital_f,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            def _link_for(pg, pr):
                return _build_grid_recover_link(
                    code, pg, pr, init_i,
                    capital=capital_f, end_date=end_date_iso, period=applied_period,
                )

            cells = []
            for gi, pg in enumerate(result["profit_gaps"]):
                for ri, pr in enumerate(result["profit_recovers"]):
                    pct = result["grid"][gi][ri]
                    is_best = (pg == result["best"]["profit_gap"] and pr == result["best"]["profit_recover"])
                    is_worst = (pg == result["worst"]["profit_gap"] and pr == result["worst"]["profit_recover"])
                    cells.append({
                        "profit_gap": pg, "profit_recover": pr, "pct": pct,
                        "total": result["initial_asset"] * (1 + pct / 100),
                        "color": _profit_color(pct, vmax),
                        "link": _link_for(pg, pr),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["profit_gaps"] = result["profit_gaps"]
            context["profit_recovers"] = result["profit_recovers"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["hold_only_asset"] = result["hold_only_asset"]
            context["best_link"] = _link_for(result["best"]["profit_gap"], result["best"]["profit_recover"])
            context["worst_link"] = _link_for(result["worst"]["profit_gap"], result["worst"]["profit_recover"])

            def _with_link(combo):
                return {**combo, "link": _link_for(combo["profit_gap"], combo["profit_recover"])}

            context["top10"] = [_with_link(c) for c in result["top10"]]
            context["bottom10"] = [_with_link(c) for c in result["bottom10"]]
            context["ranked"] = [_with_link(c) for c in result["ranked"]]
            context["ranked_raw"] = result["raw_ranked"]

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"히트맵 계산 중 오류가 발생했습니다: {e}"

    return render_template("profit_recovery_heatmap.html", **context)


# /best와 /best_heatmap이 공유하는 6개 전략 메타데이터 (key, 아이콘, 표시 이름).
_BEST_STRATEGIES = [
    ("grid_trade", "🔲", "하락 매도 상승 매수(익절)"),
    ("profit_recovery", "🏦", "이익회수"),
    ("capital_recovery", "💰", "주식평가금액 유지"),
    ("daily_reversal", "🔄", "상승 매도 하락 매수 - 전날 기준"),
    ("daily_gap", "🔃", "상승 매도 하락 매수 - min,max 기준"),
    ("daily_reference", "📌", "상승 매도 하락 매수 - 매매시 기준"),
]


def _compute_best_by_strategy(df, init_i, sweep_max_i):
    """
    6개 전략을 각자의 기본 스윕 범위(/heatmap 계열 페이지와 동일한 기본값)로 한 번씩
    계산해서, 전략 key -> 그 전략의 최고 수익률 조합(best) dict를 반환한다.
    /best와 /best_heatmap이 이 계산을 공유한다.
    """
    sweep_values = range(1, sweep_max_i + 1)
    return {
        "grid_trade": compute_profit_heatmap(
            df, sweep_values, sweep_values, initial_shares=init_i,
        )["best"],
        "profit_recovery": compute_profit_recovery_heatmap(
            df, sweep_values, range(1, 101), initial_shares=init_i,
        )["best"],
        "capital_recovery": compute_capital_recovery_heatmap(
            df, sweep_values, range(1, 101), initial_shares=init_i,
        )["best"],
        "daily_reversal": compute_daily_heatmap(
            df, sweep_values, sweep_values, initial_shares=init_i,
        )["best"],
        "daily_gap": compute_daily_gap_heatmap(
            df, sweep_values, sweep_values, initial_shares=init_i,
        )["best"],
        "daily_reference": compute_daily_reference_heatmap(
            df, sweep_values, sweep_values, initial_shares=init_i,
        )["best"],
    }


def _strategy_link_from_axes(strat_key, axis1, axis2, code, init_i, end_date_iso, period):
    """
    전략 key와 두 축 값만으로 그 조합의 실제 백테스트 페이지 링크를 만든다. 축 의미는
    전략마다 다르다: grid_trade/daily_reversal/daily_gap/daily_reference는 (gap,
    qty_pct), capital_recovery는 (gap, ratio_pct), profit_recovery는 (profit_gap,
    profit_recover).
    """
    if strat_key == "grid_trade":
        return _build_grid_link(code, axis1, axis2, init_i, end_date=end_date_iso, period=period)
    if strat_key == "profit_recovery":
        return _build_grid_recover_link(code, axis1, axis2, init_i, end_date=end_date_iso, period=period)
    if strat_key == "daily_reversal":
        return _build_daily_link(code, axis1, axis2, init_i, False, False, end_date=end_date_iso, period=period)
    if strat_key == "daily_gap":
        return _build_daily2_link(code, axis1, axis2, init_i, end_date=end_date_iso, period=period)
    if strat_key == "daily_reference":
        return _build_daily3_link(code, axis1, axis2, init_i, end_date=end_date_iso, period=period)
    if strat_key == "capital_recovery":
        return _build_recovery_link(code, axis1, axis2, init_i, end_date=end_date_iso, period=period)
    raise ValueError(f"알 수 없는 전략 key: {strat_key}")


# 전략 key -> best 조합 dict에서 (축1, 축2) 값을 꺼낼 때 쓸 키 이름.
_STRATEGY_AXIS_KEYS = {
    "grid_trade": ("gap", "qty_pct"),
    "profit_recovery": ("profit_gap", "profit_recover"),
    "capital_recovery": ("gap", "ratio_pct"),
    "daily_reversal": ("gap", "qty_pct"),
    "daily_gap": ("gap", "qty_pct"),
    "daily_reference": ("gap", "qty_pct"),
}


def _strategy_detail_link(strat_key, b, code, init_i, end_date_iso, period):
    """
    _compute_best_by_strategy()가 돌려준 best 조합(b)을 그 전략의 실제 백테스트
    페이지(/grid_trade, /profit_recovery 등)로 이동하는 링크로 바꾼다.
    /best_heatmap의 각 칸이 비교 페이지(/best)가 아니라 그 조합의 실제 데이터
    페이지로 바로 연결되도록 쓴다.
    """
    k1, k2 = _STRATEGY_AXIS_KEYS[strat_key]
    return _strategy_link_from_axes(strat_key, b[k1], b[k2], code, init_i, end_date_iso, period)


@app.route("/best", methods=["GET"])
def best():
    """
    각 전략별 히트맵(하락 매도 상승 매수(익절)/이익회수/일별매매/일별매매2/일별매매3/
    주식평가금액유지)을 각 페이지의 기본 스윕범위·고정값 그대로 한 번씩 계산해서, 그 안에서
    나온 **최고 수익률 조합**만 한 화면에 모아 비교하는 요약 페이지. 저장된 로컬 CSV만 사용.

    sweep_max_pct: gap·수량·매수 트리거처럼 기본 상한이 50%였던 축들에 공통으로 적용되는
    스윕 상한(%) — 하나의 값을 6개 히트맵의 해당 축에 그대로 전달한다. 이미 100%가 자연스러운
    축(이익회수 회수율, 주식평가금액유지 매매율)은 그대로 둔다(스윕 대상 아니거나 이미
    100%).
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

            # 1) 하락 매도 상승 매수(익절) — /heatmap8과 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r1 = compute_profit_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b1 = r1["best"]
            rows.append({
                "key": "grid_trade", "icon": "🔲", "label": "하락 매도 상승 매수(익절)",
                "profit_pct": b1["profit_pct"], "total": b1["total"],
                "condition": f"gap {b1['gap']}% / 수량 {b1['qty_pct']}% ({b1['qty']}주)",
                "counts": f"매도 {b1['매도횟수']}회 / 매수 {b1['매수횟수']}회",
                "strategy_link": _build_grid_link(
                    code, b1["gap"], b1["qty_pct"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/grid_trade_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_min': 1, 'gap_max': sweep_max_i, 'qty_pct_min': 1, 'qty_pct_max': sweep_max_i})}",
            })

            # 2) 이익회수 — /heatmap9와 동일한 기본값 (이익gap 1~sweep_max_pct%, 회수율 1~100%)
            r2 = compute_profit_recovery_heatmap(df, sweep_values, range(1, 101), initial_shares=init_i)
            b2 = r2["best"]
            rows.append({
                "key": "profit_recovery", "icon": "🏦", "label": "이익회수",
                "profit_pct": b2["profit_pct"], "total": b2["total"],
                "condition": f"이익gap {b2['profit_gap']}% / 회수율 {b2['profit_recover']}%",
                "counts": f"이익회수 {b2['이익회수횟수']}회",
                "strategy_link": _build_grid_recover_link(
                    code, b2["profit_gap"], b2["profit_recover"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/profit_recovery_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'profit_gap_min': 1, 'profit_gap_max': sweep_max_i, 'profit_recover_min': 1, 'profit_recover_max': 100})}",
            })

            # 3) 주식평가금액 유지 — /heatmap7과 동일한 기본값 (gap 1~sweep_max_pct%, 매매율 1~100%, 기준가=첫날 종가)
            r6 = compute_capital_recovery_heatmap(df, sweep_values, range(1, 101), initial_shares=init_i)
            b6 = r6["best"]
            rows.append({
                "key": "capital_recovery", "icon": "💰", "label": "주식평가금액 유지",
                "profit_pct": b6["profit_pct"], "total": b6["total"],
                "condition": f"gap {b6['gap']}% / 매매율 {b6['ratio_pct']}%",
                "counts": f"매도 {b6['매도횟수']}회 / 매수 {b6['매수횟수']}회",
                "strategy_link": _build_recovery_link(
                    code, b6["gap"], b6["ratio_pct"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/capital_recovery_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_pct_min': 1, 'gap_pct_max': sweep_max_i, 'ratio_pct_min': 1, 'ratio_pct_max': 100})}",
            })

            # 4) 상승 매도 하락 매수 - 전날 기준 — /heatmap4와 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r3 = compute_daily_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b3 = r3["best"]
            rows.append({
                "key": "daily_reversal", "icon": "🔄", "label": "상승 매도 하락 매수 - 전날 기준",
                "profit_pct": b3["profit_pct"], "total": b3["total"],
                "condition": f"gap {b3['gap']}% / 수량 {b3['qty_pct']}% ({b3['qty']}주)",
                "counts": f"매도 {b3['매도횟수']}회 / 매수 {b3['매수횟수']}회",
                "strategy_link": _build_daily_link(
                    code, b3["gap"], b3["qty_pct"], init_i, False, False, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/daily_reversal_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_pct_min': 1, 'gap_pct_max': sweep_max_i, 'qty_pct_min': 1, 'qty_pct_max': sweep_max_i})}",
            })

            # 5) 상승 매도 하락 매수 - min,max 기준 — /heatmap5와 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r4 = compute_daily_gap_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b4 = r4["best"]
            rows.append({
                "key": "daily_gap", "icon": "🔃", "label": "상승 매도 하락 매수 - min,max 기준",
                "profit_pct": b4["profit_pct"], "total": b4["total"],
                "condition": f"gap {b4['gap']}% / 수량 {b4['qty_pct']}% ({b4['qty']}주)",
                "counts": f"매도 {b4['매도횟수']}회 / 매수 {b4['매수횟수']}회",
                "strategy_link": _build_daily2_link(code, b4["gap"], b4["qty_pct"], init_i, end_date=end_date_iso, period=applied_period),
                "heatmap_link": f"/daily_gap_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_pct_min': 1, 'gap_pct_max': sweep_max_i, 'qty_pct_min': 1, 'qty_pct_max': sweep_max_i})}",
            })

            # 6) 상승 매도 하락 매수 - 매매시 기준 — /heatmap6과 동일한 기본값 (gap/수량 1~sweep_max_pct%)
            r5 = compute_daily_reference_heatmap(df, sweep_values, sweep_values, initial_shares=init_i)
            b5 = r5["best"]
            rows.append({
                "key": "daily_reference", "icon": "📌", "label": "상승 매도 하락 매수 - 매매시 기준",
                "profit_pct": b5["profit_pct"], "total": b5["total"],
                "condition": f"gap {b5['gap']}% / 수량 {b5['qty_pct']}% ({b5['qty']}주)",
                "counts": f"매도 {b5['매도횟수']}회 / 매수 {b5['매수횟수']}회",
                "strategy_link": _build_daily3_link(
                    code, b5["gap"], b5["qty_pct"], init_i, end_date=end_date_iso, period=applied_period,
                ),
                "heatmap_link": f"/daily_reference_heatmap?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': applied_period, 'gap_pct_min': 1, 'gap_pct_max': sweep_max_i, 'qty_pct_min': 1, 'qty_pct_max': sweep_max_i})}",
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


_BEST_HEATMAP_PERIODS = [180, 90, 60, 30, 14, 7]


@app.route("/best_heatmap", methods=["GET"])
def best_heatmap():
    """
    6개 전략 × 최대 6개 기간(180/90/60/30/14/7일)의 최고 수익률을 한 번에 비교하는
    히트맵. 각 칸은 그 전략을 각자의 기본 스윕 범위로, 그 기간만큼의 데이터로 계산했을
    때 나온 최고 수익률이다. 매도/매수 없이 그냥 들고만 있었을 때(단순 보유)의 기간별
    수익률도 별도 행으로 함께 보여준다. 종료일은 모든 기간에 공통이며(비우면 오늘),
    라디오 버튼으로 고른 기준 기간(max_period, 기본 90일) 이하의 기간만 분석 대상으로
    삼고, 기간마다 따로 데이터를 받지 않고 그중 가장 긴 기간의 데이터를 한 번만 받아
    필요한 만큼씩 잘라 쓴다. 기간(열 머리글)을 클릭하면 그 기간의 /best로, 각 칸을
    클릭하면 그 조합 그대로 해당 전략의 실제 백테스트 페이지로 이동한다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    sweep_max_pct = request.args.get("sweep_max_pct", "100").strip()
    max_period_str = request.args.get("max_period", "90").strip()

    context = {
        "active": "best_heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "init_shares": init_shares,
        "sweep_max_pct": sweep_max_pct,
        "max_period": max_period_str,
        "all_periods": _BEST_HEATMAP_PERIODS,
        "period_links": None,
        "error": None,
        "fetch_note": None,
        "rows": None,
        "hold_row": None,
    }

    if code:
        try:
            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            sweep_max_i = int(sweep_max_pct)
            if sweep_max_i < 1:
                raise ValueError("스윕 상한(%)은 1 이상이어야 합니다.")
            max_period_i = int(max_period_str)
            if max_period_i not in _BEST_HEATMAP_PERIODS:
                raise ValueError("기준 기간은 180/90/60/30/14/7일 중 하나여야 합니다.")

            # 선택한 기준 기간 이하의 기간들만 분석 대상으로 삼는다.
            periods = [p for p in _BEST_HEATMAP_PERIODS if p <= max_period_i]

            data_period = max(periods)
            start_date, end_date, _ = _resolve_query_range(end_date_str, "", str(data_period))

            df, fetch_note, _, start_date = _ensure_range_df(code, start_date, end_date)
            if df.empty:
                raise ValueError(fetch_note or "데이터가 없습니다. 종목 코드를 확인해주세요.")
            end_date_iso = end_date.strftime("%Y-%m-%d")
            context["end_date"] = end_date_iso
            context["fetch_note"] = fetch_note

            sorted_df = df.sort_values("날짜")

            # 기간마다 따로 조회하지 않고, 선택한 기준 기간 중 가장 긴 데이터를 날짜로
            # 잘라 재사용한다.
            best_by_period = {}
            hold_by_period = {}  # 매도/매수 없이 그냥 들고만 있었을 때(단순 보유) 수익률
            for period in periods:
                period_start = end_date - timedelta(days=period)
                sub_df = sorted_df[sorted_df["날짜"] >= pd.Timestamp(period_start)]
                if len(sub_df) < 2:
                    best_by_period[period] = None
                    hold_by_period[period] = None
                    continue
                try:
                    best_by_period[period] = _compute_best_by_strategy(sub_df, init_i, sweep_max_i)
                except ValueError:
                    best_by_period[period] = None

                first_price = float(sub_df["종가"].iloc[0])
                last_price = float(sub_df["종가"].iloc[-1])
                hold_total = init_i * last_price
                hold_profit_pct = (last_price - first_price) / first_price * 100 if first_price else 0.0
                hold_by_period[period] = {"profit_pct": hold_profit_pct, "total": hold_total}

            vmax = 1e-9
            for best_by_strategy in best_by_period.values():
                if best_by_strategy is None:
                    continue
                for b in best_by_strategy.values():
                    vmax = max(vmax, abs(b["profit_pct"]))
            for hold in hold_by_period.values():
                if hold is not None:
                    vmax = max(vmax, abs(hold["profit_pct"]))

            # 단순보유 대비(%) 값들을 먼저 모아, 그 색상 스케일(vmax_vs)을 수익률과
            # 따로 구한다 — 수익률은 양수인데 단순보유 대비는 음수인 경우처럼 두 값의
            # 부호가 다를 수 있어 색을 공유하면 안 된다.
            vs_hold_by_period = {}
            vmax_vs = 1e-9
            for period in periods:
                best_by_strategy = best_by_period.get(period)
                hold = hold_by_period.get(period)
                if best_by_strategy is None or hold is None or not hold["total"]:
                    vs_hold_by_period[period] = {}
                    continue
                vs_map = {}
                for strat_key, b in best_by_strategy.items():
                    vs_pct = (b["total"] - hold["total"]) / hold["total"] * 100
                    vs_map[strat_key] = vs_pct
                    vmax_vs = max(vmax_vs, abs(vs_pct))
                vs_hold_by_period[period] = vs_map

            rows = []
            for strat_key, icon, label in _BEST_STRATEGIES:
                cells = []
                for period in periods:
                    best_by_strategy = best_by_period.get(period)
                    if best_by_strategy is None:
                        cells.append({
                            "period": period, "profit_pct": None, "total": None,
                            "color": "#e5e7eb", "vs_hold_color": "#e5e7eb", "link": None,
                        })
                        continue
                    b = best_by_strategy[strat_key]
                    vs_hold_pct = vs_hold_by_period.get(period, {}).get(strat_key)
                    cells.append({
                        "period": period,
                        "profit_pct": b["profit_pct"],
                        "total": b["total"],
                        "vs_hold_pct": vs_hold_pct,
                        "color": _profit_color(b["profit_pct"], vmax),
                        "vs_hold_color": (
                            _profit_color(vs_hold_pct, vmax_vs) if vs_hold_pct is not None else "#e5e7eb"
                        ),
                        "link": _strategy_detail_link(strat_key, b, code, init_i, end_date_iso, period),
                    })
                rows.append({"key": strat_key, "icon": icon, "label": label, "cells": cells})

            context["period_links"] = [
                {
                    "period": period,
                    "link": f"/best?{urlencode({'code': code, 'init_shares': init_i, 'end_date': end_date_iso, 'period': period, 'sweep_max_pct': sweep_max_i})}",
                }
                for period in periods
            ]

            hold_cells = []
            for period in periods:
                hold = hold_by_period.get(period)
                if hold is None:
                    hold_cells.append({
                        "period": period, "profit_pct": None, "total": None,
                        "color": "#e5e7eb", "vs_hold_pct": None, "vs_hold_color": "#e5e7eb",
                    })
                    continue
                hold_cells.append({
                    "period": period,
                    "profit_pct": hold["profit_pct"],
                    "total": hold["total"],
                    "vs_hold_pct": 0.0,
                    "color": _profit_color(hold["profit_pct"], vmax),
                    "vs_hold_color": _profit_color(0.0, vmax_vs),
                })

            context["rows"] = rows
            context["hold_row"] = {"icon": "🤚", "label": "매도/매수 없음 (단순 보유)", "cells": hold_cells}

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"계산 중 오류가 발생했습니다: {e}"

    return render_template("best_heatmap.html", **context)


def _compute_full_sweep_grids(df, init_i, max_n=100):
    """
    6개 전략을 각각 축1(gap/profit_gap/gap_pct)·축2(qty_pct/회수율/매매율) 1~max_n
    전체로 한 번씩 계산해서, 전략 key -> {"grid": [[profit_pct, ...], ...],
    "initial_asset": float} 를 반환한다. grid[i][j]의 축1 값은 i+1, 축2 값은 j+1이다.
    """
    axis_full = range(1, max_n + 1)
    full_100 = range(1, 101)

    r1 = compute_profit_heatmap(df, axis_full, axis_full, initial_shares=init_i)
    r2 = compute_profit_recovery_heatmap(df, axis_full, full_100, initial_shares=init_i)
    r3 = compute_daily_heatmap(df, axis_full, axis_full, initial_shares=init_i)
    r4 = compute_daily_gap_heatmap(df, axis_full, axis_full, initial_shares=init_i)
    r5 = compute_daily_reference_heatmap(df, axis_full, axis_full, initial_shares=init_i)
    r6 = compute_capital_recovery_heatmap(df, axis_full, full_100, initial_shares=init_i)

    return {
        "grid_trade": {"grid": r1["grid"], "initial_asset": r1["initial_asset"]},
        "profit_recovery": {"grid": r2["grid"], "initial_asset": r2["initial_asset"]},
        "capital_recovery": {"grid": r6["grid"], "initial_asset": r6["initial_asset"]},
        "daily_reversal": {"grid": r3["grid"], "initial_asset": r3["initial_asset"]},
        "daily_gap": {"grid": r4["grid"], "initial_asset": r4["initial_asset"]},
        "daily_reference": {"grid": r5["grid"], "initial_asset": r5["initial_asset"]},
    }


def _best_axis1_at_axis2(grid, axis2_value):
    """
    grid[i][j](0-indexed, 축값은 i+1/j+1)에서 축2(qty_pct/회수율/매매율 등)를
    axis2_value로 고정하고 축1(gap류)을 1~100 전체로 스윕했을 때의 최댓값과 그
    축1 값을 (profit_pct, axis1) 튜플로 반환한다.
    """
    col = axis2_value - 1
    best_val = float("-inf")
    best_i = 0
    for i, row in enumerate(grid):
        v = row[col]
        if v > best_val:
            best_val, best_i = v, i
    return best_val, best_i + 1


_BEST_SWEEP_VALUES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@app.route("/best_sweep_heatmap", methods=["GET"])
def best_sweep_heatmap():
    """
    6개 전략 × 수량류 축(qty_pct/회수율/매매율) 11개 지점(1, 10, 20, ..., 100%)의
    최고 수익률을 한 번에 비교하는 히트맵. 각 칸은 그 수량값을 고정하고 gap류 축을
    1~100% 전체로 스윕했을 때 나오는 최고 수익률이다("그 수량에서 gap을 최적으로
    골랐을 때 최선의 결과"). 전략마다 1~100×1~100 전체 그리드를 한 번만 계산한 뒤,
    표시할 11개 수량값에 해당하는 열(column)에서만 최댓값을 뽑아 쓴다(전략당 다시
    계산하지 않는다). 각 칸을 클릭하면 그 수량값과 그때의 최적 gap 조합 그대로 해당
    전략의 실제 백테스트 페이지로 이동한다.
    """
    codes = _list_local_codes()
    default_code = max(codes, key=lambda c: c["max_date"])["code"] if codes else ""

    code = request.args.get("code", "102110").strip()
    end_date_str = request.args.get("end_date", "").strip()
    start_date_str = request.args.get("start_date", "").strip()
    period_str = request.args.get("period", "").strip()
    end_date_str, start_date_str, period_str = _default_display_range(end_date_str, start_date_str, period_str)
    init_shares = request.args.get("init_shares", "100").strip()

    context = {
        "active": "best_sweep_heatmap",
        "codes": codes,
        "code": code,
        "default_code": default_code,
        "end_date": end_date_str,
        "start_date": start_date_str,
        "period": period_str,
        "init_shares": init_shares,
        "sweep_values": _BEST_SWEEP_VALUES,
        "error": None,
        "applied_period": None,
        "fetch_note": None,
        "rows": None,
        "hold_row": None,
        "initial_asset": None,
        "hold_only_asset": None,
    }

    if code:
        try:
            start_date, end_date, applied_period = _resolve_query_range(end_date_str, start_date_str, period_str)

            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

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
            hold_profit_pct = (last_price - first_price) / first_price * 100 if first_price else 0.0

            full_by_strategy = _compute_full_sweep_grids(df, init_i, max_n=100)

            vmax = max(1e-9, abs(hold_profit_pct))
            best_by_strategy = {}
            for strat_key, icon, label in _BEST_STRATEGIES:
                full = full_by_strategy[strat_key]
                best_by_n = [_best_axis1_at_axis2(full["grid"], n) for n in context["sweep_values"]]
                best_by_strategy[strat_key] = (best_by_n, full["initial_asset"])
                for profit_pct, _ in best_by_n:
                    vmax = max(vmax, abs(profit_pct))

            rows = []
            for strat_key, icon, label in _BEST_STRATEGIES:
                best_by_n, strat_initial_asset = best_by_strategy[strat_key]
                cells = []
                for n, (profit_pct, best_axis1) in zip(context["sweep_values"], best_by_n):
                    total = strat_initial_asset * (1 + profit_pct / 100)
                    cells.append({
                        "n": n,
                        "profit_pct": profit_pct,
                        "total": total,
                        "color": _profit_color(profit_pct, vmax),
                        "link": _strategy_link_from_axes(
                            strat_key, best_axis1, n, code, init_i, end_date_iso, applied_period,
                        ),
                    })
                rows.append({"key": strat_key, "icon": icon, "label": label, "cells": cells})

            hold_cells = [
                {"n": n, "profit_pct": hold_profit_pct, "total": hold_only_asset, "color": _profit_color(hold_profit_pct, vmax)}
                for n in context["sweep_values"]
            ]
            context["hold_row"] = {"icon": "🤚", "label": "매도/매수 없음 (단순 보유)", "cells": hold_cells}

            context["rows"] = rows
            context["initial_asset"] = initial_asset
            context["hold_only_asset"] = hold_only_asset

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"계산 중 오류가 발생했습니다: {e}"

    return render_template("best_sweep_heatmap.html", **context)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
