"""
app.py — Flask 웹 앱: 종목 코드와 기간을 입력받아 네이버 금융 데이터를 보여줌.

실행:
    python app.py
접속:
    http://localhost:8000
"""

import os
import re
from datetime import datetime
from io import StringIO
from urllib.parse import urlencode

import pandas as pd
from flask import Flask, render_template, request, Response
from core import (
    get_stock_data, get_stock_name, grid_trade_strategy, resolve_trade_qty,
    compute_profit_heatmap, compute_profit_heatmap2, compute_profit_heatmap_2d,
    HEATMAP_FEATURES, SISE_DAY_URL,
)

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
_CSV_NAME_RE = re.compile(r"^(\d+)_(\d+)days_(\d{8})\.csv$")


def _first_page_url(code: str) -> str:
    """데이터를 가져오는 첫 번째 요청 URL (page=1)을 만들어 반환."""
    return f"{SISE_DAY_URL}?{urlencode({'code': code, 'page': 1})}"


def _local_csv_path(code: str, days: int, created: str = None) -> str:
    created = created or datetime.now().strftime("%Y%m%d")
    return os.path.join(DATA_DIR, f"{code}_{days}days_{created}.csv")


def _save_local_csv(df: pd.DataFrame, code: str, days: int) -> str:
    """조회한 데이터를 data/ 폴더에 (조회 당일 날짜를 붙여) 저장해, 이후 백테스트/히트맵
    페이지에서 네이버를 다시 호출하지 않고 재사용할 수 있게 한다."""
    path = _local_csv_path(code, days)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _list_local_csvs():
    """
    data/ 폴더에 저장된 CSV 목록을 (종목코드, 기간, 생성일자) 단위로 반환.
    같은 종목/기간이라도 조회한 날짜가 다르면 별도 항목으로 나열된다.
    """
    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        m = _CSV_NAME_RE.match(fname)
        if m:
            code, days, created = m.group(1), m.group(2), m.group(3)
            files.append({
                "filename": fname,
                "code": code,
                "days": days,
                "created": created,
                "created_display": f"{created[:4]}-{created[4:6]}-{created[6:]}",
                "group_key": f"{code}_{days}",
            })
    # 최신 생성일자가 먼저 오도록 정렬
    files.sort(key=lambda f: (f["group_key"], f["created"]), reverse=True)
    return files


def _grouped_local_csvs(files):
    """
    _list_local_csvs() 결과를 (종목코드+기간) 단위로 묶는다.
    프론트에서 '종목 선택 -> 생성일자 선택' 2단계 드롭다운을 만들 때 사용.
    """
    groups = {}
    order = []
    for f in files:
        key = f["group_key"]
        if key not in groups:
            groups[key] = {"code": f["code"], "days": f["days"], "items": []}
            order.append(key)
        groups[key]["items"].append({"filename": f["filename"], "created_display": f["created_display"]})
    return [groups[k] for k in order]


def _most_recent_file(files):
    """가장 최근에 생성된 로컬 CSV의 파일명을 반환한다 (없으면 빈 문자열)."""
    if not files:
        return ""
    return max(files, key=lambda f: f["created"])["filename"]


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
    selected_file, gap, qty_pct, init_shares, no_sell, no_buy, allow_negative_cash,
    profit_gap=None, profit_recover=None, capital=None,
):
    """
    히트맵/히트맵2 셀·요약 클릭 시 그 조건 그대로 백테스트 페이지로 이동하는 링크를 만든다.
    profit_gap/profit_recover를 함께 넘기면(히트맵2용) "이익 회수 사용"을 켠 상태로 연결한다.
    """
    params = {
        "file": selected_file,
        "sell_gap": gap,
        "buy_gap": gap,
        "qty_pct": qty_pct,
        "init_shares": init_shares,
    }
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


@app.route("/", methods=["GET"])
def index():
    code = request.args.get("code", "").strip()
    days = request.args.get("days", "30").strip()

    context = {
        "code": code,
        "days": days,
        "error": None,
        "table": None,
        "name": None,
        "chart_labels": None,
        "chart_prices": None,
        "source_url": None,
    }

    if code:
        try:
            days_int = int(days)
            if days_int <= 0:
                raise ValueError("기간은 1 이상의 숫자여야 합니다.")

            context["source_url"] = _first_page_url(code)

            try:
                context["name"] = get_stock_name(code)
            except Exception:
                context["name"] = None

            df = get_stock_data(code, days_int)

            if df.empty:
                context["error"] = "데이터가 없습니다. 종목 코드를 확인해주세요."
            else:
                _save_local_csv(df, code, days_int)

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
    """화면에 표시된 것과 동일한 조건(code, days)으로 다시 조회해 CSV로 내려준다."""
    code = request.args.get("code", "").strip()
    days = request.args.get("days", "30").strip()

    if not code:
        return "code 파라미터가 필요합니다.", 400

    try:
        days_int = int(days)
        if days_int <= 0:
            raise ValueError("기간은 1 이상의 숫자여야 합니다.")
    except ValueError as e:
        return f"입력 오류: {e}", 400

    df = get_stock_data(code, days_int)
    if df.empty:
        return "데이터가 없습니다. 종목 코드를 확인해주세요.", 400

    _save_local_csv(df, code, days_int)

    buf = StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)

    filename = f"{code}_{days_int}days_{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


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
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)
    default_file = _most_recent_file(files)

    selected_file = request.args.get("file", "").strip()
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
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "default_file": default_file,
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
        "code": None,
        "days": None,
        "created_display": None,
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

    if selected_file:
        path = os.path.join(DATA_DIR, selected_file)
        m = _CSV_NAME_RE.match(selected_file)

        if not m or not os.path.isfile(path):
            context["error"] = "선택한 파일을 찾을 수 없습니다. 메인 페이지에서 먼저 종목을 조회해주세요."
            return render_template("backtest.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
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

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

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


@app.route("/heatmap", methods=["GET"])
def heatmap():
    """
    gap 1~50%(1% 단위) x 매매수량(시작 보유 주식수 대비 %) 1~50%(1% 단위) = 2,500가지 조합의
    수익률을 계산해 히트맵으로 보여준다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)
    default_file = _most_recent_file(files)

    selected_file = request.args.get("file", "").strip()
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
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "default_file": default_file,
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
        "code": None,
        "days": None,
        "created_display": None,
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
        "qty_stats": None,
        "recommended_qty": None,
        "recommended_qty_link": None,
    }

    if selected_file:
        path = os.path.join(DATA_DIR, selected_file)
        m = _CSV_NAME_RE.match(selected_file)

        if not m or not os.path.isfile(path):
            context["error"] = "선택한 파일을 찾을 수 없습니다. 메인 페이지에서 먼저 종목을 조회해주세요."
            return render_template("heatmap.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
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

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

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
                        "link": _build_backtest_link(selected_file, g, qp, init_i, no_sell, no_buy, allow_negative_cash),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["qty_pcts"] = result["qty_pcts"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["best_link"] = _build_backtest_link(
                selected_file, result["best"]["gap"], result["best"]["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash
            )
            context["worst_link"] = _build_backtest_link(
                selected_file, result["worst"]["gap"], result["worst"]["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash
            )

            context["qty_stats"] = result["qty_stats"]
            context["recommended_qty"] = result["recommended_qty"]
            if result["recommended_qty"] is not None:
                recover_params = {
                    "file": selected_file,
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
                        selected_file, combo["gap"], combo["qty_pct"], init_i, no_sell, no_buy, allow_negative_cash
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
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)
    default_file = _most_recent_file(files)

    selected_file = request.args.get("file", "").strip()
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
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "default_file": default_file,
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
        "code": None,
        "days": None,
        "created_display": None,
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
        "capital_used": None,
    }

    if selected_file:
        path = os.path.join(DATA_DIR, selected_file)
        m = _CSV_NAME_RE.match(selected_file)

        if not m or not os.path.isfile(path):
            context["error"] = "선택한 파일을 찾을 수 없습니다. 메인 페이지에서 먼저 종목을 조회해주세요."
            return render_template("heatmap2.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
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

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

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
                            selected_file, g, qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                            profit_gap=pg, profit_recover=profit_recover_f, capital=capital_f,
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
            context["capital_used"] = result["capital"]
            context["best_link"] = _build_backtest_link(
                selected_file, result["best"]["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                profit_gap=result["best"]["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
            )
            context["worst_link"] = _build_backtest_link(
                selected_file, result["worst"]["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                profit_gap=result["worst"]["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_backtest_link(
                        selected_file, combo["gap"], qty_pct_f, init_i, no_sell, no_buy, allow_negative_cash,
                        profit_gap=combo["profit_gap"], profit_recover=profit_recover_f, capital=capital_f,
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
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)
    default_file = _most_recent_file(files)

    selected_file = request.args.get("file", "").strip()
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
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "default_file": default_file,
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
        "code": None,
        "days": None,
        "created_display": None,
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
    }

    if selected_file:
        path = os.path.join(DATA_DIR, selected_file)
        m = _CSV_NAME_RE.match(selected_file)

        if not m or not os.path.isfile(path):
            context["error"] = "선택한 파일을 찾을 수 없습니다. 메인 페이지에서 먼저 종목을 조회해주세요."
            return render_template("heatmap3.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
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

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

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
                    selected_file, params["gap"], params["qty_pct"], init_i, no_sell, no_buy,
                    allow_negative_cash, profit_gap=params["profit_gap"],
                    profit_recover=params["profit_recover"], capital=capital_f,
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
