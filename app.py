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
    get_stock_data, get_stock_name, grid_trade_strategy,
    compute_profit_heatmap, compute_profit_heatmap2, SISE_DAY_URL,
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


def _build_backtest_link(selected_file, gap, qty, init_shares, sell_only, allow_negative_cash):
    """히트맵 셀/요약 클릭 시 그 조건 그대로 백테스트 페이지로 이동하는 링크를 만든다."""
    params = {
        "file": selected_file,
        "sell_gap": gap,
        "buy_gap": gap,
        "qty": qty,
        "init_shares": init_shares,
    }
    if sell_only:
        params["sell_only"] = "on"
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    return f"/backtest?{urlencode(params)}"


def _build_backtest2_link(selected_file, gap, profit_gap, qty, profit_recover, init_shares, sell_only, allow_negative_cash):
    """히트맵2 셀/요약 클릭 시 그 조건 그대로 백테스트2 페이지로 이동하는 링크를 만든다."""
    params = {
        "file": selected_file,
        "sell_gap": gap,
        "buy_gap": gap,
        "qty": qty,
        "init_shares": init_shares,
        "profit_gap": profit_gap,
        "profit_recover": profit_recover,
    }
    if sell_only:
        params["sell_only"] = "on"
    if allow_negative_cash:
        params["allow_negative_cash"] = "on"
    return f"/backtest2?{urlencode(params)}"


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
    네이버에 다시 접속하지 않는다 (부하/차단 방지).
    """
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)

    selected_file = request.args.get("file", "").strip()
    sell_gap = request.args.get("sell_gap", "5").strip()
    buy_gap = request.args.get("buy_gap", "").strip()  # 비워두면 sell_gap과 동일하게 처리
    qty = request.args.get("qty", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    sell_only = request.args.get("sell_only") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"

    context = {
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "sell_gap": sell_gap,
        "buy_gap": buy_gap,
        "qty": qty,
        "init_shares": init_shares,
        "sell_only": sell_only,
        "allow_negative_cash": allow_negative_cash,
        "error": None,
        "summary": None,
        "trade_log": None,
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
            qty_i = int(qty)
            init_i = int(init_shares)
            if sell_gap_f <= 0:
                raise ValueError("매도 gap은 0보다 커야 합니다.")
            if buy_gap_f is not None and buy_gap_f <= 0:
                raise ValueError("매수 gap은 0보다 커야 합니다.")
            if qty_i <= 0:
                raise ValueError("거래 수량은 1 이상이어야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

            result = grid_trade_strategy(
                df,
                trade_qty=qty_i,
                sell_gap_percent=sell_gap_f,
                buy_gap_percent=buy_gap_f,
                initial_shares=init_i,
                sell_only=sell_only,
                allow_negative_cash=allow_negative_cash,
            )
            context["effective_buy_gap"] = buy_gap_f if buy_gap_f is not None else sell_gap_f

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            sorted_df = df.sort_values("날짜")
            chart_labels = sorted_df["날짜"].dt.strftime("%Y-%m-%d").tolist()
            chart_prices = sorted_df["종가"].tolist()
            chart_sell_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매도"
            ]
            chart_buy_points = [
                {"x": row["날짜"], "y": row["가격"]} for row in trade_log if row["구분"] == "매수"
            ]

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
            context["chart_labels"] = chart_labels
            context["chart_prices"] = chart_prices
            context["chart_sell_points"] = chart_sell_points
            context["chart_buy_points"] = chart_buy_points

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("backtest.html", **context)


@app.route("/backtest2", methods=["GET"])
def backtest2():
    """
    그리드 매매 백테스트 + "이익 회수" 기능.
    매수/매도 규칙은 /backtest와 완전히 동일하고, 추가로: 기준가(첫날 종가로 시작) 대비
    profit_gap% 이상 오르면 그 시점 평가차익 중 profit_recover%를 현금에서 떼어 별도
    적립금으로 옮긴다(이후 매매에 쓰이지 않음). 이벤트 발동 시 기준가는 그 시점 주가로 갱신된다.
    """
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)

    selected_file = request.args.get("file", "").strip()
    sell_gap = request.args.get("sell_gap", "5").strip()
    buy_gap = request.args.get("buy_gap", "").strip()  # 비워두면 sell_gap과 동일하게 처리
    qty = request.args.get("qty", "10").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    sell_only = request.args.get("sell_only") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    profit_gap = request.args.get("profit_gap", "5").strip()
    profit_recover = request.args.get("profit_recover", "50").strip()

    context = {
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "sell_gap": sell_gap,
        "buy_gap": buy_gap,
        "qty": qty,
        "init_shares": init_shares,
        "sell_only": sell_only,
        "allow_negative_cash": allow_negative_cash,
        "profit_gap": profit_gap,
        "profit_recover": profit_recover,
        "error": None,
        "summary": None,
        "trade_log": None,
        "recover_log": None,
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
    }

    if selected_file:
        path = os.path.join(DATA_DIR, selected_file)
        m = _CSV_NAME_RE.match(selected_file)

        if not m or not os.path.isfile(path):
            context["error"] = "선택한 파일을 찾을 수 없습니다. 메인 페이지에서 먼저 종목을 조회해주세요."
            return render_template("backtest2.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
            sell_gap_f = float(sell_gap)
            buy_gap_f = float(buy_gap) if buy_gap else None
            qty_i = int(qty)
            init_i = int(init_shares)
            profit_gap_f = float(profit_gap)
            profit_recover_f = float(profit_recover)
            if sell_gap_f <= 0:
                raise ValueError("매도 gap은 0보다 커야 합니다.")
            if buy_gap_f is not None and buy_gap_f <= 0:
                raise ValueError("매수 gap은 0보다 커야 합니다.")
            if qty_i <= 0:
                raise ValueError("거래 수량은 1 이상이어야 합니다.")
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")
            if profit_gap_f <= 0:
                raise ValueError("이익 회수 gap은 0보다 커야 합니다.")
            if not (1 <= profit_recover_f <= 100):
                raise ValueError("이익 회수율은 1~100 사이여야 합니다.")

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

            result = grid_trade_strategy(
                df,
                trade_qty=qty_i,
                sell_gap_percent=sell_gap_f,
                buy_gap_percent=buy_gap_f,
                initial_shares=init_i,
                sell_only=sell_only,
                allow_negative_cash=allow_negative_cash,
                profit_gap_percent=profit_gap_f,
                profit_recover_percent=profit_recover_f,
            )
            context["effective_buy_gap"] = buy_gap_f if buy_gap_f is not None else sell_gap_f

            trade_log = result.pop("매매일지")
            for row in trade_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

            recover_log = result.pop("이익회수일지")
            for row in recover_log:
                row["날짜"] = pd.Timestamp(row["날짜"]).strftime("%Y-%m-%d")

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

        except ValueError as e:
            context["error"] = f"입력 오류: {e}"
        except Exception as e:
            context["error"] = f"백테스트 계산 중 오류가 발생했습니다: {e}"

    return render_template("backtest2.html", **context)


@app.route("/heatmap", methods=["GET"])
def heatmap():
    """
    gap 1~50%(1% 단위) x 매매수량 1~50(1단위) = 2,500가지 조합의 수익률을 계산해
    히트맵으로 보여준다. 저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)

    selected_file = request.args.get("file", "").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    sell_only = request.args.get("sell_only") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    gap_min = request.args.get("gap_min", "1").strip()
    gap_max = request.args.get("gap_max", "50").strip()
    qty_min = request.args.get("qty_min", "1").strip()
    qty_max = request.args.get("qty_max", "50").strip()

    context = {
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "init_shares": init_shares,
        "sell_only": sell_only,
        "allow_negative_cash": allow_negative_cash,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "qty_min": qty_min,
        "qty_max": qty_max,
        "error": None,
        "code": None,
        "days": None,
        "created_display": None,
        "gaps": None,
        "qtys": None,
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
            return render_template("heatmap.html", **context)

        context["code"] = m.group(1)
        context["days"] = m.group(2)
        created = m.group(3)
        context["created_display"] = f"{created[:4]}-{created[4:6]}-{created[6:]}"

        try:
            init_i = int(init_shares)
            if init_i < 0:
                raise ValueError("시작 주식 수는 0 이상이어야 합니다.")

            gap_min_i = int(gap_min)
            gap_max_i = int(gap_max)
            qty_min_i = int(qty_min)
            qty_max_i = int(qty_max)
            if gap_min_i < 1 or qty_min_i < 1:
                raise ValueError("gap/수량 하한은 1 이상이어야 합니다.")
            if gap_max_i < gap_min_i:
                raise ValueError("gap 상한은 하한보다 크거나 같아야 합니다.")
            if qty_max_i < qty_min_i:
                raise ValueError("수량 상한은 하한보다 크거나 같아야 합니다.")

            df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["날짜"])
            if df.empty:
                raise ValueError("CSV에 데이터가 없습니다.")

            gap_values = range(gap_min_i, gap_max_i + 1)  # 1% 단위
            qty_values = range(qty_min_i, qty_max_i + 1)  # 1주 단위

            result = compute_profit_heatmap(
                df, gap_values, qty_values, initial_shares=init_i,
                sell_only=sell_only, allow_negative_cash=allow_negative_cash,
            )

            vmax = max(abs(result["best"]["profit_pct"]), abs(result["worst"]["profit_pct"]), 1e-9)

            cells = []
            for gi, g in enumerate(result["gaps"]):
                for qi, q in enumerate(result["qtys"]):
                    pct = result["grid"][gi][qi]
                    is_best = (g == result["best"]["gap"] and q == result["best"]["qty"])
                    is_worst = (g == result["worst"]["gap"] and q == result["worst"]["qty"])
                    cells.append({
                        "gap": g, "qty": q, "pct": pct,
                        "color": _profit_color(pct, vmax),
                        "link": _build_backtest_link(selected_file, g, q, init_i, sell_only, allow_negative_cash),
                        "is_best": is_best,
                        "is_worst": is_worst,
                    })

            context["gaps"] = result["gaps"]
            context["qtys"] = result["qtys"]
            context["cells"] = cells
            context["best"] = result["best"]
            context["worst"] = result["worst"]
            context["initial_asset"] = result["initial_asset"]
            context["best_link"] = _build_backtest_link(
                selected_file, result["best"]["gap"], result["best"]["qty"], init_i, sell_only, allow_negative_cash
            )
            context["worst_link"] = _build_backtest_link(
                selected_file, result["worst"]["gap"], result["worst"]["qty"], init_i, sell_only, allow_negative_cash
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_backtest_link(
                        selected_file, combo["gap"], combo["qty"], init_i, sell_only, allow_negative_cash
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
    백테스트2(이익 회수) 전용 히트맵: 매매 gap(%) x 이익회수 gap(%) 조합의 수익률을 계산한다.
    거래 수량과 회수율은 폼에서 고정 숫자로 입력받는다 (스윕 대상이 아님).
    저장된 로컬 CSV만 사용 (네이버 재접속 없음).
    """
    files = _list_local_csvs()
    groups = _grouped_local_csvs(files)

    selected_file = request.args.get("file", "").strip()
    init_shares = request.args.get("init_shares", "100").strip()
    sell_only = request.args.get("sell_only") == "on"
    allow_negative_cash = request.args.get("allow_negative_cash") == "on"
    qty = request.args.get("qty", "10").strip()
    profit_recover = request.args.get("profit_recover", "50").strip()
    gap_min = request.args.get("gap_min", "1").strip()
    gap_max = request.args.get("gap_max", "50").strip()
    profit_gap_min = request.args.get("profit_gap_min", "1").strip()
    profit_gap_max = request.args.get("profit_gap_max", "50").strip()

    context = {
        "files": files,
        "groups": groups,
        "selected_file": selected_file,
        "init_shares": init_shares,
        "sell_only": sell_only,
        "allow_negative_cash": allow_negative_cash,
        "qty": qty,
        "profit_recover": profit_recover,
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
        "initial_asset": None,
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

            qty_i = int(qty)
            if qty_i <= 0:
                raise ValueError("거래 수량은 1 이상이어야 합니다.")

            profit_recover_f = float(profit_recover)
            if not (1 <= profit_recover_f <= 100):
                raise ValueError("이익 회수율은 1~100 사이여야 합니다.")

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
                df, gap_values, profit_gap_values, trade_qty=qty_i,
                profit_recover_percent=profit_recover_f, initial_shares=init_i,
                sell_only=sell_only, allow_negative_cash=allow_negative_cash,
            )

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
                        "link": _build_backtest2_link(
                            selected_file, g, pg, qty_i, profit_recover_f, init_i, sell_only, allow_negative_cash
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
            context["best_link"] = _build_backtest2_link(
                selected_file, result["best"]["gap"], result["best"]["profit_gap"],
                qty_i, profit_recover_f, init_i, sell_only, allow_negative_cash,
            )
            context["worst_link"] = _build_backtest2_link(
                selected_file, result["worst"]["gap"], result["worst"]["profit_gap"],
                qty_i, profit_recover_f, init_i, sell_only, allow_negative_cash,
            )

            def _with_link(combo):
                return {
                    **combo,
                    "link": _build_backtest2_link(
                        selected_file, combo["gap"], combo["profit_gap"],
                        qty_i, profit_recover_f, init_i, sell_only, allow_negative_cash,
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
