"""
전세사기 위험도 분석 시스템
=========================
서울시 전세 거래 데이터와 전세사기 피해 데이터를 기반으로
구별 위험도를 분석하고 사용자 맞춤형 위험도 점수를 산출합니다.

사용법:
    python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
import koreanize_matplotlib  # noqa: F401 (한글 폰트 설정)


def load_data():
    """공공데이터 CSV 파일을 로드합니다."""
    fraud_data = pd.read_csv(
        "data/주택도시보증공사_경공매지원서비스 신청자의 전세사기피해주택 소재지(시군구)_20241031.csv",
        encoding="CP949",
    )

    transaction_data = pd.read_csv(
        "data/아파트(전월세)_실거래가_20241205203326.csv",
        encoding="CP949",
        skiprows=15,
        low_memory=False,
    )

    return fraud_data, transaction_data


def preprocess_data(fraud_data, transaction_data):
    """데이터 전처리: 정규식 기반 시군구 추출, 이상치 제거(IQR)"""
    # 보증금 숫자 변환
    transaction_data.loc[:, "보증금(만원)"] = pd.to_numeric(
        transaction_data["보증금(만원)"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    # 시군구 추출 (정규식)
    fraud_data["시군구"] = fraud_data["시군구"].str.extract(r"(?!.*시)(?!.*군)([가-힣]+구)$")
    fraud_data = fraud_data.dropna(subset=["시군구"]).groupby("시군구", as_index=False)["피해주택수"].sum()

    # 전세 데이터 필터링
    transaction_data = transaction_data[["시군구", "전월세구분", "보증금(만원)", "월세금(만원)"]]
    transaction_data["시군구"] = transaction_data["시군구"].str.extract(r"([가-힣]+구)")
    transaction_data = transaction_data[transaction_data["전월세구분"] == "전세"]
    transaction_data["보증금(만원)"] = pd.to_numeric(
        transaction_data["보증금(만원)"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    # IQR 방식 이상치 제거
    q1 = transaction_data["보증금(만원)"].quantile(0.25)
    q3 = transaction_data["보증금(만원)"].quantile(0.75)
    iqr = q3 - q1
    transaction_data = transaction_data[
        (transaction_data["보증금(만원)"] >= q1 - 1.5 * iqr)
        & (transaction_data["보증금(만원)"] <= q3 + 1.5 * iqr)
    ]

    return fraud_data, transaction_data


def analyze_and_visualize(fraud_data, transaction_data):
    """데이터 분석 및 시각화"""
    # 구별 평균 보증금 집계
    avg_deposit = (
        transaction_data.groupby("시군구")["보증금(만원)"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_deposit.columns = ["시군구", "평균보증금", "거래수"]

    # 공통 구 필터링
    common_districts = set(fraud_data["시군구"]) & set(avg_deposit["시군구"])
    fraud_filtered = fraud_data[fraud_data["시군구"].isin(common_districts)].copy()
    transaction_filtered = avg_deposit[avg_deposit["시군구"].isin(common_districts)]
    transaction_filtered = transaction_filtered.sort_values("평균보증금", ascending=False)
    district_order = transaction_filtered["시군구"].tolist()

    # --- 시각화 1: 피해건수 & 평균보증금 ---
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 1, 1)
    fraud_filtered["시군구"] = pd.Categorical(
        fraud_filtered["시군구"], categories=district_order, ordered=True
    )
    fraud_filtered = fraud_filtered.sort_values("시군구")
    ax1 = sns.barplot(data=fraud_filtered, x="시군구", y="피해주택수", color="skyblue")
    plt.title("구별 전세사기 피해건수", pad=20, size=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("피해건수")
    for i, v in enumerate(fraud_filtered["피해주택수"]):
        ax1.text(i, v, f"{v:,}건", ha="center", va="bottom")

    plt.subplot(2, 1, 2)
    ax2 = sns.barplot(data=transaction_filtered, x="시군구", y="평균보증금", color="lightcoral")
    plt.title("구별 평균 전세보증금 (거래수 표시)", pad=20, size=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("보증금(만원)")
    for i, (v, c) in enumerate(
        zip(transaction_filtered["평균보증금"], transaction_filtered["거래수"])
    ):
        ax2.text(i, v, f"{v:,.0f}만원\n(n={c:,}건)", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    # --- 상관관계 분석 ---
    merged_data = pd.merge(fraud_data, avg_deposit, on="시군구", how="inner")
    merged_data["피해율"] = merged_data["피해주택수"] / merged_data["거래수"] * 100

    correlation, p_value = pearsonr(merged_data["평균보증금"], merged_data["피해주택수"])

    print(f"\n{'='*50}")
    print(f"피어슨 상관계수: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"\n전세사기 피해 상위 5개 구:")
    print(merged_data.nlargest(5, "피해주택수")[["시군구", "피해주택수", "피해율"]].to_string(index=False))
    print(f"\n평균 보증금 상위 5개 구:")
    print(merged_data.nlargest(5, "평균보증금")[["시군구", "평균보증금", "피해주택수"]].to_string(index=False))

    # --- 시각화 2: 산점도 + 회귀선 ---
    plt.figure(figsize=(12, 8))
    sns.regplot(
        data=merged_data, x="평균보증금", y="피해주택수",
        scatter_kws={"alpha": 0.5}, line_kws={"color": "red"},
    )
    plt.title("전세 보증금과 사기 피해 건수의 관계")
    plt.xlabel("평균 보증금 (만원)")
    plt.ylabel("사기 피해 건수")
    for _, row in merged_data.iterrows():
        plt.annotate(
            row["시군구"],
            (row["평균보증금"], row["피해주택수"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.show()

    # --- 회귀분석 ---
    x_const = sm.add_constant(merged_data["평균보증금"])
    model = sm.OLS(merged_data["피해주택수"], x_const).fit()
    print(f"\n{'='*50}")
    print("OLS 회귀분석 결과:")
    print(model.summary().tables[1])

    # --- Quartile 분석 ---
    merged_data["보증금_quartile"] = pd.qcut(
        merged_data["평균보증금"], q=4, labels=["1분위(하)", "2분위", "3분위", "4분위(상)"]
    )
    quartile_analysis = (
        merged_data.groupby("보증금_quartile", observed=False)["피해주택수"]
        .agg(["mean", "sum", "count"])
        .round(2)
    )
    print(f"\n보증금 구간별 사기 피해 현황:")
    print(quartile_analysis)

    # --- 시각화 3: 구간별 박스플롯 ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_data, x="보증금_quartile", y="피해주택수")
    plt.title("보증금 구간별 사기 피해 건수 분포")
    plt.xlabel("보증금 구간")
    plt.ylabel("사기 피해 건수")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return fraud_data, avg_deposit, merged_data


def risk_scoring(fraud_data, avg_deposit):
    """사용자 입력 기반 위험도 스코어링 시스템"""
    print(f"\n{'='*50}")
    print("전세사기 위험도 분석 시스템")
    print(f"{'='*50}")
    print("\n분석 가능한 구 목록:")
    print(", ".join(sorted(avg_deposit["시군구"].unique())))

    district = input("\n분석을 원하시는 구를 입력하세요 (예: 강남구): ")
    while district not in avg_deposit["시군구"].values:
        print("잘못된 구 이름입니다. 다시 입력해주세요.")
        district = input("\n분석을 원하시는 구를 입력하세요 (예: 강남구): ")

    deposit = 0
    while True:
        try:
            deposit = float(input("구매하시려는 보증금(만원)을 입력하세요 (예: 20000): "))
            break
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

    # 위험도 점수 계산
    district_fraud = fraud_data[fraud_data["시군구"] == district]["피해주택수"].iloc[0]
    district_trans = avg_deposit[avg_deposit["시군구"] == district]
    avg_deposit_value = district_trans["평균보증금"].iloc[0]
    transactions = district_trans["거래수"].iloc[0]

    deposit_diff_percentage = abs(deposit - avg_deposit_value) / avg_deposit_value * 100
    deposit_diff_score = min(100, deposit_diff_percentage)

    max_fraud = fraud_data["피해주택수"].max()
    fraud_score = min(100, (district_fraud / max_fraud * 150))

    fraud_rate = (district_fraud / transactions) * 100
    fraud_rate_score = min(100, fraud_rate * 30)

    risk_score = (
        deposit_diff_score * 0.3
        + fraud_score * 0.35
        + fraud_rate_score * 0.35
    )

    print(f"\n{'='*50}")
    print(f"{district} 전세 위험도 분석 결과")
    print(f"{'='*50}")
    print(f"종합 위험도 점수: {risk_score:.1f}/100")
    print(f"해당 구 평균 보증금: {avg_deposit_value:,.0f}만원")
    print(f"입력하신 보증금: {deposit:,.0f}만원")
    print(f"전세사기 피해건수: {district_fraud:,}건")
    print(f"전체 거래량: {transactions:,}건")
    print(f"거래량 대비 피해율: {fraud_rate:.2f}%")
    print(f"\n위험도 상세:")
    print(f"  보증금 차이: {deposit_diff_score:.1f}/100")
    print(f"  피해건수: {fraud_score:.1f}/100")
    print(f"  피해율: {fraud_rate_score:.1f}/100")

    if risk_score >= 75:
        print("\n⚠ 매우 높은 위험: 거래 시 각별한 주의가 필요합니다.")
    elif risk_score >= 50:
        print("\n⚠ 중간 위험: 부동산 및 계약 관련 전문가와 상담을 권장합니다.")
    else:
        print("\n✓ 비교적 안전: 일반적인 주의사항을 지켜주세요.")


def main():
    print("데이터 로드 중...")
    fraud_data, transaction_data = load_data()

    print("데이터 전처리 중...")
    fraud_data, transaction_data = preprocess_data(fraud_data, transaction_data)

    print("분석 및 시각화 진행 중...")
    fraud_data, avg_deposit, _ = analyze_and_visualize(fraud_data, transaction_data)

    risk_scoring(fraud_data, avg_deposit)
    print("\n분석이 모두 완료되었습니다.")


if __name__ == "__main__":
    main()
