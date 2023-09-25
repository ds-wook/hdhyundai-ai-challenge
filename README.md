# hdhyundai-ai-challenge
항만 내 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발

## Feature Information
+ ARI_CO 도착항의 소속국가(도착항 앞 2글자)
+ ARI_PO 도착항의 항구명(도착항 뒷 글자)
+ SHIP_TYPE_CATEGORY 선종 통합 바탕으로 5대 선종으로 분류
+ DIST 정박지(ber_port)와 접안지 사이의 거리
+ ATA anc_port에 도착한 시점으 utc. 실제 정박 시간(Actual Time of Arrival)
+ ID 선박식별 일련 번호
+ BEARDTH 선박의 폭
+ BUILT 선박의 연령
+ DEADWEIGHT 선박 재화중량톤수
+ DEPTH 선박의 깊이
+ GT 용적톤수(Groos Tonnage) 값
+ LENGTH 선박의 길이
+ SHIPMANAGER 선박 소유주
+ FLAG 선박의 국적
+ U_WIND 풍향 u벡터
+ V_WIND 풍향 v벡터
+ AIR_TEMPERATURE 기온
+ BN 보퍼트 풍력 계급
+ ATA_LT anc_port에 도착한 시점의 형지 정박 시간(Local Time of Arrival)(단위: H)
+ DUBAI 해당일의 두바이유 값
+ BRENT 해당일의 브렌트유 값
+ WTI 해당일의 WTI 값
+ BDI_ADJ 조정된 벌크운임지수
+ PORT_SIZE 접안지 폴리곤 영역의 크기
+ CI_HOUR 대기시간(target)