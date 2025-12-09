# ============================================
# 0. IMPORT 파트
# ============================================
import json
import os
import re
import sqlite3
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache
import folium
import googlemaps
import requests
import streamlit as st
import hashlib
import threading
import queue
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from folium.features import DivIcon
from streamlit_folium import st_folium
from apify_client import ApifyClient 
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None
try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from crewai import Agent, Task, Crew, Process, LLM
except ImportError:
    Agent = None
    Task = None
    Crew = None
    Process = None
    LLM = None
try:
    from docx import Document
except ImportError:
    Document = None

# Initialize Logger for flight/hotel search
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================
# 1. 함수 정의 파트
# ============================================

def parse_duration_to_days(duration_str):
    """
    "3박 4일" 또는 "4일" 같은 문자열을 정수로 변환합니다.
    예: "3박 4일" -> 4, "6박 7일" -> 7, "1박 2일" -> 2
    """
    if not duration_str:
        return 1  # 기본값 1일
    
    # "박" 또는 "일" 앞의 숫자 찾기
    match_night = re.search(r'(\d+)\s*박', duration_str)
    match_day = re.search(r'(\d+)\s*일', duration_str)
    
    try:
        if match_night:
            # X박 (X+1)일
            return int(match_night.group(1)) + 1
        elif match_day:
            # X일
            return int(match_day.group(1))
        else:
            # 숫자만 있는 경우 (예: "4")
            num_match = re.search(r'(\d+)', duration_str)
            if num_match:
                return int(num_match.group(1))
    except Exception:
        pass
    
    return 1  # 파싱 실패 시 기본값


def geocode_location(gmaps_client, location_name):
    """
    장소 이름을 받아서 위도, 경도를 반환합니다.
    """
    if not gmaps_client or not location_name:
        return None
    
    try:
        geocode_result = gmaps_client.geocode(location_name, language="ko")
        if geocode_result:
            loc = geocode_result[0]['geometry']['location']
            return [loc['lat'], loc['lng']]
    except Exception as e:
        st.error(f"Geocoding 오류: {e}")
    
    return None

@st.cache_resource
def load_sentiment_analyzer():
    """허깅페이스 감정 분석 모델 로드 및 캐싱"""
    try:
        # 한국어 감정 분석 모델 (beomi/kcbert-base-v2-sentiment) 사용
        return pipeline("text-classification", model="beomi/kcbert-base-v2-sentiment")
    except Exception as e:
        print(f"감정 분석 모델 로딩 실패: {e}")
        return None

sentiment_analyzer = load_sentiment_analyzer()

def get_sentiment_score(review_text: str) -> float:
    """주어진 텍스트를 분석하여 0.0 (부정) ~ 1.0 (긍정) 점수를 반환"""
    if not sentiment_analyzer or not review_text.strip():
        return 0.5 
    
    try:
        result = sentiment_analyzer(review_text)[0]
        label = result['label']
        score = result['score']
        
        if "positive" in label.lower():
            return score
        elif "negative" in label.lower():
            return 1.0 - score
        else:
            return 0.5
            
    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e}")
        return 0.5

# 전역에서 분석기 로드
sentiment_analyzer = load_sentiment_analyzer()

def get_sentiment_score(review_text: str) -> float:
    """
    주어진 리뷰 텍스트에 대해 감정 점수(0.0 ~ 1.0)를 반환합니다.
    """
    if not sentiment_analyzer:
        return 0.5 # 모델 로딩 실패 시 중립 값 반환
    
    if not review_text.strip():
        return 0.5 # 리뷰 텍스트가 없을 경우 중립 값 반환

    try:
        # 파이프라인 실행
        result = sentiment_analyzer(review_text)[0]
        label = result['label']
        score = result['score']
        
        # '긍정(positive)' 라벨이면 score를, '부정(negative)' 라벨이면 1 - score를 반환하여
        # 0.0 (강한 부정) ~ 1.0 (강한 긍정) 스케일로 통일합니다.
        if "positive" in label.lower():
            return score
        elif "negative" in label.lower():
            return 1.0 - score
        else:
            return 0.5 # 중립적이거나 알 수 없는 경우
            
    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e}")
        return 0.5


def create_map(gmaps_client, center_location):
    """
    지도를 생성하고 여행지 중심으로 표시합니다.
    """
    # 지도 중심 설정
    if center_location:
        map_center = center_location
        zoom_level = 12
    else:
        # 기본값: 서울
        map_center = [37.5665, 126.9780]
        zoom_level = 10
    
    # 지도 생성
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_level
    )
    
    return m


def get_region_cities():
    """
    지역별 도시 딕셔너리를 반환합니다.
    """
    return {
        "일본": ["도쿄", "후쿠오카", "삿포로", "오사카"],
        "중화/중국": ["상하이", "가오슝", "타이베이", "홍콩", "베이징"],
        "한국": ["가평/양평", "강릉/속초", "경주", "부산", "여수", "인천", "전주", "제주", 
                "춘천/홍천", "태안", "통영/거제/남해", "포항/안동"],
        "미주": ["벤쿠버", "샌프란시스코", "토론토", "하와이", "뉴욕", "로스앤젤레스"],
        "유럽": ["리스본", "밀라노", "브뤼셀", "포르투", "파리", "프라하", "로마", "런던", 
                "바르셀로나", "빈", "인터라켄", "마드리드", "부다페스트", "프랑크푸르트", 
                "뮌헨", "암스테르담", "베를린"],
        "동남아시아": ["나트랑", "치앙마이", "푸꾸옥", "라오스", "쿠알라룸프르", "다낭", "방콕", 
                      "세부", "코타키나발루", "싱가포르", "하노이", "호치민", "발리", "푸켓", "보라카이"],
        "남태평양": ["시드니", "멜버른", "괌", "사이판"]
    }


def initialize_session_state():
    """
    세션 상태를 초기화합니다.
    """
    if "map_center" not in st.session_state:
        st.session_state.map_center = [37.5665, 126.9780]  # 기본값: 서울
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 10
    if "num_days" not in st.session_state:
        st.session_state.num_days = 1
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = None
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = None
    if "vector_db_status" not in st.session_state:
        st.session_state.vector_db_status = None
    if "vector_db_last_region" not in st.session_state:
        st.session_state.vector_db_last_region = None
    if "vector_db_in_progress" not in st.session_state:
        st.session_state.vector_db_in_progress = False
    if "vector_db_progress" not in st.session_state:
        st.session_state.vector_db_progress = 0.0
    if "vector_db_current_status" not in st.session_state:
        st.session_state.vector_db_current_status = None
    if "day_chats" not in st.session_state:
        st.session_state.day_chats = {}  # {day: [{"role": "user/assistant", "content": "...", "recommendations": [...]}]}
    if "confirmed_plans" not in st.session_state:
        st.session_state.confirmed_plans = {}  # {day: [{"place_id": "...", "name": "...", "metadata": {...}}]}
    if "pending_places" not in st.session_state:
        st.session_state.pending_places = []  # 전역 챗봇에서 선택된 장소들 (확정 전)
    if "confirmed_places" not in st.session_state:
        st.session_state.confirmed_places = []  # 확정된 장소들 (day별이 아닌 전체)


# --- 벡터 DB 관련 상수 및 유틸 ---
VECTOR_DB_DIR = Path("vector_dbs")
VECTOR_SQLITE_PATH = Path("vector_store.db")
CHROMA_DIR = Path("chroma_store")
VECTOR_META_TABLE = "vector_meta_v2"
VECTOR_ENTRIES_TABLE = "vector_entries_v2"
APIFY_ACTOR_ID = "compass/google-maps-crawler" 
MAX_PLACES_PER_REGION = 30


def ensure_vector_db_dir():
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def init_sqlite_store():
    conn = sqlite3.connect(VECTOR_SQLITE_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_meta_v2 (
            db_key TEXT PRIMARY KEY,
            display_name TEXT,
            region TEXT,
            city TEXT,
            record_count INTEGER,
            updated_at TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_entries_v2 (
            place_id TEXT PRIMARY KEY,
            db_key TEXT,
            city TEXT,
            name TEXT,
            payload TEXT,
            embedding TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def sanitize_name(name: str) -> str:
    """
    ChromaDB 컬렉션 이름 규칙에 맞게 이름을 변환합니다.
    규칙: 3-63자, 알파벳/숫자/._- 만 허용, 알파벳/숫자로 시작과 끝.
    한글이 포함되어 있거나 규칙에 맞지 않으면 안전하게 해시값으로 변환합니다.
    """
    # 1. 먼저 영어, 숫자, _, - 만 남기고 나머지는 제거해봅니다.
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "", name.strip())
    
    # 2. ChromaDB 규칙 검사
    # (1) 길이가 3글자 미만이거나 (한글만 있어서 다 지워진 경우 포함)
    # (2) 첫 글자나 마지막 글자가 알파벳/숫자가 아닌 경우 (언더바로 시작하는 경우 등)
    if len(clean_name) < 3 or not clean_name[0].isalnum() or not clean_name[-1].isalnum():
        # 입력받은 원본 이름(한글 포함)을 MD5 해시로 변환하여 고유한 영문 ID 생성
        # 예: "영국_런던" -> "vec_5d41402abc..."
        hash_val = hashlib.md5(name.encode('utf-8')).hexdigest()
        return f"vec_{hash_val}"
        
    return clean_name


def get_vector_db_path(name: str) -> Path:
    return VECTOR_DB_DIR / f"{sanitize_name(name)}.json"


@dataclass
class VectorDBNames:
    base: str
    sqlite: str
    chroma: str
    english: str


def get_english_city_name(city_name: Optional[str], gmaps_client) -> str:
    if not city_name:
        return "UnknownCity"
    if not gmaps_client:
        return city_name
    try:
        geocode_result = gmaps_client.geocode(city_name, language="en")
        if geocode_result:
            components = geocode_result[0].get("address_components", [])
            for component in components:
                if "locality" in component.get("types", []):
                    return component.get("long_name") or city_name
            formatted = geocode_result[0].get("formatted_address")
            if formatted:
                return formatted.split(",")[0]
    except Exception as exc:
        st.warning(f"도시 영문명 변환에 실패했습니다. 원문을 사용합니다. (사유: {exc})")
    return city_name


def build_vector_db_names(city_name: Optional[str], gmaps_client) -> VectorDBNames:
    english_name = get_english_city_name(city_name, gmaps_client)
    base = sanitize_name(english_name or city_name or "UnknownCity")
    if not base:
        base = "UnknownCity"
    return VectorDBNames(
        base=base,
        sqlite=f"{base}_SQLite",
        chroma=f"{base}_Chroma",
        english=english_name or city_name or "UnknownCity",
    )


def vector_db_exists(db_key: str) -> bool:
    ensure_vector_db_dir()
    init_sqlite_store()
    conn = sqlite3.connect(VECTOR_SQLITE_PATH)
    c = conn.cursor()
    c.execute(f"SELECT 1 FROM {VECTOR_META_TABLE} WHERE db_key = ? LIMIT 1", (db_key,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 지점 간의 거리를 계산합니다 (Haversine 공식, 단위: km).
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # 지구 반지름 (km)
    
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance


def fetch_places_by_category_and_sort(
    city_name: str,
    gmaps_client,
    label: str,
    place_type: str,
    limit_per_category: int,
    center_coordinates: Optional[List[float]] = None,
    max_distance_km: float = 50.0,
    use_streamlit: bool = False,  # 스레드 내부에서는 False로 설정
):
    """
    카테고리별로 검색 후 리뷰 수(user_ratings_total) 기준으로 상위 N개를 추출합니다.
    중심 좌표가 제공되면 해당 지역 내의 장소만 필터링합니다.
    타임아웃과 예외 처리를 강화하여 무한 대기 방지.
    use_streamlit: False일 경우 Streamlit 함수를 호출하지 않음 (스레드 내부에서 사용 시)
    """
    # results 변수 초기화 (모든 분기에서 사용 가능하도록)
    results = []
    
    if not city_name or not gmaps_client:
        if use_streamlit:
            st.write(f"❌ [검색 시작] '{label}' 검색: city_name 또는 gmaps_client가 없습니다.")
        return []

    query = f"{city_name} {label}"
    if use_streamlit:
        st.write(f"🔍 [검색 시작] '{label}' 검색 시작 - 쿼리: {query}")
    
    # Google Places API 호출을 별도 스레드에서 실행하여 타임아웃 적용
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    api_thread = None
    
    def api_call_worker():
        try:
            if use_streamlit:
                st.write(f"🔄 [API 호출] Google Places API 검색 시작 - 쿼리: {query}, 목표: {limit_per_category}개")
            
            all_results = []
            next_page_token = None
            max_pages = max(5, (limit_per_category // 20) + 1)  # 목표 개수에 맞춰 페이지 수 계산 (한 페이지당 약 20개)
            
            # 페이지네이션을 통해 더 많은 결과 수집
            for page_num in range(max_pages):
                try:
                    if next_page_token:
                        # 다음 페이지 요청 (next_page_token 사용 시 약간의 대기 필요)
                        import time
                        time.sleep(2)  # next_page_token 사용 시 최소 2초 대기 필요
                        # page_token을 사용할 때는 query 없이 page_token만 전달
                        response = gmaps_client.places(page_token=next_page_token, language="ko")
                    else:
                        # 첫 페이지 요청
                        response = gmaps_client.places(query=query, language="ko")
                    
                    if not response:
                        if use_streamlit:
                            st.write(f"❌ [API 응답] Google Places API 응답이 없습니다. 쿼리: {query}, 페이지: {page_num + 1}")
                        break
                    
                    page_results = response.get("results", [])
                    all_results.extend(page_results)
                    
                    if use_streamlit:
                        st.write(f"✅ [API 응답] 페이지 {page_num + 1}: {len(page_results)}개 결과 (누적: {len(all_results)}개)")
                    
                    # 다음 페이지 토큰 확인
                    next_page_token = response.get("next_page_token")
                    if not next_page_token:
                        # 더 이상 페이지가 없음
                        if use_streamlit:
                            st.write(f"📄 [API 응답] 모든 페이지 수집 완료 (총 {len(all_results)}개)")
                        break
                    
                    # 목표 개수에 도달했으면 중단
                    if len(all_results) >= limit_per_category * 2:  # 여유있게 2배 수집 (필터링 후에도 충분하도록)
                        if use_streamlit:
                            st.write(f"🎯 [API 응답] 목표 개수 충족 (총 {len(all_results)}개 수집)")
                        break
                        
                except Exception as e:
                    if use_streamlit:
                        st.write(f"⚠️ [API 응답] 페이지 {page_num + 1} 수집 실패: {type(e).__name__}: {str(e)[:100]}")
                    # 페이지 수집 실패해도 계속 진행
                    break
            
            if use_streamlit:
                if all_results:
                    st.write(f"✅ [API 응답] Google Places API 검색 완료 - 쿼리: {query}, 총 결과: {len(all_results)}개")
                else:
                    st.write(f"⚠️ [API 응답] Google Places API 응답은 있지만 결과가 없습니다. 쿼리: {query}")
            
            result_queue.put(all_results)
        except Exception as e:
            if use_streamlit:
                st.write(f"❌ [API 응답] Google Places API 호출 실패 - 쿼리: {query}, 오류: {type(e).__name__}: {str(e)[:100]}")
            exception_queue.put(e)
            result_queue.put([])  # 예외 발생 시에도 빈 리스트 추가하여 결과 큐 보장
    
    # API 호출을 별도 스레드에서 실행
    try:
        api_thread = threading.Thread(target=api_call_worker, daemon=True)
        api_thread.start()
        api_thread.join(timeout=20)  # 20초 타임아웃
        
        if api_thread.is_alive():
            # 타임아웃 발생 - 스레드 강제 종료 시도
            if use_streamlit:
                st.write(f"⏱️ [API 응답] Google Places API 타임아웃 발생 - 쿼리: {query} (20초 초과)")
            # daemon 스레드는 메인 스레드 종료 시 자동 종료되지만, 명시적으로 처리
            results = []
            return []
        elif not exception_queue.empty():
            # 예외 발생
            exc = exception_queue.get()
            if use_streamlit:
                st.write(f"❌ [API 응답] Google Places API 예외 발생 - 쿼리: {query}, 예외: {type(exc).__name__}: {str(exc)[:100]}")
            results = []
            return []
        elif not result_queue.empty():
            # 성공
            results = result_queue.get()
            if use_streamlit:
                st.write(f"✅ [API 응답] '{label}' 검색 완료: {len(results)}개 결과")
        else:
            # 결과가 없음
            if use_streamlit:
                st.write(f"⚠️ [API 응답] Google Places API 결과 큐가 비어있습니다. 쿼리: {query}")
            results = []
            return []
    except Exception as e:
        # 스레드 생성/실행 중 예외 발생
        if use_streamlit:
            st.write(f"❌ [검색 오류] '{label}' 검색 중 예외 발생: {type(e).__name__}: {str(e)[:100]}")
        results = []
        return []

    # results 변수가 정의되지 않았거나 비어있으면 빈 리스트 반환
    if not results:
        if use_streamlit:
            st.write(f"⚠️ [검색 완료] '{label}' 검색 결과가 없습니다.")
        return []
    
    # 중심 좌표가 있으면 거리 필터링
    if center_coordinates and len(center_coordinates) >= 2:
        try:
            if use_streamlit:
                st.write(f"📍 [거리 필터링] '{label}' 거리 필터링 시작: {len(results)}개 장소")
            center_lat, center_lng = center_coordinates[0], center_coordinates[1]
            filtered_results = []
            
            for place in results:
                try:
                    geometry = place.get("geometry", {})
                    location = geometry.get("location", {})
                    place_lat = location.get("lat")
                    place_lng = location.get("lng")
                    
                    if place_lat is not None and place_lng is not None:
                        distance = calculate_distance(center_lat, center_lng, place_lat, place_lng)
                        if distance <= max_distance_km:
                            place["distance_from_center"] = distance
                            filtered_results.append(place)
                except Exception as e:
                    # 스레드 내부에서는 조용히 스킵
                    if use_streamlit:
                        st.write(f"⚠️ [거리 필터링] 장소 거리 계산 실패: {type(e).__name__}")
                    continue
            
            results = filtered_results
            if use_streamlit:
                st.write(f"✅ [거리 필터링] '{label}' 거리 필터링 완료: {len(results)}개 장소")
        except Exception as e:
            if use_streamlit:
                st.write(f"❌ [거리 필터링] '{label}' 거리 필터링 실패: {type(e).__name__}: {str(e)[:200]}")
            # 필터링 실패 시 원본 결과 사용 (results는 이미 정의되어 있음)

    # 정렬 및 상위 N개 추출
    try:
        if use_streamlit:
            st.write(f"📊 [정렬] '{label}' 정렬 시작: {len(results)}개 장소")
        sorted_results = sorted(
            results,
            key=lambda x: x.get("user_ratings_total", 0),
            reverse=True,
        )
        top_n = []
        collected_count = 0
        for place in sorted_results:
            try:
                place["custom_category_label"] = label
                place["custom_category_type"] = place_type
                top_n.append(place)
                collected_count += 1
                # 목표 개수에 도달하면 중단
                if collected_count >= limit_per_category:
                    break
            except Exception as e:
                if use_streamlit:
                    st.write(f"⚠️ [정렬] 장소 추가 실패: {type(e).__name__}")
                continue
        
        if use_streamlit:
            if collected_count >= limit_per_category:
                st.write(f"✅ [정렬 완료] '{label}' 정렬 완료: 목표 {limit_per_category}개 모두 수집 ({len(top_n)}개)")
            else:
                st.write(f"⚠️ [정렬 완료] '{label}' 정렬 완료: 목표 {limit_per_category}개 중 {len(top_n)}개만 수집 (API에서 더 이상 데이터 없음)")
        return top_n
    except Exception as e:
        if use_streamlit:
            st.write(f"❌ [정렬 오류] '{label}' 정렬 실패: {type(e).__name__}: {str(e)[:200]}")
        return []


def fetch_google_place_details(gmaps_client, place_id: str):
    """
    Google Places API를 사용하여 장소 상세 정보를 가져옵니다.
    타임아웃 없이 응답을 기다립니다.
    
    Args:
        gmaps_client: Google Maps 클라이언트
        place_id: 장소 ID
    
    Returns:
        장소 상세 정보 딕셔너리, 실패 시 빈 딕셔너리
    """
    if not place_id or not gmaps_client:
        st.write(f"❌ [API 응답] place_id 또는 gmaps_client가 없습니다. place_id: {place_id}")
        return {}
    
    try:
        st.write(f"🔄 [API 호출] Google Places API 호출 시작 - place_id: {place_id[:20]}...")
        
        # [수정됨] API 필드명 변경 (복수형 -> 단수형)
        # types -> type, photos -> photo, reviews -> review
        fields = [
            "place_id",
            "name",
            "geometry",
            "formatted_address",
            "formatted_phone_number",
            "website",
            "rating",
            "user_ratings_total",
            "type",    # [수정] types -> type
            "opening_hours",
            "photo",   # [수정] photos -> photo
            "review",  # [수정] reviews -> review
            "price_level",
            "url",
            "editorial_summary",  # Google Places API의 장소 설명 추가
        ]
        
        response = gmaps_client.place(place_id=place_id, fields=fields, language="ko")
        
        if not response:
            st.write(f"❌ [API 응답] Google Places API 응답이 없습니다. place_id: {place_id[:20]}...")
            return {}
        
        result = response.get("result", {})
        
        if not result:
            st.write(f"❌ [API 응답] Google Places API result가 비어있습니다. place_id: {place_id[:20]}...")
            return {}

        # 단수형으로 요청했지만 결과는 기존 로직과 호환되도록 매핑
        # API 결과 키값도 'photos'가 아니라 'photo'로 올 수 있으므로 안전하게 처리
        # 보통 googlemaps 파이썬 클라이언트는 내부적으로 매핑해주기도 하지만, 
        # 원본 응답 키를 확인하여 변환해주는 것이 안전함.
        
        # 'photo' 키로 들어온 것을 'photos' 키로 복사 (기존 코드 호환성 유지)
        if "photo" in result:
            result["photos"] = result["photo"]
            
        # 'type' 키로 들어온 것을 'types' 키로 복사
        if "type" in result:
            result["types"] = result["type"]
            
        # 'review' 키로 들어온 것을 'reviews' 키로 복사
        if "review" in result:
            result["reviews"] = result["review"]

        place_name = result.get("name", "알 수 없는 장소")
        st.write(f"✅ [API 응답] Google Places API 응답 성공 - 장소명: {place_name}, place_id: {place_id[:20]}...")
        
        return result
        
    except Exception as exc:
        st.write(f"❌ [API 응답] Google Places API 호출 실패 - place_id: {place_id[:20]}..., 오류: {type(exc).__name__}: {str(exc)[:100]}")
        st.warning(f"Google 장소 상세 조회 실패: {exc}")
        return {}


def search_naver_blog_api(query: str, naver_client_id: str, naver_client_secret: str, display: int = 5):
    """
    네이버 검색 Open API (Blog Search) 로 블로그 목록을 가져오는 함수.
    description(요약문)까지 활용.
    """
    if not naver_client_id or not naver_client_secret:
        return []

    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }
    params = {
        "query": query,
        "display": display,
        "sort": "sim",  # 정확도순
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        # st.warning(f"네이버 블로그 검색 중 오류: {e}")
        return []

    blogs = []
    for item in data.get("items", []):
        title = re.sub(r"<.*?>", "", item.get("title", ""))  # HTML 태그 제거
        desc = re.sub(r"<.*?>", "", item.get("description", ""))
        link = item.get("link")
        blogs.append(
            {
                "title": title,
                "description": desc,
                "url": link,
            }
        )

    return blogs


def get_naver_blog_summary(place_name: str, openai_client, naver_client_id: str, naver_client_secret: str, max_blogs: int = 5, timeout: int = 10):
    """
    네이버 Search API에서 가져온 title + description만 가지고
    GPT에게 요약을 요청하는 함수.
    타임아웃을 추가하여 무한 대기 방지.
    """
    if not openai_client:
        return None, []

    try:
        blogs = search_naver_blog_api(f"{place_name} 후기", naver_client_id, naver_client_secret, display=max_blogs)
        if not blogs:
            return None, []

        context_parts = []
        for b in blogs:
            if not b.get("description"):
                continue
            context_parts.append(f"[{b['title']}]\n{b['description']}")

        if not context_parts:
            return None, blogs

        context = "\n\n---\n\n".join(context_parts)

        system_msg = """
        당신은 여행지를 소개하는 블로거입니다.
        아래에 여러 블로그의 제목과 요약(description)이 주어집니다.
        이를 바탕으로 해당 장소의 전반적인 분위기, 장단점, 추천 포인트를 한국어로 요약해 주세요.

        - 맛 / 분위기 / 가격 / 동선 팁 / 주의할 점 등이 보이면 항목별로 정리해 주세요.
        - 제공된 내용 범위 안에서만 요약하고, 과장하거나 없는 내용은 만들지 마세요.
        """

        user_msg = f"""
        [장소 이름]
        {place_name}

        [블로그 검색 결과 요약문]
        {context}
        """

        # OpenAI API 호출 (타임아웃은 requests 레벨에서 처리)
        try:
            # timeout 파라미터가 지원되는 경우 사용, 아니면 기본값 사용
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    timeout=timeout,  # OpenAI SDK의 timeout 파라미터 (지원되는 경우)
                    max_tokens=300,  # 토큰 수 제한으로 빠른 응답
                )
            except (TypeError, AttributeError):
                # timeout 파라미터가 지원되지 않는 경우
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=300,  # 토큰 수 제한으로 빠른 응답
                )
            summary = resp.choices[0].message.content
            return summary, blogs
        except Exception as api_error:
            # 타임아웃 또는 기타 API 오류 (조용히 실패)
            return None, blogs
    except Exception as e:
        # 네이버 API 호출 실패 또는 기타 오류
        return None, []


def fetch_serpapi_place_description(place_name: str, city_name: str, serpapi_key: str) -> Optional[str]:
    """
    SerpAPI를 사용하여 장소에 대한 설명 데이터를 가져옵니다.
    타임아웃을 추가하여 무한 대기 방지.
    """
    if not serpapi_key or not GoogleSearch:
        return None
    
    try:
        params = {
            "q": f"{place_name} {city_name}",
            "api_key": serpapi_key,
            "engine": "google",
            "hl": "ko",
            "gl": "kr"
        }
        
        # 타임아웃을 위해 threading과 queue 사용
        result_queue = queue.Queue()
        
        def search_with_timeout():
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                result_queue.put(("success", results))
            except Exception as e:
                result_queue.put(("error", e))
        
        search_thread = threading.Thread(target=search_with_timeout)
        search_thread.daemon = True
        search_thread.start()
        search_thread.join(timeout=10)  # 10초 타임아웃
        
        if search_thread.is_alive():
            # 타임아웃 발생
            return None
        
        if result_queue.empty():
            return None
        
        status, data = result_queue.get()
        
        if status == "error":
            return None
        
        results = data
        
        # knowledge_graph 또는 organic_results에서 설명 찾기
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            if "description" in kg:
                return kg["description"]
            if "about" in kg:
                return kg["about"]
        
        # organic_results에서 첫 번째 결과의 snippet 사용
        if "organic_results" in results and results["organic_results"]:
            first_result = results["organic_results"][0]
            if "snippet" in first_result:
                return first_result["snippet"]
        
        return None
    except Exception as e:
        # 조용히 실패 (너무 많은 경고 방지)
        return None


def fetch_apify_details(place_name: str, apify_token: str, timeout: int = 15):
    """
    ApifyClient 라이브러리를 사용하도록 변경 및 응답 처리 로직 수정
    타임아웃 추가로 무한 대기 방지
    """
    if not apify_token:
        return {
            "reviews": [],
            "crowd_levels": None,
            "feature_tags": [],
            "price_range": None,
            "keywords": [],
            "source": "token_missing",
        }
    
    try:
        client = ApifyClient(apify_token)
        
        run_input = {
            "searchStrings": [place_name],
            "maxCrawledPlacesPerSearch": 1,
            "language": "ko",
            "maxReviews": 5,
            "maxImages": 0,
            "scrapeReviewerName": True,  # 리뷰 작성자 정보 수집
            "scrapeReviewerId": True,    # 리뷰 작성자 ID 수집
        }
        
        # Actor 실행 (타임아웃을 짧게 설정)
        # Apify는 느릴 수 있으므로 최대 15초만 대기
        try:
            # call 메서드는 기본적으로 동기 실행이므로, 
            # 타임아웃을 위해 threading을 사용하거나 간단히 시도만 하고 실패 시 스킵
            run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        except Exception as call_error:
            # 호출 자체가 실패하면 빈 데이터 반환
            return {
                "reviews": [],
                "crowd_levels": None,
                "feature_tags": [],
                "price_range": None,
                "keywords": [],
                "source": "call_failed",
            }
        
        # 데이터셋에서 결과 가져오기
        if not run or "defaultDatasetId" not in run:
            return {
                "reviews": [],
                "crowd_levels": None,
                "feature_tags": [],
                "price_range": None,
                "keywords": [],
                "source": "no_dataset",
            }
        
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        
        if dataset_items:
            item = dataset_items[0]
            # 리뷰 데이터 수집 (최신 5개, 작성자 정보 포함)
            reviews_data = []
            for r in item.get("reviews", [])[:5]:  # 최신 5개만
                if r.get('text'):
                    review_info = {
                        "text": r.get('text'),
                        "author_name": r.get('authorName') or r.get('author_name') or r.get('authorName'),
                        "author_id": r.get('authorId') or r.get('author_id') or r.get('authorId'),
                        "rating": r.get('rating'),
                        "time": r.get('time'),
                    }
                    reviews_data.append(review_info)
            
            # 리뷰에서 키워드 추출 (간단한 방법: 빈도수 높은 단어)
            keywords = []
            if reviews_data:
                import re
                from collections import Counter
                all_text = " ".join([r.get("text", "") for r in reviews_data])
                # 한글 단어 추출 (2글자 이상)
                words = re.findall(r'[가-힣]{2,}', all_text)
                # 빈도수 상위 10개 키워드
                word_counts = Counter(words)
                keywords = [word for word, count in word_counts.most_common(10)]
            
            # 속성 태그 추출
            feature_tags = []
            if item.get("attributes"):
                feature_tags = list(item.get("attributes", {}).keys())
            elif item.get("tags"):
                feature_tags = item.get("tags", [])
            elif item.get("placeId"):
                # placeId는 태그가 아니므로 제외
                pass
            
            return {
                "reviews": reviews_data,  # 리뷰 정보 (작성자 포함)
                "crowd_levels": item.get("popularTimesHistogram"),
                "feature_tags": feature_tags,  # 장소 속성 태그
                "price_range": item.get("price"),
                "keywords": keywords,  # 리뷰에서 추출한 키워드
                "source": "apify",
            }
            
    except Exception as exc:
        # st.warning(f"Apify 호출 실패({place_name}): {exc}") # 너무 잦은 경고 방지
        pass
        
    return {
        "reviews": [],
        "crowd_levels": None,
        "feature_tags": [],
        "price_range": None,
        "keywords": [],
        "source": "apify_error",
    }


def build_embedding_payload(place: dict) -> str:
    # None 타입 처리 강화 (join 함수 사용 시 에러 방지)
    def safe_get(key, default=""):
        val = place.get(key)
        if val is None:
            return default
        return val

    def safe_join(lst, sep=", "):
        if not lst: return ""
        return sep.join([str(x) for x in lst if x])

    lines = [
        f"이름: {safe_get('name')}",
        f"주소: {safe_get('address')}",
        f"위치: {safe_get('lat')}, {safe_get('lng')}",
        f"카테고리: {safe_join(place.get('categories', []))}",
        f"사용자 정의 카테고리: {safe_get('custom_category')}",
        f"평점: {safe_get('rating', '정보없음')} / 리뷰 수: {safe_get('user_ratings_total', 0)}",
        f"설명: {safe_get('description')}",
        f"전화번호: {safe_get('phone_number')}",
        f"영업시간: {safe_join(place.get('opening_hours_text', []), ' | ')}",
        f"가격대: {safe_get('price_level')}",
        f"리뷰 요약: {safe_join(place.get('review_snippets', []), ' | ')}",
        f"네이버 블로그 요약: {safe_get('naver_blog_summary')}",
        f"역사 및 팁: {safe_get('history_and_tips')}",
    ]
    return "\n".join(line for line in lines if line and line.strip())


def store_vector_db(db_name: str, payload: dict):
    ensure_vector_db_dir()
    path = get_vector_db_path(db_name)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path


def persist_records_to_sqlite(db_key: str, display_name: str, region: str, city: str, records: List[dict]):
    init_sqlite_store()
    conn = sqlite3.connect(VECTOR_SQLITE_PATH)
    c = conn.cursor()
    c.execute(f"DELETE FROM {VECTOR_ENTRIES_TABLE} WHERE db_key = ?", (db_key,))
    for record in records:
        c.execute(
            f"""
            INSERT OR REPLACE INTO {VECTOR_ENTRIES_TABLE} (place_id, db_key, city, name, payload, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                db_key,
                city,
                record["name"],
                json.dumps(record["metadata"], ensure_ascii=False),
                json.dumps(record["embedding"]),
            ),
        )
    c.execute(
        f"""
        INSERT OR REPLACE INTO {VECTOR_META_TABLE} (db_key, display_name, region, city, record_count, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            db_key,
            display_name,
            region,
            city,
            len(records),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def persist_records_to_chroma(collection_name: str, records: List[dict]):
    if not chromadb:
        st.warning("chromadb 패키지가 설치되어 있지 않아 ChromaDB 저장을 건너뜁니다.")
        return
    ensure_vector_db_dir()
    # ChromaDB 경로 문자열 변환
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"collection": collection_name},
    )
    if not records:
        return
    ids = [record["id"] for record in records]
    documents = [build_embedding_payload(record["metadata"]) for record in records]
    metadatas = [
        {
            "collection": collection_name,
            "city": record["city"],
            "name": record["name"],
        }
        for record in records
    ]
    embeddings = [record["embedding"] for record in records]
    # Remove existing entries with the same ids to prevent duplicates
    if ids:
        collection.delete(ids=ids) # ID가 있을 때만 삭제 시도
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def load_places_from_vector_db(db_key: str) -> List[dict]:
    """
    벡터DB에서 장소 데이터를 읽어옵니다.
    """
    init_sqlite_store()
    conn = sqlite3.connect(VECTOR_SQLITE_PATH)
    c = conn.cursor()
    c.execute(
        f"SELECT place_id, name, payload, embedding FROM {VECTOR_ENTRIES_TABLE} WHERE db_key = ?",
        (db_key,)
    )
    results = c.fetchall()
    conn.close()
    
    places = []
    for place_id, name, payload_json, embedding_json in results:
        try:
            payload = json.loads(payload_json) if payload_json else {}
            embedding = json.loads(embedding_json) if embedding_json else None
            lat = payload.get("lat")
            lng = payload.get("lng")
            if lat is not None and lng is not None:
                places.append({
                    "place_id": place_id,
                    "name": name,
                    "lat": lat,
                    "lng": lng,
                    "metadata": payload,
                    "category": payload.get("custom_category", "기타"),
                    "embedding": embedding
                })
        except Exception:
            continue
    
    return places


def search_similar_places_from_vector_db(
    db_key: str,
    user_query: str,
    openai_client,
    top_k: int = 10,
    group_id: Optional[int] = None
) -> List[dict]:
    """
    벡터DB에서 사용자 쿼리와 유사한 장소를 검색합니다.
    group_id가 제공되면 해당 그룹의 장소만 검색합니다.
    """
    if not openai_client:
        return []
    
    # 1. 사용자 쿼리 임베딩 생성
    try:
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding, dtype="float32")
    except Exception as e:
        st.error(f"쿼리 임베딩 생성 실패: {e}")
        return []
    
    # 2. 벡터DB에서 장소 로드
    places = load_places_from_vector_db(db_key)
    
    if not places:
        return []
    
    # 3. 그룹 필터링 (group_id가 제공된 경우)
    if group_id is not None:
        init_sqlite_store()
        conn = sqlite3.connect(VECTOR_SQLITE_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT place_id FROM place_groups WHERE db_key = ? AND group_id = ?",
            (db_key, group_id)
        )
        group_place_ids = {row[0] for row in c.fetchall()}
        conn.close()
        
        places = [p for p in places if p["place_id"] in group_place_ids]
    
    # 4. 유사도 계산
    similarities = []
    for place in places:
        if place.get("embedding"):
            try:
                place_embedding = np.array(place["embedding"], dtype="float32")
                # 코사인 유사도 계산
                similarity = float(np.dot(
                    query_embedding / np.linalg.norm(query_embedding),
                    place_embedding / np.linalg.norm(place_embedding)
                ))
                similarities.append((similarity, place))
            except Exception:
                continue
    
    # 5. 유사도 기준 정렬 및 상위 k개 반환
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [place for _, place in similarities[:top_k]]


def render_place_cards(places: List[dict], google_maps_api_key: str, is_global_chatbot: bool = False):
    if not places:
        return
    
    # ❌ 카테고리별 그룹화 및 정렬 로직 (이 부분을 모두 삭제하거나 주석 처리하세요)
    # places_by_category = {} 
    # ... (중략: 카테고리별 정렬 코드) ...
    # sorted_categories = sorted(category_review_counts.items(), key=lambda x: x[1], reverse=True)
    
    
    # 💡 수정된 로직: 복합 순위가 적용된 places 리스트를 바로 순회합니다.
    # st.markdown(f"### {category}") <- 이 카테고리 헤더도 삭제합니다.
    
    # 2개씩 묶어서 표시 (기존의 가로 배치 로직 유지)
    for i in range(0, len(places), 2):
        # 한 줄에 2개의 카드뷰 배치
        col_left, col_right = st.columns(2)
        
        # 왼쪽 카드
        with col_left:
            place = places[i]
            # 💡 _render_single_place_card 호출 시 index를 복합 순위에 맞게 조정: i+1 대신 idx를 사용하거나,
            #    여기서는 0부터 시작하는 리스트 인덱스를 전달합니다. (i)
            #    _render_single_place_card 내부에서 idx+1로 순위가 표시되므로 그대로 둡니다.
            _render_single_place_card(place, i, place.get("category", "기타"), google_maps_api_key, is_global_chatbot)
        
        # 오른쪽 카드 (장소가 홀수개일 경우 마지막은 비어있을 수 있음)
        with col_right:
            if i + 1 < len(places):
                place = places[i + 1]
                _render_single_place_card(place, i + 1, place.get("category", "기타"), google_maps_api_key, is_global_chatbot)
        
        st.markdown("")  # 카드 행 간 간격


def _render_single_place_card(place: dict, index: int, category: str, google_maps_api_key: str, is_global_chatbot: bool):
    """
    단일 장소 카드를 렌더링합니다. 높이를 통일하기 위해 사용됩니다.
    """
    metadata = place.get("metadata", {})
    place_name = place.get("name") or metadata.get("name", "알 수 없는 장소")
    rating = metadata.get("rating")
    address = metadata.get("address", "")
    description = metadata.get("description", "") or metadata.get("history_and_tips", "")
    reviews = metadata.get("reviews", []) or metadata.get("review_snippets", [])
    photos = metadata.get("photos", [])
    photo_references = metadata.get("photo_references", [])
    
    # 카드 컨테이너 (높이 통일을 위해 고정 높이 컨테이너 사용)
    with st.container(border=True):
        # 왼쪽: 사진, 오른쪽: 정보
        col_img, col_info = st.columns([2, 5])
        
        with col_img:
            # 이미지
            if photo_references and google_maps_api_key:
                photo_ref = photo_references[0]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=600&photo_reference={photo_ref}&key={google_maps_api_key}"
                st.image(photo_url, width=10, use_container_width=True)
            else:
                st.markdown("### 📸")
                st.caption("이미지 없음")
        
        with col_info:
            # 헤더 섹션: 이름, 평점, 카테고리
            col_title, col_badge = st.columns([3, 1])
            with col_title:
                #st.markdown(f"**{place_name}**")
                st.markdown(f"**🏅 {index + 1}위** - {place_name}")
            with col_badge:
                category = place.get("category") or metadata.get("custom_category", "기타")
                # 카테고리 배지 스타일
                category_colors = {
                    "관광지": "🔵",
                    "음식점": "🍽️",
                    "카페": "☕",
                    "쇼핑": "🛍️",
                    "숙박": "🏨",
                    "액티비티": "🎯",
                    "기타": "📍"
                }
                category_icon = category_colors.get(category, "📍")
                st.markdown(f"**{category_icon} {category}**")
            
            # 평점과 기본 정보 (한 줄에 표시)
            info_cols = st.columns([2, 2, 2])
            with info_cols[0]:
                if rating:
                    st.markdown(f"⭐ **{rating}** / 5.0")
                else:
                    st.caption("평점 없음")
            
            with info_cols[1]:
                # 가격대 표시
                price_level = metadata.get("price_level")
                price_range = metadata.get("price_range")
                if price_level is not None:
                    price_symbols = ["💰", "💰💰", "💰💰💰", "💰💰💰💰"]
                    price_text = price_symbols[min(price_level - 1, 3)] if 1 <= price_level <= 4 else "💰"
                    st.markdown(f"**{price_text}** 가격대")
                elif price_range:
                    st.markdown(f"**{price_range}**")
                else:
                    st.caption("가격 정보 없음")
            
            with info_cols[2]:
                # 리뷰 개수
                user_ratings_total = metadata.get("user_ratings_total")
                if user_ratings_total:
                    st.markdown(f"💬 리뷰 {user_ratings_total:,}개")
                else:
                    st.caption("리뷰 없음")
            
            st.markdown("")
            
            # 간단한 설명
            if description:
                description_clean = description.split('\n')[0].strip()
                # 리뷰 패턴이 포함되어 있지 않은 경우에만 표시
                if not any(keyword in description_clean for keyword in ["⭐", "리뷰", "후기", "Review", "review", "작성자", "별점"]):
                    # 설명이 길면 첫 100자만 표시 (카드 크기 축소에 맞춤)
                    description_short = description_clean[:100]
                    if len(description_clean) > 100:
                        description_short += "..."
                    with st.container(border=False):
                        st.markdown(f"*{description_short}*")
                else:
                    st.caption(f"{place_name}에 대한 정보")
            else:
                st.caption("설명 없음")
            
            st.markdown("")
            
            # 상세 정보 섹션 (주소, 영업시간, 가격대)
            # 주소
            if address:
                st.markdown(f"📍 **주소**")
                st.caption(address[:50] + "..." if len(address) > 50 else address)
            
            # 영업시간과 가격대를 왼쪽/오른쪽으로 배치
            col_hours, col_price = st.columns(2)
            
            with col_hours:
                # 영업시간 (토글로 표시)
                opening_hours_text = metadata.get("opening_hours_text", [])
                opening_hours_raw = metadata.get("opening_hours_raw", {})
                has_opening_hours = bool(opening_hours_text or (opening_hours_raw and opening_hours_raw.get("weekday_text")))
                
                if has_opening_hours:
                    with st.expander("🕐 영업시간", expanded=False):
                        if opening_hours_text:
                            for day_schedule in opening_hours_text[:3]:  # 최대 3개만 표시
                                st.markdown(f"• {day_schedule}")
                        elif opening_hours_raw:
                            weekday_text = opening_hours_raw.get("weekday_text", [])
                            if weekday_text:
                                for day_schedule in weekday_text[:3]:  # 최대 3개만 표시
                                    st.markdown(f"• {day_schedule}")
                else:
                    with st.expander("🕐 영업시간", expanded=False):
                        st.caption("영업시간 정보가 없습니다.")
            
            with col_price:
                # 가격대 (토글로 표시)
                price_level = metadata.get("price_level")
                price_range = metadata.get("price_range")
                has_price_info = price_level is not None or price_range
                
                if has_price_info:
                    with st.expander("💰 가격대", expanded=False):
                        if price_level is not None:
                            price_symbols = ["💰", "💰💰", "💰💰💰", "💰💰💰💰"]
                            price_text = price_symbols[min(price_level - 1, 3)] if 1 <= price_level <= 4 else "💰"
                            price_labels = {1: "저렴함", 2: "보통", 3: "비쌈", 4: "매우 비쌈"}
                            price_label = price_labels.get(price_level, "정보 없음")
                            st.markdown(f"**{price_text} {price_label}**")
                        elif price_range:
                            st.markdown(f"**{price_range}**")
                else:
                    with st.expander("💰 가격대", expanded=False):
                        st.caption("가격 정보가 없습니다.")
            
            st.markdown("")
            
            # Google Places API 리뷰 (최신 5개)
            google_reviews = metadata.get("google_reviews", []) or metadata.get("reviews", [])
            # Google Places API의 reviews 필드에서 직접 가져온 리뷰인지 확인
            if not google_reviews and metadata.get("place_details"):
                place_details = metadata.get("place_details", {})
                google_reviews = place_details.get("reviews", [])
            
            if google_reviews:
                # 최신 5개 표시
                with st.expander(f"⭐ Google 리뷰 ({len(google_reviews)}개)", expanded=False):
                    for idx, review in enumerate(google_reviews[:5], 1):  # 최대 5개 표시
                        if isinstance(review, dict):
                            review_text = review.get("text", "") or review.get("review_text", "")
                            review_rating = review.get("rating", "")
                            author_name = review.get("author_name", "") or review.get("author", "익명")
                            
                            if review_text:
                                with st.container(border=True):
                                    review_header_cols = st.columns([3, 1])
                                    with review_header_cols[0]:
                                        if review_rating:
                                            st.markdown(f"⭐ **{review_rating}/5**")
                                        else:
                                            st.markdown("⭐ 리뷰")
                                    with review_header_cols[1]:
                                        st.caption(f"by {author_name}")
                                    st.markdown(review_text)
                        elif isinstance(review, str):
                            with st.container(border=True):
                                st.markdown(review)
            else:
                with st.expander("⭐ Google 리뷰 (0개)", expanded=False):
                    st.caption("Google 리뷰 없음")
            
            # 네이버 블로그 리뷰 (관광지 카테고리에만 표시)
            if category == "관광지":
                naver_summary = metadata.get("naver_blog_summary")
                naver_blogs = metadata.get("naver_blogs", [])
                
                if naver_summary:
                    with st.expander(f"📝 네이버 블로그 리뷰 요약", expanded=False):
                        st.markdown(naver_summary)
                elif naver_blogs:
                    with st.expander(f"📝 네이버 블로그 리뷰 ({len(naver_blogs)}개)", expanded=False):
                        for idx, blog in enumerate(naver_blogs, 1):  # 전체 표시
                            with st.container(border=True):
                                st.markdown(f"**{blog.get('title', '제목 없음')}**")
                                if blog.get('description'):
                                    st.caption(blog['description'])
                                if blog.get('url'):
                                    st.markdown(f"[원문 보기 →]({blog['url']})")
                else:
                    with st.expander("📝 네이버 블로그 리뷰 (0개)", expanded=False):
                        st.caption("네이버 블로그 리뷰 없음")
            
            st.markdown("")
            
            # 일정에 추가 버튼 (고유 키 생성)
            place_id = place.get("place_id", "unknown")
            button_key = f"add_{place_id}_{category}_{index}_{hash(place_name)}"
            if st.button("➕ 일정에 추가", key=button_key, use_container_width=True, type="primary"):
                if is_global_chatbot:
                    # 전역 챗봇: pending_places에 추가
                    place_id = place.get("place_id")
                    if not any(p.get("place_id") == place_id for p in st.session_state.pending_places):
                        st.session_state.pending_places.append({
                            "place_id": place_id,
                            "name": place_name,
                            "metadata": metadata
                        })
                        st.success(f"{place_name}이(가) 선택되었습니다! 아래 '일정 확정' 버튼을 클릭하여 확정하세요.")
                        st.rerun()
                else:
                    # Day별 챗봇: 기존 로직 유지
                    day_num = st.session_state.get("current_day_num", 1)
                    if day_num not in st.session_state.confirmed_plans:
                        st.session_state.confirmed_plans[day_num] = []
                    
                    # 중복 체크
                    place_id = place.get("place_id")
                    if not any(p.get("place_id") == place_id for p in st.session_state.confirmed_plans[day_num]):
                        st.session_state.confirmed_plans[day_num].append({
                            "place_id": place_id,
                            "name": place_name,
                            "metadata": metadata
                        })
                        st.success(f"{place_name}이(가) Day {day_num} 일정에 추가되었습니다!")
                        st.rerun()
                
                st.markdown("")  # 카드 간 간격


def group_places_by_distance(places: List[dict], num_groups: int, min_per_group: int = 4, max_per_group: int = 7, gmaps_client=None) -> List[List[dict]]:
    """
    장소들을 거리 기반으로 그룹화합니다.
    
    Args:
        places: 그룹화할 장소 리스트 [{"place_id": "...", "name": "...", "metadata": {...}}]
        num_groups: 그룹 수 (여행 일정에 따라 결정)
        min_per_group: 그룹당 최소 장소 수
        max_per_group: 그룹당 최대 장소 수
        gmaps_client: Google Maps 클라이언트 (위도/경도 가져오기용)
    
    Returns:
        그룹화된 장소 리스트 [[place1, place2, ...], [place3, place4, ...], ...]
    """
    if not places or num_groups <= 0:
        return []
    
    # 1. 장소들의 위도/경도 추출
    places_with_coords = []
    for place in places:
        metadata = place.get("metadata", {})
        location = metadata.get("location") or metadata.get("geometry", {}).get("location", {})
        
        lat = location.get("lat")
        lng = location.get("lng")
        
        # 위도/경도가 없으면 place_id로 geocoding 시도
        if lat is None or lng is None:
            if gmaps_client and place.get("place_id"):
                try:
                    place_details = gmaps_client.place(place.get("place_id"), fields=["geometry"])
                    if place_details.get("result", {}).get("geometry", {}).get("location"):
                        loc = place_details["result"]["geometry"]["location"]
                        lat = loc.get("lat")
                        lng = loc.get("lng")
                except:
                    pass
        
        if lat is not None and lng is not None:
            places_with_coords.append({
                "place": place,
                "lat": lat,
                "lng": lng
            })
    
    if not places_with_coords:
        # 위도/경도가 없으면 단순히 균등 분할
        places_per_group = len(places) // num_groups
        remainder = len(places) % num_groups
        groups = []
        idx = 0
        for i in range(num_groups):
            group_size = places_per_group + (1 if i < remainder else 0)
            groups.append([p["place"] for p in places[idx:idx+group_size]])
            idx += group_size
        return groups
    
    # 2. K-means 클러스터링 (간단한 구현)
    if np is None:
        # numpy가 없으면 단순 균등 분할
        places_per_group = len(places_with_coords) // num_groups
        remainder = len(places_with_coords) % num_groups
        groups = []
        idx = 0
        for i in range(num_groups):
            group_size = places_per_group + (1 if i < remainder else 0)
            groups.append([p["place"] for p in places_with_coords[idx:idx+group_size]])
            idx += group_size
        return groups
    
    # numpy를 사용한 K-means 클러스터링
    coords = np.array([[p["lat"], p["lng"]] for p in places_with_coords])
    
    # 초기 중심점 선택 (무작위)
    np.random.seed(42)  # 재현성을 위해
    centroids = coords[np.random.choice(len(coords), num_groups, replace=False)]
    
    # K-means 반복
    for _ in range(100):  # 최대 100회 반복
        # 각 점을 가장 가까운 중심점에 할당
        distances = np.sqrt(((coords[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # 새로운 중심점 계산
        new_centroids = np.array([coords[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] 
                                  for i in range(num_groups)])
        
        # 수렴 확인
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    # 3. 그룹별로 장소 할당
    groups = [[] for _ in range(num_groups)]
    for i, label in enumerate(labels):
        groups[label].append(places_with_coords[i]["place"])
    
    # 4. 그룹 크기 조정 (최소 4개, 최대 7개)
    # 그룹이 너무 작으면 인접 그룹과 병합, 너무 크면 분할
    final_groups = []
    for group in groups:
        if len(group) < min_per_group:
            # 작은 그룹은 다음 그룹과 병합 (마지막 그룹이면 이전 그룹과)
            if final_groups:
                final_groups[-1].extend(group)
            else:
                final_groups.append(group)
        elif len(group) > max_per_group:
            # 큰 그룹은 분할
            for i in range(0, len(group), max_per_group):
                final_groups.append(group[i:i+max_per_group])
        else:
            final_groups.append(group)
    
    # 그룹 수가 num_groups보다 적으면 빈 그룹 추가
    while len(final_groups) < num_groups:
        final_groups.append([])
    
    # 그룹 수가 num_groups보다 많으면 마지막 그룹들을 병합
    if len(final_groups) > num_groups:
        # 마지막 그룹들을 하나로 병합
        merged = []
        for group in final_groups[num_groups-1:]:
            merged.extend(group)
        final_groups = final_groups[:num_groups-1] + [merged]
    
    return final_groups


def _local_haversine_km(origin: dict, dest: dict) -> float:
    """
    Haversine 공식을 사용하여 두 지점 간의 직선 거리를 계산합니다 (km).
    
    Args:
        origin: {"lat": float, "lng": float}
        dest: {"lat": float, "lng": float}
    
    Returns:
        거리 (km)
    """
    from math import radians, sin, cos, sqrt, atan2
    
    lat1, lon1 = origin.get("lat"), origin.get("lng")
    lat2, lon2 = dest.get("lat"), dest.get("lng")
    
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float('inf')
    
    R = 6371.0  # 지구 반지름 (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def compute_optimal_route_order(places: List[dict]) -> List[int]:
    """
    Nearest Neighbor 알고리즘을 사용하여 장소들의 최적 방문 순서를 계산합니다.
    
    Args:
        places: [{"name": str, "metadata": {"lat": float, "lng": float}}, ...]
    
    Returns:
        최적 순서의 인덱스 리스트
    """
    if len(places) <= 1:
        return list(range(len(places)))
    
    # 각 장소의 좌표 추출
    coords = []
    for place in places:
        metadata = place.get("metadata", {})
        lat = metadata.get("lat")
        lng = metadata.get("lng")
        if lat is not None and lng is not None:
            coords.append({"lat": lat, "lng": lng})
        else:
            coords.append(None)
    
    best_order = None
    best_total = None
    
    # 각 시작점에서 Nearest Neighbor 알고리즘 실행
    for start in range(len(places)):
        remaining = list(range(len(places)))
        order = [start]
        remaining.remove(start)
        total_dist = 0.0
        
        while remaining:
            current_idx = order[-1]
            current_coord = coords[current_idx]
            best_next = None
            best_dist = None
            
            for cand_idx in remaining:
                cand_coord = coords[cand_idx]
                if not current_coord or not cand_coord:
                    continue
                try:
                    dist_km = _local_haversine_km(current_coord, cand_coord)
                except Exception:
                    continue
                
                if best_dist is None or dist_km < best_dist:
                    best_dist = dist_km
                    best_next = cand_idx
            
            if best_next is None:
                # 더 이상 계산 불가 → 남은 것들 그냥 뒤에 붙임
                order.extend(remaining)
                break
            
            order.append(best_next)
            remaining.remove(best_next)
            total_dist += best_dist if best_dist is not None else 0.0
        
        if best_order is None or (best_total is None or total_dist < best_total):
            best_order = order
            best_total = total_dist
    
    return best_order if best_order is not None else list(range(len(places)))


def optimize_route_for_day(
    gmaps_client,
    day_plans: List[dict],
    day_num: int
) -> dict:
    """
    하루치 일정의 최적 경로를 계산하고 교통수단 정보를 반환합니다.
    
    Args:
        gmaps_client: Google Maps 클라이언트
        day_plans: 해당 날짜의 확정 일정 리스트
        day_num: 날짜 번호
    
    Returns:
        {
            "optimal_order": [인덱스 리스트],
            "route_info": [{"from": str, "to": str, "transport": str, "distance_km": float, "duration_min": float}, ...]
        }
    """
    if len(day_plans) < 2:
        return {
            "optimal_order": list(range(len(day_plans))),
            "route_info": []
        }
    
    # 최적 순서 계산
    optimal_order = compute_optimal_route_order(day_plans)
    
    route_info = []
    
    # 각 구간별 최적 교통수단 계산
    for i in range(len(optimal_order) - 1):
        from_idx = optimal_order[i]
        to_idx = optimal_order[i + 1]
        
        from_place = day_plans[from_idx]
        to_place = day_plans[to_idx]
        
        from_metadata = from_place.get("metadata", {})
        to_metadata = to_place.get("metadata", {})
        
        from_lat = from_metadata.get("lat")
        from_lng = from_metadata.get("lng")
        to_lat = to_metadata.get("lat")
        to_lng = to_metadata.get("lng")
        
        if not all([from_lat, from_lng, to_lat, to_lng]):
            continue
        
        from_name = from_place.get("name", "출발지")
        to_name = to_place.get("name", "도착지")
        
        # 각 교통수단별 거리/시간 계산 (자동차 제외)
        candidates = []
        walking_info = None
        transit_info = None
        
        # 1. 도보 경로 확인
        try:
            if gmaps_client:
                walking_routes = gmaps_client.directions(
                    origin=(from_lat, from_lng),
                    destination=(to_lat, to_lng),
                    mode="walking",
                    language="ko"
                )
                if walking_routes:
                    leg = walking_routes[0]["legs"][0]
                    dist_km = leg["distance"]["value"] / 1000.0
                    dur_min = leg["duration"]["value"] / 60.0
                    walking_info = {
                        "api_mode": "walking",
                        "label": "도보",
                        "dist_km": dist_km,
                        "dur_min": dur_min,
                        "route_details": None
                    }
                    # 20분 이내면 추천 옵션으로 추가
                    if dur_min <= 20:
                        candidates.append(walking_info)
        except Exception:
            # API 실패 시 Haversine 거리로 추정
            dist_km = _local_haversine_km(
                {"lat": from_lat, "lng": from_lng},
                {"lat": to_lat, "lng": to_lng}
            )
            dur_min = dist_km / 4.0 * 60  # 4 km/h
            walking_info = {
                "api_mode": "walking",
                "label": "도보",
                "dist_km": dist_km,
                "dur_min": dur_min,
                "route_details": None
            }
            if dur_min <= 20:
                candidates.append(walking_info)
        
        # 2. 대중교통 경로 확인
        try:
            if gmaps_client:
                transit_routes = gmaps_client.directions(
                    origin=(from_lat, from_lng),
                    destination=(to_lat, to_lng),
                    mode="transit",
                    language="ko"
                )
                if transit_routes:
                    route = transit_routes[0]
                    leg = route["legs"][0]
                    dist_km = leg["distance"]["value"] / 1000.0
                    dur_min = leg["duration"]["value"] / 60.0
                    
                    # 대중교통 상세 정보 추출
                    transit_details = []
                    steps = leg.get("steps", [])
                    for step in steps:
                        travel_mode = step.get("travel_mode", "")
                        if travel_mode == "TRANSIT":
                            transit_step = step.get("transit_details", {})
                            line = transit_step.get("line", {})
                            vehicle = transit_step.get("vehicle", {})
                            vehicle_type = vehicle.get("type", "").upper()
                            
                            # 각 교통수단별 상세 정보
                            step_dist_km = step.get("distance", {}).get("value", 0) / 1000.0
                            step_dur_min = step.get("duration", {}).get("value", 0) / 60.0
                            
                            # 교통수단 종류 및 번호/이름
                            if vehicle_type == "BUS":
                                line_name = line.get("short_name") or line.get("name", "버스")
                                num_stops = transit_step.get("num_stops", 0)
                                transit_details.append({
                                    "type": "버스",
                                    "number": line_name,
                                    "departure_stop": transit_step.get("departure_stop", {}).get("name", ""),
                                    "arrival_stop": transit_step.get("arrival_stop", {}).get("name", ""),
                                    "num_stops": num_stops,
                                    "distance_km": round(step_dist_km, 2),
                                    "duration_min": round(step_dur_min, 2)
                                })
                            elif vehicle_type == "SUBWAY" or vehicle_type == "HEAVY_RAIL":
                                line_name = line.get("short_name") or line.get("name", "지하철")
                                num_stops = transit_step.get("num_stops", 0)
                                transit_details.append({
                                    "type": "지하철",
                                    "line": line_name,
                                    "departure_station": transit_step.get("departure_stop", {}).get("name", ""),
                                    "arrival_station": transit_step.get("arrival_stop", {}).get("name", ""),
                                    "num_stops": num_stops,
                                    "distance_km": round(step_dist_km, 2),
                                    "duration_min": round(step_dur_min, 2)
                                })
                            elif vehicle_type == "TRAIN" or vehicle_type == "RAIL":
                                line_name = line.get("short_name") or line.get("name", "기차")
                                num_stops = transit_step.get("num_stops", 0)
                                transit_details.append({
                                    "type": "기차",
                                    "line": line_name,
                                    "departure_station": transit_step.get("departure_stop", {}).get("name", ""),
                                    "arrival_station": transit_step.get("arrival_stop", {}).get("name", ""),
                                    "num_stops": num_stops,
                                    "distance_km": round(step_dist_km, 2),
                                    "duration_min": round(step_dur_min, 2)
                                })
                            else:
                                # 기타 대중교통
                                line_name = line.get("short_name") or line.get("name", "대중교통")
                                num_stops = transit_step.get("num_stops", 0)
                                transit_details.append({
                                    "type": "대중교통",
                                    "line": line_name,
                                    "departure_station": transit_step.get("departure_stop", {}).get("name", ""),
                                    "arrival_station": transit_step.get("arrival_stop", {}).get("name", ""),
                                    "num_stops": num_stops,
                                    "distance_km": round(step_dist_km, 2),
                                    "duration_min": round(step_dur_min, 2)
                                })
                        elif travel_mode == "WALKING":
                            # 환승을 위한 도보 구간
                            pass
                    
                    # Google Maps URL 생성
                    google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_lat},{from_lng}&destination={to_lat},{to_lng}&travelmode=transit"
                    
                    transit_info = {
                        "api_mode": "transit",
                        "label": "대중교통",
                        "dist_km": dist_km,
                        "dur_min": dur_min,
                        "route_details": transit_details,
                        "google_maps_url": google_maps_url
                    }
                    candidates.append(transit_info)
        except Exception:
            # API 실패 시 Haversine 거리로 추정
            dist_km = _local_haversine_km(
                {"lat": from_lat, "lng": from_lng},
                {"lat": to_lat, "lng": to_lng}
            )
            dur_min = dist_km / 25.0 * 60  # 25 km/h
            transit_info = {
                "api_mode": "transit",
                "label": "대중교통",
                "dist_km": dist_km,
                "dur_min": dur_min,
                "route_details": []
            }
            candidates.append(transit_info)
        
        if candidates:
            # 가장 빠른 교통수단 선택
            best = min(candidates, key=lambda x: x["dur_min"])
            
            # 추천 교통편 텍스트 생성 (단순하게: 버스, 도보, 지하철, 기차 중 하나)
            if best["api_mode"] == "walking":
                recommended_transport = "도보"
            elif best["api_mode"] == "transit" and best.get("route_details"):
                # 대중교통의 경우 route_details를 확인하여 버스/지하철/기차 구분
                details = best["route_details"]
                if details:
                    # 첫 번째 교통수단의 타입으로 결정
                    first_detail = details[0]
                    if first_detail["type"] == "버스":
                        recommended_transport = "버스"
                    elif first_detail["type"] == "지하철":
                        recommended_transport = "지하철"
                    elif first_detail["type"] == "기차":
                        recommended_transport = "기차"
                    else:
                        recommended_transport = "대중교통"
                else:
                    recommended_transport = "대중교통"
            else:
                recommended_transport = best["label"]
            
            route_entry = {
                "구간": f"{from_name} → {to_name}",
                "from": from_name,
                "to": to_name,
                "추천 교통편": recommended_transport,
                "transport": recommended_transport,
                "거리(km)": round(best["dist_km"], 2),
                "distance_km": round(best["dist_km"], 2),
                "예상 소요 시간(분)": round(best["dur_min"], 2),
                "duration_min": round(best["dur_min"], 2),
                "route_details": best.get("route_details", []),
                "google_maps_url": best.get("google_maps_url", "")
            }
            
            # 도보인 경우도 Google Maps URL 생성
            if best["api_mode"] == "walking":
                route_entry["google_maps_url"] = f"https://www.google.com/maps/dir/?api=1&origin={from_lat},{from_lng}&destination={to_lat},{to_lng}&travelmode=walking"
            
            # 도보가 20분 이내이고 대중교통보다 느리지 않으면 도보도 추천 옵션으로 추가
            if walking_info and walking_info["dur_min"] <= 20 and best["api_mode"] != "walking":
                # 도보가 대중교통보다 5분 이내 차이면 도보도 추천
                if walking_info["dur_min"] <= best["dur_min"] + 5:
                    route_entry["도보 추천"] = f"도보 {round(walking_info['dur_min'], 1)}분"
            
            route_info.append(route_entry)
    
    return {
        "optimal_order": optimal_order,
        "route_info": route_info
    }


def generate_travel_guide_multicrew(confirmed_plans: dict, destination: str, num_days: int):
    """
    CrewAI를 사용하여 여행 가이드북을 생성합니다.
    
    Args:
        confirmed_plans: {day: [places]} 형태의 확정된 일정
        destination: 여행지 이름
        num_days: 여행 일수
    
    Returns:
        생성된 가이드북 텍스트
    """
    if not Agent or not Task or not Crew:
        st.error("CrewAI 패키지가 설치되지 않았습니다. 'pip install crewai' 명령으로 설치해주세요.")
        return None
    
    # 일정 데이터를 문자열로 변환
    plans_text = ""
    for day in range(1, num_days + 1):
        day_plans = confirmed_plans.get(day, [])
        if day_plans:
            plans_text += f"\n[Day {day}]\n"
            for idx, plan in enumerate(day_plans, 1):
                plan_name = plan.get("name", "알 수 없는 장소")
                metadata = plan.get("metadata", {})
                address = metadata.get("address", "") or metadata.get("formatted_address", "")
                plans_text += f"{idx}. {plan_name}"
                if address:
                    plans_text += f" - {address}"
                plans_text += "\n"
    
    # -------------------------
    # 1) Agents 정의
    # -------------------------
    historian = Agent(
        role="Travel Historian",
        goal="각 여행지의 역사·문화적 의미와 배경을 깊이 있게 설명하는 여행 전문가.",
        backstory="20년 경력의 역사 여행 전문 기자. 현지 문화의 맥락과 숨겨진 이야기를 이끌어내는 전문가.",
        verbose=True
    )

    foodie = Agent(
        role="Culinary Expert",
        goal="각 일정 주변에서 가치 있는 맛집을 미식 가이드북 수준으로 분석하여 소개한다.",
        backstory="세계 각국 레스토랑을 리뷰한 미식 칼럼니스트.",
        verbose=True
    )

    navigator = Agent(
        role="Transit Navigator",
        goal="여행자가 실제로 따라갈 수 있는 실용적인 이동 설명을 제공한다.",
        backstory="지도 기반 여행동선 최적화를 전문으로 하는 교통 분석가.",
        verbose=True
    )

    compiler = Agent(
        role="Travel Guide Compiler",
        goal="여러 전문가가 제공한 정보를 종합하여 '장소 중심 가이드북'을 완성한다.",
        backstory="전문 여행 가이드북 편집자. 정보 재구성, 정리, 문서 구조화 전문가.",
        verbose=True
    )

    # -------------------------
    # 2) Task 정의
    # -------------------------

    historian_task = Task(
        description=f"""
        아래 일정에 포함된 **각 장소의 역사·스토리·배경 설명**을 작성하라.

        ● 목적지: {destination}
        ● 일정 데이터:
        {plans_text}

        각 장소에 대해 다음을 포함하여 작성하라:
        - 장소의 역사적 배경과 의미
        - 문화적 중요성
        - 숨겨진 이야기나 트리비아
        - 방문 시 주목할 포인트
        - 포토 스팟 추천
        """,
        expected_output="각 장소별 장문의 역사·스토리 중심 설명",
        agent=historian
    )

    foodie_task = Task(
        description=f"""
        아래 일정의 각 장소 주변에서 여행자가 방문할 만한 맛집을 1~2곳 추천하라.

        ● 목적지: {destination}
        ● 일정 데이터:
        {plans_text}

        각 장소 근처의 맛집에 대해 다음을 포함하여 작성하라:
        - 맛집 이름과 위치
        - 대표 메뉴와 특징
        - 가격대와 분위기
        - 방문 팁
        """,
        expected_output="각 장소 근처의 맛집 설명",
        agent=foodie
    )

    navigator_task = Task(
        description=f"""
        일정에 포함된 장소 간 이동 방법을 쉽고 간단하게 설명하라.

        ● 목적지: {destination}
        ● 일정 데이터:
        {plans_text}

        각 Day별로 다음을 포함하여 작성하라:
        - 장소 간 이동 방법 (대중교통, 도보 등)
        - 예상 소요 시간
        - 이동 팁과 주의사항
        """,
        expected_output="각 Day별 이동 요약 정보",
        agent=navigator
    )

    # 3명 결과물 종합하는 마지막 Compiler Task
    compiler_task = Task(
        description=f"""
        아래는 3명의 전문가(Historian, Foodie, Navigator)가 생성한 자료이다.

        너의 역할은:
        - 이 3개의 Task 결과물을 종합하여
        - **'장소 중심 여행 가이드북'**을 완성하는 것이다.

        반드시 다음 구조로 정리하라:

        [Day 1]
        - 장소 1: 깊이 있는 역사/스토리 설명 + Trivia + 포토 스팟
        - 주변 맛집 1~2곳 소개
        - 이동 요약(보조적)

        [Day 2]
        (반복)

        [전체 가이드북 스타일 요구사항]
        - 장소 설명이 문서의 중심이 되도록 구성
        - 동선은 짧고 간결하게 보조적으로 구성
        - 여행자가 읽기 쉽도록 문단, 제목, 소제목 활용
        - 여행자가 실제로 '아, 이 장소는 이런 의미가 있구나!' 하고 느끼도록 작성
        """,
        expected_output="3개 Task 결과물을 하나의 완전한 여행 가이드북으로 구조화한 최종 문서",
        agent=compiler,
        context=[historian_task, foodie_task, navigator_task]  # Task 결과 전달
    )

    # -------------------------
    # 3) Crew 실행
    # -------------------------
    crew = Crew(
        agents=[historian, foodie, navigator, compiler],
        tasks=[historian_task, foodie_task, navigator_task, compiler_task],
        verbose=True
    )

    result = crew.kickoff()

    # CrewAI 버전별 output 처리
    if hasattr(result, "output"):
        return result.output
    elif hasattr(result, "raw"):
        return result.raw
    elif hasattr(result, "final_output"):
        return result.final_output
    else:
        return str(result)


# ============================================
# 항공권/숙박 검색 관련 함수들 (gemini2_travel_v2.py 통합)
# ============================================

# 데이터 모델
if BaseModel:
    class FlightRequest(BaseModel):
        origin: str
        destination: str
        outbound_date: str
        return_date: str

    class HotelRequest(BaseModel):
        location: str
        check_in_date: str
        check_out_date: str

    class FlightInfo(BaseModel):
        airline: str
        price: str
        duration: str
        stops: str
        departure: str
        arrival: str
        travel_class: str
        return_date: str
        airline_logo: str

    class HotelInfo(BaseModel):
        name: str
        price: str
        rating: float
        location: str
        link: str
else:
    # Pydantic이 없는 경우 기본 클래스 사용
    class FlightRequest:
        def __init__(self, origin, destination, outbound_date, return_date):
            self.origin = origin
            self.destination = destination
            self.outbound_date = outbound_date
            self.return_date = return_date

    class HotelRequest:
        def __init__(self, location, check_in_date, check_out_date):
            self.location = location
            self.check_in_date = check_in_date
            self.check_out_date = check_out_date

    class FlightInfo:
        def __init__(self, airline, price, duration, stops, departure, arrival, travel_class, return_date, airline_logo=""):
            self.airline = airline
            self.price = price
            self.duration = duration
            self.stops = stops
            self.departure = departure
            self.arrival = arrival
            self.travel_class = travel_class
            self.return_date = return_date
            self.airline_logo = airline_logo
        
        def model_dump(self):
            return {
                "airline": self.airline,
                "price": self.price,
                "duration": self.duration,
                "stops": self.stops,
                "departure": self.departure,
                "arrival": self.arrival,
                "travel_class": self.travel_class,
                "return_date": self.return_date,
                "airline_logo": self.airline_logo
            }
    
    class HotelInfo:
        def __init__(self, name, price, rating, location, link):
            self.name = name
            self.price = price
            self.rating = rating
            self.location = location
            self.link = link
        
        def model_dump(self):
            return {
                "name": self.name,
                "price": self.price,
                "rating": self.rating,
                "location": self.location,
                "link": self.link
            }

# LLM 초기화 함수
@lru_cache(maxsize=1)
def initialize_flight_hotel_llm():
    """Initialize and cache the LLM instance for flight/hotel search."""
    if not LLM:
        return None
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    return LLM(
        model="gpt-4o",
        provider="openai",
        api_key=openai_key
    )

# 검색 함수들
async def run_search(params):
    """Generic function to run SerpAPI searches asynchronously."""
    if not GoogleSearch:
        raise Exception("SerpAPI가 설치되지 않았습니다. 'pip install google-search-results' 명령으로 설치해주세요.")
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
    except Exception as e:
        logger.exception(f"SerpAPI search error: {str(e)}")
        raise Exception(f"Search API error: {str(e)}")


async def search_flights_async(flight_request: FlightRequest, serp_api_key: str):
    """Fetch real-time flight details from Google Flights using SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} to {flight_request.destination}")

    params = {
        "api_key": serp_api_key,
        "engine": "google_flights",
        "hl": "ko",
        "gl": "kr",
        "departure_id": flight_request.origin.strip().upper(),
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "KRW"
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Flight search error: {search_results['error']}")
        return {"error": search_results["error"]}

    best_flights = search_results.get("best_flights", [])
    if not best_flights:
        logger.warning("No flights found in search results")
        return []

    formatted_flights = []
    # 요청한 출발지와 도착지 코드 (대소문자 무시)
    requested_origin = flight_request.origin.strip().upper()
    requested_destination = flight_request.destination.strip().upper()
    
    for flight in best_flights:
        if not flight.get("flights") or len(flight["flights"]) == 0:
            continue

        first_leg = flight["flights"][0]
        
        # 출발지와 도착지 확인 (경유편인 경우 마지막 구간 확인)
        dep_airport = first_leg.get('departure_airport', {})
        dep_id = dep_airport.get('id', '').strip().upper() if dep_airport.get('id') else ''
        
        # 마지막 구간의 도착지를 확인 (경유편의 경우)
        last_leg = flight["flights"][-1]
        arr_airport = last_leg.get('arrival_airport', {})
        arr_id = arr_airport.get('id', '').strip().upper() if arr_airport.get('id') else ''
        
        # 출발지와 도착지가 정확히 일치하는 항공편만 포함
        if dep_id != requested_origin or arr_id != requested_destination:
            logger.warning(f"항공편 필터링: 출발지 {dep_id} != {requested_origin} 또는 도착지 {arr_id} != {requested_destination}")
            continue
        
        airline = first_leg.get("airline") or "정보 없음"
        if airline == "Unknown Airline":
            airline = "정보 없음"
            
        flight_price = flight.get("price")
        price = str(flight_price) if flight_price and flight_price != "N/A" else "정보 없음"
        
        duration_min = flight.get('total_duration')
        duration = f"{duration_min}분" if duration_min and duration_min != "N/A" else "정보 없음"
        
        stops = "직항" if len(flight["flights"]) == 1 else f"{len(flight['flights']) - 1}경유"
        
        dep_name = dep_airport.get('name') or "정보 없음"
        dep_time = dep_airport.get('time') or "정보 없음"
        departure = f"{dep_name} ({dep_id}) {dep_time}"
        
        arr_name = arr_airport.get('name') or "정보 없음"
        arr_time = arr_airport.get('time') or "정보 없음"
        arrival = f"{arr_name} ({arr_id}) {arr_time}"
        
        travel_class = first_leg.get("travel_class") or "정보 없음"
        
        formatted_flights.append(FlightInfo(
            airline=airline,
            price=price,
            duration=duration,
            stops=stops,
            departure=departure,
            arrival=arrival,
            travel_class=travel_class,
            return_date=flight_request.return_date,
            airline_logo=first_leg.get("airline_logo", "")
        ))

    logger.info(f"Found {len(formatted_flights)} flights matching origin={requested_origin}, destination={requested_destination}")
    return formatted_flights


async def search_hotels_async(hotel_request: HotelRequest, serp_api_key: str):
    """Fetch hotel information from SerpAPI."""
    logger.info(f"Searching hotels for: {hotel_request.location}")

    params = {
        "api_key": serp_api_key,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "ko",
        "gl": "kr",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "KRW",
        "sort_by": 3,
        "rating": 8
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Hotel search error: {search_results['error']}")
        return {"error": search_results["error"]}

    hotel_properties = search_results.get("properties", [])
    if not hotel_properties:
        logger.warning("No hotels found in search results")
        return []

    formatted_hotels = []
    for hotel in hotel_properties:
        try:
            location = None
            
            if hotel.get("location"):
                loc_val = hotel.get("location")
                if isinstance(loc_val, str) and loc_val.strip():
                    location = loc_val.strip()
                elif isinstance(loc_val, dict):
                    location = loc_val.get("address") or loc_val.get("name") or loc_val.get("locality")
            
            if not location and hotel.get("address"):
                addr_val = hotel.get("address")
                if isinstance(addr_val, str) and addr_val.strip():
                    location = addr_val.strip()
            
            if not location and hotel.get("vicinity"):
                location = hotel.get("vicinity")
            
            if not location and hotel.get("locality"):
                location = hotel.get("locality")
            
            if not location and isinstance(hotel.get("gps_coordinates"), dict):
                gps_data = hotel.get("gps_coordinates", {})
                location = gps_data.get("address") or gps_data.get("name")
            
            if not location and isinstance(hotel.get("structured_location"), dict):
                loc_data = hotel.get("structured_location", {})
                location = loc_data.get("address") or loc_data.get("locality") or loc_data.get("region") or loc_data.get("name")
            
            if not location and hotel.get("region"):
                location = hotel.get("region")
            
            if not location or location == "N/A" or location == "" or (isinstance(location, str) and location.strip() == ""):
                location = "정보 없음"
            
            price_data = hotel.get("rate_per_night", {})
            if isinstance(price_data, dict):
                price = price_data.get("lowest") or price_data.get("extracted") or price_data.get("high") or None
                if not price:
                    price = "정보 없음"
            elif price_data:
                price = price_data
            else:
                price = "정보 없음"
            
            name = hotel.get("name") or "정보 없음"
            if name in ["Unknown Hotel", "N/A", ""]:
                name = "정보 없음"
            
            link = hotel.get("link") or hotel.get("booking_link") or hotel.get("website") or "정보 없음"
            if link in ["N/A", ""]:
                link = "정보 없음"
            
            formatted_hotels.append(HotelInfo(
                name=name,
                price=str(price) if price and price != "N/A" else "정보 없음",
                rating=hotel.get("overall_rating", 0.0),
                location=location,
                link=link
            ))
        except Exception as e:
            logger.warning(f"호텔 데이터 포맷팅 오류: {str(e)}")

    logger.info(f"Found {len(formatted_hotels)} hotels")
    return formatted_hotels


def format_travel_data(data_type, data, origin: Optional[str] = None, destination: Optional[str] = None):
    """Generic formatter for both flight and hotel data."""
    if not data:
        return f"No {data_type} available."

    if data_type == "flights":
        route_info = ""
        if origin and destination:
            route_info = f"\n**🚩 검색 경로: {origin} → {destination}**\n\n"
        
        formatted_text = f"✈️ **Available flight options**:{route_info}"
        for i, flight in enumerate(data):
            airline = flight.get('airline') if isinstance(flight, dict) else flight.airline
            price = flight.get('price') if isinstance(flight, dict) else flight.price
            duration = flight.get('duration') if isinstance(flight, dict) else flight.duration
            stops = flight.get('stops') if isinstance(flight, dict) else flight.stops
            departure = flight.get('departure') if isinstance(flight, dict) else flight.departure
            arrival = flight.get('arrival') if isinstance(flight, dict) else flight.arrival
            travel_class = flight.get('travel_class') if isinstance(flight, dict) else flight.travel_class
            
            formatted_text += (
                f"**Flight {i + 1}:**\n"
                f"✈️ **Airline:** {airline}\n"
                f"💰 **Price:** ₩{price}\n"
                f"⏱️ **Duration:** {duration}\n"
                f"🛑 **Stops:** {stops}\n"
                f"🕔 **Departure:** {departure}\n"
                f"🕖 **Arrival:** {arrival}\n"
                f"💺 **Class:** {travel_class}\n\n"
            )
    elif data_type == "hotels":
        formatted_text = "🏨 **Available Hotel Options**:\n\n"
        for i, hotel in enumerate(data):
            name = hotel.get('name') if isinstance(hotel, dict) else hotel.name
            price = hotel.get('price') if isinstance(hotel, dict) else hotel.price
            rating = hotel.get('rating') if isinstance(hotel, dict) else hotel.rating
            location = hotel.get('location') if isinstance(hotel, dict) else hotel.location
            link = hotel.get('link') if isinstance(hotel, dict) else hotel.link
            
            formatted_text += (
                f"**Hotel {i + 1}:**\n"
                f"🏨 **Name:** {name}\n"
                f"💰 **Price:** ₩{price}\n"
                f"⭐ **Rating:** {rating}\n"
                f"📍 **Location:** {location}\n"
                f"🔗 **More Info:** [Link]({link})\n\n"
            )
    else:
        return "Invalid data type."

    return formatted_text.strip()


async def get_ai_recommendation_async(data_type, formatted_data, origin: Optional[str] = None, destination: Optional[str] = None):
    """Unified function for getting AI recommendations for both flights and hotels."""
    if not Agent or not Task or not Crew or not Process or not LLM:
        return f"{data_type} AI 추천을 생성하려면 CrewAI가 설치되어 있어야 합니다."
    
    logger.info(f"Getting {data_type} analysis from AI")
    llm_model = initialize_flight_hotel_llm()
    
    if not llm_model:
        return "OpenAI API 키가 설정되지 않았습니다."

    if data_type == "flights":
        role = "AI 항공편 분석 전문가"
        goal = "가격, 소요 시간, 경유지, 전반적인 편의성을 고려하여 최적의 항공편을 추천합니다."
        backstory = "다양한 요소를 종합적으로 분석하여 항공편 옵션을 비교하는 AI 전문가입니다."
        
        # 출발지와 도착지 정보를 프롬프트에 명확히 포함
        route_info = ""
        if origin and destination:
            route_info = f"\n\n**⚠️ 중요: 반드시 {origin} 출발, {destination} 도착인 항공편만 추천해주세요. 다른 출발지나 도착지를 가진 항공편은 절대 추천하지 마세요.**\n"
        
        description = f"""
        아래 제공된 정보를 바탕으로 이용 가능한 항공편 중 최선의 선택을 추천해주세요.
        {route_info}
        **추천 이유:**
        - **💰 가격:** 이 항공편이 다른 항공편 대비 최고의 가성비를 제공하는 이유를 자세히 설명해주세요.
        - **⏱️ 소요 시간:** 이 항공편의 소요 시간이 다른 항공편 대비 최적인 이유를 설명해주세요.
        - **🛑 경유:** 이 항공편의 경유 횟수가 최소이거나 최적인 이유를 논의해주세요.
        - **💺 좌석 등급:** 이 항공편이 최고의 편안함과 편의 시설을 제공하는 이유를 설명해주세요.

        제공된 항공편 데이터를 바탕으로 추천을 해주세요. 각 속성에 대해 명확한 논리로 선택을 정당화해주세요. 응답에 항공편 세부 정보를 반복하지 마세요.
        {route_info}
        **중요: 모든 응답은 반드시 한국어로 작성해주세요.**
        """
    elif data_type == "hotels":
        role = "AI 호텔 분석 전문가"
        goal = "가격, 평점, 위치, 편의 시설을 고려하여 최적의 호텔을 추천합니다."
        backstory = "다양한 요소를 종합적으로 분석하여 호텔 옵션을 비교하는 AI 전문가입니다."
        description = """
        다음 분석을 바탕으로 최선의 호텔에 대한 상세한 추천을 생성해주세요. 가격, 평점, 위치, 편의 시설을 기반으로 명확한 추론을 포함해야 합니다.

        **🏆 AI 호텔 추천**
        다음 분석을 바탕으로 최선의 호텔을 추천합니다:

        **추천 이유**:
        - **💰 가격:** 추천 호텔은 다른 옵션 대비 가격 대비 최선의 선택으로, 제공되는 편의 시설과 서비스에 대해 최고의 가치를 제공합니다. 이를 자세히 설명해주세요.
        - **⭐ 평점:** 다른 대안들보다 높은 평점을 가지고 있어 더 나은 전반적인 게스트 경험을 보장합니다. 이것이 최선의 선택인 이유를 설명해주세요.
        - **📍 위치:** 호텔은 주요 명소에 가까운 최고의 위치에 있어 여행객에게 편리합니다. 위치의 장점을 설명해주세요.
        - **🛋️ 편의 시설:** 호텔은 Wi-Fi, 수영장, 체육관, 무료 조식 등의 편의 시설을 제공합니다. 이러한 편의 시설이 경험을 향상시키고 다양한 유형의 여행객에게 적합한 이유를 논의해주세요.

        📝 **추천 요구사항**:
        - 각 섹션에서 가격, 평점, 위치, 편의 시설의 요소를 바탕으로 이 호텔이 최선의 선택인 이유를 명확히 설명해주세요.
        - 다른 옵션과 비교하여 이 호텔이 두각을 나타내는 이유를 설명해주세요.
        - 추천이 여행객에게 명확하도록 간결하고 잘 구조화된 추론을 제공해주세요.
        - 추천은 단 하나의 요소가 아닌 여러 요소를 바탕으로 정보에 입각한 결정을 내릴 수 있도록 도와야 합니다.

        **중요: 모든 응답은 반드시 한국어로 작성해주세요.**
        """
    else:
        raise ValueError("Invalid data type for AI recommendation")

    analyze_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_model,
        verbose=False
    )

    analyze_task = Task(
        description=f"{description}\n\n분석할 데이터:\n{formatted_data}\n\n**모든 응답은 반드시 한국어로 작성해주세요.**",
        agent=analyze_agent,
        expected_output=f"제공된 세부 사항에 대한 분석을 바탕으로 최선의 {data_type} 선택을 설명하는 구조화된 추천입니다. 모든 내용은 한국어로 작성되어야 합니다."
    )

    analyst_crew = Crew(
        agents=[analyze_agent],
        tasks=[analyze_task],
        process=Process.sequential,
        verbose=False
    )

    try:
        crew_results = await asyncio.to_thread(analyst_crew.kickoff)

        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            if data_type == "flights":
                return crew_results.get(role, "항공편 추천을 생성할 수 없습니다.")
            else:
                return crew_results.get(role, "호텔 추천을 생성할 수 없습니다.")
        else:
            return str(crew_results)
    except Exception as e:
        logger.exception(f"Error in AI {data_type} analysis: {str(e)}")
        if data_type == "flights":
            return f"항공편 추천 생성 중 오류가 발생했습니다: {str(e)}"
        else:
            return f"호텔 추천 생성 중 오류가 발생했습니다: {str(e)}"


async def generate_itinerary_async(destination, flights_text, hotels_text, check_in_date, check_out_date):
    """Generate a detailed travel itinerary based on flight and hotel information."""
    if not Agent or not Task or not Crew or not Process or not LLM:
        return "여행 일정을 생성하려면 CrewAI가 설치되어 있어야 합니다."
    
    try:
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        days = (check_out - check_in).days

        llm_model = initialize_flight_hotel_llm()
        if not llm_model:
            return "OpenAI API 키가 설정되지 않았습니다."

        analyze_agent = Agent(
            role="AI 여행 계획 전문가",
            goal="항공편 및 호텔 정보를 바탕으로 사용자를 위한 상세한 여행 일정을 작성합니다.",
            backstory="항공편 세부 정보, 호텔 숙박, 목적지의 필수 방문 장소를 포함한 일별 여행 일정을 생성하는 AI 여행 전문가입니다.",
            llm=llm_model,
            verbose=False
        )

        analyze_task = Task(
            description=f"""
            다음 세부 정보를 바탕으로 {days}일간의 여행 일정을 작성해주세요:

            **항공편 정보**:
            {flights_text}

            **호텔 정보**:
            {hotels_text}

            **여행지**: {destination}

            **여행 날짜**: {check_in_date}부터 {check_out_date}까지 ({days}일)

            일정에는 다음이 포함되어야 합니다:
            - 항공편 도착 및 출발 정보
            - 호텔 체크인 및 체크아웃 세부 정보
            - 일별 활동 내역
            - 필수 방문 명소 및 예상 방문 시간
            - 식사 추천 레스토랑
            - 현지 교통 수단 팁

            📝 **형식 요구사항**:
            - 명확한 제목이 있는 마크다운 형식 사용 (# 메인 제목, ## 날짜, ### 섹션)
            - 다양한 활동 유형에 이모지 포함 (🏛️ 랜드마크, 🍽️ 레스토랑 등)
            - 활동 나열 시 불릿 포인트 사용
            - 각 활동에 예상 시간 포함
            - 시각적으로 매력적이고 읽기 쉬운 형식으로 일정 작성
            - 모든 내용은 반드시 한국어로 작성

            **중요: 모든 일정과 설명은 반드시 한국어로 작성해주세요.**
            """,
            agent=analyze_agent,
            expected_output="항공편, 호텔, 일별 내역이 포함된 이모지, 제목, 불릿 포인트가 있는 마크다운 형식의 잘 구조화되고 시각적으로 매력적인 여행 일정입니다. 모든 내용은 한국어로 작성되어야 합니다."
        )

        itinerary_planner_crew = Crew(
            agents=[analyze_agent],
            tasks=[analyze_task],
            process=Process.sequential,
            verbose=False
        )

        crew_results = await asyncio.to_thread(itinerary_planner_crew.kickoff)

        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get("AI 여행 계획 전문가", "여행 일정을 생성할 수 없습니다.")
        else:
            return str(crew_results)

    except Exception as e:
        logger.exception(f"Error generating itinerary: {str(e)}")
        return f"여행 일정 생성 중 오류가 발생했습니다. 나중에 다시 시도해주세요. 오류 내용: {str(e)}"


# Synchronous wrapper functions
def search_flights_sync(flight_data: Dict[str, str], serp_api_key: str) -> List[Dict[str, Any]]:
    """Synchronous wrapper for flight search."""
    flight_request = FlightRequest(**flight_data)
    flights = asyncio.run(search_flights_async(flight_request, serp_api_key))
    
    if isinstance(flights, dict) and "error" in flights:
        raise Exception(flights["error"])
    
    return [flight.model_dump() if hasattr(flight, 'model_dump') else flight.__dict__ if hasattr(flight, '__dict__') else flight for flight in flights]


def search_hotels_sync(hotel_data: Dict[str, str], serp_api_key: str) -> List[Dict[str, Any]]:
    """Synchronous wrapper for hotel search."""
    hotel_request = HotelRequest(**hotel_data)
    hotels = asyncio.run(search_hotels_async(hotel_request, serp_api_key))
    
    if isinstance(hotels, dict) and "error" in hotels:
        raise Exception(hotels["error"])
    
    return [hotel.model_dump() if hasattr(hotel, 'model_dump') else hotel.__dict__ if hasattr(hotel, '__dict__') else hotel for hotel in hotels]


def get_ai_recommendation_sync(data_type: str, data: List, origin: Optional[str] = None, destination: Optional[str] = None) -> str:
    """Synchronous wrapper for AI recommendation."""
    formatted_data = format_travel_data(data_type, data, origin, destination)
    return asyncio.run(get_ai_recommendation_async(data_type, formatted_data, origin, destination))


def generate_itinerary_sync(destination: str, flights: List[Dict], hotels: List[Dict], 
                           check_in_date: str, check_out_date: str) -> str:
    """Synchronous wrapper for itinerary generation."""
    flights_text = format_travel_data("flights", flights)
    hotels_text = format_travel_data("hotels", hotels)
    return asyncio.run(generate_itinerary_async(destination, flights_text, hotels_text, check_in_date, check_out_date))


def _render_flight_hotel_search_ui():
    """AI 항공/숙박 모드의 새로운 메인 화면을 렌더링합니다."""
    # API 키 확인
    serp_api_key = os.getenv("SERP_API_KEY") or os.getenv("SERPER_API_KEY")
    
    if not serp_api_key:
        st.error("⚠️ SERP_API_KEY 또는 SERPER_API_KEY가 설정되지 않았습니다.")
        st.info("💡 .env 파일에 SERP_API_KEY를 추가해주세요.")
        return
    
    if not GoogleSearch:
        st.error("⚠️ SerpAPI 패키지가 설치되지 않았습니다.")
        st.info("💡 'pip install google-search-results' 명령으로 설치해주세요.")
        return
    
    # 사이드바: 검색 모드 선택
    with st.sidebar:
        st.markdown("#### 🔍 검색 모드")
        search_mode = st.radio(
            "",
            ["전체 검색 (항공편 + 호텔 + 일정)", "항공편만", "호텔만"],
            label_visibility="collapsed"
        )
        st.markdown("---")
    
    # 메인 검색 폼
    st.markdown("### ✈️ 여행 검색")
    st.markdown("AI를 활용하여 항공편과 호텔을 찾고 맞춤형 추천을 받아보세요!")
    st.markdown("")
    
    with st.form(key="travel_search_form"):
        cols = st.columns([1, 1])

        with cols[0]:
            st.subheader("🛫 항공편 정보")
            origin = st.text_input("출발 공항 (IATA 코드)", "ICN", help="예: ICN (인천), GMP (김포), PUS (부산), JFK (뉴욕)")
            destination = st.text_input("도착 공항 (IATA 코드)", "NRT", help="예: NRT (나리타), LAX (로스앤젤레스), BKK (방콕)")

            tomorrow = datetime.now() + timedelta(days=1)
            next_week = tomorrow + timedelta(days=7)

            outbound_date = st.date_input("출발 날짜", tomorrow)
            return_date = st.date_input("귀국 날짜", next_week)

        with cols[1]:
            st.subheader("🏨 호텔 정보")
            use_flight_destination = st.checkbox("항공편 도착지와 같은 지역 호텔 검색", value=True)

            if use_flight_destination:
                location = destination
                st.info(f"항공편 도착지 ({destination})와 같은 지역의 호텔을 검색합니다")
            else:
                location = st.text_input("호텔 위치", "", help="도시명 또는 공항 코드 입력")

            check_in_date = st.date_input("체크인 날짜", outbound_date)
            check_out_date = st.date_input("체크아웃 날짜", return_date)

        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col2:
            submit_button = st.form_submit_button("🔍 검색", use_container_width=True)

    # 폼 제출 처리
    if submit_button:
        if not origin or not destination:
            st.error("출발 공항과 도착 공항을 모두 입력해주세요.")
        elif outbound_date >= return_date:
            st.error("귀국 날짜는 출발 날짜보다 늦어야 합니다.")
        elif check_in_date >= check_out_date:
            st.error("체크아웃 날짜는 체크인 날짜보다 늦어야 합니다.")
        else:
            flight_data = {
                "origin": origin,
                "destination": destination,
                "outbound_date": str(outbound_date),
                "return_date": str(return_date)
            }

            hotel_data = {
                "location": location,
                "check_in_date": str(check_in_date),
                "check_out_date": str(check_out_date)
            }

            with st.spinner("최적의 여행 옵션을 검색하고 있습니다..."):
                try:
                    flights = []
                    hotels = []
                    ai_flight_recommendation = ""
                    ai_hotel_recommendation = ""
                    itinerary = ""

                    if search_mode == "전체 검색 (항공편 + 호텔 + 일정)":
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            flight_future = executor.submit(search_flights_sync, flight_data, serp_api_key)
                            hotel_future = executor.submit(search_hotels_sync, hotel_data, serp_api_key)
                            
                            flights = flight_future.result()
                            hotels = hotel_future.result()
                        
                        if flights:
                            ai_flight_recommendation = get_ai_recommendation_sync("flights", flights, origin=flight_data.get("origin"), destination=flight_data.get("destination"))
                        if hotels:
                            ai_hotel_recommendation = get_ai_recommendation_sync("hotels", hotels)
                        
                        if flights and hotels:
                            itinerary = generate_itinerary_sync(
                                destination=destination,
                                flights=flights,
                                hotels=hotels,
                                check_in_date=str(check_in_date),
                                check_out_date=str(check_out_date)
                            )

                    elif search_mode == "항공편만":
                        flights = search_flights_sync(flight_data, serp_api_key)
                        if flights:
                            ai_flight_recommendation = get_ai_recommendation_sync("flights", flights, origin=flight_data.get("origin"), destination=flight_data.get("destination"))

                    elif search_mode == "호텔만":
                        hotels = search_hotels_sync(hotel_data, serp_api_key)
                        if hotels:
                            ai_hotel_recommendation = get_ai_recommendation_sync("hotels", hotels)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    import traceback
                    with st.expander("상세 오류 정보"):
                        st.code(traceback.format_exc(), language="python")
                    st.stop()

            # 결과 표시
            if search_mode == "항공편만":
                tabs = st.tabs(["✈️ 항공편", "🏆 AI 추천"])
            elif search_mode == "호텔만":
                tabs = st.tabs(["🏨 호텔", "🏆 AI 추천"])
            else:
                tabs = st.tabs(["✈️ 항공편", "🏨 호텔", "🏆 AI 추천", "📅 여행 일정"])

            # 항공편 탭
            if search_mode != "호텔만":
                with tabs[0]:
                    st.subheader(f"✈️ {origin} → {destination} 항공편 검색 결과")

                    if flights:
                        flight_cols = st.columns(2)

                        for i, flight in enumerate(flights):
                            col_idx = i % 2
                            with flight_cols[col_idx]:
                                with st.container(border=True):
                                    stops_text = flight['stops'] if flight['stops'] != "Nonstop" else "직항"
                                    st.markdown(f"""
                                    ### ✈️ {flight['airline']} - {stops_text}

                                    🕒 **출발**: {flight['departure']}  
                                    🕘 **도착**: {flight['arrival']}  
                                    ⏱️ **소요 시간**: {flight['duration']}  
                                    💰 **가격**: **₩{flight['price']}**  
                                    💺 **좌석 등급**: {flight['travel_class']}
                                    """)
                                    st.button(f"🔖 이 항공편 선택", key=f"flight_{i}")
                    else:
                        st.info("검색 조건에 맞는 항공편을 찾을 수 없습니다.")

            # 호텔 탭
            if search_mode != "항공편만":
                with tabs[1 if search_mode == "호텔만" else 1]:
                    st.subheader(f"🏨 {location} 지역 호텔 검색 결과")

                    if hotels:
                        hotel_cols = st.columns(3)

                        for i, hotel in enumerate(hotels):
                            col_idx = i % 3
                            with hotel_cols[col_idx]:
                                with st.container(border=True):
                                    st.markdown(f"""
                                    ### 🏨 {hotel['name']}

                                    💰 **가격**: ₩{hotel['price']} / 1박  
                                    ⭐ **평점**: {hotel['rating']}  
                                    📍 **위치**: {hotel['location']}
                                    """)
                                    cols = st.columns([1, 1])
                                    with cols[0]:
                                        st.button(f"🔖 선택", key=f"hotel_{i}")
                                    with cols[1]:
                                        if hotel['link'] and hotel['link'] != "정보 없음":
                                            st.link_button("🔗 상세 정보", hotel['link'])
                    else:
                        st.info("검색 조건에 맞는 호텔을 찾을 수 없습니다.")

            # AI 추천 탭
            recommendation_tab_index = 1 if search_mode in ["항공편만", "호텔만"] else 2
            with tabs[recommendation_tab_index]:
                if search_mode != "호텔만" and ai_flight_recommendation:
                    st.subheader("✈️ AI 항공편 추천")
                    with st.container(border=True):
                        st.markdown(ai_flight_recommendation)

                if search_mode != "항공편만" and ai_hotel_recommendation:
                    st.subheader("🏨 AI 호텔 추천")
                    with st.container(border=True):
                        st.markdown(ai_hotel_recommendation)

            # 일정 탭
            if search_mode == "전체 검색 (항공편 + 호텔 + 일정)" and itinerary:
                with tabs[3]:
                    st.subheader("📅 여행 일정")
                    with st.container(border=True):
                        st.markdown(itinerary)

                    st.download_button(
                        label="📥 일정 다운로드",
                        data=itinerary,
                        file_name=f"여행일정_{destination}_{outbound_date}.md",
                        mime="text/markdown"
                    )


def _render_customizing_main_screen():
    """커스터마이징 모드의 기존 메인 화면을 렌더링합니다."""
    # 메인 영역: 지도 표시 (확정 일정)
    st.markdown("## 🗺️ 여행 일정 지도")
    st.markdown("")


def save_to_word(content: str, filename: str = "travel_guide.docx") -> str:
    """
    생성된 가이드북 텍스트를 Word 파일로 저장합니다.
    
    Args:
        content: 가이드북 텍스트
        filename: 저장할 파일명
    
    Returns:
        저장된 파일 경로
    """
    if not Document:
        st.error("python-docx 패키지가 설치되지 않았습니다. 'pip install python-docx' 명령으로 설치해주세요.")
        return None
    
    doc = Document()
    
    for line in content.split("\n"):
        doc.add_paragraph(line)
    
    filepath = filename
    doc.save(filepath)
    return filepath


def export_plans_to_notion(
    confirmed_plans: dict,
    destination: str,
    num_days: int,
    notion_api_key: Optional[str],
    notion_database_id: Optional[str],
    openai_client: Optional[OpenAI]
):
    """
    확정된 일정을 Notion 데이터베이스로 내보냅니다.
    
    Args:
        confirmed_plans: {day_num: [plan1, plan2, ...]} 형태의 딕셔너리
        destination: 여행지 이름
        num_days: 여행 일수
        notion_api_key: Notion API 키
        notion_database_id: Notion 데이터베이스 ID
        openai_client: OpenAI 클라이언트 (경로 요약 생성용)
    """
    if not (notion_api_key and notion_database_id):
        st.error("Notion API 설정이 없습니다. .env 파일의 NOTION_API_KEY / NOTION_DATABASE_ID를 확인해주세요.")
        return
    
    # 확정된 일정이 있는지 확인
    total_plans = sum(len(plans) for plans in confirmed_plans.values())
    if total_plans == 0:
        st.warning("내보낼 일정이 없습니다.")
        return
    
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    
    def make_route_summary_korean(day_label: str, dest_name: str, place_sequence: List[dict]) -> str:
        """
        자연어 추천 경로 요약 생성
        place_sequence: [{"order": 1, "name": "...", "address": "..."}, ...]
        """
        if not place_sequence:
            return ""
        
        names = [p.get("name", "") for p in place_sequence if p.get("name")]
        if not names:
            return ""
        
        # OpenAI 클라이언트가 없으면 간단 템플릿으로 생성
        if not openai_client:
            if len(names) == 1:
                return f"{day_label}에는 {dest_name}의 {names[0]}를 여유롭게 즐겨보세요."
            elif len(names) == 2:
                return f"{day_label}에는 먼저 {dest_name}의 {names[0]}를 방문한 뒤, {names[1]}로 이동하며 하루를 보내보세요."
            else:
                first = names[0]
                middle = " → ".join(names[1:-1])
                last = names[-1]
                return (
                    f"{day_label}에는 먼저 {dest_name}의 {first}를 방문한 후, "
                    f"{middle}를 거쳐 마지막으로 {last}까지 둘러보며 여행을 만끽해보세요."
                )
        
        # OpenAI 사용해 자연어 요약 생성
        try:
            user_prompt = f"""
너는 한국인 여행 플래너야.

아래는 '{dest_name}' 여행 {day_label} 일정에서 방문하는 장소 목록이야.
각 장소는 방문 순서와 주소가 포함되어 있어.

장소 목록:
{json.dumps(place_sequence, ensure_ascii=False, indent=2)}

이 정보를 바탕으로 자연스러운 한국어 한 단락으로 일정을 요약해줘.

요구사항:
- 예시 느낌: "1일차에는 먼저 도쿄의 Aoyama Flower Market Green House를 자동차로 방문한 후, 한국 인천의 Cafe Comma & Yann Couvreur로 이동합니다. 그 다음에는 대중교통을 이용해 서울의 Aqua garden cafe Lotte World Tower로 가고, 마지막으로 다시 대중교통을 타고 Marie n Zoo로 이동합니다. 하루 동안 다양한 장소를 즐기며 여행을 만끽해보세요!"
- 2~4문장 정도
- 존댓말 사용
- 마크다운 기호 없이 순수 문장만 출력
- 문장 앞에 날짜(예: "1일차에는")를 자연스럽게 포함해도 좋음
"""
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 한국어로 여행 일정을 매끄럽게 요약해 주는 여행 플래너입니다."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception:
            # 실패 시 템플릿으로 fallback
            if len(names) == 1:
                return f"{day_label}에는 {dest_name}의 {names[0]}를 여유롭게 즐겨보세요."
            else:
                seq = " → ".join(names)
                return f"{day_label}에는 {dest_name} 일대에서 {seq} 순서로 이동하며 여행을 즐겨보세요."
    
    try:
        exported_days = 0
        for day in range(1, num_days + 1):
            day_plans = confirmed_plans.get(day, [])
            if not day_plans:
                continue
            
            day_label = f"Day {day}"
            
            # 장소 리스트 문자열 / sequence 만들기
            place_lines = []
            place_sequence = []
            start_name = ""
            end_name = ""
            
            for idx, plan in enumerate(day_plans):
                plan_name = plan.get("name", "알 수 없는 장소")
                metadata = plan.get("metadata", {})
                address = metadata.get("address", "") or metadata.get("formatted_address", "")
                
                if idx == 0:
                    start_name = plan_name
                end_name = plan_name
                
                line = f"{idx+1}. {plan_name}"
                if address:
                    line += f" - {address}"
                place_lines.append(line)
                place_sequence.append({
                    "order": idx + 1,
                    "name": plan_name,
                    "address": address
                })
            
            place_text = "\n".join(place_lines)
            route_text = f"{start_name} → {end_name}" if start_name and end_name else ""
            
            # 자연어 추천 경로 요약 생성
            route_summary = make_route_summary_korean(day_label, destination, place_sequence)
            
            # Notion 속성 매핑
            title_text = f"{destination} {day_label}".strip()
            
            properties = {
                "Name": {
                    "title": [
                        {"text": {"content": title_text or "여행 일정"}}
                    ]
                },
                "날짜": {
                    "rich_text": [
                        {"text": {"content": day_label}}
                    ]
                },
                "도시": {
                    "rich_text": [
                        {"text": {"content": destination}}
                    ]
                },
                "장소 리스트": {
                    "rich_text": [
                        {"text": {"content": place_text}}
                    ]
                },
                "출발지/도착지": {
                    "rich_text": [
                        {"text": {"content": route_text}}
                    ]
                },
                "추천 경로": {
                    "rich_text": [
                        {"text": {"content": route_summary}}
                    ]
                },
            }
            
            payload = {
                "parent": {"database_id": notion_database_id},
                "properties": properties,
            }
            
            resp = requests.post(url, headers=headers, json=payload)
            
            if resp.status_code not in (200, 201):
                st.error(f"Notion 오류 (Day {day}): {resp.status_code} - {resp.text}")
                continue
            
            exported_days += 1
        
        if exported_days > 0:
            st.success(f"✅ {exported_days}일차 일정이 Notion DB에 저장되었습니다!")
        else:
            st.warning("내보낼 일정이 없습니다.")
    
    except Exception as e:
        st.error(f"Notion Export 오류: {e}")


def get_destination_info_from_gpt(destination: str, openai_client) -> Optional[dict]:
    """
    OpenAI GPT-4o를 사용하여 여행지 기본 정보를 가져옵니다.
    
    Args:
        destination: 여행지 이름 (예: "런던", "도쿄")
        openai_client: OpenAI 클라이언트
    
    Returns:
        구조화된 여행지 정보 딕셔너리 또는 None
    """
    if not openai_client or not destination:
        return None
    
    # 프롬프트 생성 (가독성 좋은 형식으로 요청)
    prompt = f"""{destination}의 여행 정보를 다음 형식으로 제공해주세요:

**기본정보:**
- 시차: (명확한 설명)
- 통화: (통화명과 환율 정보)
- 언어: (주요 언어)
- 기후: (계절별 날씨 특징)
- 교통: (주요 교통수단과 이용 방법)
- 전압: (전압, 플러그 타입)

**역사:**
(역사에 대한 간결하고 읽기 좋은 설명, 2-3문단)

**정치/경제/문화:**
- 정치: (정치 체제와 특징)
- 경제: (주요 산업과 경제 특징)
- 문화: (문화적 특징과 전통)

**명소 (10개):**
각 명소마다 다음 형식으로:
1. [명소 이름]
   - 설명: (상세 설명)
   - 추천 이유: (왜 가볼 만한지)

2. [명소 이름]
   ...

**음식 (10개):**
각 음식마다 다음 형식으로:
1. [음식 이름] - (간단한 설명)
2. [음식 이름] - (간단한 설명)
...

**여행 팁:**
- 팁 1: (구체적인 팁)
- 팁 2: (구체적인 팁)
- 팁 3: (구체적인 팁)
...

다음 JSON 형식으로 응답해주세요 (마크다운 형식의 텍스트 포함):
{{
    "기본정보": "마크다운 형식의 텍스트 (시차, 통화, 언어, 기후, 교통, 전압 등을 명확하게 구분하여 작성)",
    "역사": "마크다운 형식의 텍스트 (읽기 좋게 문단으로 구분)",
    "정치경제문화": "마크다운 형식의 텍스트 (정치, 경제, 문화를 명확하게 구분하여 작성)",
    "명소": [
        {{"이름": "명소 이름", "설명": "상세 설명", "추천이유": "추천 이유"}},
        ...
    ],
    "음식": [
        {{"이름": "음식 이름", "설명": "간단한 설명"}},
        ...
    ],
    "여행팁": "마크다운 형식의 텍스트 (각 팁을 명확하게 구분하여 작성)"
}}"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 여행 정보 전문가입니다. 요청한 정보를 정확하고 상세하게 JSON 형식으로 제공해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        import json
        info_dict = json.loads(content)
        return info_dict
        
    except Exception as e:
        st.error(f"여행지 정보를 가져오는 중 오류가 발생했습니다: {str(e)}")
        return None


def generate_recommendation_message(
    openai_client,
    user_query: str,
    recommendations: List[dict],
    day_num: Optional[int],
    destination: str
) -> str:
    """
    추천 장소의 이름만 나열하는 메시지를 생성합니다.
    """
    if not recommendations:
        if day_num:
            return f"죄송합니다. Day {day_num}에 맞는 장소를 찾지 못했습니다. 다른 키워드로 검색해보세요."
        else:
            return f"죄송합니다. {destination}에 맞는 장소를 찾지 못했습니다. 다른 키워드로 검색해보세요."
    
    # 카테고리별로 장소 그룹화
    places_by_category = {}
    for rec in recommendations:
        metadata = rec.get("metadata", {})
        category = rec.get("category") or metadata.get("custom_category", "기타")
        if category not in places_by_category:
            places_by_category[category] = []
        places_by_category[category].append(rec)
    
    # 카테고리별로 장소 이름만 나열
    category_summaries = []
    for category, places in places_by_category.items():
        place_names = []
        for rec in places:
            metadata = rec.get("metadata", {})
            place_name = rec.get("name") or metadata.get("name", "알 수 없는 장소")
            place_names.append(place_name)
        
        category_info = f"**{category}**\n"
        for name in place_names:
            category_info += f"- {name}\n"
        category_summaries.append(category_info)
    
    # 최종 메시지: 장소 이름만 나열
    places_summary = "\n".join(category_summaries)
    if day_num:
        return f"Day {day_num} 추천 장소:\n\n{places_summary}"
    else:
        return f"{destination} 추천 장소:\n\n{places_summary}"


def create_vector_database(
    region_name: str,
    city_name: str,
    gmaps_client,
    openai_client,
    apify_token: str,
    progress_callback,
    status_callback,
    center_coordinates: Optional[List[float]] = None,
    db_names: Optional[VectorDBNames] = None,
    num_days: int = 1,
    naver_client_id: Optional[str] = None,
    naver_client_secret: Optional[str] = None,
    serpapi_key: Optional[str] = None,
):
    try:
        ensure_vector_db_dir()
    except Exception as e:
        st.error(f"[오류] 벡터DB 디렉토리 생성 실패: {e}")
        raise
    
    target_city = city_name or region_name
    try:
        if db_names is None:
            db_names = build_vector_db_names(target_city, gmaps_client)
    except Exception as e:
        st.error(f"[오류] 벡터DB 이름 생성 실패: {e}")
        raise
    
    try:
        status_callback("카테고리별 인기 장소를 탐색 중입니다...")
        progress_callback(0.05)
    except Exception as e:
        st.error(f"[오류] 상태 메시지 업데이트 실패: {e}")
        raise
    
    categories_config = [
        ("관광지", "tourist_attraction", 100),
        ("서점/라이브러리", "book_store", 10),
        ("사원/성당/종교명소", "place_of_worship", 10),
        ("테마파크/액티비티", "amusement_park", 5),
        ("스파/온천", "spa", 10),
        ("맛집", "restaurant", 100),
        ("카페", "cafe", 20),
        ("베이커리/디저트", "bakery", 20),
        ("쇼핑", "shopping_mall", 10),
        ("바/술집", "bar", 20),
        ("박물관/미술관", "museum", 10),
        ("공원", "park", 5),
    ]
    
    place_candidates = []
    seen_place_ids = set()
    total_target_places = sum(limit for _, _, limit in categories_config)  # 전체 목표 장소 수
    
    for label, place_type, limit in categories_config:
        try:
            status_callback(f"'{label}' 데이터 수집 중... (상위 {limit}개)")
        except Exception as e:
            st.warning(f"[경고] 상태 메시지 업데이트 실패 ({label}): {e}")
        
        try:
            # 각 카테고리 검색에 최대 20초 타임아웃 설정
            import threading
            import queue
            import time
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            start_time = time.time()
            
            def search_worker():
                try:
                    # 스레드 내부에서는 use_streamlit=False로 설정하여 Streamlit 함수 호출 방지
                    places = fetch_places_by_category_and_sort(
                        city_name=target_city,
                        gmaps_client=gmaps_client,
                        label=label,
                        place_type=place_type,
                        limit_per_category=limit,
                        center_coordinates=center_coordinates,
                        max_distance_km=50.0,  # 중심 좌표로부터 50km 이내의 장소만 포함
                        use_streamlit=False,  # 스레드 내부에서는 Streamlit 함수 호출하지 않음
                    )
                    result_queue.put(places)
                except Exception as e:
                    exception_queue.put(e)
            
            # 검색을 별도 스레드에서 실행
            search_thread = threading.Thread(target=search_worker, daemon=True)
            search_thread.start()
            search_thread.join(timeout=20)  # 20초 타임아웃
            
            elapsed_time = time.time() - start_time
            
            if search_thread.is_alive():
                # 타임아웃 발생 - 조용히 빈 리스트 반환하고 계속 진행
                category_places = []
                # 에러 메시지는 저장하되 경고만 표시
                error_msg = f"[경고] '{label}' 카테고리 검색 타임아웃 ({elapsed_time:.1f}초 초과) - 계속 진행"
                try:
                    st.session_state.vector_db_error = error_msg
                    st.session_state.vector_db_current_status = error_msg
                except Exception:
                    pass
            elif not exception_queue.empty():
                # 예외 발생 - 조용히 빈 리스트 반환하고 계속 진행
                exc = exception_queue.get()
                category_places = []
                error_msg = f"[경고] '{label}' 카테고리 검색 실패: {type(exc).__name__} - 계속 진행"
                try:
                    st.session_state.vector_db_error = error_msg
                    st.session_state.vector_db_current_status = error_msg
                except Exception:
                    pass
            elif not result_queue.empty():
                # 성공
                category_places = result_queue.get()
                # 목표 개수보다 적게 수집된 경우, 다음 카테고리로 넘어가기 위해 계속 진행
                if len(category_places) < limit:
                    # 부족한 경우에도 계속 진행 (다음 카테고리에서 보충)
                    pass
            else:
                # 결과가 없음
                category_places = []
        except Exception as e:
            # 카테고리 검색 실패해도 계속 진행 (조용히 처리)
            category_places = []
            try:
                error_msg = f"[경고] '{label}' 카테고리 검색 중 예외 발생 - 계속 진행"
                st.session_state.vector_db_error = error_msg
                st.session_state.vector_db_current_status = error_msg
            except Exception:
                pass
        
        for place in category_places:
            try:
                pid = place.get("place_id")
                if pid and pid not in seen_place_ids:
                    place["custom_category_label"] = label
                    place["custom_category_type"] = place_type
                    place_candidates.append(place)
                    seen_place_ids.add(pid)
            except Exception as e:
                # 장소 후보 추가 실패는 조용히 스킵하고 계속 진행
                continue
    
    st.write(f"✅ [디버그] 총 {len(place_candidates)}개 장소 후보 수집 완료")
    
    if not place_candidates:
        error_msg = "[오류] 장소 후보를 찾지 못했습니다. 도시 이름을 확인해주세요."
        st.error(error_msg)
        st.session_state.vector_db_error = error_msg
        raise ValueError("장소 후보를 찾지 못했습니다. 도시 이름을 확인해주세요.")

    st.toast(f"총 {len(place_candidates)}개의 장소 데이터를 확보했습니다!", icon="✅")

    enriched_places = []
    status_callback("장소별 상세 정보(리뷰, 혼잡도, 설명) 수집 중...")
    
    total_candidates = len(place_candidates)
    
    # Apify 병렬 처리를 위한 준비
    apify_tasks = []  # (idx, candidate, target_city, category_label) 튜플 리스트
    
    # 먼저 Google Details를 수집하고 Apify 작업 준비
    places_with_details = []
    for idx, candidate in enumerate(place_candidates, start=1):
        try:
            place_id = candidate.get("place_id")
            place_name = candidate.get("name", "알 수 없는 장소")
            
            # 진행 상황 표시 (안전하게 처리)
            try:
                status_callback(f"장소 정보 수집 중... ({idx}/{total_candidates}): {place_name}")
            except Exception:
                pass  # 상태 메시지 업데이트 실패해도 계속 진행
            
            # [중요] 카테고리 라벨 가져오기
            category_label = candidate.get("custom_category_label", "기타")
            
            # Google Details (필수)
            try:
                details = fetch_google_place_details(gmaps_client, place_id)
            except Exception as e:
                # Google API 호출 실패 시 스킵
                st.warning(f"[경고] Google Places API 호출 실패 ({place_name}): {type(e).__name__}: {str(e)[:100]}")
                try:
                    progress_callback(0.05 + 0.4 * (idx / total_candidates))
                except Exception:
                    pass
                continue
            
            if not details:
                # Google 정보가 없으면 스킵
                try:
                    progress_callback(0.05 + 0.4 * (idx / total_candidates))
                except Exception:
                    pass
                continue
            
            # 중심 좌표와의 거리 검증 (이중 확인)
            if center_coordinates and len(center_coordinates) >= 2:
                geometry = details.get("geometry", {}).get("location", {})
                place_lat = geometry.get("lat")
                place_lng = geometry.get("lng")
                
                if place_lat is not None and place_lng is not None:
                    try:
                        distance = calculate_distance(
                            center_coordinates[0], center_coordinates[1],
                            place_lat, place_lng
                        )
                        # 50km를 초과하면 해당 지역이 아니므로 스킵
                        if distance > 50.0:
                            try:
                                progress_callback(0.05 + 0.4 * (idx / total_candidates))
                            except Exception:
                                pass
                            continue
                    except Exception as e:
                        # 거리 계산 실패 시 스킵
                        st.warning(f"[경고] 거리 계산 실패 ({place_name}): {type(e).__name__}: {str(e)[:100]}")
                        try:
                            progress_callback(0.05 + 0.4 * (idx / total_candidates))
                        except Exception:
                            pass
                        continue
                else:
                    # 좌표가 없으면 스킵
                    try:
                        progress_callback(0.05 + 0.4 * (idx / total_candidates))
                    except Exception:
                        pass
                    continue
            
            # Google Details 수집 완료 - Apify 작업 목록에 추가 (나중에 병렬 처리)
            places_with_details.append({
                "idx": idx,
                "candidate": candidate,
                "details": details,
                "place_name": place_name,
                "category_label": category_label,
                "geometry": geometry
            })
            
            # 네이버 블로그 리뷰 요약 (관광지 카테고리에만 수집, 선택적, 실패해도 계속 진행)
            # 타임아웃을 짧게 설정하여 빠르게 실패하도록 함
            naver_summary = None
            naver_blogs = []
            # 관광지 카테고리에만 네이버 블로그 요약 수집
            if category_label == "관광지":
                try:
                    if naver_client_id and naver_client_secret and openai_client:
                        # 타임아웃을 8초로 설정하여 빠르게 실패
                        naver_summary, naver_blogs = get_naver_blog_summary(
                            place_name, 
                            openai_client, 
                            naver_client_id, 
                            naver_client_secret,
                            max_blogs=5,
                            timeout=8
                        )
                except Exception as e:
                    # 네이버 리뷰 실패해도 계속 진행
                    st.warning(f"[경고] 네이버 블로그 요약 실패 ({place_name}): {type(e).__name__}: {str(e)[:100]}")
                    naver_summary = None
                    naver_blogs = []

            # 진행 상황 업데이트
            try:
                progress_callback(0.05 + 0.2 * (idx / total_candidates))
            except Exception:
                pass
        except Exception as e:
            # 개별 장소 처리 실패 시에도 계속 진행
            try:
                progress_callback(0.05 + 0.2 * (idx / total_candidates))
            except Exception:
                pass
            continue
    
    # Apify 병렬 처리 (최대 5개 동시 실행)
    status_callback("Apify 리뷰 데이터 병렬 수집 중...")
    apify_results = {}  # {idx: apify_data}
    
    if apify_token and places_with_details:
        def fetch_apify_wrapper(place_info):
            idx = place_info["idx"]
            place_name = place_info["place_name"]
            category_label = place_info["category_label"]
            try:
                search_query = f"{target_city} {place_name} {category_label}"
                return (idx, fetch_apify_details(search_query, apify_token))
            except Exception as e:
                return (idx, {})
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_apify_wrapper, place_info): place_info 
                      for place_info in places_with_details}
            
            completed = 0
            for future in as_completed(futures):
                try:
                    idx, apify_data = future.result()
                    apify_results[idx] = apify_data
                    completed += 1
                    try:
                        progress_callback(0.25 + 0.1 * (completed / len(places_with_details)))
                    except Exception:
                        pass
                except Exception:
                    place_info = futures[future]
                    apify_results[place_info["idx"]] = {}
    
    # 최종 데이터 조합 (Apify 결과 포함)
    status_callback("최종 데이터 조합 중...")
    total_places = len(places_with_details)
    for place_idx, place_info in enumerate(places_with_details, start=1):
        idx = place_info["idx"]
        candidate = place_info["candidate"]
        details = place_info["details"]
        place_name = place_info["place_name"]
        category_label = place_info["category_label"]
        geometry = place_info["geometry"]
        place_id = candidate.get("place_id")
        
        # 진행 상황 업데이트 (주기적으로)
        if place_idx % 10 == 0 or place_idx == total_places:
            try:
                status_callback(f"최종 데이터 조합 중... ({place_idx}/{total_places})")
                progress_callback(0.35 + 0.05 * (place_idx / total_places))
            except Exception:
                pass
        
        # Apify 결과 가져오기
        apify_data = apify_results.get(idx, {})
        
        # 네이버 블로그 리뷰 요약 (관광지 카테고리에만 수집, 선택적, 실패해도 계속 진행, 타임아웃 단축)
        naver_summary = None
        naver_blogs = []
        # 관광지 카테고리에만 네이버 블로그 요약 수집
        if category_label == "관광지":
            try:
                if naver_client_id and naver_client_secret and openai_client:
                    # 타임아웃을 5초로 단축하고, 실패 시 즉시 스킵
                    naver_summary, naver_blogs = get_naver_blog_summary(
                        place_name, 
                        openai_client, 
                        naver_client_id, 
                        naver_client_secret,
                        max_blogs=3,  # 블로그 개수도 줄임
                        timeout=5  # 타임아웃 단축
                    )
            except Exception:
                # 네이버 블로그 실패는 조용히 스킵
                naver_summary = None
                naver_blogs = []
        
        photos = (details.get("photos") or [])[:1]  # 사진 1개만 저장
        opening_hours = details.get("opening_hours", {}).get("weekday_text", [])
        
        # Google Places API 리뷰 수집 (최신 5개)
        google_reviews_raw = details.get("reviews", [])
        google_reviews = []
        if google_reviews_raw:
            # 시간순으로 정렬 (최신순)
            sorted_google_reviews = sorted(
                google_reviews_raw,
                key=lambda x: x.get("time", 0) if isinstance(x, dict) else 0,
                reverse=True
            )[:5]  # 최신 5개만
            
            # 리뷰 데이터 정리
            for review in sorted_google_reviews:
                if isinstance(review, dict):
                    google_reviews.append({
                        "text": review.get("text", ""),
                        "rating": review.get("rating", ""),
                        "author_name": review.get("author_name", "익명"),
                        "time": review.get("time", 0),
                        "author": review.get("author_name", "익명")
                    })
        
        # SerpAPI를 통한 장소 설명 데이터 가져오기 (선택적, 실패해도 계속 진행, 타임아웃 단축)
        serpapi_description = None
        try:
            if serpapi_key:
                # 타임아웃을 5초로 단축
                serpapi_description = fetch_serpapi_place_description(place_name, target_city, serpapi_key)
        except Exception:
            # SerpAPI 실패는 조용히 스킵
            serpapi_description = None
        
        # description 생성
        editorial_summary = details.get("editorial_summary", {})
        editorial_text = editorial_summary.get("overview", "") if isinstance(editorial_summary, dict) else ""
        
        if serpapi_description:
            final_description = serpapi_description
        elif editorial_text:
            final_description = editorial_text
        else:
            final_description = (
                candidate.get("vicinity") or 
                details.get("formatted_address", "").split(",")[0] or 
                f"{place_name}에 대한 정보"
            )

        enriched_places.append(
            {
                "place_id": place_id,
                "name": details.get("name") or place_name,
                "address": details.get("formatted_address") or candidate.get("formatted_address"),
                "lat": geometry.get("lat"),
                "lng": geometry.get("lng"),
                "rating": details.get("rating"),
                "user_ratings_total": details.get("user_ratings_total"),
                "photos": photos,
                "photo_references": [photo.get("photo_reference") for photo in photos],
                "categories": details.get("types") or candidate.get("types", []),
                "custom_category": category_label,
                "phone_number": details.get("formatted_phone_number"),
                "website": details.get("website"),
                "opening_hours_raw": details.get("opening_hours"),
                "opening_hours_text": opening_hours,
                "price_level": details.get("price_level"),
                "google_url": details.get("url"),
                "google_reviews": google_reviews,  # Google Places API 리뷰 (최신 5개)
                "reviews": apify_data.get("reviews", [])[:5],  # Apify 리뷰 5개 (기존 호환성 유지)
                "review_snippets": [review.get("text", "") for review in apify_data.get("reviews", [])[:5]],  # Apify 리뷰 텍스트
                "apify_reviews": apify_data.get("reviews", []),
                "place_details": details,  # Google Places API 전체 결과 저장 (리뷰 접근용)
                "feature_tags": apify_data.get("feature_tags", []),
                "keywords": apify_data.get("keywords", []),
                "crowd_levels": apify_data.get("crowd_levels"),
                "price_range": apify_data.get("price_range"),
                "naver_blog_summary": naver_summary,
                "naver_blogs": naver_blogs,
                "history_and_tips": "",
                "description": final_description,
                "source_city": target_city,
            }
        )
        
        # 진행 상황 업데이트
        try:
            progress_callback(0.35 + 0.1 * (len(enriched_places) / len(places_with_details)))
        except Exception:
            pass

    # 수집된 장소가 없으면 오류
    if not enriched_places:
        raise ValueError("수집된 장소 정보가 없습니다. API 호출을 확인해주세요.")
    
    try:
        status_callback("임베딩 생성 중...")
    except Exception as e:
        st.warning(f"[경고] 상태 메시지 업데이트 실패 (임베딩 생성): {e}")
    
    # [수정] 임베딩 페이로드 생성 시 custom_category 정보도 포함되면 좋음 (build_embedding_payload 함수 수정 필요할 수 있음)
    try:
        documents = [build_embedding_payload(place) for place in enriched_places]
    except Exception as e:
        # 임베딩 페이로드 생성 실패 시 기본값 사용
        st.warning(f"[경고] 임베딩 페이로드 생성 실패, 기본값 사용: {type(e).__name__}: {str(e)[:200]}")
        documents = [f"{place.get('name', '장소')} {place.get('address', '')}" for place in enriched_places]
    
    if not openai_client:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")

    try:
        # 임베딩 생성 시 타임아웃 적용 (지원되는 경우)
        try:
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=documents,
                timeout=60,  # 60초 타임아웃
            )
        except TypeError:
            # timeout 파라미터가 지원되지 않는 경우
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=documents,
            )
        sorted_embeddings = sorted(embedding_response.data, key=lambda item: item.index)
    except Exception as e:
        st.error(f"[오류] OpenAI 임베딩 생성 실패: {type(e).__name__}: {str(e)[:200]}")
        raise ValueError(f"OpenAI 임베딩 생성 실패: {e}")
    
    try:
        progress_callback(0.85)
    except Exception:
        pass

    records = []
    for place, embedding_data in zip(enriched_places, sorted_embeddings):
        records.append(
            {
                "id": place["place_id"],
                "name": place["name"],
                "region": region_name,
                "city": target_city,
                "metadata": place,
                "embedding": embedding_data.embedding,
            }
        )

    try:
        status_callback("벡터 DB 저장 중...")
    except Exception as e:
        st.warning(f"[경고] 상태 메시지 업데이트 실패 (저장 중): {e}")
    
    try:
        payload = {
            "region": region_name,
            "city": target_city,
            "record_count": len(records),
            "places": records,
        }
        path = store_vector_db(db_names.base, payload)
    except Exception as e:
        st.error(f"[오류] 벡터DB JSON 저장 실패: {type(e).__name__}: {str(e)[:200]}")
        raise ValueError(f"벡터 DB 저장 실패: {e}")
    
    try:
        persist_records_to_sqlite(
            db_key=db_names.sqlite,
            display_name=db_names.english,
            region=region_name,
            city=target_city,
            records=records,
        )
    except Exception as e:
        st.error(f"[오류] SQLite 저장 실패: {type(e).__name__}: {str(e)[:200]}")
        raise ValueError(f"SQLite 저장 실패: {e}")
    
    try:
        persist_records_to_chroma(db_names.chroma, records)
    except Exception as e:
        st.warning(f"[경고] ChromaDB 저장 실패 (선택적): {type(e).__name__}: {str(e)[:200]}")
        # ChromaDB는 선택적이므로 실패해도 계속 진행
    
    try:
        progress_callback(0.9)
        status_callback("벡터 DB 저장 완료")
    except Exception:
        pass
    
    # 벡터DB 저장 완료 확인
    try:
        status_callback("벡터DB 저장 완료 확인 중...")
        progress_callback(0.92)
    except Exception:
        pass
    
    try:
        # 벡터DB에서 장소 데이터 읽어오기
        places = load_places_from_vector_db(db_names.sqlite)
        
        if places:
            try:
                progress_callback(0.98)
                status_callback(f"벡터DB 생성 완료: {len(places)}개 장소 저장됨")
            except Exception:
                pass
        else:
            try:
                status_callback("저장된 장소가 없습니다.")
            except Exception:
                pass
    except Exception as e:
        # 벡터DB 읽기 실패해도 벡터DB는 생성되었으므로 계속 진행
        try:
            status_callback("벡터DB 읽기 실패 (벡터DB는 생성됨)")
        except Exception:
            pass
    
    try:
        progress_callback(1.0)
        status_callback(f"벡터 DB 생성 완료: {path}")
    except Exception:
        pass
    
    return path


# ============================================
# 2. 메인 함수 파트
# ============================================

def main():
    """
    메인 애플리케이션 함수
    """
    filtered_recommendations = []
    
    # .env 파일 로드 (파싱 오류 방지)
    try:
        load_dotenv()
    except Exception as e:
        st.warning(f"[경고] .env 파일 로드 중 오류 발생: {e}. 일부 환경 변수를 읽지 못할 수 있습니다.")
    
    # 페이지 설정
    st.set_page_config(
        page_title="FITrip - AI 여행 플래너", 
        layout="wide",
        page_icon="✈️",
        initial_sidebar_state="expanded"
    )
    
    # 메인 타이틀 (planning_mode에 따라 변경)
    if "planning_mode" in st.session_state and st.session_state.planning_mode == "AI 항공/숙박":
        st.markdown("# ✈️ FITrip - AI 항공권/숙박 도우미")
    else:
        st.markdown("# ✈️ FITrip")
        st.markdown("### 🤖 AI 기반 맞춤형 여행 계획 서비스")
    st.markdown("")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # API 키 로드
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    
    # Google Maps / OpenAI 클라이언트 초기화
    gmaps = None
    if GOOGLE_MAPS_API_KEY:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    else:
        st.error("🚨 .env 파일에 GOOGLE_MAPS_API_KEY가 설정되지 않았습니다!")
        st.info("Google Maps API 키를 .env 파일에 추가해주세요.")
        return
    
    openai_client = None
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        st.warning("OPENAI_API_KEY가 설정되지 않았습니다. 벡터 DB 생성 기능을 사용하려면 API 키가 필요합니다.")
    
    # 사이드바: 사용자 입력 받기
    with st.sidebar:
        # 헤더 스타일 개선
        st.markdown("## ✈️ 여행 계획 설정")
        st.markdown("---")
        
        # 플래닝 모드 선택
        st.markdown("#### 🎯 플래닝 모드")
        if "planning_mode" not in st.session_state:
            st.session_state.planning_mode = "AI 항공/숙박"
        
        planning_mode = st.radio(
            "",
            ["AI 항공/숙박", "커스터마이징"],
            index=0 if st.session_state.planning_mode == "AI 항공/숙박" else 1,
            key="planning_mode_radio",
            label_visibility="collapsed"
        )
        st.session_state.planning_mode = planning_mode
        
        if planning_mode == "AI 항공/숙박":
            st.caption("✈️ AI가 항공권과 숙박을 찾아드립니다")
        else:
            st.caption("✏️ 직접 원하는 장소와 일정을 선택하여 계획합니다")
        
        # "커스터마이징" 모드일 때만 여행지 정보 및 데이터 수집 섹션 표시
        if planning_mode == "커스터마이징":
            st.markdown("---")
            
            # 여행지 정보 섹션
            st.markdown("#### 📍 여행지 정보")
            st.markdown("")
            
            # 지역별 도시 딕셔너리 가져오기
            region_cities = get_region_cities()
            
            # 첫 번째 드롭다운: 지역 선택
            regions = list(region_cities.keys())
            selected_region = st.selectbox(
                "🌏 지역",
                options=["선택하세요"] + regions,
                index=0 if st.session_state.get("selected_region") is None else regions.index(st.session_state.selected_region) + 1 if st.session_state.selected_region in regions else 0,
                key="region_select",
                help="여행할 지역을 선택하세요"
            )
            
            # 지역이 선택되면 세션 상태에 저장
            if selected_region != "선택하세요":
                st.session_state.selected_region = selected_region
            else:
                st.session_state.selected_region = None
                st.session_state.selected_city = None
            
            # 두 번째 드롭다운: 도시 선택 (지역 선택 시에만 표시)
            selected_city = None
            if st.session_state.selected_region and st.session_state.selected_region in region_cities:
                cities = region_cities[st.session_state.selected_region]
                if cities:  # 도시 목록이 있는 경우에만
                    current_index = 0
                    if st.session_state.get("selected_city") in cities:
                        current_index = cities.index(st.session_state.selected_city) + 1
                    
                    selected_city = st.selectbox(
                        "🏙️ 도시",
                        options=["선택하세요"] + cities,
                        index=current_index,
                        key="city_select",
                        help="여행할 도시를 선택하세요"
                    )
                    
                    if selected_city != "선택하세요":
                        st.session_state.selected_city = selected_city
                        # 도시 선택 시 자동으로 destination에 설정
                        st.session_state.destination = selected_city
                    else:
                        st.session_state.selected_city = None
                        st.session_state.destination = ""
                else:
                    st.info("⚠️ 해당 지역의 도시 목록이 없습니다.")
            
            # 여행지 입력 (드롭다운 선택 시 자동으로 채워짐)
            destination = st.text_input(
                "✈️ 여행지",
                placeholder="예: 프랑스 파리, 일본 도쿄",
                value=st.session_state.get("destination", ""),
                help="도시를 선택하거나 직접 입력하세요"
            )
            
            # 여행 기간 입력
            duration = st.text_input(
                "📅 여행 기간",
                placeholder="예: 3박 4일, 6박 7일",
                value=st.session_state.get("duration", ""),
                help="여행 일정을 입력하세요 (예: 3박 4일)"
            )
            
            # 입력값 저장
            if destination:
                st.session_state.destination = destination
            
            if duration:
                st.session_state.duration = duration
                st.session_state.num_days = parse_duration_to_days(duration)
            
            st.markdown("---")
            
            # 데이터 수집 버튼
            st.markdown("#### 🚀 시작하기")
            st.markdown("")
            start_vector_generation = st.button(
                "📊 데이터 수집하기", 
                use_container_width=True,
                type="primary",
                help="여행지 데이터를 수집하고 벡터DB를 생성합니다"
            )
            
            # 입력 정보 요약 표시
            if st.session_state.get("destination") and st.session_state.get("duration"):
                st.markdown("---")
                st.markdown("#### 📋 입력 정보 요약")
                with st.container(border=True):
                    st.markdown(f"**✈️ 여행지**")
                    st.caption(st.session_state.destination)
                    st.markdown("")
                    st.markdown(f"**📅 기간**")
                    st.caption(f"{st.session_state.duration} ({st.session_state.num_days}일)")
    
    # planning_mode에 따라 다른 메인 화면 표시
    if st.session_state.get("planning_mode") == "AI 항공/숙박":
        # AI 항공/숙박 모드: 새로운 메인 화면 표시만 하고 종료
        _render_flight_hotel_search_ui()
        return  # 지도, 여행 정보, 일정별 계획 섹션은 표시하지 않음
    else:
        # 커스터마이징 모드: 기존 메인 화면 표시
        _render_customizing_main_screen()
    
    # 아래 코드들은 "커스터마이징" 모드일 때만 실행됨
    
    # 확정된 일정 요약 (메인 화면 상단, 토글 형식)
    all_confirmed_count = sum(len(st.session_state.confirmed_plans.get(day, [])) for day in range(1, st.session_state.num_days + 1))
    if all_confirmed_count > 0:
        with st.expander(f"📋 확정된 일정 요약 ({all_confirmed_count}개 장소)", expanded=False):
            st.markdown("")
            for day in range(1, st.session_state.num_days + 1):
                day_plans = st.session_state.confirmed_plans.get(day, [])
                if day_plans:
                    st.markdown(f"### 📆 Day {day} ({len(day_plans)}개 장소)")
                    for idx, plan in enumerate(day_plans, 1):
                        plan_name = plan.get("name", "알 수 없는 장소")
                        metadata = plan.get("metadata", {})
                        rating = metadata.get("rating", "")
                        category = metadata.get("custom_category", "기타")
                        
                        col_name, col_info = st.columns([3, 1])
                        with col_name:
                            st.markdown(f"**{idx}.** {plan_name}")
                        with col_info:
                            if rating:
                                st.caption(f"⭐ {rating}")
                            st.caption(category)
                    st.markdown("")
            
            # Notion 내보내기 및 가이드북 생성 버튼
            col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
            with col_export1:
                if st.button("📤 Notion으로 내보내기", use_container_width=True, type="primary"):
                    export_plans_to_notion(
                        confirmed_plans=st.session_state.confirmed_plans,
                        destination=st.session_state.get("destination", ""),
                        num_days=st.session_state.num_days,
                        notion_api_key=NOTION_API_KEY,
                        notion_database_id=NOTION_DATABASE_ID,
                        openai_client=openai_client
                    )
            with col_export2:
                st.caption("일정을 Notion 데이터베이스로 내보냅니다")
            with col_export3:
                if st.button("📖 Crew AI로 가이드북 생성", use_container_width=True, type="primary"):
                    if not Agent or not Task or not Crew:
                        st.error("CrewAI 패키지가 설치되지 않았습니다. 'pip install crewai' 명령으로 설치해주세요.")
                    elif not Document:
                        st.error("python-docx 패키지가 설치되지 않았습니다. 'pip install python-docx' 명령으로 설치해주세요.")
                    else:
                        with st.spinner("가이드북을 생성하는 중입니다..."):
                            try:
                                content = generate_travel_guide_multicrew(
                                    confirmed_plans=st.session_state.confirmed_plans,
                                    destination=st.session_state.get("destination", ""),
                                    num_days=st.session_state.num_days
                                )
                                
                                if content:
                                    filepath = save_to_word(content)
                                    
                                    if filepath:
                                        with open(filepath, "rb") as f:
                                            st.download_button(
                                                label="📥 Word 파일 다운로드",
                                                data=f,
                                                file_name="여행_가이드북.docx",
                                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                            )
                            except Exception as e:
                                st.error(f"가이드북 생성 중 오류가 발생했습니다: {str(e)}")
                                import traceback
                                with st.expander("상세 오류 정보"):
                                    st.code(traceback.format_exc(), language="python")
    
    # 아래 코드들은 "커스터마이징" 모드일 때만 실행됨
    
    # 여행지 중심 좌표 계산
    center_location = None
    if st.session_state.get("destination"):
        center_location = geocode_location(gmaps, st.session_state.destination)
        if center_location:
            st.session_state.map_center = center_location
            st.session_state.map_zoom = 12
    
    # 모든 날짜의 확정 일정을 지도에 표시
    all_confirmed_places = []
    for day in range(1, st.session_state.num_days + 1):
        if day in st.session_state.confirmed_plans:
            day_plans = st.session_state.confirmed_plans[day]
            
            # 경로 최적화 로직:
            # 1. 경로 최적화 버튼을 누르면 confirmed_plans[day]가 최적 순서로 재정렬됨
            # 2. "이 경로를 지도에서 보기" 버튼을 누르면 use_optimal_route_for_map_{day}가 True가 됨
            # 3. 지도 생성 시 confirmed_plans[day]의 현재 순서를 그대로 사용
            #    (이미 최적 순서로 재정렬되어 있으므로 그대로 사용하면 됨)
            
            # confirmed_plans는 경로 최적화 후 이미 최적 순서로 재정렬되어 있음
            # 따라서 그대로 사용하면 최적 순서가 지도에 반영됨
            for plan in day_plans:
                metadata = plan.get("metadata", {})
                lat = metadata.get("lat")
                lng = metadata.get("lng")
                if lat and lng:
                    all_confirmed_places.append({
                        "day": day,
                        "name": plan.get("name", "알 수 없는 장소"),
                        "lat": lat,
                        "lng": lng,
                        "metadata": metadata
                    })
    
    if all_confirmed_places:
        # 지도 중심점 설정
        if st.session_state.get("map_center"):
            center_lat, center_lng = st.session_state.map_center[0], st.session_state.map_center[1]
        else:
            # 확정 일정의 평균 좌표 사용
            center_lat = sum(p["lat"] for p in all_confirmed_places) / len(all_confirmed_places)
            center_lng = sum(p["lng"] for p in all_confirmed_places) / len(all_confirmed_places)
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
        
        # 날짜별 색상
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        
        # 날짜별로 그룹화하여 선으로 연결
        # all_confirmed_places는 이미 최적 경로 순서로 정렬되어 있음
        for day in range(1, st.session_state.num_days + 1):
            day_places = [p for p in all_confirmed_places if p["day"] == day]
            if day_places:
                color = colors[(day - 1) % len(colors)]
                locations = []
                
                # 최적 경로가 적용되어 있는지 확인
                use_optimal = st.session_state.get(f"use_optimal_route_for_map_{day}", False)
                
                # all_confirmed_places는 이미 최적 순서로 정렬되어 있으므로 그대로 사용
                for idx, place in enumerate(day_places):
                    lat, lng = place["lat"], place["lng"]
                    locations.append([lat, lng])
                    
                    # 마커 추가
                    marker_label = f"{idx + 1}. {place['name']}" if use_optimal else place['name']
                    folium.Marker(
                        [lat, lng],
                        popup=folium.Popup(f"<b>Day {day} - {marker_label}</b>", max_width=300),
                        tooltip=f"Day {day} - {marker_label}",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(m)
                
                # 선으로 연결 (순서대로)
                if len(locations) > 1:
                    route_label = f"Day {day} 최적 경로" if use_optimal else f"Day {day} 경로"
                    folium.PolyLine(
                        locations,
                        color=color,
                        weight=4 if use_optimal else 3,
                        opacity=0.8 if use_optimal else 0.7,
                        popup=route_label
                    ).add_to(m)
        
        # 고유한 키 생성 (확정 일정 개수 기반)
        places_count = len(all_confirmed_places)
        map_key = f"confirmed_plans_map_{places_count}_{hash(str(all_confirmed_places)) % 100000}"
        st_folium(m, width="100%", height=500, key=map_key)
        
        # 간단한 여행 일정 표시 (지도 밑)
        st.markdown("")
        st.markdown("### 📅 여행 일정 요약")
        st.markdown("")
        for day in range(1, st.session_state.num_days + 1):
            if day in st.session_state.confirmed_plans and st.session_state.confirmed_plans[day]:
                day_places = [p for p in all_confirmed_places if p["day"] == day]
                if day_places:
                    place_names = [p["name"] for p in day_places]
                    with st.container(border=True):
                        st.markdown(f"**📌 Day {day}:** {' → '.join(place_names)}")
    else:
        # 확정 일정이 없으면 기본 지도 표시
        travel_map = create_map(
            gmaps,
            st.session_state.map_center
        )
        # 고유한 키 생성 (목적지 기반)
        dest_hash = hash(st.session_state.get("destination", "default")) % 100000
        default_map_key = f"default_travel_map_{dest_hash}"
        st_folium(travel_map, width='100%', height=500, key=default_map_key)
        st.markdown("")
        st.info("💡 확정된 일정이 없습니다. 일정을 추가하면 지도에 표시됩니다.")
    
    vector_status_container = st.container()
    
    st.divider()
    
    # 진행 중이거나 에러가 발생한 경우 진행 상황 표시 (rerun 후에도 유지)
    if st.session_state.get("vector_db_in_progress") or st.session_state.get("vector_db_error"):
        with vector_status_container:
            # 진행 상황 표시 (에러 발생 시에도 마지막 진행률 표시)
            current_progress = st.session_state.get("vector_db_progress", 0.0)
            st.progress(current_progress)
            
            # 현재 상태 메시지 표시
            current_status = st.session_state.get("vector_db_current_status")
            if current_status:
                if st.session_state.get("vector_db_error"):
                    st.error(current_status)  # 에러인 경우 error로 표시
                else:
                    st.info(current_status)  # 정상 진행인 경우 info로 표시
            
            # 에러가 발생한 경우 상세 정보 표시
            if st.session_state.get("vector_db_error"):
                error_msg = st.session_state.vector_db_error
                if error_msg not in (current_status or ""):  # 중복 표시 방지
                    st.error(error_msg)
                if st.session_state.get("vector_db_traceback"):
                    with st.expander("상세 에러 정보 보기"):
                        st.code(st.session_state.vector_db_traceback, language="python")
    
    # 완료된 상태 표시
    if st.session_state.vector_db_status and not st.session_state.get("vector_db_in_progress"):
        with vector_status_container:
            st.info(st.session_state.vector_db_status)
    
    # 사이드바에서 정의된 버튼 변수 사용 (IAP copy.py 방식)
    if start_vector_generation:
        # 에러 상태 및 진행 상태 초기화
        if "vector_db_error" in st.session_state:
            del st.session_state.vector_db_error
        if "vector_db_traceback" in st.session_state:
            del st.session_state.vector_db_traceback
        st.session_state.vector_db_progress = 0.0
        st.session_state.vector_db_current_status = None
        with vector_status_container:
            if not st.session_state.get("destination"):
                st.error("여행지를 먼저 선택하거나 입력해주세요.")
            else:
                db_names = build_vector_db_names(
                    st.session_state.destination,
                    gmaps,
                )
                vector_db_name = st.session_state.selected_region or st.session_state.destination
                region_label = vector_db_name or st.session_state.destination
                if not region_label:
                    st.error("벡터 DB 이름을 결정할 수 없습니다. 선택한 지역을 확인해주세요.")
                elif vector_db_exists(db_names.sqlite):
                    st.session_state.vector_db_status = f"'{db_names.english}' 벡터 DB는 이미 생성되어 있어 단계를 건너뜁니다."
                    st.session_state.vector_db_last_region = db_names.english
                    st.success(st.session_state.vector_db_status)
                else:
                    if not openai_client:
                        st.error("OPENAI_API_KEY가 없어 벡터 DB를 생성할 수 없습니다.")
                    else:
                        progress_bar = st.progress(0.0)
                        status_placeholder = st.empty()
                        st.session_state.vector_db_in_progress = True
                        
                        # 안전한 콜백 함수 생성 (세션 상태에도 저장)
                        def safe_progress_callback(value):
                            try:
                                progress_value = min(max(value, 0.0), 1.0)
                                progress_bar.progress(progress_value)
                                st.session_state.vector_db_progress = progress_value  # 세션 상태에 저장
                            except Exception:
                                pass  # 진행 상황 업데이트 실패해도 계속 진행
                        
                        def safe_status_callback(message):
                            try:
                                status_placeholder.info(message)
                                st.session_state.vector_db_current_status = message  # 세션 상태에 저장
                            except Exception:
                                pass  # 상태 메시지 업데이트 실패해도 계속 진행
                        
                        try:
                            create_vector_database(
                                region_name=region_label,
                                city_name=st.session_state.destination,
                                db_names=db_names,
                                gmaps_client=gmaps,
                                openai_client=openai_client,
                                apify_token=APIFY_API_TOKEN,
                                progress_callback=safe_progress_callback,
                                status_callback=safe_status_callback,
                                center_coordinates=st.session_state.get("map_center"),
                                num_days=st.session_state.num_days,
                                naver_client_id=NAVER_CLIENT_ID,
                                naver_client_secret=NAVER_CLIENT_SECRET,
                                serpapi_key=SERPAPI_API_KEY,
                            )
                            st.session_state.vector_db_status = f"'{db_names.english}' 벡터 DB가 성공적으로 생성되었습니다."
                            st.session_state.vector_db_last_region = db_names.english
                            try:
                                progress_bar.progress(1.0)
                                status_placeholder.success(st.session_state.vector_db_status)
                            except Exception:
                                pass
                        except Exception as exc:
                            import traceback
                            error_msg = f"[오류] 벡터 DB 생성 중 예외 발생: {type(exc).__name__}: {str(exc)[:500]}"
                            full_traceback = traceback.format_exc()
                            
                            # 세션 상태에 에러 저장 (rerun 후에도 표시되도록)
                            st.session_state.vector_db_error = error_msg
                            st.session_state.vector_db_traceback = full_traceback
                            st.session_state.vector_db_current_status = f"오류 발생: {error_msg}"
                            
                            # 즉시 에러 표시
                            st.error(error_msg)
                            st.code(full_traceback, language="python")
                            
                            try:
                                status_placeholder.error(f"벡터 DB 생성 중 오류가 발생했습니다: {exc}")
                            except Exception:
                                pass
                        finally:
                            st.session_state.vector_db_in_progress = False
                            # 진행 상태는 유지 (에러 발생 시점의 진행률 표시)
    
    
    st.divider()
    
    # 지역 선택/입력 시 두 개의 탭 생성 (기본 정보, AI 챗봇)
    destination = st.session_state.get("destination", "")
    if destination:
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        st.markdown(f"## 🗺️ {destination} 여행 정보")
        st.markdown("")
        
        # 두 개의 탭 생성
        info_tab, chatbot_tab = st.tabs(["📋 기본 정보", "💬 AI 챗봇"])
        
        # 첫 번째 탭: 지역 기본 정보
        with info_tab:
            # 벡터 DB 상태 정보
            if gmaps:
                db_names = build_vector_db_names(destination, gmaps)
                db_exists = vector_db_exists(db_names.sqlite)
                
                if db_exists:
                    st.success(f"✅ '{db_names.english}' 벡터 DB가 생성되어 있습니다.")
                    st.info("💡 AI 챗봇 탭에서 여행지에 대한 장소 추천을 받을 수 있습니다.")
                else:
                    st.warning("⚠️ 벡터 DB가 아직 생성되지 않았습니다.")
                    st.info("💡 사이드바에서 '📊 데이터 수집하기' 버튼을 눌러 여행지 데이터를 수집해주세요.")
            
            st.markdown("")
            
            # OpenAI GPT를 사용한 여행지 상세 정보
            if destination and openai_client:
                # 세션 상태에 정보 저장 (재호출 방지)
                info_key = f"destination_info_{destination}"
                if info_key not in st.session_state:
                    with st.spinner(f"{destination}의 상세 정보를 가져오는 중..."):
                        destination_info = get_destination_info_from_gpt(destination, openai_client)
                        if destination_info:
                            st.session_state[info_key] = destination_info
                        else:
                            st.session_state[info_key] = None
                
                destination_info = st.session_state.get(info_key)
                
                if destination_info:
                    st.markdown("---")
                    st.markdown("### 📚 여행지 상세 정보")
                    st.markdown("")
                    
                    # 기본정보
                    if destination_info.get("기본정보"):
                        with st.expander("📋 기본정보", expanded=False):
                            info_text = destination_info["기본정보"]
                            # 딕셔너리 형태의 텍스트를 읽기 좋게 변환
                            if isinstance(info_text, str):
                                # 마크다운 형식으로 표시
                                st.markdown(info_text)
                            else:
                                st.json(info_text)
                    
                    # 역사
                    if destination_info.get("역사"):
                        with st.expander("📜 역사", expanded=False):
                            history_text = destination_info["역사"]
                            if isinstance(history_text, str):
                                st.markdown(history_text)
                            else:
                                st.json(history_text)
                    
                    # 정치/경제/문화
                    if destination_info.get("정치경제문화"):
                        with st.expander("🏛️ 정치/경제/문화", expanded=False):
                            culture_text = destination_info["정치경제문화"]
                            if isinstance(culture_text, str):
                                st.markdown(culture_text)
                            else:
                                st.json(culture_text)
                    
                    # 명소
                    if destination_info.get("명소"):
                        with st.expander(f"🏛️ 명소 ({len(destination_info['명소'])}개)", expanded=False):
                            for idx, place in enumerate(destination_info["명소"], 1):
                                if isinstance(place, dict):
                                    st.markdown(f"#### {idx}. {place.get('이름', '알 수 없음')}")
                                    if place.get("설명"):
                                        st.markdown(f"**설명:** {place['설명']}")
                                    if place.get("추천이유"):
                                        st.markdown(f"**추천 이유:** {place['추천이유']}")
                                elif isinstance(place, str):
                                    st.markdown(f"**{idx}. {place}**")
                                if idx < len(destination_info["명소"]):
                                    st.markdown("---")
                    
                    # 음식
                    if destination_info.get("음식"):
                        with st.expander(f"🍽️ 음식 ({len(destination_info['음식'])}개)", expanded=False):
                            for idx, food in enumerate(destination_info["음식"], 1):
                                if isinstance(food, dict):
                                    food_name = food.get("이름", food.get("name", "알 수 없음"))
                                    food_desc = food.get("설명", food.get("description", ""))
                                    if food_desc:
                                        st.markdown(f"**{idx}. {food_name}** - {food_desc}")
                                    else:
                                        st.markdown(f"**{idx}. {food_name}**")
                                elif isinstance(food, str):
                                    st.markdown(f"**{idx}. {food}**")
                    
                    # 여행 팁
                    if destination_info.get("여행팁"):
                        with st.expander("💡 여행 팁", expanded=False):
                            tips_text = destination_info["여행팁"]
                            if isinstance(tips_text, str):
                                st.markdown(tips_text)
                            else:
                                st.json(tips_text)
                elif destination_info is None:
                    st.warning("⚠️ 여행지 정보를 가져오지 못했습니다.")
        
        # 두 번째 탭: AI 챗봇
        with chatbot_tab:
            # 전역 챗봇 (Day별이 아닌 전체 여행지에 대한 챗봇)
            if "global_chat" not in st.session_state:
                st.session_state.global_chat = []
            
            with st.container(border=True):
                st.markdown("### 💬 AI 챗봇")
                st.caption("여행 스타일과 선호도를 알려주시면 맞춤 장소를 추천해드립니다.")
                st.markdown("")
                
                # 안내 메시지
                st.info("💡 챗봇을 통해 여행에서 방문할 장소(관광지)를 선택하세요")
                st.info("💡 장소를 선택한 후, 챗봇을 통해 맛집, 카페를 선택하세요")
                st.markdown("")
                
                # 대화 기록 표시
                chat_messages_container = st.container(height=1000)
                with chat_messages_container:
                    if not st.session_state.global_chat:
                        st.info(f"💡 {destination}에 대한 여행 스타일이나 선호도를 입력해보세요!")
                    else:
                        for message in st.session_state.global_chat:
                            if message["role"] == "user":
                                with st.chat_message("user"):
                                    st.write(message["content"])
                            else:
                                with st.chat_message("assistant"):
                                    st.write(message["content"])
                                    # 추천 장소가 있으면 카드뷰로 표시
                                    if message.get("recommendations"):
                                        st.markdown("---")
                                        st.markdown("#### 🎯 추천 장소")
                                        render_place_cards(
                                            message["recommendations"],
                                            GOOGLE_MAPS_API_KEY,
                                            is_global_chatbot=True
                                        )
                
                # 사용자 입력
                user_input = st.chat_input(f"{destination}에 대해 알려주세요...", key="global_chat_input")
            
            # 사용자 입력 처리
            if user_input:
                # 사용자 메시지 저장
                st.session_state.global_chat.append({
                    "role": "user",
                    "content": user_input
                })
                
                # 벡터DB가 있는 경우 추천 생성
                if destination:
                    # 현재 destination으로 벡터DB 이름 생성
                    db_names = build_vector_db_names(destination, gmaps)
                    
                    # 벡터DB 존재 여부 확인
                    db_exists = vector_db_exists(db_names.sqlite)
                    
                    # vector_db_last_region이 설정되어 있고 현재 destination으로 찾지 못한 경우
                    if not db_exists and st.session_state.get("vector_db_last_region"):
                        # 저장된 영어 이름으로도 확인 시도
                        last_region = st.session_state.vector_db_last_region
                        last_db_names = build_vector_db_names(last_region, gmaps)
                        if vector_db_exists(last_db_names.sqlite):
                            db_exists = True
                            db_names = last_db_names
                    
                    if db_exists and openai_client:
                        with st.spinner("장소를 추천하고 있습니다..."):
                            # 벡터DB에서 유사한 장소 검색 (최소 20개, 최대 50개)
                            recommendations = search_similar_places_from_vector_db(
                                db_key=db_names.sqlite,
                                user_query=user_input,
                                openai_client=openai_client,
                                top_k=50,  # 최대 50개
                                group_id=None  # 전역 챗봇이므로 그룹 제한 없음
                            )
                            # ==========================================================
                            # ✨ 장소 카테고리 필터링 및 5개 제한 로직 ✨
                            # ==========================================================

                            CATEGORIES = ["맛집", "베이커리/디저트", "관광지", "바/술집", "카페"]
                            requested_category = None

                            # 사용자 입력에서 요청 카테고리 식별
                            for cat in CATEGORIES:
                                if cat in user_input:
                                    requested_category = cat
                                    break

                            # 요청된 카테고리가 식별되면 장소 목록 필터링
                            if requested_category:
                                # 장소 항목(place)의 'category' 값이 요청된 카테고리와 일치하는 것만 추출
                                filtered_recommendations = [
                                    place for place in recommendations 
                                    if place.get('category') == requested_category
                                ]
                                
                                # 필터링된 목록으로 원본 recommendations를 대체
                                recommendations = filtered_recommendations

                            # IAP.py 파일 내, 장소 필터링 로직 이후
                            # 💡 카테고리 필터링된 목록으로 원본 recommendations를 대체
                            recommendations = filtered_recommendations 

                            # ==========================================================
                            # ✨ 복합 순위 점수 계산 및 정렬 로직 (이전에 안내된 코드) ✨
                            # ==========================================================

                            if recommendations:
                            # ----------------------------------------------------
                            # 1. 데이터 정규화에 필요한 최대/최소값 찾기 (함수 외부에 위치)
                            # ----------------------------------------------------
                            
                                review_counts = [place.get('review_count', 1) for place in recommendations]
                                max_reviews = max(review_counts) if review_counts else 1

                                similarity_scores = [place.get('similarity_score', 0.0) for place in recommendations]
                                max_similarity = max(similarity_scores) if similarity_scores else 1.0
                                min_similarity = min(similarity_scores) if similarity_scores else 0.0
                                similarity_range = max_similarity - min_similarity
                                    
                                # 1. 벡터 DB에서 유사 장소 검색 (top_k=50 유지)
                                recommendations = search_similar_places_from_vector_db(
                                    db_key=db_names.sqlite,
                                    user_query=user_input,
                                    openai_client=openai_client,
                                    top_k=50, 
                                    group_id=None
                                )

                                # ==========================================================
                                # ✨ 2. 감정 분석 점수 주입 ✨
                                # ==========================================================
                                for place in recommendations:
                                    # 💡 'review_text'는 실제 장소 데이터의 리뷰 텍스트 키 이름으로 수정해야 합니다.
                                    reviews_to_analyze = place.get('review_text', '') 
                                    
                                    sentiment = get_sentiment_score(reviews_to_analyze)
                                    place['sentiment_score'] = sentiment
                                    
                                # ==========================================================
                                # ✨ 3. 카테고리 필터링 로직 (맛집만 등) ✨
                                # ==========================================================
                                CATEGORIES = ["맛집", "베이커리/디저트", "관광지", "바/술집", "카페"]
                                requested_category = None

                                for cat in CATEGORIES:
                                    if cat in user_input:
                                        requested_category = cat
                                        break

                                if requested_category:
                                    filtered_recommendations = [
                                        place for place in recommendations 
                                        if place.get('category') == requested_category
                                    ]
                                    recommendations = filtered_recommendations 

                                # ==========================================================
                                # ✨ 4. 복합 순위 점수 계산 및 정렬 로직 (norm_sim 오류 해결) ✨
                                # ==========================================================

                                if recommendations:
                                    # ----------------------------------------------------
                                    # 4-1. 데이터 정규화(Normalization)를 위한 최대/최소값 찾기
                                    # ----------------------------------------------------
                                    review_counts = [place.get('review_count', 1) for place in recommendations]
                                    max_reviews = max(review_counts) if review_counts else 1

                                    similarity_scores = [place.get('similarity_score', 0.0) for place in recommendations]
                                    max_similarity = max(similarity_scores) if similarity_scores else 1.0
                                    min_similarity = min(similarity_scores) if similarity_scores else 0.0
                                    similarity_range = max_similarity - min_similarity
                                    
                                    # ----------------------------------------------------
                                    # 4-2. 복합 점수 계산 함수 정의 (norm_sim 정의 포함)
                                    # ----------------------------------------------------
                                    def calculate_composite_score(place, max_reviews, min_similarity, similarity_range):
                                        
                                        sim = place.get('similarity_score', 0.0)
                                        rating = place.get('rating', 0.0) 
                                        review_count = place.get('review_count', 0)
                                        norm_sentiment = place.get('sentiment_score', 0.5) 

                                        # 💡 norm_sim 정의 (NameError 해결)
                                        if similarity_range > 0:
                                            norm_sim = (sim - min_similarity) / similarity_range
                                        else:
                                            norm_sim = 1.0
                                            
                                        # norm_rating 정의
                                        norm_rating = min(rating / 5.0, 1.0) 
                                        
                                        # norm_review_count 정의
                                        norm_review_count = review_count / max_reviews if max_reviews > 0 else 0.0
                                        
                                        # 가중치 조합 (4694~4696줄 근처)
                                        composite_score = (
                                            (0.3 * norm_sim) + 
                                            (0.3 * norm_rating) + 
                                            (0.2 * norm_review_count) + 
                                            (0.2 * norm_sentiment)
                                        )
                                        return composite_score

                                    # 4-3. 모든 장소에 복합 점수 계산 및 저장
                                    for place in recommendations:
                                        place['composite_score'] = calculate_composite_score(
                                            place, 
                                            max_reviews, 
                                            min_similarity, 
                                            similarity_range 
                                        )

                                    # 4-4. 계산된 복합 점수를 기준으로 내림차순 정렬 (순위 반영)
                                    recommendations.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)

                                    # 4-5. 상위 10개로 제한
                                    recommendations = recommendations[:10]

                                # ==========================================================


                                # 5. 필터링 및 순위 정렬된 장소 목록을 바탕으로 AI 추천 메시지 생성
                                llm_recommendation_message = generate_recommendation_message(
                                    recommendations=recommendations,
                                    user_query=user_input,
                                    openai_client=openai_client,
                                    # 💡 필수 인자 추가 (TypeError 해결)
                                    day_num=None,          
                                    destination=None       
                                )   

                            # ----------------------------------------------------
                            # 3. 모든 장소에 복합 점수 계산 및 저장 (함수 호출 시 인자 전달)
                            # ----------------------------------------------------
                            for place in recommendations:
                                place['composite_score'] = calculate_composite_score(
                                    place, 
                                    max_reviews, 
                                    min_similarity, 
                                    similarity_range 
                                )

                                # 5. 상위 5개로 제한
                                recommendations = recommendations[:10]

                            # ==========================================================

                            # 💡 카테고리 필터링이 끝난 후, 최종적으로 5개만 남도록 제한 (가장 앞의 5개)
                            # 이전에 top_k=50으로 검색했다면, 여기서 5개로 잘라줍니다.
                            recommendations = recommendations[:10]

                            # ==========================================================
                            # ✨ 장소 정보에 감정 점수 추가 (새로운 로직) ✨
                            # ==========================================================
                            for place in recommendations:
                                # 💡 장소 데이터에서 리뷰 텍스트를 가져오는 부분입니다. 
                                # 'review_text'는 실제 데이터의 키 이름에 맞게 수정하세요.
                                reviews_to_analyze = place.get('review_text', '') 
                                
                                # 감정 분석 점수 계산
                                sentiment = get_sentiment_score(reviews_to_analyze)
                                
                                # place 딕셔너리에 'sentiment_score' 키로 저장
                                place['sentiment_score'] = sentiment

                            # ==========================================================

                            # 2. 필터링된 장소 목록을 바탕으로 AI 추천 메시지 생성
                            llm_recommendation_message = generate_recommendation_message(
                                recommendations=recommendations,
                                user_query=user_input,
                                openai_client=openai_client,
                                day_num=None,             # 임시로 '1' 또는 None을 사용
                                destination=None
                            )   

                            # LLM을 통한 추천 메시지 생성
                            recommendation_text = generate_recommendation_message(
                                openai_client=openai_client,
                                user_query=user_input,
                                recommendations=recommendations,
                                day_num=None,  # 전역 챗봇이므로 day_num 없음
                                destination=destination
                            )
                            
                            # 어시스턴트 메시지 저장
                            st.session_state.global_chat.append({
                                "role": "assistant",
                                "content": recommendation_text,
                                "recommendations": recommendations
                            })
                            
                            st.rerun()
                    elif not openai_client:
                        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
                    else:
                        st.warning("⚠️ 벡터DB가 생성되지 않았습니다. 먼저 '📊 데이터 수집하기' 버튼을 눌러주세요.")
                else:
                    st.warning("⚠️ 여행지를 먼저 입력해주세요.")
            
            # 선택된 장소 및 확정된 일정 표시 섹션
            st.markdown("---")
            
            # 선택된 장소 (확정 전)
            if st.session_state.pending_places:
                with st.container(border=True):
                    st.markdown("### 📝 선택된 장소 (확정 전)")
                    st.caption(f"{len(st.session_state.pending_places)}개의 장소가 선택되었습니다.")
                    
                    # 선택된 장소 목록 표시
                    for idx, place in enumerate(st.session_state.pending_places):
                        col_name, col_remove = st.columns([4, 1])
                        with col_name:
                            st.markdown(f"{idx + 1}. {place.get('name', '알 수 없는 장소')}")
                        with col_remove:
                            if st.button("❌ 제거", key=f"remove_pending_{place.get('place_id')}_{idx}"):
                                st.session_state.pending_places.pop(idx)
                                st.rerun()
                    
                    st.markdown("")
                    # 일정 확정 버튼
                    if st.button("✅ 일정 확정", key="confirm_places", use_container_width=True, type="primary"):
                        # pending_places를 confirmed_places로 추가
                        all_confirmed = list(st.session_state.confirmed_places)  # 기존 확정 장소
                        for place in st.session_state.pending_places:
                            # 중복 체크
                            if not any(p.get("place_id") == place.get("place_id") for p in all_confirmed):
                                all_confirmed.append(place)
                        
                        # 그룹화 수행
                        num_days = st.session_state.get("num_days", 1)
                        if num_days <= 0:
                            num_days = 1
                        
                        with st.spinner("장소들을 거리 기반으로 그룹화하고 있습니다..."):
                            # gmaps 클라이언트 가져오기
                            gmaps_client = None
                            if GOOGLE_MAPS_API_KEY:
                                try:
                                    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
                                except:
                                    pass
                            
                            # 그룹화
                            groups = group_places_by_distance(
                                places=all_confirmed,
                                num_groups=num_days,
                                min_per_group=4,
                                max_per_group=7,
                                gmaps_client=gmaps_client
                            )
                            
                            # 그룹화된 장소들을 day별로 할당
                            st.session_state.confirmed_plans = {}
                            for day_num in range(1, num_days + 1):
                                group_idx = day_num - 1
                                if group_idx < len(groups):
                                    st.session_state.confirmed_plans[day_num] = groups[group_idx]
                                else:
                                    st.session_state.confirmed_plans[day_num] = []
                        
                        # pending_places 비우기
                        st.session_state.pending_places = []
                        # confirmed_places는 그룹화 후 day별로 할당되었으므로 비우기
                        st.session_state.confirmed_places = []
                        
                        total_places = sum(len(plans) for plans in st.session_state.confirmed_plans.values())
                        st.success(f"{total_places}개의 장소가 {num_days}일 일정으로 그룹화되어 확정되었습니다!")
                        st.rerun()
            
            # 확정된 일정 표시
            if st.session_state.confirmed_places:
                with st.container(border=True):
                    st.markdown("### ✅ 확정된 일정")
                    st.caption(f"{len(st.session_state.confirmed_places)}개의 장소가 확정되었습니다.")
                    
                    # 확정된 장소 목록 표시
                    for idx, place in enumerate(st.session_state.confirmed_places):
                        col_name, col_remove = st.columns([4, 1])
                        with col_name:
                            st.markdown(f"{idx + 1}. {place.get('name', '알 수 없는 장소')}")
                        with col_remove:
                            if st.button("❌ 제거", key=f"remove_confirmed_{place.get('place_id')}_{idx}"):
                                st.session_state.confirmed_places.pop(idx)
                                st.rerun()
        
        st.markdown("")
        st.markdown("---")
        st.markdown("")
    
    # 여행 기간에 맞춰 Day별 탭 생성
    num_days = st.session_state.num_days
    
    if num_days > 1:
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        st.markdown("## 📅 일정별 계획")
        st.markdown("")
        
        # 탭 생성
        tab_names = [f"📆 Day {i}" for i in range(1, num_days + 1)]
        tabs = st.tabs(tab_names)
        
        # 각 탭에 챗봇 UI 추가
        for i, tab in enumerate(tabs):
            with tab:
                day_num = i + 1
                st.session_state.current_day_num = day_num  # 현재 날짜 저장 (일정 추가 버튼용)
                
                # 해당 날짜의 확정 일정 초기화
                if day_num not in st.session_state.confirmed_plans:
                    st.session_state.confirmed_plans[day_num] = []
                
                # 확정 일정과 최적화된 경로 정보를 좌우로 배치
                confirmed_count = len(st.session_state.confirmed_plans.get(day_num, []))
                
                # 좌우 분할 레이아웃
                col_left, col_right = st.columns([1, 1])
                
                # 왼쪽: 확정 일정 섹션
                with col_left:
                    with st.container(border=True):
                        # 헤더: 제목과 장소 개수
                        col_header1, col_header2 = st.columns([3, 1])
                        with col_header1:
                            st.markdown("### 📋 확정 일정")
                        with col_header2:
                            st.markdown(f"**{confirmed_count}개**")
                        
                        if confirmed_count > 0:
                            # 확정 일정 목록 (컴팩트하게 표시 - 한 줄에 여러 정보)
                            for idx, plan in enumerate(st.session_state.confirmed_plans[day_num][:10], 1):
                                plan_name = plan.get("name", "알 수 없는 장소")
                                col_num, col_name, col_remove = st.columns([0.4, 4.5, 0.6])
                                with col_num:
                                    st.markdown(f"**{idx}.**", help=None)
                                with col_name:
                                    st.markdown(plan_name)
                                with col_remove:
                                    if st.button("🗑️", key=f"quick_remove_{day_num}_{plan.get('place_id')}_{idx}", use_container_width=True):
                                        st.session_state.confirmed_plans[day_num].pop(idx - 1)
                                        st.rerun()
                            
                            if confirmed_count > 10:
                                st.caption(f"외 {confirmed_count - 10}개 장소...")
                            
                            st.markdown("")
                            # 경로 최적화 버튼
                            if st.button("🤖 경로 최적화", key=f"optimize_route_{day_num}", use_container_width=True, type="primary"):
                                # 경로 최적화 실행
                                day_plans = st.session_state.confirmed_plans[day_num]
                                if len(day_plans) >= 2:
                                    with st.spinner("경로를 최적화하고 있습니다..."):
                                        optimal_result = optimize_route_for_day(
                                            gmaps_client=gmaps,
                                            day_plans=day_plans,
                                            day_num=day_num
                                        )
                                        # 최적화된 순서를 세션 상태에 저장
                                        st.session_state[f"optimal_order_{day_num}"] = optimal_result["optimal_order"]
                                        st.session_state[f"route_info_{day_num}"] = optimal_result["route_info"]
                                        st.session_state[f"show_optimal_route_{day_num}"] = True
                                        # 최적화된 순서로 일정 재정렬
                                        if optimal_result["optimal_order"]:
                                            reordered_plans = [day_plans[i] for i in optimal_result["optimal_order"]]
                                            st.session_state.confirmed_plans[day_num] = reordered_plans
                                            # 경로 최적화 후 자동으로 지도에 최적 경로 적용
                                            st.session_state[f"use_optimal_route_for_map_{day_num}"] = True
                                    st.success("경로가 최적화되었습니다!")
                                    st.rerun()
                                else:
                                    st.warning("경로 최적화를 위해서는 최소 2개 이상의 장소가 필요합니다.")
                        else:
                            st.info("💡 챗봇과 대화하여 장소를 추천받고 '일정에 추가' 버튼을 눌러주세요.")
                
                # 오른쪽: 최적화된 경로 정보 표시
                with col_right:
                    if st.session_state.get(f"show_optimal_route_{day_num}", False) and confirmed_count >= 2:
                        route_info = st.session_state.get(f"route_info_{day_num}", [])
                        if route_info:
                            with st.container(border=True):
                                st.markdown("#### 🤖 최적화된 경로 정보")
                                st.markdown("")
                                
                                import pandas as pd
                                
                                # 경로 정보를 표시용으로 변환
                                display_data = []
                                for idx, route in enumerate(route_info, 1):
                                    display_row = {
                                        "구간": route.get("구간", f"{route.get('from', '')} → {route.get('to', '')}"),
                                        "추천 교통편": route.get("추천 교통편", route.get("transport", "")),
                                        "거리(km)": route.get("거리(km)", route.get("distance_km", 0)),
                                        "예상 소요 시간(분)": route.get("예상 소요 시간(분)", route.get("duration_min", 0))
                                    }
                                    
                                    # 도보 추천이 있으면 추가
                                    if route.get("도보 추천"):
                                        display_row["도보 추천"] = route["도보 추천"]
                                    
                                    display_data.append(display_row)
                                
                                route_df = pd.DataFrame(display_data)
                                route_df.index = route_df.index + 1
                                route_df.index.name = "No."
                                
                                # 표시할 컬럼 선택
                                display_cols = ["구간", "추천 교통편", "거리(km)", "예상 소요 시간(분)"]
                                if "도보 추천" in route_df.columns:
                                    display_cols.append("도보 추천")
                                
                                st.dataframe(route_df[display_cols], use_container_width=True, hide_index=False)
                                
                                # 총 거리 및 시간 계산
                                total_distance = sum(r.get("distance_km", 0) for r in route_info)
                                total_duration = sum(r.get("duration_min", 0) for r in route_info)
                                
                                col_sum1, col_sum2 = st.columns(2)
                                with col_sum1:
                                    st.metric("총 이동 거리", f"{total_distance:.2f} km")
                                with col_sum2:
                                    st.metric("총 소요 시간", f"{total_duration:.1f} 분")
                                
                                # 지도에 최적 경로 적용 버튼
                                if st.button("🗺️ 이 경로를 지도에서 보기", key=f"apply_route_to_map_{day_num}", use_container_width=True):
                                    # 최적 경로를 지도에 적용
                                    # confirmed_plans는 이미 최적 순서로 재정렬되어 있으므로
                                    # use_optimal_route_for_map 플래그를 설정하고 rerun하여 지도가 다시 그려지도록 함
                                    st.session_state[f"use_optimal_route_for_map_{day_num}"] = True
                                    st.success("최적 경로가 지도에 적용되었습니다!")
                                    # 지도가 다시 그려지도록 rerun (이미 재정렬된 confirmed_plans 순서로 지도 생성됨)
                                    st.rerun()
                    else:
                        # 경로 최적화 전 상태 표시
                        with st.container(border=True):
                            st.markdown("#### 🤖 최적화된 경로 정보")
                            st.info("💡 경로 최적화 버튼을 눌러 최적 경로를 계산하세요.")
                
                st.markdown("")
                st.markdown("---")
                st.markdown("")
                
                # 상세 경로 정보 섹션 (챗봇 위쪽, 확정 일정과 최적화된 경로 정보 아래쪽)
                if st.session_state.get(f"show_optimal_route_{day_num}", False) and confirmed_count >= 2:
                    route_info = st.session_state.get(f"route_info_{day_num}", [])
                    if route_info:
                        with st.container(border=True):
                            st.markdown("#### 📋 상세 경로 정보")
                            st.markdown("")
                            
                            for idx, route in enumerate(route_info, 1):
                                route_details = route.get("route_details", [])
                                google_maps_url = route.get("google_maps_url", "")
                                
                                if route_details:
                                    # 각 교통수단별 상세 정보 표시
                                    for detail_idx, detail in enumerate(route_details, 1):
                                        with st.expander(f"구간 {idx}-{detail_idx}: {route.get('from', '')} → {route.get('to', '')} ({detail.get('type', '')})", expanded=False):
                                            if detail["type"] == "버스":
                                                st.markdown(f"**🚌 버스 {detail.get('number', '')}번**")
                                                st.markdown("")
                                                
                                                col_info1, col_info2 = st.columns(2)
                                                with col_info1:
                                                    if detail.get("departure_stop"):
                                                        st.markdown(f"📍 **출발 정류장:** {detail['departure_stop']}")
                                                    if detail.get("arrival_stop"):
                                                        st.markdown(f"📍 **도착 정류장:** {detail['arrival_stop']}")
                                                with col_info2:
                                                    if detail.get("distance_km"):
                                                        st.markdown(f"📏 **이동 거리:** {detail['distance_km']} km")
                                                    if detail.get("duration_min"):
                                                        st.markdown(f"⏱️ **소요 시간:** {detail['duration_min']}분")
                                                
                                                if detail.get("num_stops") is not None:
                                                    st.markdown(f"🚏 **정류장 수:** {detail['num_stops']}개")
                                            
                                            elif detail["type"] == "지하철":
                                                st.markdown(f"**🚇 {detail.get('line', '')} 지하철**")
                                                st.markdown("")
                                                
                                                col_info1, col_info2 = st.columns(2)
                                                with col_info1:
                                                    if detail.get("departure_station"):
                                                        st.markdown(f"📍 **출발역:** {detail['departure_station']}")
                                                    if detail.get("arrival_station"):
                                                        st.markdown(f"📍 **도착역:** {detail['arrival_station']}")
                                                with col_info2:
                                                    if detail.get("distance_km"):
                                                        st.markdown(f"📏 **이동 거리:** {detail['distance_km']} km")
                                                    if detail.get("duration_min"):
                                                        st.markdown(f"⏱️ **소요 시간:** {detail['duration_min']}분")
                                                
                                                if detail.get("num_stops") is not None:
                                                    st.markdown(f"🚏 **역 수:** {detail['num_stops']}개")
                                            
                                            elif detail["type"] == "기차":
                                                st.markdown(f"**🚂 {detail.get('line', '')} 기차**")
                                                st.markdown("")
                                                
                                                col_info1, col_info2 = st.columns(2)
                                                with col_info1:
                                                    if detail.get("departure_station"):
                                                        st.markdown(f"📍 **출발역:** {detail['departure_station']}")
                                                    if detail.get("arrival_station"):
                                                        st.markdown(f"📍 **도착역:** {detail['arrival_station']}")
                                                with col_info2:
                                                    if detail.get("distance_km"):
                                                        st.markdown(f"📏 **이동 거리:** {detail['distance_km']} km")
                                                    if detail.get("duration_min"):
                                                        st.markdown(f"⏱️ **소요 시간:** {detail['duration_min']}분")
                                                
                                                if detail.get("num_stops") is not None:
                                                    st.markdown(f"🚏 **역 수:** {detail['num_stops']}개")
                                            
                                            else:
                                                st.markdown(f"**🚊 {detail.get('line', '대중교통')}**")
                                                st.markdown("")
                                                
                                                col_info1, col_info2 = st.columns(2)
                                                with col_info1:
                                                    if detail.get("departure_station"):
                                                        st.markdown(f"📍 **출발:** {detail['departure_station']}")
                                                    if detail.get("arrival_station"):
                                                        st.markdown(f"📍 **도착:** {detail['arrival_station']}")
                                                with col_info2:
                                                    if detail.get("distance_km"):
                                                        st.markdown(f"📏 **이동 거리:** {detail['distance_km']} km")
                                                    if detail.get("duration_min"):
                                                        st.markdown(f"⏱️ **소요 시간:** {detail['duration_min']}분")
                                            
                                            # Google Maps URL 표시
                                            if google_maps_url:
                                                st.markdown("")
                                                st.markdown(f"[🗺️ Google Maps에서 경로 확인하기]({google_maps_url})")
                                
                                # 도보로 이동하는 경우
                                elif route.get("transport") == "도보" or route.get("추천 교통편") == "도보" or "도보" in str(route.get("transport", "")):
                                    with st.expander(f"구간 {idx}: {route.get('from', '')} → {route.get('to', '')} (도보)", expanded=False):
                                        st.markdown(f"**🚶 도보**")
                                        st.markdown("")
                                        
                                        col_info1, col_info2 = st.columns(2)
                                        with col_info1:
                                            st.markdown(f"📏 **이동 거리:** {route.get('거리(km)', route.get('distance_km', 0))} km")
                                        with col_info2:
                                            st.markdown(f"⏱️ **소요 시간:** {route.get('예상 소요 시간(분)', route.get('duration_min', 0))}분")
                                        
                                        if google_maps_url:
                                            st.markdown("")
                                            st.markdown(f"[🗺️ Google Maps에서 경로 확인하기]({google_maps_url})")
                                
                                # 도보 추천 정보
                                if route.get("도보 추천"):
                                    with st.expander(f"구간 {idx}: {route.get('from', '')} → {route.get('to', '')} (도보 추천)", expanded=False):
                                        st.info(f"🚶 {route['도보 추천']} - 걸어서 이동 가능합니다.")
                                        if google_maps_url:
                                            st.markdown(f"[🗺️ Google Maps에서 경로 확인하기]({google_maps_url})")
                            
                            st.markdown("")
    else:
        st.markdown("")
        st.info("💡 사이드바에서 여행 기간을 입력하면 일정별 탭이 생성됩니다.")


# ============================================
# 3. 함수 실행 파트
# ============================================

if __name__ == "__main__":
    main()