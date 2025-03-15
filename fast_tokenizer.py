import re
import time
import multiprocessing
from multiprocessing import freeze_support
from tqdm import tqdm
import numpy as np
from konlpy.tag import Komoran  # MeCab 대신 Komoran 사용

# 정규식 패턴 미리 컴파일
korean_pattern = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]")

# 불용어 목록 (예시, 필요에 따라 확장)
stop_words = set([
    '이', '그', '저', '것', '수', '를', '에', '의', '가', '은', '는', '들', '좀', '잘', '과', '도', '를', '으로', '자', '에게', '와', '한', '하다'
])

# 전역 변수로 Komoran 객체 선언 (각 프로세스에서 초기화)
komoran_instance = None

def init_worker():
    """워커 프로세스 초기화 함수"""
    global komoran_instance
    komoran_instance = Komoran()

def preprocess_text_worker(text_with_filter):
    """
    워커 프로세스에서 실행되는 텍스트 전처리 함수
    
    Args:
        text_with_filter: (텍스트, 품사 필터) 튜플
    
    Returns:
        tokens: 토큰화된 단어 리스트
    """
    global komoran_instance
    
    # 인자 분리
    text, pos_filter = text_with_filter
    
    # Komoran이 초기화되지 않았으면 초기화
    if komoran_instance is None:
        komoran_instance = Komoran()
    
    # 정규화 (한글 외 문자 제거)
    text = korean_pattern.sub("", text)
    
    # 빈 문자열이면 빈 리스트 반환
    if not text.strip():
        return []
    
    # 형태소 분석
    if pos_filter:
        # 특정 품사만 추출
        morphs = []
        for word, pos in komoran_instance.pos(text):
            if pos in pos_filter:
                morphs.append(word)
    else:
        # 모든 형태소 추출
        morphs = komoran_instance.morphs(text)
    
    # 불용어 제거
    tokens = [word for word in morphs if word not in stop_words and len(word) > 1]
    
    return tokens

def process_batch_worker(batch_with_filter):
    """
    워커 프로세스에서 실행되는 배치 처리 함수
    
    Args:
        batch_with_filter: (텍스트 배치, 품사 필터) 튜플
    
    Returns:
        results: 처리된 토큰 리스트
    """
    batch, pos_filter = batch_with_filter
    return [preprocess_text_worker((text, pos_filter)) for text in batch]

def parallel_tokenize(corpus, pos_filter=None, batch_size=1000, n_jobs=None):
    """
    코퍼스를 병렬로 토큰화합니다.
    
    Args:
        corpus: 텍스트 리스트
        pos_filter: 품사 필터 (예: ['NNG', 'NNP', 'VV', 'VA'])
        batch_size: 배치 크기
        n_jobs: 사용할 프로세스 수 (None이면 CPU 코어 수 - 1)
    
    Returns:
        tokenized_data: 토큰화된 결과 리스트
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    start_time = time.time()
    print(f"총 {len(corpus)}개 텍스트 처리 중... ({n_jobs}개 프로세스 사용)")
    
    # 결과 저장 리스트
    tokenized_data = []
    
    # 배치 단위로 처리
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:min(i+batch_size, len(corpus))]
        print(f"배치 처리 중: {i+1}~{i+len(batch)} / {len(corpus)}")
        
        # 배치를 더 작은 청크로 나누기
        chunk_size = max(1, len(batch) // n_jobs)
        chunks = [batch[j:j+chunk_size] for j in range(0, len(batch), chunk_size)]
        
        # 각 청크에 품사 필터 정보 추가
        chunks_with_filter = [(chunk, pos_filter) for chunk in chunks]
        
        # 멀티프로세싱 풀 생성 및 처리
        with multiprocessing.Pool(processes=n_jobs, initializer=init_worker) as pool:
            # 각 청크를 병렬로 처리
            chunk_results = list(tqdm(
                pool.imap(process_batch_worker, chunks_with_filter),
                total=len(chunks_with_filter),
                desc="청크 처리 중"
            ))
            
            # 결과 합치기
            for result in chunk_results:
                tokenized_data.extend(result)
    
    elapsed_time = time.time() - start_time
    print(f"처리 완료: {len(tokenized_data)}개 텍스트 ({elapsed_time:.2f}초)")
    
    return tokenized_data

def sequential_tokenize(corpus, pos_filter=None):
    """
    코퍼스를 순차적으로 토큰화합니다 (비교용).
    
    Args:
        corpus: 텍스트 리스트
        pos_filter: 품사 필터
    
    Returns:
        tokenized_data: 토큰화된 결과 리스트
    """
    start_time = time.time()
    print(f"순차 처리 시작: 총 {len(corpus)}개 텍스트")
    
    # Komoran 초기화 (한 번만)
    komoran = Komoran()
    
    # 결과 저장 리스트
    tokenized_data = []
    
    # 순차 처리
    for text in tqdm(corpus, desc="텍스트 처리 중"):
        # 정규화 (한글 외 문자 제거)
        text = korean_pattern.sub("", text)
        
        # 빈 문자열이면 빈 리스트 추가
        if not text.strip():
            tokenized_data.append([])
            continue
        
        # 형태소 분석
        if pos_filter:
            # 특정 품사만 추출
            morphs = []
            for word, pos in komoran.pos(text):
                if pos in pos_filter:
                    morphs.append(word)
        else:
            # 모든 형태소 추출
            morphs = komoran.morphs(text)
        
        # 불용어 제거
        tokens = [word for word in morphs if word not in stop_words and len(word) > 1]
        tokenized_data.append(tokens)
    
    elapsed_time = time.time() - start_time
    print(f"순차 처리 완료: {len(tokenized_data)}개 텍스트 ({elapsed_time:.2f}초)")
    
    return tokenized_data

def compare_performance(corpus, sample_size=1000):
    """
    병렬 처리와 순차 처리의 성능을 비교합니다.
    
    Args:
        corpus: 텍스트 리스트
        sample_size: 샘플 크기
    """
    if len(corpus) > sample_size:
        # 랜덤 샘플링
        indices = np.random.choice(len(corpus), sample_size, replace=False)
        sample_corpus = [corpus[i] for i in indices]
    else:
        sample_corpus = corpus
    
    print(f"성능 비교 (샘플 크기: {len(sample_corpus)})")
    print("-" * 50)
    
    # 순차 처리
    sequential_result = sequential_tokenize(sample_corpus)
    
    # 병렬 처리 (다양한 프로세스 수로 테스트)
    for n_jobs in [2, 4, max(1, multiprocessing.cpu_count() - 1)]:
        parallel_result = parallel_tokenize(sample_corpus, n_jobs=n_jobs)
        
        # 결과 일치 확인
        is_equal = len(sequential_result) == len(parallel_result)
        if is_equal:
            for i in range(len(sequential_result)):
                if sequential_result[i] != parallel_result[i]:
                    is_equal = False
                    break
        
        print(f"결과 일치: {is_equal}")
    
    print("-" * 50)

# 사용 예시
if __name__ == "__main__":
    # Windows에서 멀티프로세싱을 위한 코드
    freeze_support()
    
    # 예시 코퍼스
    example_corpus = [
        "안녕하세요 반갑습니다.",
        "한국어 자연어 처리를 빠르게 해봅시다.",
        "병렬 처리를 통해 속도를 높일 수 있습니다.",
        "Komoran은 한국어 형태소 분석기입니다."
    ]
    
    # 명사와 동사만 추출하는 필터
    pos_filter = ['NNG', 'NNP', 'VV', 'VA']
    
    # 병렬 처리로 토큰화
    tokenized_data = parallel_tokenize(example_corpus, pos_filter=pos_filter)
    
    # 결과 출력
    for i, tokens in enumerate(tokenized_data):
        print(f"텍스트 {i+1}: {example_corpus[i]}")
        print(f"토큰: {tokens}")
        print()
    
    # 성능 비교 (선택적)
    # compare_performance(example_corpus) 