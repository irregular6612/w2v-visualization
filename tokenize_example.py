import pandas as pd
import time
from multiprocessing import freeze_support
from fast_tokenizer import parallel_tokenize
import os
from read_korean_parallel import read_corpus_file

# 예시 데이터 로드 (실제 데이터에 맞게 수정)
# 예: corpus = pd.read_csv('your_data.csv')['text_column'].tolist()

def main():
    # 예시 데이터
    filePath = "/Users/bagjuhyeon/Korpora"
    namuwiki_path = os.path.join(filePath, "namuwikitext")
    corpus = read_corpus_file(os.path.join(namuwiki_path, "namuwikitext_20200302.train"))

    # 시작 시간 기록
    start_time = time.time()

    # 명사와 동사만 추출하는 필터 (선택적)
    pos_filter = ['NNG', 'NNP', 'VV', 'VA']

    # 병렬 처리로 토큰화
    # n_jobs: 사용할 CPU 코어 수 (None이면 자동으로 설정)
    # batch_size: 한 번에 처리할 텍스트 수
    tokenized_data = parallel_tokenize(
        corpus, 
        pos_filter=pos_filter,  # 특정 품사만 추출하려면 사용, 아니면 None
        batch_size=1000,        # 메모리 사용량 조절을 위한 배치 크기
        n_jobs=2                # 프로세스 수를 명시적으로 지정 (None이면 CPU 코어 수 - 1)
    )

    # 종료 시간 기록 및 소요 시간 출력
    end_time = time.time()
    print(f"총 처리 시간: {end_time - start_time:.2f}초")
    print(f"텍스트당 평균 처리 시간: {(end_time - start_time) / len(corpus):.4f}초")

    # 결과 확인 (처음 5개)
    for i, tokens in enumerate(tokenized_data[:5]):
        print(f"\n텍스트 {i+1}: {corpus[i] if i < len(corpus) else ''}")
        print(f"토큰: {tokens}")

    # 결과를 DataFrame으로 변환 (선택적)
    result_df = pd.DataFrame({
        'text': corpus,
        'tokens': tokenized_data
    })

    # 토큰 수 통계 (선택적)
    result_df['token_count'] = result_df['tokens'].apply(len)
    print("\n토큰 수 통계:")
    print(result_df['token_count'].describe())

    # 결과 저장 (선택적)
    # result_df.to_csv('tokenized_results.csv', index=False)

    # Word2Vec 등 임베딩 모델 학습을 위한 준비 (선택적)
    # from gensim.models import Word2Vec
    # model = Word2Vec(tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    # model.save("word2vec.model")

if __name__ == "__main__":
    # Windows에서 멀티프로세싱을 위한 코드
    freeze_support()
    main() 