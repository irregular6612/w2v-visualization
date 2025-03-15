import os

# 코퍼스 파일 경로 설정
corpus_dir = "/Users/bagjuhyeon/Korpora/korean_parallel"
ko_train_file = os.path.join(corpus_dir, "korean-english-park.train.ko")
en_train_file = os.path.join(corpus_dir, "korean-english-park.train.en")
ko_dev_file = os.path.join(corpus_dir, "korean-english-park.dev.ko")
en_dev_file = os.path.join(corpus_dir, "korean-english-park.dev.en")

def read_corpus_file(file_path, encoding='utf-8'):
    """코퍼스 파일을 읽어 라인 리스트로 반환합니다."""
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def print_parallel_samples(ko_lines, en_lines, num_samples=5):
    """한국어-영어 병렬 샘플을 출력합니다."""
    print(f"총 문장 쌍 수: {len(ko_lines)}")
    print("\n=== 샘플 문장 쌍 ===")
    
    for i in range(min(num_samples, len(ko_lines))):
        print(f"\n[샘플 {i+1}]")
        print(f"한국어: {ko_lines[i]}")
        print(f"영어: {en_lines[i]}")

def main():
    # 학습 데이터 읽기
    print("학습 데이터 읽는 중...")
    ko_train_lines = read_corpus_file(ko_train_file)
    en_train_lines = read_corpus_file(en_train_file)
    
    # 개발 데이터 읽기
    print("개발 데이터 읽는 중...")
    ko_dev_lines = read_corpus_file(ko_dev_file)
    en_dev_lines = read_corpus_file(en_dev_file)
    
    # 학습 데이터 샘플 출력
    print("\n=== 학습 데이터 ===")
    print_parallel_samples(ko_train_lines, en_train_lines)
    
    # 개발 데이터 샘플 출력
    print("\n=== 개발 데이터 ===")
    print_parallel_samples(ko_dev_lines, en_dev_lines)
    
    # 특정 키워드로 검색
    keyword = "컴퓨터"
    print(f"\n=== '{keyword}' 키워드 검색 결과 ===")
    
    found = False
    for i, (ko, en) in enumerate(zip(ko_train_lines, en_train_lines)):
        if keyword in ko:
            print(f"\n[검색 결과 {i+1}]")
            print(f"한국어: {ko}")
            print(f"영어: {en}")
            found = True
            
            # 5개만 출력
            if i >= 4:
                break
    
    if not found:
        print(f"'{keyword}' 키워드를 포함하는 문장을 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 