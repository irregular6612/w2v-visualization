import os
import sys
import argparse

# 코퍼스 파일 경로 설정
corpus_dir = "/Users/bagjuhyeon/Korpora/korean_parallel"
ko_train_file = os.path.join(corpus_dir, "korean-english-park.train.ko")
en_train_file = os.path.join(corpus_dir, "korean-english-park.train.en")
ko_dev_file = os.path.join(corpus_dir, "korean-english-park.dev.ko")
en_dev_file = os.path.join(corpus_dir, "korean-english-park.dev.en")
ko_test_file = os.path.join(corpus_dir, "korean-english-park.test.ko")
en_test_file = os.path.join(corpus_dir, "korean-english-park.test.en")

def read_corpus_file(file_path, encoding='utf-8'):
    """코퍼스 파일을 읽어 라인 리스트로 반환합니다."""
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def search_in_corpus(ko_lines, en_lines, keyword, language='ko', max_results=10):
    """코퍼스에서 키워드를 검색하고 결과를 출력합니다."""
    results = []
    
    for i, (ko, en) in enumerate(zip(ko_lines, en_lines)):
        if (language == 'ko' and keyword in ko) or (language == 'en' and keyword in en):
            results.append((i, ko, en))
            if len(results) >= max_results:
                break
    
    return results

def print_search_results(results, keyword, language):
    """검색 결과를 출력합니다."""
    if not results:
        print(f"'{keyword}' 키워드를 포함하는 문장을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(results)}개의 결과를 찾았습니다.")
    
    for i, (idx, ko, en) in enumerate(results):
        print(f"\n[검색 결과 {i+1}] (원본 인덱스: {idx})")
        
        # 키워드 강조 표시
        if language == 'ko':
            highlighted_ko = ko.replace(keyword, f"\033[1m\033[91m{keyword}\033[0m")
            print(f"한국어: {highlighted_ko}")
            print(f"영어: {en}")
        else:
            highlighted_en = en.replace(keyword, f"\033[1m\033[91m{keyword}\033[0m")
            print(f"한국어: {ko}")
            print(f"영어: {highlighted_en}")

def main():
    parser = argparse.ArgumentParser(description='한국어-영어 병렬 코퍼스 검색 도구')
    parser.add_argument('keyword', help='검색할 키워드')
    parser.add_argument('--lang', choices=['ko', 'en'], default='ko', help='검색할 언어 (ko: 한국어, en: 영어)')
    parser.add_argument('--dataset', choices=['train', 'dev', 'test', 'all'], default='all', help='검색할 데이터셋')
    parser.add_argument('--max', type=int, default=10, help='최대 검색 결과 수')
    
    args = parser.parse_args()
    
    # 데이터셋 선택
    datasets = []
    if args.dataset == 'train' or args.dataset == 'all':
        ko_train_lines = read_corpus_file(ko_train_file)
        en_train_lines = read_corpus_file(en_train_file)
        datasets.append(('학습 데이터', ko_train_lines, en_train_lines))
    
    if args.dataset == 'dev' or args.dataset == 'all':
        ko_dev_lines = read_corpus_file(ko_dev_file)
        en_dev_lines = read_corpus_file(en_dev_file)
        datasets.append(('개발 데이터', ko_dev_lines, en_dev_lines))
    
    if args.dataset == 'test' or args.dataset == 'all':
        ko_test_lines = read_corpus_file(ko_test_file)
        en_test_lines = read_corpus_file(en_test_file)
        datasets.append(('테스트 데이터', ko_test_lines, en_test_lines))
    
    # 검색 실행
    total_results = []
    
    for dataset_name, ko_lines, en_lines in datasets:
        print(f"\n=== {dataset_name} 검색 결과 ===")
        results = search_in_corpus(ko_lines, en_lines, args.keyword, args.lang, args.max)
        print_search_results(results, args.keyword, args.lang)
        total_results.extend(results)
        
        if len(total_results) >= args.max:
            break
    
    print(f"\n총 {len(total_results)}개의 결과를 찾았습니다.")

if __name__ == "__main__":
    main() 