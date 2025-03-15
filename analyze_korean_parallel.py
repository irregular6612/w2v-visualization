import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import numpy as np
import platform

# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux
        # 사용 가능한 한글 폰트 찾기
        font_found = False
        for font in fm.findSystemFonts():
            font_name = fm.FontProperties(fname=font).get_name()
            if any(keyword in font_name.lower() for keyword in ['gothic', '고딕', 'gulim', '굴림', 'batang', '바탕', 'malgun', '맑은']):
                plt.rc('font', family=font_name)
                print(f"한글 폰트 '{font_name}'을(를) 사용합니다.")
                font_found = True
                break
        
        if not font_found:
            print("사용 가능한 한글 폰트를 찾을 수 없습니다. 그래프에서 한글이 깨질 수 있습니다.")
    
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)

# 한글 폰트 설정 적용
set_korean_font()

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

def tokenize_simple(text, language='ko'):
    """간단한 토큰화 함수입니다."""
    # 한국어는 공백 기준으로 토큰화
    if language == 'ko':
        tokens = text.split()
    # 영어는 공백과 일부 특수문자 기준으로 토큰화
    else:
        text = text.lower()
        for char in ',.!?;:()[]{}""\'':
            text = text.replace(char, ' ')
        tokens = [token for token in text.split() if token]
    return tokens

def analyze_corpus(ko_lines, en_lines, dataset_name):
    """코퍼스 데이터를 분석합니다."""
    print(f"\n=== {dataset_name} 분석 결과 ===")
    
    # 기본 통계
    print(f"문장 쌍 수: {len(ko_lines)}")
    
    # 문장 길이 통계
    ko_lengths = [len(line) for line in ko_lines]
    en_lengths = [len(line) for line in en_lines]
    
    ko_token_counts = [len(tokenize_simple(line, 'ko')) for line in ko_lines]
    en_token_counts = [len(tokenize_simple(line, 'en')) for line in en_lines]
    
    print(f"한국어 평균 문자 길이: {np.mean(ko_lengths):.2f} (최소: {min(ko_lengths)}, 최대: {max(ko_lengths)})")
    print(f"영어 평균 문자 길이: {np.mean(en_lengths):.2f} (최소: {min(en_lengths)}, 최대: {max(en_lengths)})")
    print(f"한국어 평균 토큰 수: {np.mean(ko_token_counts):.2f} (최소: {min(ko_token_counts)}, 최대: {max(ko_token_counts)})")
    print(f"영어 평균 토큰 수: {np.mean(en_token_counts):.2f} (최소: {min(en_token_counts)}, 최대: {max(en_token_counts)})")
    
    # 어휘 통계
    ko_tokens = []
    en_tokens = []
    
    for line in ko_lines:
        ko_tokens.extend(tokenize_simple(line, 'ko'))
    
    for line in en_lines:
        en_tokens.extend(tokenize_simple(line, 'en'))
    
    ko_vocab = set(ko_tokens)
    en_vocab = set(en_tokens)
    
    print(f"한국어 고유 토큰 수: {len(ko_vocab)}")
    print(f"영어 고유 토큰 수: {len(en_vocab)}")
    print(f"한국어 총 토큰 수: {len(ko_tokens)}")
    print(f"영어 총 토큰 수: {len(en_tokens)}")
    
    # 가장 많이 등장하는 단어
    ko_counter = Counter(ko_tokens)
    en_counter = Counter(en_tokens)
    
    print("\n한국어 상위 10개 토큰:")
    for token, count in ko_counter.most_common(10):
        print(f"  - {token}: {count}회")
    
    print("\n영어 상위 10개 토큰:")
    for token, count in en_counter.most_common(10):
        print(f"  - {token}: {count}회")
    
    return {
        'ko_lengths': ko_lengths,
        'en_lengths': en_lengths,
        'ko_token_counts': ko_token_counts,
        'en_token_counts': en_token_counts,
        'ko_vocab_size': len(ko_vocab),
        'en_vocab_size': len(en_vocab),
        'ko_tokens': ko_tokens,
        'en_tokens': en_tokens
    }

def plot_statistics(train_stats, dev_stats, test_stats):
    """코퍼스 통계를 시각화합니다."""
    plt.figure(figsize=(15, 10))
    
    # 문장 길이 분포 (토큰 수 기준)
    plt.subplot(2, 2, 1)
    plt.hist(train_stats['ko_token_counts'], bins=30, alpha=0.5, label='한국어 (학습)')
    plt.hist(train_stats['en_token_counts'], bins=30, alpha=0.5, label='영어 (학습)')
    plt.title('문장 길이 분포 (토큰 수)')
    plt.xlabel('토큰 수')
    plt.ylabel('문장 수')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 어휘 크기 비교
    plt.subplot(2, 2, 2)
    datasets = ['학습', '개발', '테스트']
    ko_vocab_sizes = [train_stats['ko_vocab_size'], dev_stats['ko_vocab_size'], test_stats['ko_vocab_size']]
    en_vocab_sizes = [train_stats['en_vocab_size'], dev_stats['en_vocab_size'], test_stats['en_vocab_size']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, ko_vocab_sizes, width, label='한국어')
    plt.bar(x + width/2, en_vocab_sizes, width, label='영어')
    plt.title('데이터셋별 어휘 크기')
    plt.xlabel('데이터셋')
    plt.ylabel('고유 토큰 수')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 한국어 vs 영어 문장 길이 비교 (산점도)
    plt.subplot(2, 2, 3)
    plt.scatter(train_stats['ko_token_counts'][:1000], train_stats['en_token_counts'][:1000], alpha=0.3)
    plt.title('한국어 vs 영어 문장 길이 (토큰 수)')
    plt.xlabel('한국어 토큰 수')
    plt.ylabel('영어 토큰 수')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 상위 10개 단어 빈도
    plt.subplot(2, 2, 4)
    ko_counter = Counter(train_stats['ko_tokens'])
    en_counter = Counter(train_stats['en_tokens'])
    
    ko_top10 = [count for _, count in ko_counter.most_common(10)]
    en_top10 = [count for _, count in en_counter.most_common(10)]
    
    plt.bar(range(1, 11), ko_top10, alpha=0.5, label='한국어')
    plt.bar(range(1, 11), en_top10, alpha=0.5, label='영어')
    plt.title('상위 10개 단어 빈도')
    plt.xlabel('순위')
    plt.ylabel('빈도')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('korean_parallel_stats.png')
    plt.show()

def main():
    # 현재 설정된 폰트 확인
    print(f"현재 설정된 폰트: {plt.rcParams['font.family']}")
    
    # 데이터 로드
    print("데이터 로드 중...")
    ko_train_lines = read_corpus_file(ko_train_file)
    en_train_lines = read_corpus_file(en_train_file)
    ko_dev_lines = read_corpus_file(ko_dev_file)
    en_dev_lines = read_corpus_file(en_dev_file)
    ko_test_lines = read_corpus_file(ko_test_file)
    en_test_lines = read_corpus_file(en_test_file)
    
    # 데이터 분석
    print("데이터 분석 중...")
    train_stats = analyze_corpus(ko_train_lines, en_train_lines, "학습 데이터")
    dev_stats = analyze_corpus(ko_dev_lines, en_dev_lines, "개발 데이터")
    test_stats = analyze_corpus(ko_test_lines, en_test_lines, "테스트 데이터")
    
    # 통계 시각화
    print("\n통계 시각화 중...")
    plot_statistics(train_stats, dev_stats, test_stats)
    print("분석 완료! 'korean_parallel_stats.png' 파일에 시각화 결과가 저장되었습니다.")

if __name__ == "__main__":
    main() 