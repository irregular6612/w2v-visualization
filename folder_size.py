import os
import sys
from datetime import datetime

def get_folder_size(folder_path):
    """
    폴더의 총 크기를 바이트 단위로 계산합니다.
    
    Args:
        folder_path: 크기를 계산할 폴더 경로
        
    Returns:
        total_size: 폴더의 총 크기 (바이트)
    """
    total_size = 0
    
    try:
        # 폴더가 존재하는지 확인
        if not os.path.exists(folder_path):
            print(f"오류: '{folder_path}' 경로가 존재하지 않습니다.")
            return 0
        
        # 폴더가 아닌 경우
        if not os.path.isdir(folder_path):
            return os.path.getsize(folder_path)
        
        # 폴더 내의 모든 항목 탐색
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # 파일 크기 합산
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                # 심볼릭 링크는 건너뛰기
                if not os.path.islink(file_path):
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, FileNotFoundError):
                        # 파일 접근 권한 오류 등 예외 처리
                        pass
    
    except Exception as e:
        print(f"오류 발생: {e}")
    
    return total_size

def format_size(size_in_bytes):
    """
    바이트 단위의 크기를 사람이 읽기 쉬운 형식으로 변환합니다.
    
    Args:
        size_in_bytes: 바이트 단위의 크기
        
    Returns:
        formatted_size: 변환된 크기 문자열
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"

def get_folder_info(folder_path):
    """
    폴더의 정보를 수집합니다.
    
    Args:
        folder_path: 정보를 수집할 폴더 경로
        
    Returns:
        info: 폴더 정보 딕셔너리
    """
    info = {}
    
    try:
        # 폴더 크기 계산
        start_time = datetime.now()
        print(f"'{folder_path}' 폴더 크기 계산 중...")
        size = get_folder_size(folder_path)
        end_time = datetime.now()
        
        # 폴더 내 파일 및 하위 폴더 수 계산
        file_count = 0
        dir_count = 0
        
        for dirpath, dirnames, filenames in os.walk(folder_path):
            file_count += len(filenames)
            dir_count += len(dirnames)
        
        # 정보 저장
        info['path'] = folder_path
        info['size'] = size
        info['formatted_size'] = format_size(size)
        info['file_count'] = file_count
        info['dir_count'] = dir_count
        info['calculation_time'] = (end_time - start_time).total_seconds()
        
    except Exception as e:
        print(f"폴더 정보 수집 중 오류 발생: {e}")
    
    return info

def print_folder_info(info):
    """
    폴더 정보를 출력합니다.
    
    Args:
        info: 폴더 정보 딕셔너리
    """
    if not info:
        print("폴더 정보가 없습니다.")
        return
    
    print("\n" + "=" * 50)
    print(f"폴더 경로: {info['path']}")
    print(f"폴더 크기: {info['formatted_size']} ({info['size']:,} 바이트)")
    print(f"파일 수: {info['file_count']:,}개")
    print(f"하위 폴더 수: {info['dir_count']:,}개")
    print(f"계산 시간: {info['calculation_time']:.2f}초")
    print("=" * 50)

def analyze_subfolders(folder_path, min_size_mb=1):
    """
    폴더 내의 하위 폴더 크기를 분석합니다.
    
    Args:
        folder_path: 분석할 폴더 경로
        min_size_mb: 표시할 최소 크기 (MB)
    """
    print(f"\n'{folder_path}' 내 하위 폴더 분석 중...")
    
    try:
        # 첫 번째 수준의 하위 폴더만 가져오기
        subfolders = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                # 폴더 크기 계산
                size = get_folder_size(item_path)
                # 최소 크기 이상인 경우만 추가
                if size >= min_size_mb * 1024 * 1024:
                    subfolders.append({
                        'path': item_path,
                        'name': item,
                        'size': size,
                        'formatted_size': format_size(size)
                    })
        
        # 크기 기준으로 정렬
        subfolders.sort(key=lambda x: x['size'], reverse=True)
        
        # 결과 출력
        if subfolders:
            print(f"\n=== '{os.path.basename(folder_path)}' 내 하위 폴더 크기 ({min_size_mb}MB 이상) ===")
            for i, folder in enumerate(subfolders, 1):
                print(f"{i}. {folder['name']}: {folder['formatted_size']}")
        else:
            print(f"'{folder_path}' 내에 {min_size_mb}MB 이상 크기의 하위 폴더가 없습니다.")
    
    except Exception as e:
        print(f"하위 폴더 분석 중 오류 발생: {e}")

def main():
    # 기본 경로 설정
    default_path = "/Users/bagjuhyeon/Korpora"
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # 사용자 입력 받기
        folder_path = input(f"크기를 계산할 폴더 경로를 입력하세요 (기본값: {default_path}): ")
        if not folder_path:
            folder_path = default_path
    
    # 폴더 존재 확인
    if not os.path.exists(folder_path):
        print(f"오류: '{folder_path}' 경로가 존재하지 않습니다.")
        return
    
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}'는 폴더가 아닙니다.")
        return
    
    # 폴더 정보 수집 및 출력
    info = get_folder_info(folder_path)
    print_folder_info(info)
    
    # 하위 폴더 분석 여부
    analyze_option = input("\n하위 폴더 분석을 수행하시겠습니까? (y/n): ")
    if analyze_option.lower() == 'y':
        min_size = input("표시할 최소 폴더 크기(MB)를 입력하세요 (기본값: 1): ")
        try:
            min_size = float(min_size) if min_size else 1
        except ValueError:
            min_size = 1
            print("잘못된 입력입니다. 기본값 1MB를 사용합니다.")
        
        analyze_subfolders(folder_path, min_size)

if __name__ == "__main__":
    main() 