# Aria Camera Streaming (TCP)

Docker 컨테이너 또는 원격 서버에서 Aria 카메라 RGB 이미지와 gaze point를 TCP로 스트리밍하여 로컬에서 받는 시스템입니다.

## 아키텍처

```
┌─────────────────────┐                    ┌──────────────────────┐
│  Remote Server      │      TCP/IP        │   Local Machine      │
│  (Aria Glasses)     │  ◄────────────►    │                      │
│                     │                    │   AriaCamera Client  │
│  (sends RGB+gaze)   │   port: 9999       │   (receives data)    │
└─────────────────────┘                    └──────────────────────┘
```

## 프로토콜

### Server → Client (TCP)
**Packet Structure:**
1. Header (4 bytes): `payload_size` (unsigned int)
2. Payload:
   - `timestamp` (8 bytes, uint64): 타임스탬프
   - `rgb_size` (4 bytes, uint32): RGB 이미지 크기
   - `rgb_data` (variable): JPEG 압축된 RGB 이미지
   - `gaze_x` (4 bytes, float): Gaze X 좌표 (0~1408)
   - `gaze_y` (4 bytes, float): Gaze Y 좌표 (0~1408)

**처리 과정:**
1. RGB 이미지 디코딩 (JPEG → numpy array)
2. BGR → RGB 변환
3. Gaze point 회전 (cw_180) 및 스케일링 (1408 → 256)
4. Gaze overlay 생성

## 특징

### TCP 사용 이유
- ✅ 신뢰성: 패킷 손실 없음
- ✅ 순서 보장: 프레임 순서 유지
- ✅ 간단한 구현: 멀티패킷 처리 불필요
- ✅ Aria 초기화 대기: 30초 타임아웃 지원

### Aria 초기화 처리
- 첫 프레임 수신까지 최대 30초 대기
- 대기 중 진행 상황 표시
- 초기화 완료 후 정상 동작

## 설치

```bash
cd /home/alin/projects/piper_project/edith
pip install -e .
```

필요한 패키지:
- numpy
- opencv-python (cv2)

## 사용법

### 1. 서버 실행 (Aria 안경이 연결된 머신)

서버는 Aria 안경에서 데이터를 읽어 TCP로 전송합니다.

**Note**: 실제 Aria 서버 구현은 사용자의 Aria 안경 인터페이스에 맞게 작성해야 합니다.

### 2. 로컬에서 클라이언트 실행

#### 시각화 도구 사용 (권장)

Gaze point overlay와 FPS 카운터를 포함한 실시간 시각화:

```bash
# 기본 사용
python aria_camera_vis.py --host 192.168.1.100 --port 9999

# 회전 모드 지정
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --rotation cw_90

# 비디오 녹화
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --save-video output.mp4

# FPS 카운터 숨기기
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --no-fps
```

**키보드 명령:**
- `q`: 종료
- `r`: 녹화 상태 토글
- `s`: 현재 프레임 저장 (PNG)

#### 테스트 스크립트 사용

```bash
# 로컬 테스트
python test_aria_camera.py --host localhost --port 9999

# 원격 서버 연결
python test_aria_camera.py --host 192.168.1.100 --port 9999
```

**Note**: 서버가 먼저 실행되어 있어야 합니다!

**Aria 초기화**: 첫 프레임 수신까지 최대 30초 대기합니다.

#### Python 코드에서 사용

```python
from edith.sensors.aria_camera import AriaCamera

# 카메라 초기화 (첫 프레임이 올 때까지 블로킹, 최대 30초 대기)
camera = AriaCamera(
    host="192.168.1.100",      # Aria 서버 IP
    port=9999,                 # Aria 서버 포트
    timeout=30,                # Aria 초기화 대기 시간
    visualize=False,           # 시각화 비활성화
    original_size=1408,        # 원본 이미지 크기
    target_size=256,           # 타겟 이미지 크기
    rotation='cw_90'           # Gaze 좌표 회전
)

# 최신 데이터 가져오기
while True:
    # 방법 1: 관측값 가져오기 (권장)
    rgb_image, gaze_point = camera.get_observation()
    # rgb_image: (256, 256, 3) - RGB 이미지 (BGR 형식)
    # gaze_point: (2,) numpy array [x, y] - 0~256 범위
    
    # 방법 2: 개별 메서드
    rgb_image = camera.get_image()        # RGB 이미지만
    gaze_point = camera.get_gaze_point()  # (2,) numpy array [x, y]
    timestamp = camera.get_timestamp()    # 타임스탬프
    
    if rgb_image is not None:
        # 이미지 처리 또는 로봇 제어
        # Note: 이미지는 BGR 형식입니다 (cv2.imread와 동일)
        pass
    
    # 녹화 상태 제어 (향후 사용)
    camera.set_traj_record_state(True)
    camera.set_traj_ended(False)

# 종료
camera.stop()
```

### 3. 환경 설정에 통합

```python
# piper_env_gaze.py에서
from edith.sensors.aria_camera import AriaCamera

class PiperEnvGaze:
    def __init__(self):
        # Aria 카메라 추가
        self.aria_camera = AriaCamera(
            host=os.getenv("ARIA_HOST", "192.168.1.100"),
            port=int(os.getenv("ARIA_PORT", "9999")),
            timeout=30,  # Aria 초기화 대기
            rotation='cw_90'
        )
    
    def get_observation(self):
        obs = {}
        
        # Aria 데이터 가져오기
        human_view, gaze_point = self.aria_camera.get_observation()
        
        if human_view is not None:
            obs["human_view"] = human_view      # RGB 이미지 (BGR 형식)
            obs["gaze_point"] = gaze_point      # (x, y) 좌표
        
        # ... 다른 센서 (RealSense 등) ...
        
        return obs
```

## 시각화 도구 (aria_camera_vis.py)

### 기능

`aria_camera_vis.py`는 Aria 카메라 데이터를 실시간으로 시각화하는 도구입니다:

- ✅ Gaze point overlay (십자선 + 원)
- ✅ FPS 카운터
- ✅ Gaze 좌표 표시
- ✅ 녹화 상태 표시
- ✅ 비디오 녹화 기능
- ✅ 프레임 저장 기능

### 사용 예제

```bash
# 기본 사용
python aria_camera_vis.py --host 192.168.1.100 --port 9999

# 비디오 녹화
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --save-video aria_recording.mp4

# 회전 모드 변경
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --rotation cw_180

# FPS 카운터 숨기기
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --no-fps
```

### 키보드 단축키

| 키 | 기능 |
|----|------|
| `q` | 종료 |
| `r` | 녹화 상태 토글 (REC 표시) |
| `s` | 현재 프레임을 PNG로 저장 |

### 커맨드라인 옵션

```
--host HOST          서버 호스트 (기본: 137.68.92.28)
--port PORT          서버 포트 (기본: 9999)
--rotation ROTATION  Gaze 좌표 회전 (cw_90, cw_180, cw_270, none)
--no-fps            FPS 카운터 숨기기
--save-video FILE   비디오 파일로 저장 (예: output.mp4)
```

## Docker 설정

### 네트워크 설정

#### 방화벽 설정 (필요시)

```bash
# TCP 포트 허용
sudo ufw allow 9999/tcp

# 특정 IP에서만 허용
sudo ufw allow from 192.168.1.0/24 to any port 9999
```

#### 원격 연결 테스트

```bash
# 서버 포트 확인
netstat -an | grep 9999

# 연결 테스트
telnet 192.168.1.100 9999
```

## 성능 최적화

### 1. JPEG 압축 품질 조정 (서버 측)

서버 코드에서:
```python
# 낮은 품질 (빠름, 작은 크기)
_, rgb_jpeg = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

# 중간 품질 (균형, 권장)
_, rgb_jpeg = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

# 높은 품질 (느림, 큰 크기)
_, rgb_jpeg = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

### 2. 이미지 크기 조정

클라이언트 초기화 시:
```python
camera = AriaCamera(
    host="localhost",
    port=9999,
    original_size=1408,    # Aria 원본 크기
    target_size=256,       # 다운샘플링 크기
)

# 더 작은 이미지 (처리 속도 향상)
camera = AriaCamera(target_size=128)  # 4배 빠름
```

### 3. TCP 최적화

클라이언트/서버 소켓 설정:
```python
# TCP_NODELAY (Nagle 알고리즘 비활성화 - 지연 감소)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# 송수신 버퍼 크기 증가
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2*1024*1024)  # 2MB
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)  # 2MB
```

### 4. Gaze 처리 간소화

필요 없는 경우 gaze overlay 생성 생략:
```python
# get_image() 대신 get_rgb_image()만 사용
rgb_image = camera.get_rgb_image()  # Gaze overlay 없이
```

## TCP 성능 특징

| 특징 | 설명 |
|------|------|
| 지연시간 | 20-50ms (네트워크 상태에 따라) |
| 신뢰성 | 100% (패킷 손실 없음) |
| 순서 | 보장됨 |
| 대역폭 | ~5-10 Mbps @ 256x256, JPEG 85% |
| 권장 환경 | 유선 또는 안정적인 WiFi |

## 문제 해결

### "No data received within 30 seconds"

Aria 안경이 초기화되지 않았거나 서버 연결 문제일 수 있습니다.

1. **서버 상태 확인**:
   ```bash
   # 서버가 실행 중인지 확인
   ps aux | grep aria
   ```

2. **포트 확인**:
   ```bash
   # TCP 포트가 열려있는지 확인
   netstat -an | grep 9999
   # 또는
   ss -tln | grep 9999
   ```

3. **방화벽 설정**:
   ```bash
   sudo ufw allow 9999/tcp
   ```

4. **연결 테스트**:
   ```bash
   # 서버로 연결 테스트
   telnet <server_ip> 9999
   # 또는
   nc -v <server_ip> 9999
   ```

5. **Aria 안경 상태**:
   - Aria 안경이 켜져 있는지 확인
   - USB 연결 확인
   - Aria 소프트웨어 실행 확인

### 프레임 지연 또는 느린 전송

TCP는 신뢰성을 위해 지연이 발생할 수 있습니다:

1. **네트워크 품질 개선**:
   - WiFi 대신 유선 연결 사용
   - 네트워크 대역폭 확인

2. **JPEG 압축 품질 낮추기** (서버에서):
   ```python
   [cv2.IMWRITE_JPEG_QUALITY, 70]  # 기본 85-90
   ```

3. **TCP 버퍼 크기 조정**:
   ```python
   # 서버 측
   sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
   
   # 클라이언트 측 (aria_camera.py)
   sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
   ```

4. **TCP_NODELAY 설정** (Nagle 알고리즘 비활성화):
   ```python
   sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
   ```

### 연결이 자주 끊김

1. **타임아웃 조정**:
   ```python
   camera = AriaCamera(timeout=60)  # 더 긴 타임아웃
   ```

2. **재연결 로직 확인**:
   - 클라이언트는 자동으로 재연결 시도
   - 서버 로그 확인

3. **방화벽/라우터 설정**:
   - Keep-alive 패킷 활성화
   - NAT timeout 증가

### 이미지가 표시되지 않음

1. OpenCV 설치 확인:
   ```bash
   pip install opencv-python
   ```

2. Display 환경 변수 (Linux):
   ```bash
   export DISPLAY=:0
   ```

3. 수동 표시:
   ```python
   camera = AriaCamera(visualize=False)
   # 수동으로 표시
   ```

## API 레퍼런스

### AriaCamera

#### `__init__(serial_number, host, port, timeout, visualize, original_size, target_size, rotation)`

- `serial_number` (str): 카메라 식별자 (기본: "ARIA_CAMERA")
- `host` (str): 서버 호스트 주소
- `port` (int): 서버 포트 (기본: 9999)
- `timeout` (int): 초기화 타임아웃 초 (기본: 60, Aria는 30 권장)
- `visualize` (bool): 실시간 표시 여부 (기본: False)
- `original_size` (int): 원본 이미지 크기 (기본: 1408)
- `target_size` (int): 타겟 이미지 크기 (기본: 256)
- `rotation` (str): Gaze 좌표 회전 방식 (기본: 'cw_180')
  - 'cw_90': 시계방향 90도
  - 'cw_180': 시계방향 180도
  - 'cw_270' 또는 'ccw_90': 반시계방향 90도
  - 'none': 회전 없음

#### 메서드

**데이터 가져오기:**
- `get_observation()`: **(rgb_image, gaze_point)** 튜플 반환 (권장)
  - rgb_image: (256, 256, 3) BGR numpy array
  - gaze_point: (2,) numpy array [x, y] (0~256 범위)
- `get_image()`: RGB 이미지만 반환 (BGR 형식)
- `get_gaze_point()`: Gaze 좌표 (x, y) 반환 (0~256 스케일)
- `get_timestamp()`: 타임스탬프 반환

**이미지 저장:**
- `save_gaze(filename)`: Gaze point가 그려진 이미지를 파일로 저장
  - filename: 저장할 파일명 (기본: "aria_gaze.png")
  - 반환: bool (성공 여부)

**녹화 상태 관리:**
- `set_traj_record_state(state)`: 녹화 상태 설정
- `get_traj_record_state()`: 녹화 상태 가져오기
- `set_traj_ended(state)`: 궤적 종료 상태 설정
- `get_traj_ended()`: 궤적 종료 상태 가져오기

**연결 관리:**
- `stop()`: 수신 스레드 종료 및 정리

## 빠른 시작 요약

### 시각화 사용 (추천)

```bash
# Gaze overlay와 FPS 카운터가 포함된 실시간 시각화
python aria_camera_vis.py --host 192.168.1.100 --port 9999

# 비디오 녹화
python aria_camera_vis.py --host 192.168.1.100 --port 9999 --save-video output.mp4
```

### Python API 사용

```python
# 1. AriaCamera 초기화
from edith.sensors.aria_camera import AriaCamera

camera = AriaCamera(
    host="192.168.1.100",
    port=9999,
    timeout=30,      # Aria 초기화 대기
    rotation='cw_90' # Gaze 좌표 회전
)

# 2. 관측값 가져오기
rgb_image, gaze_point = camera.get_observation()
# rgb_image: (256, 256, 3) - BGR 형식
# gaze_point: (2,) numpy array [x, y] (0~256)

# 3. 개별 데이터 가져오기
rgb_image = camera.get_image()        # BGR 이미지
gaze_point = camera.get_gaze_point()  # (x, y) 좌표
timestamp = camera.get_timestamp()    # 타임스탬프

# 4. 이미지 저장 (선택사항)
camera.save_gaze("snapshot.png")  # Gaze overlay된 이미지 저장

# 5. 종료
camera.stop()
```

### 간단한 이미지 저장 예제

```python
from edith.sensors.aria_camera import AriaCamera
import time

camera = AriaCamera(host="192.168.1.100", port=9999)

# 5초마다 스냅샷 저장
for i in range(10):
    time.sleep(5)
    camera.save_gaze(f"aria_snapshot_{i:03d}.png")
    print(f"Saved snapshot {i}")

camera.stop()
```

### 실시간 이미지 뷰어

저장된 `aria_gaze.png` 파일을 실시간으로 모니터링:

```bash
# 간단한 버전 (프로젝트 루트에서)
python view_aria_gaze.py

# 고급 버전 (커스터마이징 가능)
python edith/src/edith/sensors/aria_gaze_viewer.py
python aria_gaze_viewer.py --file /path/to/image.png --interval 0.1
```

**동작:**
- 파일이 업데이트될 때마다 자동으로 새로고침
- `q` 키로 종료
- 파일이 없으면 대기

## 라이센스

이 코드는 프로젝트 라이센스를 따릅니다.
