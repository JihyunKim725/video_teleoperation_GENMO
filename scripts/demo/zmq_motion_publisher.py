# SPDX-License-Identifier: Apache-2.0
"""
ZMQ v3 모션 퍼블리셔

SmplToG1Retargeter.retarget() 출력을 SONIC ZMQ v3 packed binary 메시지로
직렬화하여 g1_deploy에 전송.

메시지 구조 (zmq_planner_sender.py 검증):
    [topic_bytes][1280-byte JSON header][binary payload]
    header: {"v":3, "endian":"le", "count":1, "fields":[...]}
    fields: joint_pos(N,29), joint_vel(N,29), smpl_joints(N,24,3), smpl_pose(N,21,3)

사용법:
    from zmq_motion_publisher import ZmqMotionPublisher

    pub = ZmqMotionPublisher(host="localhost", port=5556)
    pub.start()

    retarget_output = retargeter.retarget(smpl_params, mode="sonic_v3")
    pub.publish_retarget(retarget_output)

    pub.stop()

출처:
    - zmq_planner_sender.py: pack_pose_message(), _build_header(), HEADER_SIZE=1280
    - zmq_poller.py: ZMQPoller — 수신 측 (CONFLATE=1, topic strip)
    - network_utils.py: resolve_interface() — sim/real 네트워크 선택
    - observation_config.yaml: encoder mode "smpl" (mode_id=2)
"""
from __future__ import annotations

import json
import logging
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zmq

logger = logging.getLogger(__name__)

# 출처: zmq_planner_sender.py line 14
HEADER_SIZE = 1280


# ============================================================================
# 설정
# ============================================================================


@dataclass
class ZmqPublisherConfig:
    """ZMQ 퍼블리셔 설정."""

    host: str = "localhost"
    port: int = 5556
    topic: str = "pose"
    protocol_version: int = 3

    # 연결 모드: "bind" (서버) 또는 "connect" (클라이언트)
    # sim2sim: bind (워크스테이션이 서버)
    # real: bind (워크스테이션이 서버, G1이 connect)
    socket_mode: str = "bind"

    # 퍼블리시 루프
    target_fps: int = 30
    queue_maxsize: int = 5

    # 재연결
    reconnect_interval: float = 2.0  # 초

    # 네트워크
    interface: str = "sim"  # "sim" | "real" | IP주소


# ============================================================================
# 메시지 패킹 (zmq_planner_sender.py 기반)
# ============================================================================


def _build_header(fields: list, version: int = 3, count: int = 1) -> bytes:
    """JSON 헤더 생성 + 1280바이트 패딩.

    출처: zmq_planner_sender.py _build_header() (line 15-23)
    """
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def pack_v3_message(
    retarget_output: dict,
    topic: str = "pose",
    version: int = 3,
    frame_index: int = 0,
) -> bytes:
    """리타겟팅 출력 → ZMQ v3 packed binary 메시지.

    출처: video_teleop_genmo.py build_packed_message() — 검증된 v3 필드 순서:
        joint_pos:   [N, 29]     f32  (wrist만 meaningful)
        joint_vel:   [N, 29]     f32
        smpl_joints: [N, 24, 3]  f32  (SMPL FK joint positions)
        smpl_pose:   [N, 21, 3]  f32  (SMPL body pose axis-angle)
        body_quat:   [N, 1, 4]   f32  (w, x, y, z) — global orient quaternion
        frame_index: [N]          i32  (monotonically increasing)

    출처: zmq_python_sender_test.cpp line 266 —
        "Protocol V2 missing required fields (need: smpl_joints, smpl_pose, body_quat, frame_index)"
    """
    # torch → numpy 변환
    def _to_np(val, dtype=np.float32):
        if hasattr(val, "numpy"):
            val = val.detach().cpu().numpy()
        return np.ascontiguousarray(val.astype(dtype))

    N = 1  # 기본값
    jp = _to_np(retarget_output["joint_pos"])
    if jp.ndim == 1:
        jp = jp.reshape(1, -1)
    N = jp.shape[0]

    jv = _to_np(retarget_output["joint_vel"])
    if jv.ndim == 1:
        jv = jv.reshape(1, -1)

    sj = _to_np(retarget_output["smpl_joints"])
    if sj.ndim == 2:
        sj = sj.reshape(1, 24, 3)

    sp = _to_np(retarget_output["smpl_pose"])
    if sp.ndim == 2:
        sp = sp.reshape(1, 21, 3)

    # body_quat: 외부에서 전달 또는 기본값 (identity quaternion w,x,y,z)
    if "body_quat" in retarget_output:
        bq = _to_np(retarget_output["body_quat"])
    else:
        bq = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    if bq.ndim == 1:
        bq = bq.reshape(1, 1, 4)
    elif bq.ndim == 2:
        bq = bq.reshape(N, 1, 4)

    # frame_index
    fi = np.array([frame_index], dtype=np.int32)

    # 필드 순서 (video_teleop_genmo.py와 동일)
    field_list = [
        ("joint_pos",   jp,  "f32"),
        ("joint_vel",   jv,  "f32"),
        ("smpl_joints", sj,  "f32"),
        ("smpl_pose",   sp,  "f32"),
        ("body_quat",   bq,  "f32"),
        ("frame_index", fi,  "i32"),
    ]

    fields = []
    binary_parts = []
    for name, arr, dtype_str in field_list:
        fields.append({"name": name, "dtype": dtype_str, "shape": list(arr.shape)})
        binary_parts.append(arr.tobytes())

    topic_bytes = topic.encode("utf-8")
    header_bytes = _build_header(fields, version=version, count=1)
    payload_bytes = b"".join(binary_parts)

    return topic_bytes + header_bytes + payload_bytes


def pack_command_message(start: bool = False, stop: bool = False, planner: bool = False) -> bytes:
    """command 토픽 메시지 (시작/정지 제어).

    출처: zmq_planner_sender.py build_command_message() (line 30-56)
    """
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = struct.pack("BBB", int(start), int(stop), int(planner))
    header = _build_header(fields, version=1, count=1)
    return b"command" + header + payload


# ============================================================================
# 메시지 언패킹 (디버깅/검증용)
# ============================================================================


def unpack_v3_message(raw: bytes, topic: str = "pose") -> dict:
    """수신된 바이트를 필드별로 분해 (디버깅용).

    출처: zmq_poller.py — 수신 측이 topic prefix를 strip 후 header+payload 파싱

    Args:
        raw: 수신된 전체 바이트 (topic 포함)

    Returns:
        dict: {"header": dict, "fields": {name: ndarray}}
    """
    topic_len = len(topic.encode("utf-8"))
    header_json = raw[topic_len:topic_len + HEADER_SIZE]
    header = json.loads(header_json.rstrip(b"\x00"))

    offset = topic_len + HEADER_SIZE
    fields = {}
    for field_info in header["fields"]:
        name = field_info["name"]
        dtype_str = field_info["dtype"]
        shape = field_info["shape"]

        dtype_map = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "u8": np.uint8}
        dtype = dtype_map.get(dtype_str, np.float32)

        n_elements = 1
        for s in shape:
            n_elements *= s
        n_bytes = n_elements * np.dtype(dtype).itemsize

        arr = np.frombuffer(raw[offset:offset + n_bytes], dtype=dtype).reshape(shape)
        fields[name] = arr
        offset += n_bytes

    return {"header": header, "fields": fields}


# ============================================================================
# 메인 퍼블리셔 클래스
# ============================================================================


class ZmqMotionPublisher:
    """SONIC g1_deploy에 모션 데이터를 전송하는 ZMQ PUB 소켓.

    스레드 구조:
        메인 스레드:  retarget → publish_retarget() → data_queue
        발행 스레드:  data_queue → pack_v3_message() → zmq.send()

    ZMQ 소켓 설정:
        - PUB 소켓 (1:N 브로드캐스트)
        - SNDHWM=1: 최신 메시지만 유지 (수신 측 CONFLATE=1과 대응)
        - LINGER=0: 종료 시 대기 없이 즉시 닫기

    출처:
        - zmq_poller.py: 수신 측 — SUB + CONFLATE=1
        - zmq_planner_sender.py: 송신 메시지 포맷
    """

    def __init__(self, config: Optional[ZmqPublisherConfig] = None):
        self._config = config or ZmqPublisherConfig()
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False
        self._pub_thread: Optional[threading.Thread] = None
        self._data_queue: queue.Queue = queue.Queue(maxsize=self._config.queue_maxsize)

        # 통계
        self._stats = {
            "sent": 0,
            "dropped": 0,
            "errors": 0,
            "bytes_sent": 0,
        }

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """ZMQ 소켓 초기화 + 발행 스레드 시작."""
        logger.info(f"[ZmqPublisher] 시작: {self._config.host}:{self._config.port}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)

        # 소켓 옵션
        self._socket.setsockopt(zmq.SNDHWM, 1)    # 최신 메시지만 유지
        self._socket.setsockopt(zmq.LINGER, 0)     # 즉시 종료

        # 바인드 또는 커넥트
        endpoint = f"tcp://{self._config.host}:{self._config.port}"
        if self._config.socket_mode == "bind":
            # 서버 모드: 워크스테이션에서 bind
            bind_addr = f"tcp://*:{self._config.port}"
            self._socket.bind(bind_addr)
            logger.info(f"[ZmqPublisher] Bind: {bind_addr}")
        else:
            # 클라이언트 모드: G1에 connect
            self._socket.connect(endpoint)
            logger.info(f"[ZmqPublisher] Connect: {endpoint}")

        # 발행 스레드 시작
        self._running = True
        self._pub_thread = threading.Thread(
            target=self._publish_loop, name="ZmqPublisher", daemon=True
        )
        self._pub_thread.start()
        logger.info("[ZmqPublisher] 발행 스레드 시작됨")

    def stop(self):
        """ZMQ 소켓 정리 + 스레드 종료."""
        logger.info("[ZmqPublisher] 종료 중 ...")
        self._running = False

        if self._pub_thread is not None:
            self._pub_thread.join(timeout=3.0)

        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()

        logger.info(
            f"[ZmqPublisher] 종료 완료 — "
            f"전송: {self._stats['sent']}, 드롭: {self._stats['dropped']}, "
            f"에러: {self._stats['errors']}, 총 {self._stats['bytes_sent']/1024:.0f}KB"
        )

    def publish_retarget(self, retarget_output: dict):
        """리타겟팅 결과를 발행 큐에 넣기.

        Args:
            retarget_output: SmplToG1Retargeter.retarget() 반환값
        """
        if not self._running:
            return

        try:
            self._data_queue.put_nowait(retarget_output)
        except queue.Full:
            # 오래된 데이터 드롭 (실시간 우선)
            try:
                self._data_queue.get_nowait()
                self._stats["dropped"] += 1
            except queue.Empty:
                pass
            self._data_queue.put_nowait(retarget_output)

    def send_command(self, start: bool = False, stop: bool = False, planner: bool = False):
        """command 토픽 전송 (시작/정지 제어).

        출처: zmq_planner_sender.py build_command_message()
        """
        if self._socket is None:
            return
        msg = pack_command_message(start=start, stop=stop, planner=planner)
        try:
            self._socket.send(msg, zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logger.warning(f"[ZmqPublisher] command 전송 실패: {e}")

    def _publish_loop(self):
        """발행 스레드 메인 루프.

        data_queue에서 리타겟팅 결과를 꺼내 → v3 메시지로 패킹 → ZMQ 전송.
        GENMO 결과 대기 중에도 기본 자세를 전송 (video_teleop_genmo.py 방식).
        """
        interval = 1.0 / self._config.target_fps
        frame_index = 0

        # 기본 자세 (GENMO 결과 대기 중 전송)
        # 출처: video_teleop_genmo.py DEFAULT_JP/JV/SP/SJ/BQ
        default_output = {
            "joint_pos": np.zeros((1, 29), dtype=np.float32),
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "smpl_joints": np.zeros((1, 24, 3), dtype=np.float32),
            "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
            "body_quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }

        print("Waiting for the first valid inference frame ...")

        retarget_output = self._data_queue.get(block=True)
        last_frame = self._extract_last_frame(retarget_output)
        print("First frame received! Starting streaming.")

        while self._running:
            # 큐에서 데이터 가져오기 (없으면 기본 자세 전송)
            try:
                retarget_output = self._data_queue.get_nowait()
                last_frame = self._extract_last_frame(retarget_output)
            except queue.Empty:
                pass
                # last_frame = default_output

            try:
                msg = pack_v3_message(
                    last_frame,
                    topic=self._config.topic,
                    version=self._config.protocol_version,
                    frame_index=frame_index,
                )
                self._socket.send(msg, zmq.NOBLOCK)
                self._stats["sent"] += 1
                self._stats["bytes_sent"] += len(msg)
                frame_index += 1

            except zmq.ZMQError as e:
                self._stats["errors"] += 1
                logger.warning(f"[ZmqPublisher] 전송 에러: {e}")
                self._try_reconnect()

            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"[ZmqPublisher] 패킹 에러: {e}")

            time.sleep(interval)

    @staticmethod
    def _extract_last_frame(retarget_output: dict) -> dict:
        """N프레임 결과에서 마지막 1프레임만 추출.

        SONIC은 1프레임씩 소비하므로, 최신 프레임만 전송.
        """
        result = {}
        for key, val in retarget_output.items():
            if hasattr(val, "shape") and len(val.shape) >= 1 and val.shape[0] > 1:
                result[key] = val[-1:]  # (1, ...) — 마지막 프레임, 배치 차원 유지
            else:
                result[key] = val
        return result

    def _try_reconnect(self):
        """ZMQ 소켓 재연결 시도."""
        logger.info("[ZmqPublisher] 재연결 시도 ...")
        try:
            if self._socket is not None:
                self._socket.close()

            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, 1)
            self._socket.setsockopt(zmq.LINGER, 0)

            if self._config.socket_mode == "bind":
                self._socket.bind(f"tcp://*:{self._config.port}")
            else:
                self._socket.connect(
                    f"tcp://{self._config.host}:{self._config.port}"
                )
            logger.info("[ZmqPublisher] 재연결 성공")

        except zmq.ZMQError as e:
            logger.error(f"[ZmqPublisher] 재연결 실패: {e}")
            time.sleep(self._config.reconnect_interval)


# ============================================================================
# CLI 테스트
# ============================================================================


def main():
    """독립 테스트: 합성 데이터로 ZMQ 퍼블리시 + 수신 검증."""
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ v3 모션 퍼블리셔 테스트")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--receiver", action="store_true", help="수신 테스트 모드")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.receiver:
        _run_receiver_test(args.host, args.port, args.duration)
    else:
        _run_publisher_test(args.host, args.port, args.duration)


def _run_publisher_test(host: str, port: int, duration: float):
    """퍼블리셔 테스트: 합성 데이터를 지속적으로 발행."""
    import torch

    logger.info("=== ZMQ 퍼블리셔 테스트 ===")

    config = ZmqPublisherConfig(host=host, port=port)
    pub = ZmqMotionPublisher(config)
    pub.start()

    # command 전송: 시작
    pub.send_command(start=True)
    logger.info("  command: start=True 전송")

    t0 = time.time()
    count = 0
    try:
        while time.time() - t0 < duration:
            # 합성 리타겟팅 출력
            retarget_output = {
                "joint_pos": torch.randn(1, 29) * 0.1,
                "joint_vel": torch.randn(1, 29) * 0.01,
                "smpl_joints": torch.randn(1, 24, 3) * 0.5,
                "smpl_pose": torch.randn(1, 21, 3) * 0.3,
            }
            pub.publish_retarget(retarget_output)
            count += 1
            time.sleep(1.0 / 30)
    except KeyboardInterrupt:
        pass

    # command: 정지
    pub.send_command(stop=True)
    pub.stop()

    logger.info(f"  {count}프레임 발행, {time.time()-t0:.1f}초")
    logger.info(f"  통계: {pub.stats}")

    # 메시지 패킹 검증
    logger.info("\n=== 메시지 검증 ===")
    test_output = {
        "joint_pos": np.zeros((1, 29), dtype=np.float32),
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "smpl_joints": np.zeros((1, 24, 3), dtype=np.float32),
        "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
        "body_quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    msg = pack_v3_message(test_output, topic="pose", version=3, frame_index=42)
    parsed = unpack_v3_message(msg, topic="pose")

    errors = []
    expected_shapes = {
        "joint_pos": (1, 29),
        "joint_vel": (1, 29),
        "smpl_joints": (1, 24, 3),
        "smpl_pose": (1, 21, 3),
        "body_quat": (1, 1, 4),
        "frame_index": (1,),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(parsed["fields"][key].shape)
        ok = actual == expected
        logger.info(f"  {'✓' if ok else '✗'} {key}: {actual} == {expected}")
        if not ok:
            errors.append(f"{key}: {actual} != {expected}")

    # 헤더 검증
    h = parsed["header"]
    logger.info(f"  ✓ version: {h['v']}")
    logger.info(f"  ✓ endian: {h['endian']}")
    logger.info(f"  ✓ fields: {len(h['fields'])}개")

    # 바이트 크기 검증
    expected_size = (
        len("pose")
        + HEADER_SIZE
        + (1*29 + 1*29 + 1*24*3 + 1*21*3) * 4  # f32 fields
        + (1*1*4) * 4                            # body_quat f32
        + 1 * 4                                  # frame_index i32
    )
    logger.info(f"  ✓ 메시지 크기: {len(msg)} bytes (예상: {expected_size})")
    if len(msg) != expected_size:
        errors.append(f"size: {len(msg)} != {expected_size}")

    logger.info(f"\n{'='*50}")
    logger.info(f"결과: {'ALL PASSED' if not errors else f'FAIL — {len(errors)}개'}")


def _run_receiver_test(host: str, port: int, duration: float):
    """수신 테스트: ZMQ SUB로 메시지 수신 및 파싱."""
    logger.info(f"=== ZMQ 수신 테스트 (tcp://{host}:{port}) ===")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://{host}:{port}")

    t0 = time.time()
    count = 0
    try:
        while time.time() - t0 < duration:
            if socket.poll(timeout=100):
                raw = socket.recv(zmq.NOBLOCK)
                parsed = unpack_v3_message(raw, topic="pose")
                count += 1
                if count <= 3 or count % 30 == 0:
                    shapes = {k: v.shape for k, v in parsed["fields"].items()}
                    logger.info(f"  #{count}: {shapes}")
    except KeyboardInterrupt:
        pass

    socket.close()
    context.term()
    logger.info(f"  {count}메시지 수신, {time.time()-t0:.1f}초")


if __name__ == "__main__":
    main()
