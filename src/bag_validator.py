"""
ROS Bag Validator Module
ROS2 bag metadata.yaml 파일의 구조, topic, Hz를 검증하는 모듈
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """에러 추가 (검증 실패)"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """경고 추가 (검증은 통과하지만 주의 필요)"""
        self.warnings.append(message)

    def add_info(self, key: str, value: Any) -> None:
        """정보 추가"""
        self.info[key] = value

    def has_warnings(self) -> bool:
        """경고가 있는지 확인"""
        return len(self.warnings) > 0


class BagValidator:
    """ROS Bag metadata.yaml 검증 클래스"""

    def __init__(self, template_path: Optional[str] = None):
        """
        초기화

        Args:
            template_path: 검증 템플릿 YAML 파일 경로
                          None이면 기본 경로 사용
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # 기본 템플릿 경로
        if template_path is None:
            template_path = Path(__file__).parent.parent / 'config' / 'bag_template.yaml'
        else:
            template_path = Path(template_path)

        # 템플릿 로드
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r') as f:
            self.template = yaml.safe_load(f)

        self.rules = self.template['validation_rules']

    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """
        중첩된 dictionary에서 값 추출

        Args:
            data: dictionary
            key_path: 'rosbag2_bagfile_information.version' 형태의 경로

        Returns:
            추출된 값 또는 None
        """
        keys = key_path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def validate_structure(self, metadata: Dict) -> ValidationResult:
        """
        YAML 구조 검증

        Args:
            metadata: metadata.yaml 내용

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # 필수 키 존재 확인
        for key_path in self.rules['required_keys']:
            value = self._get_nested_value(metadata, key_path)
            if value is None:
                result.add_error(f"Missing required key: '{key_path}'")

        # 제약 조건 확인
        constraints = self.rules['constraints']

        # duration 확인
        duration_ns = self._get_nested_value(metadata, 'rosbag2_bagfile_information.duration.nanoseconds')
        if duration_ns is not None:
            duration_s = duration_ns / 1e9
            result.add_info('duration_seconds', duration_s)

            if duration_s < constraints['min_duration_seconds']:
                result.add_error(f"Duration too short: {duration_s:.2f}s (min: {constraints['min_duration_seconds']}s)")

        # message count 확인
        msg_count = self._get_nested_value(metadata, 'rosbag2_bagfile_information.message_count')
        if msg_count is not None:
            result.add_info('total_message_count', msg_count)

            if msg_count < constraints['min_message_count']:
                result.add_error(f"Message count too low: {msg_count} (min: {constraints['min_message_count']})")

        # version 확인
        version = self._get_nested_value(metadata, 'rosbag2_bagfile_information.version')
        if version is not None and version != constraints['version']:
            result.add_warning(f"Version mismatch: {version} (expected: {constraints['version']})")

        # storage_identifier 확인
        storage = self._get_nested_value(metadata, 'rosbag2_bagfile_information.storage_identifier')
        if storage is not None and storage != constraints['storage_identifier']:
            result.add_warning(f"Storage identifier mismatch: {storage} (expected: {constraints['storage_identifier']})")

        return result

    def validate_topics(self, metadata: Dict) -> ValidationResult:
        """
        Topic 존재 및 타입 검증

        Args:
            metadata: metadata.yaml 내용

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # 실제 topic 리스트 추출
        topics_data = self._get_nested_value(metadata, 'rosbag2_bagfile_information.topics_with_message_count')
        if topics_data is None:
            result.add_error("No topics found in metadata")
            return result

        # topic name -> topic data 매핑
        actual_topics = {}
        for topic_item in topics_data:
            topic_meta = topic_item.get('topic_metadata', {})
            topic_name = topic_meta.get('name')
            if topic_name:
                actual_topics[topic_name] = topic_meta

        result.add_info('actual_topic_count', len(actual_topics))
        result.add_info('actual_topics', list(actual_topics.keys()))

        # 필수 topic 확인
        required_topics = self.rules['required_topics']
        missing_topics = []
        type_mismatches = []

        for req_topic in required_topics:
            topic_name = req_topic['name']
            expected_type = req_topic['type']

            if topic_name not in actual_topics:
                missing_topics.append(topic_name)
            else:
                actual_type = actual_topics[topic_name].get('type')
                if actual_type != expected_type:
                    type_mismatches.append(f"{topic_name}: {actual_type} (expected: {expected_type})")

        # 에러 추가
        if missing_topics:
            result.add_error(f"Missing required topics: {', '.join(missing_topics)}")

        if type_mismatches:
            result.add_error(f"Topic type mismatches: {', '.join(type_mismatches)}")

        # topic 개수 확인
        expected_count = self.rules['constraints']['required_topic_count']
        if len(actual_topics) < expected_count:
            result.add_warning(f"Topic count low: {len(actual_topics)} (expected: {expected_count})")

        return result

    def validate_frequencies(self, metadata: Dict) -> ValidationResult:
        """
        Topic Hz 동적 계산 및 검증

        Args:
            metadata: metadata.yaml 내용

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # duration 추출
        duration_ns = self._get_nested_value(metadata, 'rosbag2_bagfile_information.duration.nanoseconds')
        if duration_ns is None or duration_ns == 0:
            result.add_error("Invalid duration for Hz calculation")
            return result

        duration_s = duration_ns / 1e9

        # topic 데이터 추출
        topics_data = self._get_nested_value(metadata, 'rosbag2_bagfile_information.topics_with_message_count')
        if topics_data is None:
            result.add_error("No topics found for Hz calculation")
            return result

        # topic name -> message count 매핑
        topic_msg_counts = {}
        for topic_item in topics_data:
            topic_meta = topic_item.get('topic_metadata', {})
            topic_name = topic_meta.get('name')
            msg_count = topic_item.get('message_count', 0)
            if topic_name:
                topic_msg_counts[topic_name] = msg_count

        # 각 필수 topic의 Hz 계산 및 검증
        hz_results = {}
        hz_warnings = []

        for req_topic in self.rules['required_topics']:
            topic_name = req_topic['name']
            min_hz = req_topic['min_hz']
            max_hz = req_topic['max_hz']

            if topic_name not in topic_msg_counts:
                continue  # 이미 validate_topics에서 확인됨

            msg_count = topic_msg_counts[topic_name]
            actual_hz = msg_count / duration_s

            hz_results[topic_name] = {
                'actual_hz': actual_hz,
                'min_hz': min_hz,
                'max_hz': max_hz,
                'message_count': msg_count
            }

            # Hz 범위 확인
            if actual_hz < min_hz:
                hz_warnings.append(
                    f"{topic_name}: {actual_hz:.2f} Hz (below minimum {min_hz} Hz, {(actual_hz/min_hz)*100:.1f}% of min)"
                )
            elif actual_hz > max_hz:
                hz_warnings.append(
                    f"{topic_name}: {actual_hz:.2f} Hz (above maximum {max_hz} Hz, {(actual_hz/max_hz)*100:.1f}% of max)"
                )

        result.add_info('hz_results', hz_results)

        # 경고 추가
        for warning in hz_warnings:
            result.add_warning(warning)

        return result

    def validate_bag(self, bag_path: str) -> ValidationResult:
        """
        ROS Bag 통합 검증

        Args:
            bag_path: ROS bag 디렉토리 경로

        Returns:
            ValidationResult
        """
        bag_path = Path(bag_path)
        metadata_path = bag_path / 'metadata.yaml'

        # 최종 결과
        final_result = ValidationResult()

        # metadata.yaml 존재 확인
        if not metadata_path.exists():
            final_result.add_error(f"metadata.yaml not found: {metadata_path}")
            return final_result

        # metadata.yaml 로드
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            final_result.add_error(f"Failed to load metadata.yaml: {e}")
            return final_result

        final_result.add_info('bag_path', str(bag_path))
        final_result.add_info('metadata_path', str(metadata_path))

        # 1. 구조 검증
        structure_result = self.validate_structure(metadata)
        final_result.errors.extend(structure_result.errors)
        final_result.warnings.extend(structure_result.warnings)
        final_result.info.update(structure_result.info)

        if not structure_result.is_valid:
            final_result.is_valid = False
            return final_result  # 구조가 잘못되면 이후 검증 스킵

        # 2. Topic 검증
        topics_result = self.validate_topics(metadata)
        final_result.errors.extend(topics_result.errors)
        final_result.warnings.extend(topics_result.warnings)
        final_result.info.update(topics_result.info)

        if not topics_result.is_valid:
            final_result.is_valid = False

        # 3. Hz 검증 (topic 검증 실패해도 Hz는 계산)
        hz_result = self.validate_frequencies(metadata)
        final_result.warnings.extend(hz_result.warnings)
        final_result.info.update(hz_result.info)

        # 4. 파일 존재 확인
        relative_paths = self._get_nested_value(metadata, 'rosbag2_bagfile_information.relative_file_paths')
        if relative_paths:
            for rel_path in relative_paths:
                db_file = bag_path / rel_path
                if not db_file.exists():
                    final_result.add_error(f"Database file not found: {rel_path}")
                else:
                    final_result.add_info('db_files', relative_paths)

        return final_result

    def save_validation_log(self, result: ValidationResult, log_path: str) -> None:
        """
        검증 결과를 log 파일로 저장
        Save validation result to log file

        Args:
            result: ValidationResult
            log_path: log 파일 경로
        """
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ROS Bag Validation Result\n")
            f.write("=" * 60 + "\n\n")

            # Bag 정보
            if 'bag_path' in result.info:
                f.write(f"Bag Path: {result.info['bag_path']}\n")

            if 'duration_seconds' in result.info:
                f.write(f"Duration: {result.info['duration_seconds']:.2f}s\n")

            if 'total_message_count' in result.info:
                f.write(f"Total Messages: {result.info['total_message_count']}\n")

            f.write("\n")

            # 구조 검증
            if result.errors:
                f.write("ERRORS:\n")
                for error in result.errors:
                    f.write(f"   - {error}\n")
                f.write("\n")

            # 경고
            if result.warnings:
                f.write("WARNINGS:\n")
                for warning in result.warnings:
                    f.write(f"   - {warning}\n")
                f.write("\n")

            # Hz 결과
            if 'hz_results' in result.info:
                f.write("Topic Frequencies:\n")
                hz_results = result.info['hz_results']
                for topic_name, hz_data in hz_results.items():
                    actual = hz_data['actual_hz']
                    min_hz = hz_data['min_hz']
                    max_hz = hz_data['max_hz']

                    # 범위 체크
                    if min_hz <= actual <= max_hz:
                        status = "OK"
                    else:
                        status = "WARNING"

                    f.write(f"   [{status}] {topic_name}: {actual:.2f} Hz (range: {min_hz}-{max_hz} Hz)\n")
                f.write("\n")

            # 최종 결과
            f.write("=" * 60 + "\n")
            if result.is_valid:
                if result.has_warnings():
                    f.write("VALID (with warnings)\n")
                else:
                    f.write("VALID\n")
            else:
                f.write("INVALID\n")
            f.write("=" * 60 + "\n")

    def print_validation_result(self, result: ValidationResult) -> None:
        """
        검증 결과를 보기 좋게 출력

        Args:
            result: ValidationResult
        """
        print("=" * 60)
        print("ROS Bag Validation Result")
        print("=" * 60)

        # Bag 정보
        if 'bag_path' in result.info:
            print(f"Bag Path: {result.info['bag_path']}")

        if 'duration_seconds' in result.info:
            print(f"Duration: {result.info['duration_seconds']:.2f}s")

        if 'total_message_count' in result.info:
            print(f"Total Messages: {result.info['total_message_count']}")

        print()

        # 구조 검증
        if result.errors:
            print("ERRORS:")
            for error in result.errors:
                print(f"   - {error}")
            print()

        # 경고
        if result.warnings:
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"   - {warning}")
            print()

        # Hz 결과
        if 'hz_results' in result.info:
            print("Topic Frequencies:")
            hz_results = result.info['hz_results']
            for topic_name, hz_data in hz_results.items():
                actual = hz_data['actual_hz']
                min_hz = hz_data['min_hz']
                max_hz = hz_data['max_hz']

                # 범위 체크
                if min_hz <= actual <= max_hz:
                    status = "[OK]"
                else:
                    status = "[WARNING]"

                print(f"   {status} {topic_name}: {actual:.2f} Hz (range: {min_hz}-{max_hz} Hz)")
            print()

        # 최종 결과
        print("=" * 60)
        if result.is_valid:
            if result.has_warnings():
                print("VALID (with warnings)")
            else:
                print("VALID")
        else:
            print("INVALID")
        print("=" * 60)
