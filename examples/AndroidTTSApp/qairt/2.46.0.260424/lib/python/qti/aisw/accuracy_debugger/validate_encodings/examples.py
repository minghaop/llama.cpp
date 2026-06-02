#!/usr/bin/env python3
"""
Example usage of the ValidateEncodings class.

This script demonstrates various ways to use the ValidateEncodings class
for validating AIMET encoding files.
"""

import json
import tempfile
from pathlib import Path

from qti.aisw.accuracy_debugger.encodings.encodings import TensorEncoding
from qti.aisw.accuracy_debugger.validate_encodings import ValidateEncodings
from qti.aisw.accuracy_debugger.validate_encodings.validation_rules import (
    rule_all_weights_8bit,
    rule_conv_weights_symmetric,
    rule_rmsnorm_weights_16bit,
)


def create_sample_encoding():
    """Create a sample AIMET encoding file for testing."""
    encoding = {
        "version": "2.0.0",
        "encodings": [
            # Valid convolution weight - symmetric quantization
            {
                "name": "model.conv1.weight",
                "output_dtype": "int8",
                "y_scale": [0.01],
                "y_zero_point": [0],
            },
            # VIOLATION: Convolution weight with asymmetric quantization
            {
                "name": "model.conv2.weight",
                "output_dtype": "int8",
                "y_scale": [0.02],
                "y_zero_point": [10],
            },
            # Valid RMSNorm weight - 16-bit quantization
            {
                "name": "model.rmsnorm1.weight",
                "output_dtype": "int16",
                "y_scale": [0.001],
                "y_zero_point": [0],
            },
            # VIOLATION: RMSNorm weight with 8-bit quantization
            {
                "name": "model.rmsnorm2.weight",
                "output_dtype": "int8",
                "y_scale": [0.002],
                "y_zero_point": [0],
            },
        ],
    }
    return encoding


def example_1_basic_usage():
    """Example 1: Basic usage with default rules."""
    print("=" * 80)
    print("Example 1: Basic Usage with Default Rules")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample encoding file
        encoding = create_sample_encoding()
        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Create validator and run
        validator = ValidateEncodings()
        result = validator.run(
            encoding_file_path=encoding_file, output_dir=tmpdir / "validation_output"
        )

        # Print results
        print(f"\nTotal violations: {result['total_violations']}")
        print(f"JSON report: {result['json_report_path']}")
        print(f"CSV report: {result['csv_report_path']}")

        if result["violations"]:
            print("\nViolations found:")
            for violation in result["violations"]:
                print(f"  - {violation['tensor_name']}: {violation['rule_description']}")

    print()


def example_2_custom_rule():
    """Example 2: Adding a custom validation rule."""
    print("=" * 80)
    print("Example 2: Adding a Custom Validation Rule")
    print("=" * 80)

    def rule_custom_bitwidth(tensor_name: str, tensor: TensorEncoding) -> tuple:
        """Custom rule: All weights must use 8-bit quantization."""
        name = tensor_name.lower()
        if not ("weight" in name or "kernel" in name):
            return True, None

        is_valid = tensor.bitwidth == 8
        return is_valid, "All weights must use 8-bit quantization (custom rule)"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample encoding file
        encoding = create_sample_encoding()
        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Create validator and add custom rule
        validator = ValidateEncodings()
        validator.add_validation_rule(rule_custom_bitwidth)

        result = validator.run(
            encoding_file_path=encoding_file, output_dir=tmpdir / "validation_output"
        )

        print(f"\nTotal violations: {result['total_violations']}")
        if result["violations"]:
            print("\nViolations found:")
            for violation in result["violations"]:
                print(f"  - {violation['tensor_name']}: {violation['rule_description']}")

    print()


def example_3_rule_sets():
    """Example 3: Using different rule sets."""
    print("=" * 80)
    print("Example 3: Using Different Rule Sets")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample encoding file
        encoding = create_sample_encoding()
        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Test different rule sets
        rule_sets = ["default", "strict", "layer_type"]

        for rule_set_name in rule_sets:
            print(f"\n--- Using '{rule_set_name}' rule set ---")

            validator = ValidateEncodings()
            validator.use_rule_set(rule_set_name)

            result = validator.run(
                encoding_file_path=encoding_file,
                output_dir=tmpdir / f"validation_output_{rule_set_name}",
            )

            print(f"Total violations: {result['total_violations']}")

    print()


def example_4_selective_rules():
    """Example 4: Using only specific rules."""
    print("=" * 80)
    print("Example 4: Using Only Specific Rules")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample encoding file
        encoding = create_sample_encoding()
        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Create validator and clear default rules
        validator = ValidateEncodings()
        validator.clear_validation_rules()

        # Add only specific rules
        print("\nUsing only conv symmetric and 8-bit rules...")
        validator.add_validation_rule(rule_conv_weights_symmetric)
        validator.add_validation_rule(rule_all_weights_8bit)

        result = validator.run(
            encoding_file_path=encoding_file, output_dir=tmpdir / "validation_output"
        )

        print(f"\nTotal violations: {result['total_violations']}")
        if result["violations"]:
            print("\nViolations found:")
            for violation in result["violations"]:
                print(f"  - {violation['tensor_name']}: {violation['rule_description']}")

    print()


def example_5_programmatic_validation():
    """Example 5: Programmatic validation without file I/O."""
    print("=" * 80)
    print("Example 5: Programmatic Validation")
    print("=" * 80)

    from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample encoding file
        encoding = create_sample_encoding()
        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Load encoding manually
        model_encoding = ModelEncoding()
        model_encoding.load(artifact=encoding_file, load_json=True)

        # Create validator and validate
        validator = ValidateEncodings()
        violations = validator.validate(model_encoding)

        print(f"\nTotal violations: {len(violations)}")
        if violations:
            print("\nViolations found:")
            for violation in violations:
                print(f"  - {violation['tensor_name']}: {violation['rule_description']}")

    print()


def example_6_multiple_files():
    """Example 6: Validating multiple encoding files."""
    print("=" * 80)
    print("Example 6: Validating Multiple Encoding Files")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple encoding files
        encoding_files = []
        for i in range(3):
            encoding = create_sample_encoding()
            encoding_file = tmpdir / f"encoding_{i}.json"
            with open(encoding_file, "w") as f:
                json.dump(encoding, f, indent=2)
            encoding_files.append(encoding_file)

        # Validate all files
        validator = ValidateEncodings()
        results = {}

        for encoding_file in encoding_files:
            output_dir = tmpdir / "validation_results" / encoding_file.stem
            result = validator.run(encoding_file_path=encoding_file, output_dir=output_dir)
            results[encoding_file.name] = result["total_violations"]

        # Print summary
        print("\nValidation Summary:")
        for filename, violations in results.items():
            status = "✅ PASS" if violations == 0 else f"❌ FAIL ({violations} violations)"
            print(f"  {filename}: {status}")

    print()


def example_7_custom_scale_rule():
    """Example 7: Custom rule checking scale ranges."""
    print("=" * 80)
    print("Example 7: Custom Rule Checking Scale Ranges")
    print("=" * 80)

    def rule_scale_range(tensor_name: str, tensor: TensorEncoding) -> tuple:
        """Custom rule: Scales must be in range [0.001, 0.1]."""
        name = tensor_name.lower()
        if not ("weight" in name or "kernel" in name):
            return True, None

        if not tensor.scale:
            return True, None  # No scale to check

        is_valid = all(0.001 <= s <= 0.1 for s in tensor.scale)
        return is_valid, "Weight scales must be in range [0.001, 0.1]"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create encoding with out-of-range scales
        encoding = {
            "version": "2.0.0",
            "encodings": [
                {
                    "name": "model.conv1.weight",
                    "output_dtype": "int8",
                    "y_scale": [0.0005],  # Too small
                    "y_zero_point": [0],
                },
                {
                    "name": "model.conv2.weight",
                    "output_dtype": "int8",
                    "y_scale": [0.05],  # Valid
                    "y_zero_point": [0],
                },
                {
                    "name": "model.conv3.weight",
                    "output_dtype": "int8",
                    "y_scale": [0.2],  # Too large
                    "y_zero_point": [0],
                },
            ],
        }

        encoding_file = tmpdir / "encoding.json"
        with open(encoding_file, "w") as f:
            json.dump(encoding, f, indent=2)

        # Create validator with custom rule
        validator = ValidateEncodings()
        validator.clear_validation_rules()
        validator.add_validation_rule(rule_scale_range)

        result = validator.run(
            encoding_file_path=encoding_file, output_dir=tmpdir / "validation_output"
        )

        print(f"\nTotal violations: {result['total_violations']}")
        if result["violations"]:
            print("\nViolations found:")
            for violation in result["violations"]:
                print(f"  - {violation['tensor_name']}: {violation['rule_description']}")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ValidateEncodings Class Examples" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    examples = [
        example_1_basic_usage,
        example_2_custom_rule,
        example_3_rule_sets,
        example_4_selective_rules,
        example_5_programmatic_validation,
        example_6_multiple_files,
        example_7_custom_scale_rule,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
