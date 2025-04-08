import subprocess

def test_threat_classifier_runs():
    # Test that the script runs without errors
    result = subprocess.run(["python", "threat_classifier.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_threat_classifier_accuracy():
    # Test that the script produces a test accuracy >= 0.95
    result = subprocess.run(["python", "threat_classifier.py"], capture_output=True, text=True)
    output = result.stdout
    assert "Test Accuracy (after retraining):" in output, "Test accuracy not found in output"
    accuracy_line = [line for line in output.split("\n") if "Test Accuracy (after retraining):" in line][0]
    accuracy = float(accuracy_line.split(":")[1].strip())
    assert accuracy >= 0.95, f"Test accuracy {accuracy} is below 0.95"