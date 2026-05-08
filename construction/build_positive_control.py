"""
Positive control: a pickle file containing a __reduce__ method that invokes
os.system. Both Picklescan and ModelScan should flag this as a Critical
issue under their default rules. This validates that the scanners are
functioning correctly in the test environment.
"""
import os
import pickle

class MaliciousPayload:
    """Standard pickle-RCE pattern. Triggers when the pickle is loaded."""
    def __reduce__(self):
        return (os.system, ('echo "this should never run"',))

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/positive_control_malicious.pkl", "wb") as f:
        pickle.dump(MaliciousPayload(), f)
    print("Wrote artifacts/positive_control_malicious.pkl")
    print("This file contains a known-malicious pickle pattern (os.system via __reduce__).")
    print("Scanners with functional pickle-analysis MUST flag this.")
